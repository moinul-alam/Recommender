import logging
import time
import numpy as np
from scipy import sparse
from typing import Optional, Tuple, Dict, Iterator
import h5py
from sklearn.utils import shuffle
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ModelTraining:
    def __init__(
        self,
        n_factors: int = 50,
        learning_rate: float = 0.005,
        regularization: float = 0.02,
        n_epochs: int = 10,
        batch_size: int = 1024,
        early_stopping_patience: int = 3,
        validation_split: float = 0.1,
        chunk_size: int = 1000,
        temp_dir: str = './data/tmp'
    ):
        self._validate_parameters(n_factors, learning_rate, regularization, n_epochs, batch_size)
        
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.temp_dir = temp_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add progress tracking attributes
        self.training_start_time = None
        self.epoch_start_time = None
        self.batch_start_time = None
        self.total_batches = 0
        self.current_batch = 0
        
        
        # Create temporary directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        self.param_file = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def __del__(self):
        """Cleanup when the object is deleted."""
        self._cleanup_files()
    
    def _cleanup_files(self):
        """Clean up temporary HDF5 files."""
        try:
            if self.param_file is not None:
                self.param_file.close()
            
            # Remove temporary files if they exist
            temp_files = ['model_parameters.h5', 'best_parameters.h5']
            for file in temp_files:
                path = os.path.join(self.temp_dir, file)
                if os.path.exists(path):
                    os.remove(path)
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {str(e)}")
    
    @contextmanager
    def _parameter_file(self, filename: str, mode: str = 'r'):
        """Context manager for handling HDF5 files."""
        file_path = os.path.join(self.temp_dir, filename)
        file = h5py.File(file_path, mode)
        try:
            yield file
        finally:
            file.close()
    
    def _validate_parameters(
        self, 
        n_factors: int, 
        learning_rate: float, 
        regularization: float, 
        n_epochs: int,
        batch_size: int
    ) -> None:
        """Validate initialization parameters."""
        if not isinstance(n_factors, int) or n_factors < 1:
            raise ValueError(f"Number of factors must be an integer ≥ 1, got {n_factors}")
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError(f"Learning rate must be > 0, got {learning_rate}")
        if not isinstance(regularization, (int, float)) or regularization < 0:
            raise ValueError(f"Regularization must be ≥ 0, got {regularization}")
        if not isinstance(n_epochs, int) or n_epochs < 1:
            raise ValueError(f"Number of epochs must be an integer ≥ 1, got {n_epochs}")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"Batch size must be an integer ≥ 1, got {batch_size}")
    
    def _create_data_generator(self, matrix: sparse.csr_matrix) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Create generator that yields batches of data with improved shuffling."""
        if not sparse.isspmatrix_csr(matrix):
            raise ValueError("Input matrix must be in CSR format")
        
        user_indices, item_indices = matrix.nonzero()
        ratings = matrix.data
        
        # Create a random state for reproducibility
        rng = np.random.RandomState(42)
        
        # Pre-allocate arrays for efficiency
        n_samples = len(ratings)
        indices = np.arange(n_samples)
        
        # Yield batches with reshuffling for each epoch
        for start_idx in range(0, n_samples, self.batch_size):
            # Reshuffle every N batches (e.g., every 1000 batches)
            if start_idx % (self.batch_size * 1000) == 0:
                indices = rng.permutation(n_samples)
            
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield (user_indices[batch_indices],
                item_indices[batch_indices],
                ratings[batch_indices])

    def _initialize_parameters(self, n_users: int, n_items: int) -> None:
        """Initialize model parameters with memory-efficient HDF5 storage."""
        self.logger.info("Initializing model parameters...")
        
        # Close existing file if open
        if self.param_file is not None:
            self.param_file.close()
        
        # Create HDF5 file for storing parameters
        param_path = os.path.join(self.temp_dir, 'model_parameters.h5')
        self.param_file = h5py.File(param_path, 'w')
        
        # Initialize with smaller chunks for efficient access
        # limit = np.sqrt(6 / (n_users + n_items))
        limit = np.sqrt(1.0 / self.n_factors) # Xavier initialization
        chunk_size = min(self.chunk_size, n_users)
        
        # Create datasets with chunking
        self.param_file.create_dataset('user_factors', 
                                     shape=(n_users, self.n_factors),
                                     chunks=(chunk_size, self.n_factors),
                                     dtype='float32')
        self.param_file.create_dataset('item_factors', 
                                     shape=(n_items, self.n_factors),
                                     chunks=(chunk_size, self.n_factors),
                                     dtype='float32')
        
        # Initialize with random values
        for start_idx in range(0, n_users, chunk_size):
            end_idx = min(start_idx + chunk_size, n_users)
            self.param_file['user_factors'][start_idx:end_idx] = \
                np.random.uniform(-limit, limit, (end_idx - start_idx, self.n_factors))
        
        for start_idx in range(0, n_items, chunk_size):
            end_idx = min(start_idx + chunk_size, n_items)
            self.param_file['item_factors'][start_idx:end_idx] = \
                np.random.uniform(-limit, limit, (end_idx - start_idx, self.n_factors))
        
        # Biases can be kept in memory as they're much smaller
        self.user_biases = np.zeros(n_users, dtype='float32')
        self.item_biases = np.zeros(n_items, dtype='float32')
        self.global_bias = 0.0

    def fit(self, user_item_matrix: sparse.csr_matrix) -> Dict[str, list]:
        """Train the model using SGD with memory-efficient validation splitting."""
        self.training_start_time = time.time()
        self.logger.info("Starting model training...")
        self.logger.info(f"Matrix shape: {user_item_matrix.shape}, Non-zero elements: {user_item_matrix.nnz}")
        
        if not sparse.isspmatrix_csr(user_item_matrix):
            self.logger.info("Converting matrix to CSR format...")
            user_item_matrix = user_item_matrix.tocsr()
        
        if user_item_matrix.nnz == 0:
            raise ValueError("Input matrix is empty")
        
        n_users, n_items = user_item_matrix.shape
        self.logger.info(f"Initializing model with {n_users} users and {n_items} items")
        self._initialize_parameters(n_users, n_items)
        
        self.global_bias = float(user_item_matrix.data.mean())
        self.logger.info(f"Global bias initialized to {self.global_bias:.4f}")
        
        history = {'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        min_improvement_ratio = 0.001 # Minimum improvement ratio for early stopping
        
        # Create validation split
        self.logger.info("Creating train-validation split...")
        n_interactions = user_item_matrix.nnz
        n_val = int(n_interactions * self.validation_split)
        self.logger.info(f"Using {n_val} interactions for validation ({self.validation_split:.1%})")
        
        # Generate validation indices
        all_indices = np.arange(n_interactions)
        np.random.shuffle(all_indices)
        val_indices = all_indices[:n_val]
        train_indices = all_indices[n_val:]
        
        self.logger.info("Creating validation matrix...")
        t0 = time.time()
        val_matrix = self._create_split_matrix(user_item_matrix, val_indices, n_users)
        self.logger.info(f"Validation matrix created in {time.time() - t0:.2f}s")
        
        self.logger.info("Creating training matrix...")
        t0 = time.time()
        train_matrix = self._create_split_matrix(user_item_matrix, train_indices, n_users)
        self.logger.info(f"Training matrix created in {time.time() - t0:.2f}s")
        
        # Calculate total batches for progress tracking
        self.total_batches = train_matrix.nnz // self.batch_size + (1 if train_matrix.nnz % self.batch_size else 0)
        
        try:
            for epoch in range(self.n_epochs):
                self.epoch_start_time = time.time()
                
                # Train using mini-batches
                train_loss = self._train_epoch(train_matrix)
                val_loss = self._compute_validation_loss(val_matrix)
                
                history['loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                
                # Early stopping with relative improvement threshold
                if val_loss < best_val_loss:
                    relative_improvement = (best_val_loss - val_loss) / best_val_loss
                    if relative_improvement > min_improvement_ratio:
                        self.logger.info(f"Validation loss improved by {relative_improvement:.2%}")
                        best_val_loss = val_loss
                        patience_counter = 0
                        self._save_best_params()
                    else:
                        patience_counter += 1
                        self.logger.info(
                            f"Insufficient improvement ({relative_improvement:.2%}). "
                            f"Patience: {patience_counter}/{self.early_stopping_patience}"
                        )
                else:
                    patience_counter += 1
                    self.logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{self.early_stopping_patience}")
                
                if patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    self._load_best_params()
                    break
                
                # Log epoch results
                epoch_time = time.time() - self.epoch_start_time
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.n_epochs} completed in {epoch_time:.2f}s - "
                    f"loss: {train_loss:.4f} - val_loss: {val_loss:.4f}"
                )
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
    
    def _create_split_matrix(self, matrix: sparse.csr_matrix, indices: np.ndarray, n_users: int) -> sparse.csr_matrix:
        """Helper method to create split matrices with progress logging."""
        data = matrix.data[indices]
        rows = matrix.indices[indices]
        cols = np.empty_like(indices)
        current_idx = 0
        
        # Log progress every 5%
        progress_interval = max(n_users // 20, 1)
        last_progress = 0
        
        for i, row in enumerate(range(n_users)):
            row_start = matrix.indptr[row]
            row_end = matrix.indptr[row + 1]
            mask = (indices >= row_start) & (indices < row_end)
            n_mask = mask.sum()
            cols[current_idx:current_idx + n_mask] = row
            current_idx += n_mask
            
            # Log progress
            if (i + 1) % progress_interval == 0 and (i + 1) // progress_interval > last_progress:
                last_progress = (i + 1) // progress_interval
                self.logger.info(f"Processing users: {(i + 1) / n_users:.1%} complete")
        
        return sparse.csr_matrix(
            (data, (cols, rows)),
            shape=matrix.shape
        )
    
    def _process_batch(self, user_batch, item_batch, rating_batch):
        """Process a single batch with improved numerical stability."""
        # Convert batch indices to sorted lists for HDF5 compatibility
        unique_users, user_inv = np.unique(user_batch, return_inverse=True)
        unique_items, item_inv = np.unique(item_batch, return_inverse=True)
        
        # Load relevant factors using unique indices
        user_factors_all = self.param_file['user_factors'][unique_users]
        item_factors_all = self.param_file['item_factors'][unique_items]
        
        # Map back to original batch size using inverse indices
        user_factors_batch = user_factors_all[user_inv]
        item_factors_batch = item_factors_all[item_inv]
        
        # Check for and handle NaN values in loaded factors
        if np.any(np.isnan(user_factors_batch)) or np.any(np.isnan(item_factors_batch)):
            self.logger.warning("NaN values detected in factors, replacing with zeros")
            user_factors_batch = np.nan_to_num(user_factors_batch)
            item_factors_batch = np.nan_to_num(item_factors_batch)
        
        # Compute predictions with improved numerical stability
        dot_products = np.clip(
            np.sum(user_factors_batch * item_factors_batch, axis=1),
            -1e3, 1e3
        )
        
        predictions = (
            self.global_bias +
            self.user_biases[user_batch] +
            self.item_biases[item_batch] +
            dot_products
        )
        
        # Clip predictions to reasonable rating range
        predictions = np.clip(predictions, -10, 10)
        errors = rating_batch - predictions
        
        # Compute updates with improved gradient clipping
        lr = self.learning_rate
        reg = self.regularization
        clip_threshold = 5.0  # Reduced from 10.0 for better stability
        
        # Compute gradient updates more efficiently
        user_factors_grad = errors[:, np.newaxis] * item_factors_batch - reg * user_factors_batch
        item_factors_grad = errors[:, np.newaxis] * user_factors_batch - reg * item_factors_batch
        
        # Clip gradients before aggregating
        user_factors_grad = np.clip(user_factors_grad, -clip_threshold, clip_threshold)
        item_factors_grad = np.clip(item_factors_grad, -clip_threshold, clip_threshold)
        
        # Use np.bincount for more efficient gradient aggregation
        user_updates = np.zeros_like(user_factors_all)
        for f in range(self.n_factors):
            np.add.at(user_updates[:, f], user_inv, user_factors_grad[:, f])
        
        item_updates = np.zeros_like(item_factors_all)
        for f in range(self.n_factors):
            np.add.at(item_updates[:, f], item_inv, item_factors_grad[:, f])
        
        # Apply updates with bounds checking
        curr_user_factors = self.param_file['user_factors'][unique_users]
        curr_item_factors = self.param_file['item_factors'][unique_items]
        
        self.param_file['user_factors'][unique_users] = np.clip(
            curr_user_factors + user_updates, -50, 50  # Tighter bounds for better stability
        )
        self.param_file['item_factors'][unique_items] = np.clip(
            curr_item_factors + item_updates, -50, 50
        )
        
        # Update biases with improved clipping
        user_bias_updates = np.clip(lr * (errors - reg * self.user_biases[user_batch]), -0.5, 0.5)
        item_bias_updates = np.clip(lr * (errors - reg * self.item_biases[item_batch]), -0.5, 0.5)
        
        self.user_biases[user_batch] += user_bias_updates
        self.item_biases[item_batch] += item_bias_updates
        
        # Tighter clipping for biases
        self.user_biases = np.clip(self.user_biases, -3, 3)
        self.item_biases = np.clip(self.item_biases, -3, 3)
        
        # Return mean squared error with safety checks
        mse = np.mean(np.square(errors))
        if np.isnan(mse) or np.isinf(mse):
            self.logger.warning("NaN or inf encountered in loss calculation")
            return 1.0
        
        return mse


    def _train_epoch(self, matrix: sparse.csr_matrix) -> float:
        """Train one epoch using mini-batch SGD with progress tracking."""
        total_loss = 0
        n_batches = 0
        
        # Reset batch counter
        self.current_batch = 0
        batch_start_time = time.time()
        last_log_time = batch_start_time
        
        for user_batch, item_batch, rating_batch in self._create_data_generator(matrix):
            try:
                self.current_batch += 1
                current_time = time.time()
                
                # Log progress every 30 seconds or for the first batch
                if current_time - last_log_time > 30 or self.current_batch == 1:
                    progress = self.current_batch / self.total_batches
                    elapsed_time = current_time - batch_start_time
                    estimated_total = elapsed_time / progress if progress > 0 else 0
                    remaining_time = max(0, estimated_total - elapsed_time)
                    
                    self.logger.info(
                        f"Training progress: {progress:.1%} - "
                        f"Batch {self.current_batch}/{self.total_batches} - "
                        f"Elapsed: {elapsed_time:.0f}s - "
                        f"Remaining: {remaining_time:.0f}s"
                    )
                    last_log_time = current_time
                
                # Process batch
                batch_loss = self._process_batch(user_batch, item_batch, rating_batch)
                total_loss += batch_loss
                n_batches += 1
                
            except Exception as e:
                self.logger.error(f"Error in batch {self.current_batch}: {str(e)}")
                raise
        
        return total_loss / n_batches if n_batches > 0 else float('inf')
    
    def _compute_validation_loss(self, val_matrix: sparse.csr_matrix) -> float:
        """Compute validation loss with improved numerical stability."""
        total_loss = 0
        n_batches = 0
        
        for user_batch, item_batch, rating_batch in self._create_data_generator(val_matrix):
            try:
                # Convert indices to unique form for efficient HDF5 access
                unique_users, user_inv = np.unique(user_batch, return_inverse=True)
                unique_items, item_inv = np.unique(item_batch, return_inverse=True)
                
                # Load factors and handle potential NaN values
                user_factors_all = np.nan_to_num(self.param_file['user_factors'][unique_users])
                item_factors_all = np.nan_to_num(self.param_file['item_factors'][unique_items])
                
                # Map back to original indices
                user_factors = user_factors_all[user_inv]
                item_factors = item_factors_all[item_inv]
                
                # Compute predictions with clipping
                dot_products = np.clip(
                    np.sum(user_factors * item_factors, axis=1),
                    -1e3, 1e3
                )
                
                predictions = np.clip(
                    self.global_bias +
                    self.user_biases[user_batch] +
                    self.item_biases[item_batch] +
                    dot_products,
                    -10, 10
                )
                
                # Compute MSE loss with safety checks
                errors = rating_batch - predictions
                batch_loss = np.mean(np.square(errors))
                
                if np.isnan(batch_loss) or np.isinf(batch_loss):
                    self.logger.warning("NaN or inf encountered in validation batch")
                    batch_loss = 1.0
                
                total_loss += batch_loss
                n_batches += 1
                
            except Exception as e:
                self.logger.error(f"Error in validation batch: {str(e)}")
                raise
        
        return total_loss / n_batches if n_batches > 0 else float('inf')

    

    def _save_best_params(self) -> None:
        """Save current parameters as the best parameters."""
        self.logger.info("Saving best parameters...")
        
        best_path = os.path.join(self.temp_dir, 'best_parameters.h5')
        with h5py.File(best_path, 'w') as best_file:
            try:
                # Copy current parameters
                best_file.create_dataset('user_factors', data=self.param_file['user_factors'][:])
                best_file.create_dataset('item_factors', data=self.param_file['item_factors'][:])
                best_file.create_dataset('user_biases', data=self.user_biases)
                best_file.create_dataset('item_biases', data=self.item_biases)
                best_file.attrs['global_bias'] = self.global_bias
            except Exception as e:
                self.logger.error(f"Error saving best parameters: {str(e)}")
                raise

    def _load_best_params(self) -> None:
        """Load the best parameters back into the model."""
        self.logger.info("Loading best parameters...")
        
        best_path = os.path.join(self.temp_dir, 'best_parameters.h5')
        try:
            with h5py.File(best_path, 'r') as best_file:
                # Copy back to current parameters
                self.param_file['user_factors'][:] = best_file['user_factors'][:]
                self.param_file['item_factors'][:] = best_file['item_factors'][:]
                self.user_biases = best_file['user_biases'][:]
                self.item_biases = best_file['item_biases'][:]
                self.global_bias = best_file.attrs['global_bias']
        except Exception as e:
            self.logger.error(f"Error loading best parameters: {str(e)}")
            raise

    def save(self, path: str) -> None:
        """Save model parameters to disk."""
        try:
            with h5py.File(path, 'w') as f:
                # Save model parameters
                f.create_dataset('user_factors', data=self.param_file['user_factors'][:])
                f.create_dataset('item_factors', data=self.param_file['item_factors'][:])
                f.create_dataset('user_biases', data=self.user_biases)
                f.create_dataset('item_biases', data=self.item_biases)
                
                # Save model configuration
                f.attrs['global_bias'] = self.global_bias
                f.attrs['n_factors'] = self.n_factors
                f.attrs['learning_rate'] = self.learning_rate
                f.attrs['regularization'] = self.regularization
                
            self.logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
        
    def load(self, path: str) -> None:
        """Load model parameters from disk."""
        try:
            with h5py.File(path, 'r') as f:
                # Close existing parameter file if open
                if self.param_file is not None:
                    self.param_file.close()
                
                # Create new parameter file
                param_path = os.path.join(self.temp_dir, 'model_parameters.h5')
                self.param_file = h5py.File(param_path, 'w')
                
                # Load model parameters
                self.param_file.create_dataset('user_factors', data=f['user_factors'][:])
                self.param_file.create_dataset('item_factors', data=f['item_factors'][:])
                self.user_biases = f['user_biases'][:]
                self.item_biases = f['item_biases'][:]
                
                # Load model configuration
                self.global_bias = f.attrs['global_bias']
                self.n_factors = f.attrs['n_factors']
                self.learning_rate = f.attrs['learning_rate']
                self.regularization = f.attrs['regularization']
                
            self.logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, user_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        """Predict ratings for given user-item pairs."""
        try:
            if len(user_indices) != len(item_indices):
                raise ValueError("Length of user_indices and item_indices must match")
            
            # Convert indices to unique form for efficient HDF5 access
            unique_users, user_inv = np.unique(user_indices, return_inverse=True)
            unique_items, item_inv = np.unique(item_indices, return_inverse=True)
            
            # Load factors for unique indices
            user_factors_all = self.param_file['user_factors'][unique_users]
            item_factors_all = self.param_file['item_factors'][unique_items]
            
            # Map back to original indices
            user_factors = user_factors_all[user_inv]
            item_factors = item_factors_all[item_inv]
            
            # Compute predictions
            predictions = (
                self.global_bias +
                self.user_biases[user_indices] +
                self.item_biases[item_indices] +
                np.sum(user_factors * item_factors, axis=1)
            )
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise
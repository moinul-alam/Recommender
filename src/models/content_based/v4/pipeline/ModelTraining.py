import numpy as np
import pandas as pd
import gc
import logging
import memory_profiler
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from typing import Dict, List, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class ModelTraining:
    def __init__(
        self,
        feature_matrix: sparse.csr_matrix,
        item_ids: np.ndarray,
        n_components_svd: int = 300,
        random_state: Optional[int] = 42
    ):
        """
        Initialize model training with configurable parameters.
        
        Args:
            feature_matrix: Sparse matrix of features
            item_ids: Array of item identifiers
            n_components_svd: Number of components for SVD
            random_state: Random seed for reproducibility
        """
        logger.info("Initializing ModelTraining")
        
        # Log memory usage at initialization
        mem_usage = memory_profiler.memory_usage()[0]
        logger.info(f"Memory usage at initialization: {mem_usage:.2f} MB")
        
        # Model parameters
        self.random_state = random_state
        self.feature_matrix = feature_matrix
        self.item_ids = item_ids
        self.n_components_svd = min(n_components_svd, feature_matrix.shape[1], feature_matrix.shape[0])
        
        # Initialize SVD model
        self.svd = TruncatedSVD(
            n_components=self.n_components_svd, 
            random_state=self.random_state
        )
        
        self.is_fitted = False
        
        logger.info(f"Model Training initialized with {self.n_components_svd} components")
        logger.info(f"Feature matrix shape: {self.feature_matrix.shape}")
        logger.info(f"Item IDs shape: {self.item_ids.shape}")

    def fit(self) -> None:
        """
        Fit SVD model on the feature matrix.
        """
        logger.info("Fitting SVD model...")
        
        try:
            # Fit the SVD model
            self.svd.fit(self.feature_matrix)
            self.is_fitted = True
            
            # Log explained variance
            explained_variance = np.sum(self.svd.explained_variance_ratio_)
            logger.info(f"SVD fitting complete. Explained variance: {explained_variance:.4f}")
            
            # Log memory usage after fitting
            mem_usage = memory_profiler.memory_usage()[0]
            logger.info(f"Memory usage after fitting SVD: {mem_usage:.2f} MB")
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error fitting SVD model: {str(e)}", exc_info=True)
            raise

    def transform(self) -> np.ndarray:
        """
        Transform features using fitted SVD.
        
        Returns:
            Transformed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("SVD model must be fitted before transformation")
            
        try:
            logger.info("Transforming feature matrix using SVD...")
            transformed_features = self.svd.transform(self.feature_matrix)
            
            logger.info(f"Transformation complete. Shape of transformed features: {transformed_features.shape}")
            mem_usage = memory_profiler.memory_usage()[0]
            logger.info(f"Memory usage after transformation: {mem_usage:.2f} MB")
            
            return transformed_features

        except Exception as e:
            logger.error(f"Error transforming features: {str(e)}", exc_info=True)
            raise

    def fit_transform(self) -> np.ndarray:
        """
        Fit SVD model and transform feature matrix in one step.
        
        Returns:
            Transformed feature matrix
        """
        logger.info("Performing fit_transform operation...")
        
        try:
            # Directly use SVD's fit_transform for better efficiency
            transformed_features = self.svd.fit_transform(self.feature_matrix)
            self.is_fitted = True
            
            # Log explained variance
            explained_variance = np.sum(self.svd.explained_variance_ratio_)
            logger.info(f"SVD fit_transform complete. Explained variance: {explained_variance:.4f}")
            logger.info(f"Shape of transformed features: {transformed_features.shape}")
            
            # Force garbage collection
            gc.collect()
            
            return transformed_features
            
        except Exception as e:
            logger.error(f"Error in fit_transform: {str(e)}", exc_info=True)
            raise

    def fit_transform_and_combine(self) -> pd.DataFrame:
        """
        Fit SVD model, transform feature matrix, and combine with item IDs.
        
        Returns:
            DataFrame with item IDs and transformed features
        """
        logger.info("Performing fit, transform, and combining with item IDs...")
        
        try:
            # Use fit_transform for better efficiency
            transformed_features = self.fit_transform()
            
            # Create column names for the transformed features
            feature_columns = [f"feature_{i}" for i in range(self.n_components_svd)]
            
            # First create DataFrame from transformed features
            df_transformed = pd.DataFrame(
                data=transformed_features,
                columns=feature_columns
            )
            
            # Add item IDs as the first column
            df_transformed.insert(0, 'item_id', self.item_ids)
            
            logger.info(f"Successfully combined item IDs with transformed features. DataFrame shape: {df_transformed.shape}")
            
            # Convert data to float32 to reduce memory and prepare for FAISS
            for col in feature_columns:
                df_transformed[col] = df_transformed[col].astype(np.float32)
            
            # Force garbage collection
            del transformed_features
            gc.collect()
            
            return df_transformed
            
        except Exception as e:
            logger.error(f"Error combining item IDs with transformed features: {str(e)}", exc_info=True)
            raise

    def get_model_metadata(self) -> Dict:
        """
        Get metadata about the trained SVD model.
        
        Returns:
            Dictionary containing model metadata
        """
        if not self.is_fitted:
            raise ValueError("SVD model must be fitted before getting metadata")
            
        try:
            metadata = {
                "n_components": self.n_components_svd,
                "explained_variance_ratio": self.svd.explained_variance_ratio_.tolist(),
                "explained_variance": self.svd.explained_variance_.tolist(),
                "singular_values": self.svd.singular_values_.tolist(),
                "n_features": self.svd.n_features_in_,
                "random_state": self.random_state,
                "feature_matrix_shape": self.feature_matrix.shape,
                "item_ids_shape": self.item_ids.shape
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting model metadata: {str(e)}", exc_info=True)
            raise

    def get_model_summary(self) -> Dict:
        """
        Get a summary of the trained model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_fitted:
            raise ValueError("SVD model must be fitted before getting summary")
        
        try:
            return {
                "n_components": self.n_components_svd,
                "explained_variance": np.sum(self.svd.explained_variance_ratio_),
                "feature_matrix_shape": self.feature_matrix.shape,
                "item_ids_count": len(self.item_ids)
            }
        except Exception as e:
            logger.error(f"Error getting model summary: {str(e)}", exc_info=True)
            raise
    
    def prepare_for_faiss(self) -> np.ndarray:
        """
        Prepare transformed data for FAISS index creation.
        
        Returns:
            Transformed features as float32 numpy array
        """
        if not self.is_fitted:
            raise ValueError("SVD model must be fitted before preparing for FAISS")
        
        try:
            logger.info("Preparing data for FAISS indexing...")
            
            # Transform the feature matrix
            transformed_features = self.transform()
            
            # Convert to float32 for FAISS compatibility
            transformed_features_float32 = transformed_features.astype(np.float32)
            
            logger.info(f"Prepared features for FAISS with shape: {transformed_features_float32.shape}")
            
            return transformed_features_float32
            
        except Exception as e:
            logger.error(f"Error preparing for FAISS: {str(e)}", exc_info=True)
            raise
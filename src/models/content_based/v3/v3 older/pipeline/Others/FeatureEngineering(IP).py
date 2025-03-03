import numpy as np
import pandas as pd
import gc
import logging
from pathlib import Path
import psutil
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import IncrementalPCA
import joblib
from scipy import sparse
from typing import Dict, Optional, List, Tuple
import time
import memory_profiler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        max_cast_members: int = 20,
        max_directors: int = 3,
        n_components_pca: int = 200,
        batch_size: int = 5000  # Added batch_size parameter
    ):
        self.metrics = {
            'fit_time': 0,
            'transform_time': 0,
            'peak_memory': 0,
            'tfidf_dimensions': 0,
            'final_dimensions': 0,
            'input_dimensions': 0,
            'memory_usage_stages': {}
        }
        
        self.weights = weights or {
            "overview": 0.45,
            "genres": 0.45,
            "keywords": 0.04,
            "cast": 0.04, 
            "director": 0.02
        }
        self._validate_weights()
        
        self.max_cast_members = max_cast_members
        self.max_directors = max_directors
        self.is_fitted = False
        self.n_components_pca = n_components_pca
        self.batch_size = batch_size
        
        self._initialize_transformers()
        
    def _initialize_transformers(self):
        """Initialize all necessary transformers with improved configurations."""
        self.mlb_genres = MultiLabelBinarizer(sparse_output=True)
        
        # Improved TF-IDF configurations with stricter parameters
        tfidf_params = {
            'stop_words': 'english',
            'ngram_range': (1, 2),
            'dtype': np.float32,
            'norm': 'l2',
            'min_df': 5,  # Increased from 3
            'max_df': 0.95  # Added max_df to remove very common terms
        }
        
        self.tfidf_keywords = TfidfVectorizer(
            max_features=1000,  # Reduced from original
            **tfidf_params
        )
        self.tfidf_overview = TfidfVectorizer(
            max_features=2000,  # Reduced from 2000
            **tfidf_params
        )
        
        # Reduced feature hasher dimensions
        self.cast_hasher = FeatureHasher(
            n_features=1000,  # Reduced from 1000
            input_type='string',
            alternate_sign=False
        )
        self.director_hasher = FeatureHasher(
            n_features=200,  # Reduced from 200
            input_type='string',
            alternate_sign=False
        )
        
        self.pca = IncrementalPCA(
            n_components=self.n_components_pca,
            batch_size=self.batch_size
        )
    
    def _log_memory_status(self, stage: str) -> Tuple[float, float]:
        """Log memory usage at a given stage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        rss_mb = memory_info.rss / 1024 / 1024
        vms_mb = memory_info.vms / 1024 / 1024
        
        logger.info(f"Memory usage at {stage}:")
        logger.info(f"RSS: {rss_mb:.2f}MB")
        logger.info(f"VMS: {vms_mb:.2f}MB")
        
        self.metrics['memory_usage_stages'][stage] = {
            'rss': rss_mb,
            'vms': vms_mb,
            'timestamp': time.time()
        }
        
        return rss_mb, vms_mb
    
    def _validate_weights(self) -> None:
        """Validate feature weights."""
        if not isinstance(self.weights, dict):
            raise TypeError("Weights must be a dictionary")
        
        if not all(isinstance(v, (int, float)) for v in self.weights.values()):
            raise TypeError("All weights must be numeric")
            
        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")

    def _validate_input_data(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame structure and content."""
        required_columns = {'tmdb_id', 'overview', 'genres', 'keywords', 'cast', 'director'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df['tmdb_id'].duplicated().any():
            raise ValueError("Duplicate tmdb_id values found")
            
        if df.isna()['tmdb_id'].any():
            raise ValueError("tmdb_id contains null values")
            
        logger.info(f"Input data validation complete. Shape: {df.shape}")

    def _process_text_list(self, text: str, max_items: Optional[int]) -> List[str]:
        """Process comma-separated text into a list with maximum items."""
        if pd.isna(text) or not text:
            return []
        items = [item.strip() for item in text.split(',') if item.strip()]
        return items[:max_items] if max_items else items
    
    def _combine_features(self, feature_dict: Dict[str, sparse.csr_matrix]) -> sparse.csr_matrix:
        """Combine feature matrices with weights but WITHOUT final normalization."""
        logger.info("Combining feature matrices")
        feature_shapes = {name: matrix.shape for name, matrix in feature_dict.items()}
        logger.info(f"Feature matrix shapes: {feature_shapes}")
        
        weighted_matrices = [
            matrix * self.weights[feature_name]
            for feature_name, matrix in feature_dict.items()
        ]
        
        combined = sparse.hstack(weighted_matrices).tocsr()
        logger.info(f"Combined feature matrix shape: {combined.shape}")
        
        return combined.astype(np.float32)

    @memory_profiler.profile
    def fit_transformers(self, df: pd.DataFrame) -> None:
        """Fit all transformers with improved memory management and error handling."""
        start_time = time.time()
        self._log_memory_status("start_fit")
        
        try:
            self._validate_input_data(df)
            feature_dict = {}
            
            # Fit and transform each feature type
            for feature_name, processor, data in [
                ("overview", self.tfidf_overview, df['overview'].fillna('')),
                ("keywords", self.tfidf_keywords, df['keywords'].fillna('')),
                ("genres", self.mlb_genres, [self._process_text_list(g, None) for g in df['genres']]),
                ("cast", self.cast_hasher, [self._process_text_list(c, self.max_cast_members) for c in df['cast']]),
                ("director", self.director_hasher, [self._process_text_list(d, self.max_directors) for d in df['director']])
            ]:
                logger.info(f"Processing {feature_name}")
                feature_start_time = time.time()
                feature_dict[feature_name] = processor.fit_transform(data)
                logger.info(f"{feature_name} processing completed in {time.time() - feature_start_time:.2f}s")
                self._log_memory_status(f"after_{feature_name}")
            
            # Combine features
            logger.info("Combining features")
            combined_features = self._combine_features(feature_dict)
            self._log_memory_status("after_combine")
            
            # Record dimensions
            self.metrics['input_dimensions'] = combined_features.shape[1]
            self.metrics['tfidf_dimensions'] = feature_dict['overview'].shape[1]
            
            # Fit PCA in batches with detailed logging
            logger.info("Starting PCA fitting process")
            n_samples = combined_features.shape[0]
            n_features = combined_features.shape[1]
            n_batches = (n_samples + self.batch_size - 1) // self.batch_size
            
            logger.info(f"PCA Configuration:")
            logger.info(f"Total samples: {n_samples:,}")
            logger.info(f"Input features: {n_features:,}")
            logger.info(f"Target dimensions: {self.n_components_pca}")
            logger.info(f"Batch size: {self.batch_size:,}")
            logger.info(f"Number of batches: {n_batches}")
            
            pca_start_time = time.time()
            
            for batch_idx, start_idx in enumerate(range(0, n_samples, self.batch_size)):
                batch_start_time = time.time()
                end_idx = min(start_idx + self.batch_size, n_samples)
                
                logger.info(f"Processing batch {batch_idx + 1}/{n_batches}")
                logger.info(f"Samples {start_idx:,} to {end_idx:,}")
                
                # Convert sparse to dense array
                dense_conversion_start = time.time()
                batch = combined_features[start_idx:end_idx].toarray()
                logger.info(f"Dense conversion took: {time.time() - dense_conversion_start:.2f}s")
                
                # Fit PCA
                pca_fit_start = time.time()
                self.pca.partial_fit(batch)
                logger.info(f"PCA fit took: {time.time() - pca_fit_start:.2f}s")
                
                # Log memory and timing
                self._log_memory_status(f"pca_batch_{batch_idx}")
                batch_time = time.time() - batch_start_time
                remaining_batches = n_batches - batch_idx - 1
                estimated_remaining = batch_time * remaining_batches
                
                logger.info(f"Batch {batch_idx + 1} completed in {batch_time:.2f}s")
                logger.info(f"Estimated time remaining: {estimated_remaining:.2f}s")
                
                # Force garbage collection after each batch
                gc.collect()
            
            self.is_fitted = True
            total_time = time.time() - start_time
            self.metrics['fit_time'] = total_time
            self.metrics['final_dimensions'] = self.n_components_pca
            
            logger.info(f"Fitting complete in {total_time:.2f}s")
            logger.info(f"Metrics: {self.metrics}")
            
        except Exception as e:
            logger.error(f"Error in fit_transformers: {str(e)}", exc_info=True)
            raise
        finally:
            gc.collect()

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features with improved error handling and memory management."""
        if not self.is_fitted:
            raise ValueError("Transformers must be fitted before transformation")
            
        start_time = time.time()
        self._log_memory_status("start_transform")
        
        try:
            self._validate_input_data(df)
            feature_dict = {}
            
            # Transform each feature type
            for feature_name, processor, data in [
                ("overview", self.tfidf_overview, df['overview'].fillna('')),
                ("keywords", self.tfidf_keywords, df['keywords'].fillna('')),
                ("genres", self.mlb_genres, [self._process_text_list(g, None) for g in df['genres']]),
                ("cast", self.cast_hasher, [self._process_text_list(c, self.max_cast_members) for c in df['cast']]),
                ("director", self.director_hasher, [self._process_text_list(d, self.max_directors) for d in df['director']])
            ]:
                logger.info(f"Transforming {feature_name}")
                feature_start_time = time.time()
                feature_dict[feature_name] = processor.transform(data)
                logger.info(f"{feature_name} transform completed in {time.time() - feature_start_time:.2f}s")
                self._log_memory_status(f"transform_{feature_name}")
            
            # Combine and transform features
            combined_features = self._combine_features(feature_dict)
            self._log_memory_status("after_combine_transform")
            
            # Transform with PCA in batches
            n_samples = combined_features.shape[0]
            transformed_features_list = []
            
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch = combined_features[start_idx:end_idx].toarray()
                batch_transformed = self.pca.transform(batch)
                transformed_features_list.append(batch_transformed)
                self._log_memory_status(f"pca_transform_batch_{start_idx//self.batch_size}")
                gc.collect()
            
            transformed_features = np.vstack(transformed_features_list)
            
            # Create output DataFrame
            result = pd.DataFrame(
                transformed_features,
                columns=[f'feature_{i}' for i in range(transformed_features.shape[1])],
                index=df.index
            )
            result.insert(0, 'tmdb_id', df['tmdb_id'])
            
            self.metrics['transform_time'] = time.time() - start_time
            logger.info(f"Transform completed in {self.metrics['transform_time']:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in transform_features: {str(e)}", exc_info=True)
            raise
        finally:
            gc.collect()

    def save_transformers(self, path: str) -> None:
        """Save all transformers to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted transformers")
            
        logger.info(f"Saving transformers to: {path}")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Only save stateful transformers
        transformers = {
            'mlb_genres': self.mlb_genres,
            'tfidf_keywords': self.tfidf_keywords,
            'tfidf_overview': self.tfidf_overview,
            'pca': self.pca
        }

        for name, transformer in transformers.items():
            joblib.dump(transformer, path / f"{name}.pkl")
        
        # Save configuration
        config = {
            'weights': self.weights,
            'max_cast_members': self.max_cast_members,
            'max_directors': self.max_directors,
            'is_fitted': self.is_fitted
        }
        joblib.dump(config, path / "config.pkl")
        logger.info("Transformers saved successfully")

    def load_transformers(self, path: str) -> None:
        """Load all transformers from disk."""
        logger.info(f"Loading transformers from: {path}")
        path = Path(path)
        
        try:
            # Load configuration
            config = joblib.load(path / "config.pkl")
            self.weights = config['weights']
            self.max_cast_members = config['max_cast_members']
            self.max_directors = config['max_directors']
            self.is_fitted = config['is_fitted']
            
            # Load only stateful transformers
            self.mlb_genres = joblib.load(path / "mlb_genres.pkl")
            self.tfidf_keywords = joblib.load(path / "tfidf_keywords.pkl")
            self.tfidf_overview = joblib.load(path / "tfidf_overview.pkl")
            self.pca = joblib.load(path / "pca.pkl")
            
            # Reinitialize FeatureHashers (they are stateless)
            self.cast_hasher = FeatureHasher(n_features=500, input_type='string')
            self.director_hasher = FeatureHasher(n_features=100, input_type='string')
            
            logger.info("Transformers loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading transformers: {str(e)}", exc_info=True)
            raise
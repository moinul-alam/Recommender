import numpy as np
import pandas as pd
import gc
import logging
from pathlib import Path
import psutil
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MultiLabelBinarizer, normalize
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
        batch_size: int = 5000
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
        
        tfidf_params = {
            'stop_words': 'english',
            'ngram_range': (1, 2),
            'dtype': np.float32,
            'norm': 'l2'
        }
        
        self.tfidf_keywords = TfidfVectorizer(
            max_features=1000,
            **tfidf_params
        )
        self.tfidf_overview = TfidfVectorizer(
            max_features=2000,
            **tfidf_params
        )
        
        self.cast_hasher = FeatureHasher(
            n_features=500,
            input_type='string',
            alternate_sign=False
        )
        self.director_hasher = FeatureHasher(
            n_features=200,
            input_type='string',
            alternate_sign=False
        )
        
        self.pca = IncrementalPCA(
            n_components=self.n_components_pca,
            batch_size=self.batch_size,
            whiten=True  # Added for better distance preservation
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
        """Combine feature matrices with weights and L2 normalization."""
        logger.info("Combining feature matrices")
        feature_shapes = {name: matrix.shape for name, matrix in feature_dict.items()}
        logger.info(f"Feature matrix shapes: {feature_shapes}")
        
        weighted_matrices = [
            matrix * self.weights[feature_name]
            for feature_name, matrix in feature_dict.items()
        ]
        
        combined = sparse.hstack(weighted_matrices).tocsr()
        
        # Convert to dense for proper L2 normalization
        dense_combined = combined.toarray()
        normalized = normalize(dense_combined, norm='l2', axis=1)
        
        logger.info(f"Combined and normalized feature matrix shape: {normalized.shape}")
        
        return sparse.csr_matrix(normalized, dtype=np.float32)
    
    @memory_profiler.profile
    def fit_transformers(self, df: pd.DataFrame) -> None:
        """Fit all transformers with improved handling for L2 distance computation."""
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
            
            # Combine features with L2 normalization
            combined_features = self._combine_features(feature_dict)
            self._log_memory_status("after_combine")
            
            # Record dimensions
            self.metrics['input_dimensions'] = combined_features.shape[1]
            self.metrics['tfidf_dimensions'] = feature_dict['overview'].shape[1]
            
            # Convert to dense and normalize for PCA
            if sparse.issparse(combined_features):
                combined_features = combined_features.toarray()
            combined_features = normalize(combined_features, norm='l2', axis=1)
            
            # Fit PCA in batches
            n_samples = combined_features.shape[0]
            n_features = combined_features.shape[1]
            n_batches = (n_samples + self.batch_size - 1) // self.batch_size
            
            logger.info(f"PCA Configuration:")
            logger.info(f"Total samples: {n_samples:,}")
            logger.info(f"Input features: {n_features:,}")
            logger.info(f"Target dimensions: {self.n_components_pca}")
            logger.info(f"Batch size: {self.batch_size:,}")
            logger.info(f"Number of batches: {n_batches}")
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, n_samples)
                
                batch = combined_features[start_idx:end_idx]
                self.pca.partial_fit(batch)
                
                self._log_memory_status(f"pca_batch_{batch_idx}")
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
        """Transform features with improved handling for L2 distance computation."""
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
            
            # Combine features with L2 normalization
            combined_features = self._combine_features(feature_dict)
            self._log_memory_status("after_combine_transform")
            
            # Transform with PCA in batches
            n_samples = combined_features.shape[0]
            transformed_features_list = []
            
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch = combined_features[start_idx:end_idx]
                
                if sparse.issparse(batch):
                    batch = batch.toarray()
                
                # Normalize before PCA
                batch = normalize(batch, norm='l2', axis=1)
                
                # Apply PCA
                batch_transformed = self.pca.transform(batch)
                
                # Final normalization for L2 distance
                batch_transformed = normalize(batch_transformed, norm='l2', axis=1)
                
                transformed_features_list.append(batch_transformed)
                self._log_memory_status(f"pca_transform_batch_{start_idx//self.batch_size}")
                gc.collect()
            
            transformed_features = np.vstack(transformed_features_list)
            
            # Verify final normalization
            norms = np.linalg.norm(transformed_features, axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-5):
                logger.warning("Not all vectors are properly normalized")
                logger.debug(f"Norm range: min={norms.min():.6f}, max={norms.max():.6f}")
            
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
            'is_fitted': self.is_fitted,
            'n_components_pca': self.n_components_pca,
            'batch_size': self.batch_size
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
            self.cast_hasher = FeatureHasher(n_features=1000, input_type='string')
            self.director_hasher = FeatureHasher(n_features=200, input_type='string')
            
            logger.info("Transformers loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading transformers: {str(e)}", exc_info=True)
            raise
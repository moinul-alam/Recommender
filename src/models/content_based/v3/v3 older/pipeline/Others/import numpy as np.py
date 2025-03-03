import numpy as np
import pandas as pd
import gc
import logging
from pathlib import Path
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, IncrementalPCA
import joblib
from scipy import sparse
from typing import Dict, List, Optional
import time
import psutil
import memory_profiler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(
        self,
        method: str = 'both',  # Options: 'both', 'tsvd_pca', 'pca_only'
        weights: Optional[Dict[str, float]] = None,
        max_cast_members: int = 20,
        max_directors: int = 3,
        n_components_svd: int = 200,
        n_components_pca: int = 200,
        random_state: Optional[int] = None
    ):
        """
        Initialize feature engineering with configurable parameters and method selection.
        
        Args:
            method: Approach to use ('both', 'tsvd_pca', or 'pca_only')
        """
        self.method = method
        logger.info(f"Initializing FeatureEngineering with method: {method}")
        
        self.random_state = random_state
        self.metrics = {
            'fit_time': 0,
            'transform_time': 0,
            'peak_memory': 0,
            'tfidf_dimensions': 0,
            'final_dimensions': 0
        }
        
        # Set and validate weights
        self.weights = weights or {
            "overview": 0.50,
            "genres": 0.40,
            "keywords": 0.04,
            "cast": 0.04, 
            "director": 0.02
        }
        self._validate_weights()
        
        self.max_cast_members = max_cast_members
        self.max_directors = max_directors
        self.is_fitted = False
        
        # Initialize transformers
        self._initialize_transformers(n_components_svd, n_components_pca)
        
    def _initialize_transformers(self, n_components_svd: int, n_components_pca: int):
        """Initialize all necessary transformers based on selected method."""
        self.mlb_genres = MultiLabelBinarizer(sparse_output=True)
        self.tfidf_keywords = TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            dtype=np.float32,
            norm='l2'
        )
        self.tfidf_overview = TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            dtype=np.float32,
            norm='l2'
        )
        self.cast_hasher = FeatureHasher(n_features=1000, input_type='string')
        self.director_hasher = FeatureHasher(n_features=200, input_type='string')
        
        if self.method in ['both', 'tsvd_pca']:
            self.svd = TruncatedSVD(
                n_components=n_components_svd,
                random_state=self.random_state
            )
        
        self.pca = IncrementalPCA(
            n_components=n_components_pca,
            batch_size=5000
        )
        
    def _validate_weights(self) -> None:
        """Validate feature weights."""
        if not isinstance(self.weights, dict):
            raise TypeError("Weights must be a dictionary")
        
        if not all(isinstance(v, (int, float)) for v in self.weights.values()):
            raise TypeError("All weights must be numeric")
            
        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")

    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _process_overview(self, overview_matrix):
        """Process overview features based on selected method."""
        if self.method in ['both', 'tsvd_pca']:
            overview_reduced = self.svd.transform(overview_matrix)
            return sparse.csr_matrix(overview_reduced)
        return overview_matrix

    @memory_profiler.profile
    def fit_transformers(self, df: pd.DataFrame) -> None:
        """Fit all transformers with memory and time profiling."""
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        peak_memory = initial_memory
        
        try:
            # Record initial TF-IDF dimensions
            overview_matrix = self.tfidf_overview.fit_transform(df['overview'].fillna(''))
            self.metrics['tfidf_dimensions'] = overview_matrix.shape[1]
            
            # Process overview based on method
            overview_features = self._process_overview(overview_matrix)
            
            del overview_matrix
            gc.collect()
            
            # Fit genres with MultiLabelBinarizer
            logger.info('Fitting MultiLabelBinarizer on genres')
            genres_list = [genres.split(',') if pd.notna(genres) else [] for genres in df['genres']]
            genres_matrix = self.mlb_genres.fit_transform(genres_list)
            genres_matrix = normalize(genres_matrix, norm='l2', axis=1)
            
            del genres_list
            gc.collect()
            
            # Fit keywords
            logger.info('Fitting TF-IDF on keywords')
            keywords_matrix = self.tfidf_keywords.fit_transform(df['keywords'].fillna(''))
            
            # Process cast and director
            cast_features = [self._process_cast(cast) for cast in df['cast'].fillna('')]
            cast_matrix = self.cast_hasher.transform(cast_features)
            del cast_features
            
            director_features = [self._process_directors(director) for director in df['director'].fillna('')]
            director_matrix = self.director_hasher.transform(director_features)
            del director_features
            gc.collect()
            
            # Normalize matrices
            cast_matrix = normalize(cast_matrix, norm='l2', axis=1)
            director_matrix = normalize(director_matrix, norm='l2', axis=1)
            
            # Combine features using sparse operations
            feature_matrices = [
                keywords_matrix * self.weights['keywords'],
                overview_reduced * self.weights['overview'],
                genres_matrix * self.weights['genres'],
                cast_matrix * self.weights['cast'],
                director_matrix * self.weights['director']
            ]
            
            combined_features = sparse.hstack(feature_matrices).tocsr()
            
            del keywords_matrix, overview_reduced, genres_matrix, cast_matrix, director_matrix
            gc.collect()
            
            # Convert to float32 and normalize
            combined_features = combined_features.astype(np.float32)
            combined_features = normalize(combined_features, norm='l2', axis=1)
            
            # Fit IncrementalPCA in batches
            logger.info('Fitting IncrementalPCA')
            batch_size = 5000
            n_batches = (combined_features.shape[0] + batch_size - 1) // batch_size
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, combined_features.shape[0])
                batch = combined_features[start_idx:end_idx].toarray()
                self.pca.partial_fit(batch)
                
                del batch
                gc.collect()
            
            self.is_fitted = True
            logger.info("Transformer fitting completed successfully")
            
            del combined_features
            gc.collect()
            
            # Record metrics
            self.metrics['fit_time'] = time.time() - start_time
            self.metrics['peak_memory'] = peak_memory
            self.metrics['final_dimensions'] = self.pca.n_components_
            
            logger.info(f"Fitting completed. Metrics: {self.metrics}")
            
        except Exception as e:
            logger.error(f"Error in fit_transformers: {str(e)}", exc_info=True)
            raise

    def get_metrics(self) -> Dict:
        """Return performance metrics."""
        return self.metrics

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features with performance monitoring."""
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        try:
            required_columns = ['tmdb_id', 'overview', 'genres', 'cast', 'director', 'keywords']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            if not self.is_fitted:
                raise ValueError("Transformers must be fitted before transformation")
                
            try:
                logger.info(f"Starting feature transformation for DataFrame with shape: {df.shape}")
                
                # Transform genres using MultiLabelBinarizer
                genres_list = [genres.split(',') if pd.notna(genres) else [] for genres in df['genres']]
                genres_matrix = self.mlb_genres.transform(genres_list)
                genres_matrix = normalize(genres_matrix, norm='l2', axis=1)
                
                del genres_list
                gc.collect()
                
                # Transform keywords
                keywords_matrix = self.tfidf_keywords.transform(df['keywords'].fillna(''))
                
                # Transform overview
                df['overview'] = df['overview'].fillna('')
                overview_matrix = self.tfidf_overview.transform(df['overview'])
                overview_reduced = self.svd.transform(overview_matrix)
                overview_reduced = sparse.csr_matrix(overview_reduced)
                overview_reduced = normalize(overview_reduced, norm='l2', axis=1)
                
                del overview_matrix
                gc.collect()
                
                # Process cast and director
                cast_features = [self._process_cast(cast) for cast in df['cast'].fillna('')]
                cast_matrix = self.cast_hasher.transform(cast_features)
                del cast_features
                
                director_features = [self._process_directors(director) for director in df['director'].fillna('')]
                director_matrix = self.director_hasher.transform(director_features)
                del director_features
                gc.collect()
                
                # Normalize matrices
                cast_matrix = normalize(cast_matrix, norm='l2', axis=1)
                director_matrix = normalize(director_matrix, norm='l2', axis=1)
                
                # Combine features using sparse operations
                feature_matrices = [
                    keywords_matrix * self.weights['keywords'],
                    overview_reduced * self.weights['overview'],
                    genres_matrix * self.weights['genres'],
                    cast_matrix * self.weights['cast'],
                    director_matrix * self.weights['director']
                ]
                
                combined_features = sparse.hstack(feature_matrices).tocsr()
                
                del keywords_matrix, overview_reduced, genres_matrix, cast_matrix, director_matrix
                gc.collect()
                
                # Convert to float32 and normalize
                combined_features = combined_features.astype(np.float32)
                combined_features = normalize(combined_features, norm='l2', axis=1)
                
                # Transform with IncrementalPCA in batches
                batch_size = 5000
                n_batches = (combined_features.shape[0] + batch_size - 1) // batch_size
                pca_features = []
                
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, combined_features.shape[0])
                    batch = combined_features[start_idx:end_idx].toarray()
                    pca_batch = self.pca.transform(batch)
                    pca_features.append(pca_batch)
                    
                    del batch
                    gc.collect()
                
                pca_features = np.vstack(pca_features)
                
                # Final normalization
                norms = np.sqrt(np.sum(pca_features ** 2, axis=1))
                norms = np.maximum(norms, 1e-8)
                pca_features = pca_features / norms[:, np.newaxis]
                
                # Create final DataFrame
                final_features = pd.DataFrame(
                    pca_features,
                    columns=[f'feature_{i}' for i in range(pca_features.shape[1])],
                    index=df.index
                )
                final_features.insert(0, 'tmdb_id', df['tmdb_id'])
                
                del pca_features, combined_features
                gc.collect()
                
                self.metrics['transform_time'] = time.time() - start_time
                peak_memory = max(self._get_memory_usage() - initial_memory, self.metrics['peak_memory'])
                self.metrics['peak_memory'] = peak_memory
            
                return final_features
            
        except Exception as e:
            logger.error(f"Error in transform_features: {str(e)}", exc_info=True)
            raise
            
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
            'svd': self.svd,
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
            self.svd = joblib.load(path / "svd.pkl")
            self.pca = joblib.load(path / "pca.pkl")
            
            # Reinitialize FeatureHashers (they are stateless)
            self.cast_hasher = FeatureHasher(n_features=1000, input_type='string')
            self.director_hasher = FeatureHasher(n_features=200, input_type='string')
            
            logger.info("Transformers loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading transformers: {str(e)}", exc_info=True)
            raise
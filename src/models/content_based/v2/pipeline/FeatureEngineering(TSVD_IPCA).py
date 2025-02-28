import numpy as np
import pandas as pd
import gc
import logging
import memory_profiler
from pathlib import Path
import psutil
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, IncrementalPCA
import joblib
from scipy import sparse
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_engineering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        max_cast_members: int = 20,
        max_directors: int = 3,
        n_components_svd: int = 300,
        n_components_pca: int = 300,
        random_state: Optional[int] = 42
    ):
        """Initialize feature engineering with configurable parameters."""
        logger.info("Initializing FeatureEngineering")
        
        # Log memory usage at initialization
        logger.info(f"Memory usage at initialization: {memory_profiler.memory_usage()[0]} MB")
        
        self.random_state = random_state
        
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
        self.mlb_genres = MultiLabelBinarizer(sparse_output=True)
        self.tfidf_keywords = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            ngram_range=(1, 2),
            dtype=np.float32,
            norm='l2'
        )
        self.tfidf_overview = TfidfVectorizer(
            max_features=2000,
            stop_words="english",
            ngram_range=(1, 2),
            dtype=np.float32,
            norm='l2'
        )
        self.cast_hasher = FeatureHasher(n_features=500, input_type='string')
        self.director_hasher = FeatureHasher(n_features=100, input_type='string')
        self.svd = TruncatedSVD(
            n_components=n_components_svd, 
            random_state=self.random_state
        )
        self.pca = IncrementalPCA(
            n_components=n_components_pca,
            batch_size=10000
        )
        
        logger.info("FeatureEngineering initialized successfully")
        logger.info(f"Memory usage after initialization: {memory_profiler.memory_usage()[0]} MB")

    def _validate_weights(self) -> None:
        """Validate feature weights."""
        if not isinstance(self.weights, dict):
            raise TypeError("Weights must be a dictionary")
        
        if not all(isinstance(v, (int, float)) for v in self.weights.values()):
            raise TypeError("All weights must be numeric")
            
        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")
    
    def _process_cast(self, cast_str: str) -> list:
        """Process cast string to get top N cast members."""
        if pd.isna(cast_str) or not cast_str:
            return []
        cast_list = cast_str.split(',')
        return cast_list[:self.max_cast_members]

    def _process_directors(self, director_str: str) -> list:
        """Process director string to get top N directors."""
        if pd.isna(director_str) or not director_str:
            return []
        director_list = director_str.split(',')
        return director_list[:self.max_directors]

    def fit_transformers(self, df: pd.DataFrame) -> None:
        """Fit all transformers on the full dataset."""
        try:
            logger.info("Starting transformer fitting process")
            logger.info(f"Input DataFrame shape: {df.shape}")
            logger.info(f"Initial memory usage: {memory_profiler.memory_usage()[0]} MB")
            
            # Fit overview
            logger.info('Fitting TF-IDF on overview')
            df['overview'] = df['overview'].fillna('')
            overview_matrix = self.tfidf_overview.fit_transform(df['overview'])
            logger.info(f"Overview matrix shape: {overview_matrix.shape}")
            logger.info(f"Memory usage after overview: {memory_profiler.memory_usage()[0]} MB")
            
            # Apply SVD and normalize the result
            overview_reduced = self.svd.fit_transform(overview_matrix)
            overview_reduced = sparse.csr_matrix(overview_reduced)
            overview_reduced = normalize(overview_reduced, norm='l2', axis=1)
            logger.info(f"Overview reduced shape: {overview_reduced.shape}")
            
            del overview_matrix
            gc.collect()
            logger.info(f"Memory usage after SVD: {memory_profiler.memory_usage()[0]} MB")
            
            # Fit genres with MultiLabelBinarizer
            logger.info('Fitting MultiLabelBinarizer on genres')
            genres_list = [genres.split(',') if pd.notna(genres) else [] for genres in df['genres']]
            genres_matrix = self.mlb_genres.fit_transform(genres_list)
            genres_matrix = normalize(genres_matrix, norm='l2', axis=1)
            logger.info(f"Genres matrix shape: {genres_matrix.shape}")
            
            del genres_list
            gc.collect()
            logger.info(f"Memory usage after genres: {memory_profiler.memory_usage()[0]} MB")
            
            # Fit keywords
            logger.info('Fitting TF-IDF on keywords')
            keywords_matrix = self.tfidf_keywords.fit_transform(df['keywords'].fillna(''))
            logger.info(f"Keywords matrix shape: {keywords_matrix.shape}")
            
            # Process cast and director
            logger.info('Processing cast and director features')
            cast_features = [self._process_cast(cast) for cast in df['cast'].fillna('')]
            cast_matrix = self.cast_hasher.transform(cast_features)
            
            director_features = [self._process_directors(director) for director in df['director'].fillna('')]
            director_matrix = self.director_hasher.transform(director_features)
            
            # Normalize matrices
            cast_matrix = normalize(cast_matrix, norm='l2', axis=1)
            director_matrix = normalize(director_matrix, norm='l2', axis=1)
            logger.info(f"Cast matrix shape: {cast_matrix.shape}")
            logger.info(f"Director matrix shape: {director_matrix.shape}")
            
            # Combine features using sparse operations
            logger.info('Combining feature matrices')
            feature_matrices = [
                keywords_matrix * self.weights['keywords'],
                overview_reduced * self.weights['overview'],
                genres_matrix * self.weights['genres'],
                cast_matrix * self.weights['cast'],
                director_matrix * self.weights['director']
            ]
            
            combined_features = sparse.hstack(feature_matrices).tocsr()
            logger.info(f"Combined features shape: {combined_features.shape}")
            logger.info(f"Memory usage before PCA: {memory_profiler.memory_usage()[0]} MB")
            
            # Convert to float32
            combined_features = combined_features.astype(np.float32)
            
            # Fit IncrementalPCA in batches
            logger.info('Fitting IncrementalPCA')
            batch_size = 10000
            n_batches = (combined_features.shape[0] + batch_size - 1) // batch_size
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, combined_features.shape[0])
                batch = combined_features[start_idx:end_idx].toarray()
                
                logger.info(f"Batch {i+1}/{n_batches} - Shape: {batch.shape}")
                logger.info(f"Memory usage before partial_fit: {memory_profiler.memory_usage()[0]} MB")
                
                self.pca.partial_fit(batch)
                
                del batch
                gc.collect()
                logger.info(f"Memory usage after partial_fit: {memory_profiler.memory_usage()[0]} MB")
            
            self.is_fitted = True
            logger.info("Transformer fitting completed successfully")
            logger.info(f"Final memory usage: {memory_profiler.memory_usage()[0]} MB")
            
            del combined_features
            gc.collect()

        except Exception as e:
            logger.error(f"Error fitting transformers: {str(e)}", exc_info=True)
            raise

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted transformers."""
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
            
            # Create final DataFrame
            final_features = pd.DataFrame(
                pca_features,
                columns=[f'feature_{i}' for i in range(pca_features.shape[1])],
                index=df.index
            )
            final_features.insert(0, 'tmdb_id', df['tmdb_id'])
            
            del pca_features, combined_features
            gc.collect()
            
            return final_features

        except Exception as e:
            logger.error(f"Error transforming features: {str(e)}", exc_info=True)
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
            self.cast_hasher = FeatureHasher(n_features=500, input_type='string')
            self.director_hasher = FeatureHasher(n_features=100, input_type='string')
            
            logger.info("Transformers loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading transformers: {str(e)}", exc_info=True)
            raise
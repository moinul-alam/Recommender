import numpy as np
import pandas as pd
import joblib
import logging
import gc
import memory_profiler
from pathlib import Path
from typing import Dict, Optional
from scipy import sparse

# External libraries for embeddings
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_extraction.text import TfidfVectorizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_feature_engineering.log'),
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
        n_components_pca: int = 300,
        random_state: Optional[int] = 42
    ):
        """Initialize advanced feature engineering with configurable parameters."""
        logger.info("Initializing AdvancedFeatureEngineering")
        
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
        self.tfidf_keywords = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            ngram_range=(1, 2),
            dtype=np.float32,
            norm='l2'
        )
        self._validate_weights()
        
        self.max_cast_members = max_cast_members
        self.max_directors = max_directors
        self.is_fitted = False
        
        # Initialize advanced embeddings
        logger.info("Loading pre-trained embeddings")
        self.overview_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.genre_embeddings = api.load('glove-wiki-gigaword-300')
        
        # Initialize transformers
        self.cast_hasher = FeatureHasher(n_features=500, input_type='string')
        self.director_hasher = FeatureHasher(n_features=100, input_type='string')
        
        # PCA for dimensionality reduction
        self.pca = IncrementalPCA(
            n_components=n_components_pca,
            batch_size=10000
        )
        
        logger.info("AdvancedFeatureEngineering initialized successfully")

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
    
    def _embed_overview(self, overview: str) -> np.ndarray:
        """Embed overview using Sentence BERT."""
        return self.overview_embedder.encode(overview if pd.notna(overview) else '')
    
    def _embed_genres(self, genres: str) -> np.ndarray:
        """Embed genres using GloVe embeddings."""
        if pd.isna(genres):
            return np.zeros(300)
        
        genre_list = genres.split(',')
        genre_vecs = []
        
        for genre in genre_list:
            genre = genre.strip().lower()
            if genre in self.genre_embeddings:
                genre_vecs.append(self.genre_embeddings[genre])
        
        return np.mean(genre_vecs, axis=0) if genre_vecs else np.zeros(300)

    def fit_transformers(self, df: pd.DataFrame) -> None:
        """Fit advanced feature transformers."""
        try:
            logger.info("Starting advanced transformer fitting process")
            logger.info(f"Input DataFrame shape: {df.shape}")
            
            # Create batch processing for memory efficiency
            batch_size = 1000
            n_samples = len(df)
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            all_features = []
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                batch_df = df.iloc[start_idx:end_idx]
                
                # Process each feature type
                logger.info(f"Processing batch {batch_idx + 1}/{n_batches}")
                
                # Overview embeddings (384 dimensions)
                overview_embeddings = np.vstack([
                    self._embed_overview(overview) 
                    for overview in batch_df['overview']
                ])
                
                # Genre embeddings (300 dimensions)
                genre_embeddings = np.vstack([
                    self._embed_genres(genres) 
                    for genres in batch_df['genres']
                ])

                # Keywords TF-IDF processing
                logger.info('Processing keywords with TF-IDF')
                if batch_idx == 0:  # First batch - fit and transform
                    keywords_matrix = self.tfidf_keywords.fit_transform(
                        batch_df['keywords'].fillna('')
                    ).toarray()
                else:  # Subsequent batches - transform only
                    keywords_matrix = self.tfidf_keywords.transform(
                        batch_df['keywords'].fillna('')
                    ).toarray()
                
                # Cast features (sparse matrix)
                cast_features = [self._process_cast(cast) for cast in batch_df['cast'].fillna('')]
                cast_matrix = self.cast_hasher.transform(cast_features).toarray()
                
                # Director features (sparse matrix)
                director_features = [
                    self._process_directors(director) 
                    for director in batch_df['director'].fillna('')
                ]
                director_matrix = self.director_hasher.transform(director_features).toarray()
                
                # Normalize each feature matrix
                overview_embeddings = normalize(overview_embeddings)
                genre_embeddings = normalize(genre_embeddings)
                cast_matrix = normalize(cast_matrix)
                director_matrix = normalize(director_matrix)
                
                # Weight and combine features
                batch_features = np.hstack([
                    overview_embeddings * self.weights['overview'],
                    genre_embeddings * self.weights['genres'],
                    keywords_matrix * self.weights['keywords'],
                    cast_matrix * self.weights['cast'],
                    director_matrix * self.weights['director']
                ])
                
                all_features.append(batch_features)
                
                # Clean up batch
                del batch_features, overview_embeddings, genre_embeddings, cast_matrix, director_matrix
                gc.collect()
            
            # Combine all batches
            combined_features = np.vstack(all_features)
            
            # Fit PCA
            logger.info("Fitting PCA")
            self.pca.fit(combined_features)
            
            self.is_fitted = True
            logger.info("Advanced transformer fitting completed successfully")
            
            del combined_features, all_features
            gc.collect()

        except Exception as e:
            logger.error(f"Error fitting transformers: {str(e)}", exc_info=True)
            raise

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted transformers."""
        if not self.is_fitted:
            raise ValueError("Transformers must be fitted before transformation")
            
        try:
            batch_size = 1000
            n_samples = len(df)
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            all_transformed_features = []
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                batch_df = df.iloc[start_idx:end_idx]
                
                # Process features
                overview_embeddings = np.vstack([
                    self._embed_overview(overview) 
                    for overview in batch_df['overview']
                ])
                
                genre_embeddings = np.vstack([
                    self._embed_genres(genres) 
                    for genres in batch_df['genres']
                ])

                keywords_matrix = self.tfidf_keywords.transform(
                    batch_df['keywords'].fillna('')
                ).toarray()
                
                cast_features = [self._process_cast(cast) for cast in batch_df['cast'].fillna('')]
                cast_matrix = self.cast_hasher.transform(cast_features).toarray()
                
                director_features = [
                    self._process_directors(director) 
                    for director in batch_df['director'].fillna('')
                ]
                director_matrix = self.director_hasher.transform(director_features).toarray()
                
                # Normalize
                overview_embeddings = normalize(overview_embeddings)
                genre_embeddings = normalize(genre_embeddings)
                cast_matrix = normalize(cast_matrix)
                director_matrix = normalize(director_matrix)
                
                # Combine features
                batch_features = np.hstack([
                    overview_embeddings * self.weights['overview'],
                    genre_embeddings * self.weights['genres'],
                    keywords_matrix * self.weights['keywords'],
                    cast_matrix * self.weights['cast'],
                    director_matrix * self.weights['director']
                ])
                
                # Transform with PCA
                transformed_features = self.pca.transform(batch_features)
                all_transformed_features.append(transformed_features)
                
                del batch_features, transformed_features
                gc.collect()
            
            # Combine all transformed features
            final_features = np.vstack(all_transformed_features)
            
            # Create DataFrame
            result_df = pd.DataFrame(
                final_features,
                columns=[f'feature_{i}' for i in range(final_features.shape[1])],
                index=df.index
            )
            result_df.insert(0, 'tmdb_id', df['tmdb_id'])
            
            return result_df

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
        
        # Save stateful components
        transformers = {
            'pca': self.pca,
            'tfidf_keywords': self.tfidf_keywords
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
            
            # Reload stateful components
            self.pca = joblib.load(path / "pca.pkl")
            self.tfidf_keywords = joblib.load(path / "tfidf_keywords.pkl")
            
            # Reinitialize embeddings and hashers
            self.overview_embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.genre_embeddings = api.load('glove-wiki-gigaword-300')
            self.cast_hasher = FeatureHasher(n_features=500, input_type='string')
            self.director_hasher = FeatureHasher(n_features=100, input_type='string')
            
            logger.info("Transformers loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading transformers: {str(e)}", exc_info=True)
            raise
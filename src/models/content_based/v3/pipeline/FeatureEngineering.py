import numpy as np
import pandas as pd
import joblib
import logging
import gc
import memory_profiler
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from scipy import sparse

# External libraries for embeddings
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import PCA, TruncatedSVD
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
        n_components_svd_overview: int = 300,
        n_components_svd_keywords: int = 200,
        n_components_pca: int = 200,
        random_state: Optional[int] = 42,
        batch_size: int = 1000
    ):
        """Initialize advanced feature engineering with configurable parameters."""
        logger.info("Initializing FeatureEngineering")
        
        # Log memory usage at initialization
        logger.info(f"Memory usage at initialization: {memory_profiler.memory_usage()[0]} MB")
        
        self.random_state = random_state
        self.batch_size = batch_size
        
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
        
        # Initialize vectorizers and dimensionality reduction
        self.tfidf_keywords = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            ngram_range=(1, 2),
            dtype=np.float32,
            norm='l2'
        )
        
        # Initialize feature transformers
        self.cast_hasher = FeatureHasher(n_features=500, input_type='string')
        self.director_hasher = FeatureHasher(n_features=100, input_type='string')
        self.svd_overview = TruncatedSVD(n_components=n_components_svd_overview, random_state=random_state)
        self.svd_keywords = TruncatedSVD(n_components=n_components_svd_keywords, random_state=random_state)
        self.pca = PCA(n_components=n_components_pca, random_state=random_state)
        
        # Cache for already processed embeddings
        self.overview_cache = {}
        self.genre_cache = {}
        
        logger.info("Feature Engineering initialized successfully")
        
    def _validate_weights(self) -> None:
        """Validate feature weights."""
        if not isinstance(self.weights, dict):
            raise TypeError("Weights must be a dictionary")
        
        if not all(isinstance(v, (int, float)) for v in self.weights.values()):
            raise TypeError("All weights must be numeric")
            
        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")
    
    def _load_embedders(self):
        """Lazy load the embedding models when first needed."""
        if not hasattr(self, 'overview_embedder') or self.overview_embedder is None:
            logger.info("Loading Sentence Transformer model")
            self.overview_embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
        if not hasattr(self, 'genre_embeddings') or self.genre_embeddings is None:
            logger.info("Loading GloVe embeddings")
            self.genre_embeddings = api.load('glove-wiki-gigaword-300')
    
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
        """Embed overview using Sentence BERT with caching."""
        if pd.isna(overview) or not overview:
            return np.zeros(384)  # SentenceBERT 'all-MiniLM-L6-v2' has 384 dimensions
            
        # Check if we have this in cache
        if overview in self.overview_cache:
            return self.overview_cache[overview]
            
        self._load_embedders()
        embedding = self.overview_embedder.encode(overview)
        self.overview_cache[overview] = embedding
        return embedding
    
    def _embed_genres(self, genres: str) -> np.ndarray:
        """Embed genres using GloVe embeddings with caching."""
        if pd.isna(genres) or not genres:
            return np.zeros(300)  # GloVe has 300 dimensions
            
        # Check if we have this in cache
        if genres in self.genre_cache:
            return self.genre_cache[genres]
            
        self._load_embedders()
        genre_list = genres.split(',')
        genre_vecs = []
        
        for genre in genre_list:
            genre = genre.strip().lower()
            if genre in self.genre_embeddings:
                genre_vecs.append(self.genre_embeddings[genre])
        
        result = np.mean(genre_vecs, axis=0) if genre_vecs else np.zeros(300)
        self.genre_cache[genres] = result
        return result

    def process_batch(self, batch_df: pd.DataFrame, fit: bool = False) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
        """
        Process a batch of data to extract embeddings and features.
        Returns processed features and cached matrices for reuse.
        """
        # Create feature matrices
        matrices = {}
        
        # Process overview embeddings
        logger.info("Processing overview embeddings")
        overview_embeddings = np.vstack([
            self._embed_overview(overview) for overview in batch_df['overview']
        ])
        matrices['overview'] = overview_embeddings

        # Process genre embeddings
        logger.info("Processing genre embeddings")
        genre_embeddings = np.vstack([
            self._embed_genres(genres) for genres in batch_df['genres']
        ])
        matrices['genres'] = genre_embeddings
        
        # Process keywords with TF-IDF
        logger.info("Processing keywords")
        if fit:
            keywords_matrix = self.tfidf_keywords.fit_transform(batch_df['keywords'].fillna(''))
        else:
            keywords_matrix = self.tfidf_keywords.transform(batch_df['keywords'].fillna(''))
        matrices['keywords'] = keywords_matrix
        
        # Process cast features
        logger.info("Processing cast features")
        cast_features = [self._process_cast(cast) for cast in batch_df['cast'].fillna('')]
        cast_matrix = self.cast_hasher.transform(cast_features).toarray()
        matrices['cast'] = cast_matrix
        
        # Process director features
        logger.info("Processing director features")
        director_features = [self._process_directors(director) for director in batch_df['director'].fillna('')]
        director_matrix = self.director_hasher.transform(director_features).toarray()
        matrices['director'] = director_matrix
        
        return matrices

    def fit_transformers(self, df: pd.DataFrame) -> None:
        """Fit all feature transformers in a single pass."""
        try:
            logger.info("Starting transformer fitting process")
            logger.info(f"Input DataFrame shape: {df.shape}")
            
            # Clear any existing caches
            self.overview_cache = {}
            self.genre_cache = {}
            
            # Calculate number of batches
            n_samples = len(df)
            n_batches = (n_samples + self.batch_size - 1) // self.batch_size
            
            # First pass: Process all data to fit TF-IDF and collect embeddings for SVD
            all_overview_embeddings = []
            all_keywords_matrices = []
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, n_samples)
                batch_df = df.iloc[start_idx:end_idx].copy()
                
                logger.info(f"Processing batch {batch_idx+1}/{n_batches} for SVD fitting")
                
                # Process batch to get raw matrices
                fit_tfidf = batch_idx == 0  # Only fit TF-IDF on first batch
                matrices = self.process_batch(batch_df, fit=fit_tfidf)
                
                # Collect matrices needed for SVD fitting
                all_overview_embeddings.append(matrices['overview'])
                if fit_tfidf:
                    all_keywords_matrices.append(matrices['keywords'])
                
                # Clear memory after each batch
                del matrices
                gc.collect()
            
            # Combine all matrices and fit SVDs
            combined_overviews = np.vstack(all_overview_embeddings)
            logger.info(f"Fitting SVD for overviews on {combined_overviews.shape} matrix")
            self.svd_overview.fit(combined_overviews)
            
            if all_keywords_matrices:
                combined_keywords = sparse.vstack(all_keywords_matrices)
                logger.info(f"Fitting SVD for keywords on {combined_keywords.shape} matrix")
                self.svd_keywords.fit(combined_keywords)
            
            # Clear memory
            del combined_overviews, all_overview_embeddings
            if 'combined_keywords' in locals():
                del combined_keywords, all_keywords_matrices
            gc.collect()
            
            # Second pass: Apply SVD transformations and fit PCA
            all_combined_features = []
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, n_samples)
                batch_df = df.iloc[start_idx:end_idx].copy()
                
                logger.info(f"Processing batch {batch_idx+1}/{n_batches} for PCA fitting")
                
                # Get raw matrices (reuse from cache where possible)
                matrices = self.process_batch(batch_df)
                
                # Apply SVD transformations
                overview_reduced = self.svd_overview.transform(matrices['overview'])
                keywords_reduced = self.svd_keywords.transform(matrices['keywords'])
                
                # Normalize all matrices
                overview_reduced = normalize(overview_reduced)
                genre_embeddings = normalize(matrices['genres'])
                keywords_reduced = normalize(keywords_reduced)
                cast_matrix = normalize(matrices['cast'])
                director_matrix = normalize(matrices['director'])
                
                # Combine features with weights
                batch_features = np.hstack([
                    overview_reduced * self.weights['overview'],
                    genre_embeddings * self.weights['genres'],
                    keywords_reduced * self.weights['keywords'],
                    cast_matrix * self.weights['cast'],
                    director_matrix * self.weights['director']
                ])
                
                all_combined_features.append(batch_features)
                
                # Clean up batch
                del matrices, overview_reduced, genre_embeddings, keywords_reduced, cast_matrix, director_matrix
                gc.collect()
            
            # Combine all features and fit PCA
            combined_features = np.vstack(all_combined_features)
            combined_features = combined_features.astype(np.float32)
            
            # Normalize before PCA
            combined_features = normalize(combined_features)
            
            # Fit PCA
            logger.info(f"Fitting PCA on {combined_features.shape} matrix")
            self.pca.fit(combined_features)
            
            self.is_fitted = True
            logger.info("Transformer fitting completed successfully")
            
            # Clean up
            del combined_features, all_combined_features
            gc.collect()
            
            # Clear caches after fitting
            self.overview_cache = {}
            self.genre_cache = {}
            
        except Exception as e:
            logger.error(f"Error fitting transformers: {str(e)}", exc_info=True)
            raise

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted transformers in one efficient pass."""
        if not self.is_fitted:
            raise ValueError("Transformers must be fitted before transformation")
            
        try:
            # Calculate number of batches
            n_samples = len(df)
            n_batches = (n_samples + self.batch_size - 1) // self.batch_size
            
            all_transformed_features = []
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, n_samples)
                batch_df = df.iloc[start_idx:end_idx].copy()
                
                logger.info(f"Transforming batch {batch_idx+1}/{n_batches}")
                
                # Process this batch to get raw matrices
                matrices = self.process_batch(batch_df)
                
                # Apply SVD transformations
                overview_reduced = self.svd_overview.transform(matrices['overview'])
                keywords_reduced = self.svd_keywords.transform(matrices['keywords'])
                
                # Normalize all matrices
                overview_reduced = normalize(overview_reduced)
                genre_embeddings = normalize(matrices['genres'])
                keywords_reduced = normalize(keywords_reduced)
                cast_matrix = normalize(matrices['cast'])
                director_matrix = normalize(matrices['director'])
                
                # Combine features with weights
                batch_features = np.hstack([
                    overview_reduced * self.weights['overview'],
                    genre_embeddings * self.weights['genres'],
                    keywords_reduced * self.weights['keywords'],
                    cast_matrix * self.weights['cast'],
                    director_matrix * self.weights['director']
                ])
                
                # Apply PCA transformation
                batch_features = normalize(batch_features)
                transformed_features = self.pca.transform(batch_features)
                all_transformed_features.append(transformed_features)
                
                # Clean up
                del matrices, batch_features, transformed_features
                gc.collect()
            
            # Combine all transformed features
            final_features = np.vstack(all_transformed_features)
            final_features = final_features.astype(np.float32)
            
            # Create DataFrame
            result_df = pd.DataFrame(
                final_features,
                columns=[f'feature_{i}' for i in range(final_features.shape[1])],
                index=df.index
            )
            
            # Add item_id column if it exists
            if 'item_id' in df.columns:
                result_df.insert(0, 'item_id', df['item_id'])
            
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
            'tfidf_keywords': self.tfidf_keywords,
            'svd_overview': self.svd_overview,
            'svd_keywords': self.svd_keywords
        }
    
        for name, transformer in transformers.items():
            joblib.dump(transformer, path / f"{name}.pkl")
        
        # Save configuration
        config = {
            'weights': self.weights,
            'max_cast_members': self.max_cast_members,
            'max_directors': self.max_directors,
            'is_fitted': self.is_fitted,
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
            
            if 'batch_size' in config:
                self.batch_size = config['batch_size']
            
            # Reload stateful components
            self.pca = joblib.load(path / "pca.pkl")
            self.tfidf_keywords = joblib.load(path / "tfidf_keywords.pkl")
            self.svd_overview = joblib.load(path / "svd_overview.pkl")
            self.svd_keywords = joblib.load(path / "svd_keywords.pkl")
            
            # Reset caches
            self.overview_cache = {}
            self.genre_cache = {}
            
            logger.info("Transformers loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading transformers: {str(e)}", exc_info=True)
            raise
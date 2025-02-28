import numpy as np
import pandas as pd
import gc
import logging
from pathlib import Path
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import normalize, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
import joblib
from scipy import sparse
from typing import Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FeatureEngineering:
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        max_cast_members: int = 20,
        max_directors: int = 3,
        n_components_svd: int = 200,
        n_components_pca: int = 200,
        random_state: int = 42
    ):
        """Initialize feature engineering with configurable parameters."""
        logger.info("Initializing FeatureEngineering")
        
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
        self.svd = TruncatedSVD(n_components=n_components_svd, random_state=random_state)
        self.pca = PCA(n_components=n_components_pca, random_state=random_state)
        
        logger.info("FeatureEngineering initialized successfully")

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

    def _combine_sparse_features(self, features_list: List[sparse.csr_matrix], weights: List[float]) -> sparse.csr_matrix:
        """Combine sparse feature matrices with weights."""
        return sparse.hstack([
            feature * weight for feature, weight in zip(features_list, weights)
        ]).tocsr()

    def fit_transformers(self, df: pd.DataFrame) -> None:
        """Fit all transformers on the full dataset."""
        try:
            logger.info("Starting transformer fitting process")

            # Fit genres
            genres_list = df['genres'].str.split(',').tolist()
            genres_matrix = self.mlb_genres.fit_transform(genres_list)
            genres_matrix = normalize(genres_matrix, norm='l2', axis=1)
            
            # Fit keywords
            keywords_matrix = self.tfidf_keywords.fit_transform(df['keywords'].fillna(''))
            
            # Fit overview
            df['overview'] = df['overview'].fillna('')
            overview_matrix = self.tfidf_overview.fit_transform(df['overview'])
            
            # Apply SVD and normalize the result
            overview_reduced = self.svd.fit_transform(overview_matrix)
            overview_reduced = normalize(overview_reduced, norm='l2', axis=1)
            
            # Process cast and director
            cast_features = [self._process_cast(cast) for cast in df['cast'].fillna('')]
            director_features = [self._process_directors(director) for director in df['director'].fillna('')]
            
            cast_matrix = self.cast_hasher.transform(cast_features)
            director_matrix = self.director_hasher.transform(director_features)
            
            # Normalize cast and director matrices
            cast_matrix = normalize(cast_matrix, norm='l2', axis=1)
            director_matrix = normalize(director_matrix, norm='l2', axis=1)
            
            # Combine features with weights
            combined_features = np.hstack([
                keywords_matrix.toarray() * self.weights['keywords'],
                overview_reduced * self.weights['overview'],
                genres_matrix.toarray() * self.weights['genres'],
                cast_matrix.toarray() * self.weights['cast'],
                director_matrix.toarray() * self.weights['director']
            ])
            
            # Convert to float32 for FAISS compatibility
            combined_features = combined_features.astype(np.float32)
            
            # Normalize combined features before PCA using numpy for FAISS compatibility
            norms = np.sqrt(np.sum(combined_features ** 2, axis=1))
            norms = np.maximum(norms, 1e-8)  # Avoid division by zero
            combined_features = combined_features / norms[:, np.newaxis]
            
            # Fit PCA
            self.pca.fit(combined_features)
            
            self.is_fitted = True
            logger.info("Transformer fitting completed successfully")
            
            # Clear memory
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
            
            # Transform genres
            genres_list = df['genres'].str.split(',').tolist()
            genres_matrix = self.mlb_genres.transform(genres_list)
            genres_matrix = normalize(genres_matrix, norm='l2', axis=1)
            
            # Transform keywords (already normalized by TF-IDF)
            keywords_matrix = self.tfidf_keywords.transform(df['keywords'].fillna(''))
            
            # Transform overview and tagline
            df['overview'] = df['overview'].fillna('')
            overview_matrix = self.tfidf_overview.transform(df['overview'])
            overview_reduced = self.svd.transform(overview_matrix)
            overview_reduced = normalize(overview_reduced, norm='l2', axis=1)
            
            # Process cast and director
            cast_features = [self._process_cast(cast) for cast in df['cast'].fillna('')]
            director_features = [self._process_directors(director) for director in df['director'].fillna('')]
            
            cast_matrix = self.cast_hasher.transform(cast_features)
            director_matrix = self.director_hasher.transform(director_features)
            
            # Normalize cast and director matrices
            cast_matrix = normalize(cast_matrix, norm='l2', axis=1)
            director_matrix = normalize(director_matrix, norm='l2', axis=1)
            
            # Combine all features
            combined_features = np.hstack([
                keywords_matrix.toarray() * self.weights['keywords'],
                overview_reduced * self.weights['overview'],
                genres_matrix.toarray() * self.weights['genres'],
                cast_matrix.toarray() * self.weights['cast'],
                director_matrix.toarray() * self.weights['director']
            ])
            
            # Convert to float32 for FAISS compatibility
            combined_features = combined_features.astype(np.float32)
            
            # Normalize combined features before PCA using numpy for FAISS compatibility
            norms = np.sqrt(np.sum(combined_features ** 2, axis=1))
            norms = np.maximum(norms, 1e-8)  # Avoid division by zero
            combined_features = combined_features / norms[:, np.newaxis]
            
            # Apply PCA transformation
            pca_features = self.pca.transform(combined_features)
            
            # Final normalization for FAISS using numpy
            pca_features = pca_features.astype(np.float32)
            faiss_norm = np.sqrt(np.sum(pca_features ** 2, axis=1))
            faiss_norm = np.maximum(faiss_norm, 1e-8)  # Avoid division by zero
            pca_features = pca_features / faiss_norm[:, np.newaxis]
            
            # Create final DataFrame
            final_features = pd.DataFrame(
                pca_features,
                columns=[f'feature_{i}' for i in range(pca_features.shape[1])],
                index=df.index
            )
            final_features.insert(0, 'tmdb_id', df['tmdb_id'])
            
            # Clear memory
            del combined_features, pca_features
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
            self.cast_hasher = FeatureHasher(n_features=1000, input_type='string')
            self.director_hasher = FeatureHasher(n_features=200, input_type='string')
            
            logger.info("Transformers loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading transformers: {str(e)}", exc_info=True)
            raise
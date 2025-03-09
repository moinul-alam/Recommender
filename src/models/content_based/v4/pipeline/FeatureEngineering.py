import numpy as np
import pandas as pd
import gc
import logging
from sklearn.preprocessing import normalize, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from typing import Dict, Optional, List, Tuple

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FeatureEngineering:
    def __init__(
        self,
        max_cast_members: int = 20,
        max_directors: int = 3,
        weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize feature engineering with configurable parameters."""
        logger.info("Initializing FeatureEngineering")
        
        self.max_cast_members = max_cast_members
        self.max_directors = max_directors
        self.is_fitted = False
        
        self.weights = weights or {
            "overview": 0.50,
            "genres": 0.40,
            "keywords": 0.04,
            "cast": 0.04, 
            "director": 0.02
        }
        self._validate_weights()
        
        # Initialize transformers
        self.mlb_genres = MultiLabelBinarizer(sparse_output=True)
        self.tfidf_keywords = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2), min_df=3, dtype=np.float32, norm='l2'
        )
        self.tfidf_overview = TfidfVectorizer(
            max_features=3000, stop_words="english", ngram_range=(1, 2), min_df=3, dtype=np.float32, norm='l2'
        )
        self.tfidf_cast = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 1), min_df=2, dtype=np.float32, norm='l2'
        )
        self.tfidf_director = TfidfVectorizer(
            max_features=200, stop_words="english", ngram_range=(1, 1), min_df=2, dtype=np.float32, norm='l2'
        )
        
        logger.info("FeatureEngineering initialized successfully")

    def _validate_weights(self) -> None:
        """Validate feature weights."""
        if not isinstance(self.weights, dict):
            raise TypeError("Weights must be a dictionary")
        
        if not all(isinstance(v, (int, float)) for v in self.weights.values()):
            raise TypeError("All weights must be numeric")
            
        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")
    
    def fit_transformers(self, df: pd.DataFrame) -> None:
        """Fit all transformers on the full dataset."""
        try:
            logger.info("Starting transformer fitting process")
            
            # Fit TF-IDF for overview
            logger.info("Fitting Overview TF-IDF")
            self.tfidf_overview.fit(df['overview'].fillna(''))
            
            # Fit MultiLabelBinarizer for genres
            logger.info("Fitting Genres MLB")
            genres_list = df['genres'].str.split(',').tolist()
            self.mlb_genres.fit(genres_list)
            
            # Fit TF-IDF for keywords
            logger.info("Fitting Keywords TF-IDF")
            self.tfidf_keywords.fit(df['keywords'].fillna(''))

            # Fit TF-IDF for cast (top N members)
            logger.info("Fitting Cast TF-IDF")
            cast_data = df['cast'].fillna('').apply(lambda x: ' '.join(x.split(',')[:self.max_cast_members]))
            self.tfidf_cast.fit(cast_data)

            # Fit TF-IDF for director (top N directors)
            logger.info("Fitting Director TF-IDF")
            director_data = df['director'].fillna('').apply(lambda x: ' '.join(x.split(',')[:self.max_directors]))
            self.tfidf_director.fit(director_data)
            
            self.is_fitted = True
            logger.info("Transformer fitting completed successfully")
            
            gc.collect()

        except Exception as e:
            logger.error(f"Error fitting transformers: {str(e)}", exc_info=True)
            raise

    def transform_features_sparse(self, df: pd.DataFrame) -> Tuple[List[str], sparse.csr_matrix]:
        """Transform features using fitted transformers and return item_ids and sparse matrix."""
        required_columns = ['item_id', 'overview', 'genres', 'cast', 'director', 'keywords']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        if not self.is_fitted:
            raise ValueError("Transformers must be fitted before transformation")
            
        try:
            logger.info(f"Starting feature transformation for DataFrame with shape: {df.shape}")

            # Transform overview
            overview_matrix = self.tfidf_overview.transform(df['overview'].fillna(''))
            
            # Transform genres
            genres_list = df['genres'].str.split(',').tolist()
            genres_matrix = self.mlb_genres.transform(genres_list)
            
            # Transform keywords
            keywords_matrix = self.tfidf_keywords.transform(df['keywords'].fillna(''))

            # Transform cast
            cast_data = df['cast'].fillna('').apply(lambda x: ' '.join(x.split(',')[:self.max_cast_members]))
            cast_matrix = self.tfidf_cast.transform(cast_data)

            # Transform director
            director_data = df['director'].fillna('').apply(lambda x: ' '.join(x.split(',')[:self.max_directors]))
            director_matrix = self.tfidf_director.transform(director_data)
            
            # Combine all features into a single sparse matrix
            combined_matrix = sparse.hstack([
                overview_matrix * self.weights['overview'],
                keywords_matrix * self.weights['keywords'],
                genres_matrix * self.weights['genres'],
                cast_matrix * self.weights['cast'],
                director_matrix * self.weights['director']
            ])
            
            # Get item IDs
            item_ids = df['item_id'].values.tolist()
            
            logger.info("Feature transformation completed successfully")
            
            gc.collect()
            
            return item_ids, combined_matrix

        except Exception as e:
            logger.error(f"Error transforming features: {str(e)}", exc_info=True)
            raise
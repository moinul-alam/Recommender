import numpy as np
import pandas as pd
import gc
import logging
from pathlib import Path
from sklearn.preprocessing import normalize, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FeatureEngineering:
    def __init__(
        self,
        model_components: Optional[Dict[str, int]] = None,
        feature_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize feature engineering with configurable parameters."""
        logger.info("Initializing FeatureEngineering")
        
        # Set default model components if not provided
        self.model_components = model_components or {
            "tfidf_overview_max_features": 5000,
            "tfidf_keywords_max_features": 1000,
            "max_cast_members": 20,
            "max_directors": 3,
            "overview_tsvd_components": 100,  # Parameter for overview dimension reduction
            "keywords_tsvd_components": 50,   # Parameter for keywords dimension reduction
            "cast_tsvd_components": 100,      # New parameter for cast dimension reduction
            "director_tsvd_components": 50,   # New parameter for director dimension reduction
            "random_state": 42
        }
        
        # Set and validate weights
        self.weights = feature_weights or {
            "overview": 0.40,
            "genres": 0.35,
            "keywords": 0.10,
            "cast": 0.10, 
            "director": 0.05
        }
        self._validate_weights()
        
        # Initialize TF-IDF vectorizer for overview with improved parameters
        self.tfidf_overview = TfidfVectorizer(
            max_features=self.model_components['tfidf_overview_max_features'],
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            dtype=np.float32,
            norm='l2'
        )
        
        # Initialize TF-IDF vectorizer for keywords
        self.tfidf_keywords = TfidfVectorizer(
            max_features=self.model_components['tfidf_keywords_max_features'],
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            dtype=np.float32,
            norm='l2'
        )
        
        # Initialize MultiLabelBinarizer for genres
        self.mlb_genres = MultiLabelBinarizer(sparse_output=True)
        
        # Initialize MultiLabelBinarizer for cast and directors (one-hot encoding)
        self.mlb_cast = MultiLabelBinarizer(sparse_output=True)
        self.mlb_director = MultiLabelBinarizer(sparse_output=True)
        
        # Extract configuration values
        self.max_cast_members = self.model_components['max_cast_members']
        self.max_directors = self.model_components['max_directors']
        self.is_fitted = False
        
        # Initialize TSVD reducers for high-dimensional features
        self.overview_tsvd = TruncatedSVD(
            n_components=self.model_components['overview_tsvd_components'],
            random_state=self.model_components['random_state'],
            algorithm='randomized'
        )
        
        self.keywords_tsvd = TruncatedSVD(
            n_components=self.model_components['keywords_tsvd_components'],
            random_state=self.model_components['random_state'],
            algorithm='randomized'
        )
        
        # Initialize TSVD for cast and director
        self.cast_tsvd = TruncatedSVD(
            n_components=self.model_components['cast_tsvd_components'],
            random_state=self.model_components['random_state'],
            algorithm='randomized'
        )
        
        self.director_tsvd = TruncatedSVD(
            n_components=self.model_components['director_tsvd_components'],
            random_state=self.model_components['random_state'],
            algorithm='randomized'
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

    def _process_cast(self, cast_str: str) -> list:
        """Process cast string to get top N cast members."""
        if pd.isna(cast_str) or not cast_str.strip():
            return []
        
        # Normalize separators and split, then strip spaces
        cast_list = [name.strip() for name in cast_str.replace(';', ',').replace('|', ',').split(',') if name.strip()]
        
        return cast_list[:self.max_cast_members]

    def _process_directors(self, director_str: str) -> list:
        """Process director string to get top N directors."""
        if pd.isna(director_str) or not director_str.strip():
            return []
        
        # Normalize separators and split, then strip spaces
        director_list = [name.strip() for name in director_str.replace(';', ',').replace('|', ',').split(',') if name.strip()]

        return director_list[:self.max_directors]

    def _combine_weighted_features(
        self, 
        features_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Combine pre-reduced feature matrices with their respective weights.
        All inputs should already be dense numpy arrays.
        
        Args:
            features_dict: Dictionary of feature name to feature matrix (numpy array)
            
        Returns:
            Combined weighted features as numpy array
        """
        # Compute weighted features
        weighted_features = []
        for name, matrix in features_dict.items():
            if name in self.weights:
                weighted_features.append(matrix * self.weights[name])
        
        # Stack horizontally
        combined = np.hstack(weighted_features)
        
        # Ensure float32 type for efficiency and FAISS compatibility
        combined = combined.astype(np.float32)
        
        return combined

    def fit_transformers(self, df: pd.DataFrame) -> None:
        """Fit all transformers on the full dataset."""
        required_columns = ['item_id', 'overview', 'genres', 'cast', 'director', 'keywords']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        try:
            logger.info("Starting transformer fitting process")
            
            # Fit TF-IDF for overview and apply TSVD
            logger.info("Fitting Overview: TF-IDF + TSVD")
            overview_matrix = self.tfidf_overview.fit_transform(df['overview'].fillna(''))
            logger.info(f"Overview TF-IDF matrix shape: {overview_matrix.shape}")
            self.overview_tsvd.fit(overview_matrix)
            
            # Fit MultilabelBinarizer for genres
            logger.info("Fitting Genres using MLB")
            genres_list = df['genres'].fillna('').str.split(',').tolist()
            genres_matrix = self.mlb_genres.fit_transform(genres_list)
            logger.info(f"Genres matrix shape: {genres_matrix.shape}")
            
            # Fit TF-IDF for keywords and apply TSVD
            logger.info("Fitting Keywords: TF-IDF + TSVD")
            keywords_matrix = self.tfidf_keywords.fit_transform(df['keywords'].fillna(''))
            logger.info(f"Keywords TF-IDF matrix shape: {keywords_matrix.shape}")
            self.keywords_tsvd.fit(keywords_matrix)
            
            # Process and one-hot encode cast
            logger.info("Processing and one-hot encoding cast")
            cast_features = [self._process_cast(cast) for cast in df['cast'].fillna('')]
            cast_matrix = self.mlb_cast.fit_transform(cast_features)
            logger.info(f"Cast one-hot encoded matrix shape: {cast_matrix.shape}")
            # Apply TSVD to cast matrix
            logger.info("Applying TSVD to cast matrix")
            self.cast_tsvd.fit(cast_matrix)
            
            # Process and one-hot encode director
            logger.info("Processing and one-hot encoding director")
            director_features = [self._process_directors(director) for director in df['director'].fillna('')]
            director_matrix = self.mlb_director.fit_transform(director_features)
            logger.info(f"Director one-hot encoded matrix shape: {director_matrix.shape}")
            # Apply TSVD to director matrix
            logger.info("Applying TSVD to director matrix")
            self.director_tsvd.fit(director_matrix)
            
            self.is_fitted = True
            logger.info("Transformer fitting completed successfully")
            
            # Clear memory
            del overview_matrix, genres_matrix, keywords_matrix, cast_matrix, director_matrix
            gc.collect()

        except Exception as e:
            logger.error(f"Error fitting transformers: {str(e)}", exc_info=True)
            raise

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted transformers and return concatenated feature matrix."""
        required_columns = ['item_id', 'overview', 'genres', 'cast', 'director', 'keywords']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        if not self.is_fitted:
            raise ValueError("Transformers must be fitted before transformation")
            
        try:
            logger.info(f"Starting feature transformation for DataFrame with shape: {df.shape}")

            # Transform overview using TF-IDF + TSVD
            overview_matrix = self.tfidf_overview.transform(df['overview'].fillna(''))
            overview_reduced = self.overview_tsvd.transform(overview_matrix)
            
            # Transform genres using MLB
            genres_list = df['genres'].fillna('').str.split(',').tolist()
            genres_matrix = self.mlb_genres.transform(genres_list)
            
            # Transform keywords using TF-IDF + TSVD
            keywords_matrix = self.tfidf_keywords.transform(df['keywords'].fillna(''))
            keywords_reduced = self.keywords_tsvd.transform(keywords_matrix)
            
            # Process and transform cast using one-hot encoding + TSVD
            cast_features = [self._process_cast(cast) for cast in df['cast'].fillna('')]
            cast_matrix = self.mlb_cast.transform(cast_features)
            cast_reduced = self.cast_tsvd.transform(cast_matrix)
            
            # Process and transform director using one-hot encoding + TSVD
            director_features = [self._process_directors(director) for director in df['director'].fillna('')]
            director_matrix = self.mlb_director.transform(director_features)
            director_reduced = self.director_tsvd.transform(director_matrix)
            
            # Normalize matrices
            overview_norm = normalize(overview_reduced, norm='l2', axis=1)
            keywords_norm = normalize(keywords_reduced, norm='l2', axis=1)
            genres_matrix_norm = normalize(genres_matrix, norm='l2', axis=1).toarray()
            cast_norm = normalize(cast_reduced, norm='l2', axis=1)
            director_norm = normalize(director_reduced, norm='l2', axis=1)
            
            # Combine all features with weights (without final TSVD)
            logger.info("Combining features with weights (no final TSVD)")
            feature_dict = {
                "overview": overview_norm,
                "genres": genres_matrix_norm,
                "keywords": keywords_norm,
                "cast": cast_norm,
                "director": director_norm
            }
            
            combined_features = self._combine_weighted_features(feature_dict)
            
            # Final normalization for consistency
            combined_features = normalize(combined_features, norm='l2', axis=1)
            combined_features = combined_features.astype(np.float32)  # Ensure float32 for FAISS compatibility
            
            # Create final DataFrame
            logger.info("Creating final feature DataFrame")
            final_features = pd.DataFrame(
                combined_features,
                columns=[f'feature_{i}' for i in range(combined_features.shape[1])],
                index=df.index
            )
            final_features.insert(0, 'item_id', df['item_id'])
            
            # Clear memory
            del combined_features
            del overview_matrix, genres_matrix, keywords_matrix, cast_matrix, director_matrix
            del overview_reduced, keywords_reduced, cast_reduced, director_reduced
            del overview_norm, keywords_norm, genres_matrix_norm, cast_norm, director_norm
            gc.collect()
            
            return final_features

        except Exception as e:
            logger.error(f"Error transforming features: {str(e)}", exc_info=True)
            raise
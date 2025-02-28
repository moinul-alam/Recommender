import logging
import pickle
from typing import Dict, List, Tuple, Optional
import numpy as np
from functools import lru_cache
from pathlib import Path
from src.models.collaborative.v3.pipeline.Recommender import Recommender

logger = logging.getLogger(__name__)

class RecommendationService:
    def __init__(self, processed_dir_path: str, model_dir_path: str):
        """
        Initialize the recommendation service with configurable paths.
        
        Args:
            processed_dir_path: Path to processed data files
            model_dir_path: Path to model files
        """
        try:
            # Convert string paths to Path objects
            self.processed_dir_path = Path(processed_dir_path)
            self.model_dir_path = Path(model_dir_path)
            
            logger.info(f"Initializing RecommendationService with processed_dir_path: {processed_dir_path}")
            logger.info(f"Model directory path: {model_dir_path}")
            
            # Load mappings
            self._load_mappings()
            
            # Initialize recommender
            self._initialize_recommender()
            
            logger.info("RecommendationService initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RecommendationService: {str(e)}")
            raise

    def _load_mappings(self):
        """Load all required mapping files."""
        try:
            with open(self.processed_dir_path / "item_mapping.pkl", "rb") as f:
                self.item_mapping = pickle.load(f)
            logger.info("Loaded item_mapping.pkl")
            
            with open(self.processed_dir_path / "item_reverse_mapping.pkl", "rb") as f:
                self.item_reverse_mapping = pickle.load(f)
            logger.info("Loaded item_reverse_mapping.pkl")
            
            with open(self.processed_dir_path / "user_item_matrix.pkl", "rb") as f:
                self.user_item_matrix = pickle.load(f)
            logger.info("Loaded user_item_matrix.pkl")
            
            with open(self.processed_dir_path / "user_mapping.pkl", "rb") as f:
                self.user_mapping = pickle.load(f)
            logger.info("Loaded user_mapping.pkl")
            
            with open(self.processed_dir_path / "user_reverse_mapping.pkl", "rb") as f:
                self.user_reverse_mapping = pickle.load(f)
            logger.info("Loaded user_reverse_mapping.pkl")
            
        except FileNotFoundError as e:
            logger.error(f"Required file not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading mappings: {str(e)}")
            raise

    def _initialize_recommender(self):
        """Initialize the recommender with loaded data."""
        try:
            model_path = self.model_dir_path / "matrix_factorization_model.pkl"
            self.recommender = Recommender(
                model_path=model_path,
                user_item_matrix=self.user_item_matrix,
                item_mapping=self.item_mapping,
                item_reverse_mapping=self.item_reverse_mapping
            )
            logger.info("Recommender initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing recommender: {str(e)}")
            raise

    @lru_cache(maxsize=1000)
    def get_similar_movies(
        self, 
        tmdb_ids: Tuple[int, ...],
        num_recommendations: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Get similar movies based on tmdb_ids.
        Returns list of (tmdb_id, similarity_score) tuples.
        """
        try:
            logger.info(f"Getting similar movies for tmdb_ids: {tmdb_ids}")
            recommendations = self.recommender.get_similar_items(
                tmdb_ids=list(tmdb_ids),
                n_recommendations=num_recommendations
            )
            logger.info(f"Successfully generated {len(recommendations)} similar movie recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting similar movies: {str(e)}")
            raise Exception(f"Error getting similar movies: {str(e)}")

    def get_recommendations_for_user(
        self,
        tmdb_ids: List[int],
        ratings: List[float],
        num_recommendations: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Get personalized recommendations based on user's ratings.
        Returns list of (tmdb_id, prediction_score) tuples.
        """
        try:
            logger.info(f"Getting recommendations for user with {len(tmdb_ids)} ratings")
            recommendations = self.recommender.get_recommendations(
                tmdb_ids=tmdb_ids,
                ratings=ratings,
                n_recommendations=num_recommendations
            )
            logger.info(f"Successfully generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise Exception(f"Error getting recommendations: {str(e)}")
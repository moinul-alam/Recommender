from typing import Dict, List, Optional
import logging
from fastapi import HTTPException
from src.models.collaborative.v2.pipeline.recommender import UserRecommender
from src.models.collaborative.v2.services.base_recommendation_service import BaseRecommendationService
from src.models.common.file_config import file_names

logger = logging.getLogger(__name__)

class UserRecommendationService:
    @staticmethod
    def get_user_recommendations(
        user_ratings: Dict[str, float],
        collaborative_dir_path: str,
        n_recommendations: int,
        similarity_metric: str,
        min_similarity: float,
        n_neighbors: Optional[int] = 50
    ) -> List[Dict]:
        """
        Generate user-based recommendations based on user ratings.
        
        Args:
            user_ratings: Dictionary mapping item IDs to user ratings
            collaborative_dir_path: Directory path containing model files
            file_names: Dictionary mapping component names to filenames
            n_recommendations: Number of recommendations to return
            min_similarity: Minimum similarity threshold for recommendations
            n_neighbors: Number of neighbors to consider for recommendations (default: 50)
            
        Returns:
            List of recommendation dictionaries with tmdb_id, similarity, and predicted_rating
        """
        try:
            logger.info(f"User ratings received: {user_ratings}")

            try:
                user_ratings = {int(key): float(value) for key, value in user_ratings.items()}
                logger.info(f"Converted user ratings: {user_ratings}")
            except Exception as e:
                logger.error(f"Error in converting user ratings: {str(e)}")
                raise HTTPException(status_code=400, detail="Invalid rating format")

            # Load model components
            components = BaseRecommendationService.load_model_components(collaborative_dir_path, file_names)
            user_item_matrix, user_mapping, user_reverse_mapping, user_matrix, item_mapping, item_reverse_mapping, item_matrix, faiss_user_index, faiss_item_index, svd_user_model = components

            # Initialize recommender
            recommender = UserRecommender(
                faiss_user_index=faiss_user_index,
                user_embedding_matrix=user_matrix,
                faiss_item_index=faiss_item_index,
                item_embedding_matrix=item_matrix,
                user_item_matrix=user_item_matrix,
                item_mapping=item_mapping,
                item_reverse_mapping=item_reverse_mapping,
                svd_user_model=svd_user_model,
                similarity_metric=similarity_metric,
                min_similarity=min_similarity,
                n_neighbors=n_neighbors
            )

            recommendations = recommender.generate_recommendations(
                user_ratings=user_ratings,
                n_recommendations=n_recommendations
            )

            logger.info(f"Generated {len(recommendations)} user-based recommendations")
            return recommendations if recommendations else []

        except Exception as e:
            logger.error(f"Error in generating user-based recommendations: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")
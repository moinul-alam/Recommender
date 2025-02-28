from typing import Dict, List
import logging
from fastapi import HTTPException
from src.models.collaborative.v2.pipeline.Recommender import UserRecommender
from src.models.collaborative.v2.services.base_recommendation_service import BaseRecommendationService

logger = logging.getLogger(__name__)

class SwitchingRecommendationService:
    @staticmethod
    def get_user_recommendations(
        user_ratings: Dict[str, float],
        content_based_dir_path: str,
        collaborative_dir_path: str,
        n_recommendations: int,
        min_similarity: float
    ) -> List[Dict]:
        try:
            logger.info(f"User ratings received: {user_ratings}")

            try:
                user_ratings = {int(key): float(value) for key, value in user_ratings.items()}
                logger.info(f"Converted user ratings: {user_ratings}")
            except Exception as e:
                logger.error(f"Error in converting user ratings: {str(e)}")
                raise HTTPException(status_code=400, detail="Invalid rating format")

            # Load model components
            collaborative_components = BaseRecommendationService.load_model_components(collaborative_dir_path)
            _, _, _, item_mapping, item_reverse_mapping, item_matrix, svd_user_model, _, _, _, faiss_item_index = collaborative_components

            # Initialize recommender
            recommender = UserRecommender(
                faiss_index=faiss_item_index,
                embedding_matrix=item_matrix,
                svd_model=svd_user_model,
                item_mapping=item_mapping,
                item_reverse_mapping=item_reverse_mapping,
                min_similarity=min_similarity
            )

            recommendations = recommender.generate_recommendations(
                items=user_ratings,
                n_recommendations=n_recommendations
            )

            logger.info(f"Generated {len(recommendations)} user-based recommendations")
            return recommendations if recommendations else {"message": "No recommendations found"}

        except Exception as e:
            logger.error(f"Error in generating user-based recommendations: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate recommendations")

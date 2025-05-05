from typing import List, Dict, Optional
import logging
from fastapi import HTTPException
from src.models.collaborative.v2.pipeline.recommender import ItemRecommender
from src.models.collaborative.v2.services.base_recommendation_service import BaseRecommendationService

from src.schemas.content_based_schema import PipelineResponse


logger = logging.getLogger(__name__)

class ItemRecommendationService:
    @staticmethod
    def get_item_recommendations(
        items: List[int],
        collaborative_dir_path: str,
        file_names: dict,
        n_recommendations: int,
        min_similarity: float
    ) -> Optional[PipelineResponse]:
        try:
            logger.info(f"Received items for item-based recommendations: {items}")

            try:
                items = [int(item) for item in items]
                logger.info(f"Converted items: {items}")
            except Exception as e:
                logger.error(f"Error converting items: {str(e)}")
                raise HTTPException(status_code=400, detail="Invalid item format")

            # Load model components
            components = BaseRecommendationService.load_model_components(collaborative_dir_path, file_names)
            _, _, _, _, item_mapping, item_reverse_mapping, item_matrix, _, faiss_item_index = components

            # Initialize recommender
            recommender = ItemRecommender(
                faiss_index=faiss_item_index,
                embedding_matrix=item_matrix,
                item_mapping=item_mapping,
                item_reverse_mapping=item_reverse_mapping,
                min_similarity=min_similarity
            )

            recommendations = recommender.generate_recommendations(
                item_ids=items,
                n_recommendations=n_recommendations
            )

            logger.info(f"Generated {len(recommendations)} item-based recommendations")
            return recommendations if recommendations else {"message": "No recommendations found"}

        except Exception as e:
            logger.error(f"Error in generating item-based recommendations: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate recommendations")
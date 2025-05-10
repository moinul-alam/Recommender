from typing import Optional
from fastapi import HTTPException
from pathlib import Path
import faiss
import logging
from src.schemas.content_based_schema import Recommendation, RecommendationResponse, RecommendationRequest, RecommendationRequestedItem
from src.models.common.file_config import file_names

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CollaborativeRecommendationService:
    @staticmethod
    def get_recommendations(
        recommendation_request: RecommendationRequest,
        collaborative_dir_path: str,
        n_recommendations: int,
        similarity_metric: str,
        min_similarity: float,
        n_neighbors: Optional[int] = 50
    ) -> RecommendationResponse:
        try:
            logger.info(f"Received recommendation request: {recommendation_request}")
        except Exception as e:
            logger.error(f"Error in processing recommendation request: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid recommendation request format")
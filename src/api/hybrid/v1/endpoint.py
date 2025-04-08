import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Body, HTTPException, Query
from src.config.config import BaseConfig
from src.models.hybrid.v1.services.switching_recommendation_service import SwitchingRecommendationService
from src.models.hybrid.v1.services.weighted_recommendation_service import WeighedRecommendationService


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

version = 2
config = BaseConfig()
content_based_dir_path = config.CONTENT_BASED_PATH / f"v{version}"
collaborative_dir_path = config.COLLABORATIVE_PATH / f"v{version}"
hybrid_router_v1 = APIRouter()

@hybrid_router_v1.post("/recommendations/weighted")
def get_user_recommendations(
    ratings: Dict[str, float] = Body(
        ..., 
        description="Dictionary of {tmdb_id: rating} pairs for user-based recommendation"
    ),
    request_data: List[Dict[str, Any]] = Body(
        ..., 
        description="List of {'tmdb_id': str, 'metadata': { ... }} objects for content-based recommendation"
    ),
    content_based_dir_path: str = Query(
        default=str(content_based_dir_path),
        description="Path to content-based model"
    ),
    collaborative_dir_path: str = Query(
        default=str(collaborative_dir_path),
        description="Path to collaborative model"
    ),
    n_recommendations: int = Query(
        default=20,
        ge=1,
        le=100
    ),
    min_similarity: float = Query(
        default=0.1,
        ge=0.0,
        le=1.0
    )
):
    """Generate hybrid recommendations using user ratings and metadata."""

    try:
        if not ratings or not request_data:
            raise HTTPException(status_code=400, detail="Ratings and data cannot be empty")

        logger.info(f'Generating recommendations for {len(ratings)} rated items')

        recommendations = WeighedRecommendationService.get_user_recommendations(
            user_ratings=ratings,
            request_data=request_data,
            content_based_dir_path=content_based_dir_path,
            collaborative_dir_path=collaborative_dir_path,
            n_recommendations=n_recommendations,
            min_similarity=min_similarity
        )

        if not recommendations:
            logger.info("No recommendations found for the given ratings and metadata")
            return {"message": "No recommendations found"}

        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations

    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@hybrid_router_v1.post("/recommendations/switching")
def get_user_recommendations(
    ratings: Dict[str, float] = Body(
        ..., 
        description="Dictionary of {tmdb_id: rating} pairs for user-based recommendation"
    ),
    request_data: List[Dict[str, Any]] = Body(
        ..., 
        description="List of {'tmdb_id': str, 'metadata': { ... }} objects for content-based recommendation"
    ),
    content_based_dir_path: str = Query(
        default=str(content_based_dir_path),
        description="Path to content-based model"
    ),
    collaborative_dir_path: str = Query(
        default=str(collaborative_dir_path),
        description="Path to collaborative model"
    ),
    n_recommendations: int = Query(
        default=20,
        ge=1,
        le=100
    ),
    min_similarity: float = Query(
        default=0.000001,
        ge=0.0,
        le=1.0
    )
):
    """Generate hybrid recommendations using user ratings and metadata."""

    try:
        if not ratings or not request_data:
            raise HTTPException(status_code=400, detail="Ratings and data cannot be empty")

        logger.info(f'Generating recommendations for {len(ratings)} rated items')

        recommendations = SwitchingRecommendationService.get_user_recommendations(
            user_ratings=ratings,
            request_data=request_data,
            content_based_dir_path=content_based_dir_path,
            collaborative_dir_path=collaborative_dir_path,
            n_recommendations=n_recommendations,
            min_similarity=min_similarity
        )

        if not recommendations:
            logger.info("No recommendations found for the given ratings and metadata")
            return {"message": "No recommendations found"}

        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations

    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
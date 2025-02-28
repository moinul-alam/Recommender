import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, Body, HTTPException, Query
from src.config.content_based_config import ContentBasedConfigV2
from src.config.collaborative_config import CollaborativeConfigV2
from src.models.hybrid.v1.services.switching_recommendation_service import SwitchingRecommendationService
from src.models.hybrid.v1.services.weighted_recommendation_service import WeighedRecommendationService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

hybrid_router_v1 = APIRouter()

content_based_dir_path = ContentBasedConfigV2().DIR_PATH
collaborative_dir_path = CollaborativeConfigV2().DIR_PATH

@hybrid_router_v1.post("/recommendations/weighed")
def get_user_recommendations(
    user_ratings: Dict[str, float] = Body(
        ..., 
        description="Dictionary of {tmdb_id: rating} pairs for user-based recommendation"
    ),
    content_based_dir_path: str = Query(
        default=str(content_based_dir_path),
        description="Path to content based model"
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
    """Generate user-based recommendations using ratings given by a user."""
    try:
        logger.info(f'Generating user-based recommendations for {len(user_ratings)} rated items')

        recommendations = WeighedRecommendationService.get_user_recommendations(
            user_ratings=user_ratings,
            content_based_dir_path=content_based_dir_path,
            collaborative_dir_path=collaborative_dir_path,
            n_recommendations=n_recommendations,
            min_similarity=min_similarity
        )

        if not recommendations:
            logger.info("No recommendations found for the given user ratings")
            return {"message": "No recommendations found"}

        logger.info(f"Generated {len(recommendations)} user-based recommendations")
        return recommendations

    except Exception as e:
        logger.error(f"Error generating user-based recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@hybrid_router_v1.post("/recommendations/switching")
def get_user_recommendations(
    user_ratings: Dict[str, float] = Body(
        ..., 
        description="Dictionary of {tmdb_id: rating} pairs for user-based recommendation"
    ),
    content_based_dir_path: str = Query(
        default=str(content_based_dir_path),
        description="Path to content based model"
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
    """Generate user-based recommendations using ratings given by a user."""
    try:
        logger.info(f'Generating user-based recommendations for {len(user_ratings)} rated items')

        recommendations = SwitchingRecommendationService.get_user_recommendations(
            user_ratings=user_ratings,
            content_based_dir_path=content_based_dir_path,
            collaborative_dir_path=collaborative_dir_path,
            n_recommendations=n_recommendations,
            min_similarity=min_similarity
        )

        if not recommendations:
            logger.info("No recommendations found for the given user ratings")
            return {"message": "No recommendations found"}

        logger.info(f"Generated {len(recommendations)} user-based recommendations")
        return recommendations

    except Exception as e:
        logger.error(f"Error generating user-based recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

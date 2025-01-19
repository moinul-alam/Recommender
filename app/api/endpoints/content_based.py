# app/api/endpoints/content_based.py
from fastapi import APIRouter, HTTPException, Depends
from app.models.content_based.recommender import ContentBasedRecommender
from app.schemas.responses import RecommendationResponse
from app.services.recommendation_service import get_recommender

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@router.get("/similar/{tmdb_id}", response_model=RecommendationResponse)
async def get_recommendations(
    tmdb_id: int,
    k: int = 20,
    recommender: ContentBasedRecommender = Depends(get_recommender)
):
    try:
        return recommender.find_similar_movies(tmdb_id, k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

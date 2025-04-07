from fastapi import APIRouter

from src.recommenders.content_based.v1.endpoint import content_based_router_v1

router = APIRouter()

router.include_router(content_based_router_v1, prefix="/content-based/v1", tags=["Content-Based Recommender (Version 1: TF-IDF, SVD, FAISS)"])

@router.get("/health", tags=["Utility"])
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "healthy"}

@router.get("/info", tags=["Utility"])
async def info():
    """
    Provides basic service information and available recommendation methods.
    """
    return {
        "status": "success",
        "data": {
            "methods": ["content-based", "collaborative", "hybrid"],
            "version": "1.0.0",
            "description": "Recommendation system backend"
        },
        "message": "Service information"
    }
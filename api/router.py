from fastapi import APIRouter
from api.endpoints.content_based_route import content_based_router
from api.endpoints.collaborative_route import collaborative_router
from api.endpoints.hybrid_route import hybrid_router

router = APIRouter()

# Include recommendation-specific routers
router.include_router(content_based_router, prefix="/content-based/v1", tags=["Content-Based Recommender (Version 1)"])
router.include_router(collaborative_router, prefix="/collaborative/v1", tags=["Collaborative Recommender"])
router.include_router(hybrid_router, prefix="/hybrid/v1", tags=["Hybrid Recommender"])


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
from fastapi import APIRouter
from src.api.content_based.v1.endpoint import content_based_router_v1
from src.api.content_based.v2.endpoint import content_based_router_v2
from src.api.collaborative.v1.endpoint import collaborative_router
from src.api.hybrid.v1.endpoint import hybrid_router

router = APIRouter()

router.include_router(content_based_router_v2, prefix="/content-based/v2", tags=["Content-Based Recommender (Version 2)"])
router.include_router(content_based_router_v1, prefix="/content-based/v1", tags=["Content-Based Recommender (Version 1)"])
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
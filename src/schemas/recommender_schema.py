# schemas/responses.py
import logging
from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Optional

class PipelineResponse(BaseModel):
    status: str
    message: Optional[str] = None
    output: Optional[str] = None

class Item(BaseModel):
    """Item to be rated by user"""
    movieId: Optional[int] = None
    tmdbId: Optional[int] = None
    rating: Optional[float] = None

class RecommendationRequest(BaseModel):
    """Request model for recommendation API"""
    req_source: str = "tmdb" # "tmdb" or "movieId"
    n_recommendations: int = 20
    items: List[Item]

class Recommendation(BaseModel):
    """Individual movie recommendation"""
    movie_source: str
    movieId: Optional[int] = None
    tmdbId: Optional[int] = None
    item_title: Optional[str] = None
    predicted_rating: Optional[float] = None
    similarity: float

class RecommendationCategory(BaseModel):
    """Category of recommendations (item-based, user-based, content-based, etc.)"""
    recommendation_type: str
    recommendations: List[Recommendation]

class RecommendationResponse(BaseModel):
    """Response model for recommendation API"""
    status: str
    recommendations: Dict[str, RecommendationCategory]
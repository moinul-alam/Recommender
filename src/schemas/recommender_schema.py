# schemas/responses.py
import logging
from pydantic import BaseModel, Field
from typing import List, Optional

class PipelineResponse(BaseModel):
    status: str
    message: Optional[str] = None
    output: Optional[str] = None

class RecommendationRequestedItem(BaseModel):
    tmdb_id: int
    rating: Optional[float] = None

class RecommendationRequest(BaseModel):
    items: List[RecommendationRequestedItem]
    n_recommendations: Optional[int] = 10
    
class Recommendation(BaseModel):
    tmdb_id: int
    item_title: Optional[str] = None
    predicted_rating: Optional[float] = None
    similarity: float

class RecommendationResponse(BaseModel):
    status: str
    recommendations: List[Recommendation]
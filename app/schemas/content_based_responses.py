# schemas/responses.py
from pydantic import BaseModel, Field
from typing import List, Optional

class PreprocessingResponse(BaseModel):
    message: str
    processed_segments: int
    saved_path: str

class EngineeringResponse(BaseModel):
    message: str
    engineered_dataset: int
    saved_path: str
    engineered_dataset: Optional[List[str]] = None

class SimilarityResponse(BaseModel):
    message: str
    saved_path: str

class Metadata(BaseModel):
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    tagline: Optional[str] = None
    genres: List[str] = Field(default_factory=list)  
    director: List[str] = Field(default_factory=list)
    cast: List[str] = Field(default_factory=list)

class RecommendationRequest(BaseModel):
    tmdbId: int
    metadata: Optional[Metadata] = None
     
class Recommendation(BaseModel):
    tmdbId: int
    similarity: str

class RecommendationResponse(BaseModel):
    message: str
    queriedMedia: int
    similarMedia: List[Recommendation]
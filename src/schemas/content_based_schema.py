# schemas/responses.py
import logging
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PipelineResponse(BaseModel):
    status: str
    message: Optional[str] = None
    output: Optional[str] = None
    
class Metadata(BaseModel):
    media_type: str
    title: Optional[str]
    overview: Optional[str] = None
    spoken_languages: Optional[List[str]] = Field(default_factory=list)
    vote_average: Optional[float] = None
    release_year: Optional[str] = None
    genres: Optional[List[str]] = Field(default_factory=list)
    director: Optional[List[str]] = Field(default_factory=list)
    cast: Optional[List[str]] = Field(default_factory=list)
    keywords: Optional[List[str]] = Field(default_factory=list)

class RecommendationItem(BaseModel):
    tmdb_id: int
    rating: Optional[float] = None
    metadata: Optional[Metadata] = None

class RecommendationRequest(BaseModel):
    items: List[RecommendationItem]
    num_recommendations: Optional[int] = 10
    
class Recommendation(BaseModel):
    tmdb_id: int
    item_title: str
    similarity: float

class RecommendationResponse(BaseModel):
    status: str
    recommendations: List[Recommendation]




class EvaluationRequest(BaseModel):
    index_path: str
    feature_matrix_path: str
    num_test_queries: Optional[int] = 100

class IndexStats(BaseModel):
    total_vectors: int
    dimension: int
    is_trained: bool
    index_type: str

class EvaluationResponse(BaseModel):
    stats: IndexStats  
    recall: float
    precision: float
    mAP: float
    NDCG: float
    query_coverage: float
    latency: Tuple[float, float] 
    compression_ratio: float
    memory_usage: int


# schemas/responses.py
import logging
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PipelineResponse(BaseModel):
    status: str
    output: Optional[int] = None
    output_path: Optional[str] = None

class EngineeringResponse(BaseModel):
    status: str
    featured_segments: int
    saved_path: str

class TrainingResponse(BaseModel):
    status: str
    saved_path: str

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
    
    # @field_validator('*')
    # @classmethod
    # def check_fields(cls, v, info):
    #     logger.info(f"Validating {info.field_name}: {v}")
    #     return v

class RecommendationRequest(BaseModel):
    tmdb_id: int
    metadata: Optional[Metadata] = None
    
    # @field_validator('*')
    # @classmethod
    # def check_fields(cls, v, info):
    #     logger.info(f"Validating {info.field_name}: {v}")
    #     return v
     
class Recommendation(BaseModel):
    tmdb_id: int
    similarity: str

class RecommendationResponse(BaseModel):
    status: str
    queriedMedia: int
    similarMedia: List[Recommendation]

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


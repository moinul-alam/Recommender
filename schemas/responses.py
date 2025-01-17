# schemas/responses.py
from pydantic import BaseModel
from typing import List

class SimilarMedia(BaseModel):
    tmdbId: int
    similarity: float

class RecommendationResponse(BaseModel):
    queriedMedia: int
    similarMedia: List[SimilarMedia]

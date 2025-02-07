# schemas/responses.py
import logging
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class UserRating(BaseModel):
    user_id: Optional[str] = None
    ratings: Optional[Dict[int, float]] = None

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, float]]
    is_guest: bool

class TrainingStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class ModelTrainingResponse(BaseModel):
    status: TrainingStatus
    message: str
    metrics: Optional[Dict[str, float]] = None
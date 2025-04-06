# schemas/responses.py
import logging
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PipelineResponse(BaseModel):
    status: str
    message: str
    output_path: Optional[str] = None
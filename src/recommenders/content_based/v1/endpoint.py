import logging
from fastapi import APIRouter, HTTPException, Query, Request
from src.config.config import BaseConfig
from src.recommenders.content_based.v1.DataPreparation import DataPreparation

from src.schemas.content_based_schema import PipelineResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

version = 1
config = BaseConfig()
content_based_dir_path = config.CONTENT_BASED_PATH / f"v{version}"
logger.info(f"Using content-based directory: {content_based_dir_path}")
content_based_router_v1 = APIRouter()

"""
Data Preparation from raw data
"""
@content_based_router_v1.post("/data-preparation")
async def prepare_data(
    content_based_dir_path: str = Query(
        default=str(content_based_dir_path),  
        description="Path to the directory"
    ),
    dataset_name: str = Query(
        default="coredb.media.json",
        description="Path to the dataset file (json)"
    )
):
    """
    API endpoint to trigger data preparation for content-based recommendation.
    """
    try:
        return DataPreparation.prepare_data(
            content_based_dir_path=content_based_dir_path,
            dataset_name=dataset_name
        )
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error preparing data: {str(e)}")
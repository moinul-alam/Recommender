import logging
from fastapi import APIRouter, HTTPException, Query, Request
from src.config.config import BaseConfig
from src.services.content_based import DataPreparation

from src.schemas.content_based_schema import PipelineResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Defining the version and directory of the content-based recommender system
version = 1 
config = BaseConfig()
content_based_dir_path = config.CONTENT_BASED_PATH / f"v{version}"

logger.info(f"Using content-based directory: {content_based_dir_path}")

content_based_router = APIRouter()

"""
Data Preparation from raw data
"""
@content_based_router.post("/data-preparation")
async def prepare_data(
    content_based_dir_path: str = Query(
        default=str(content_based_dir_path),  
        description="Path to the raw dataset file (json)"
    ),
    raw_dataset_name: str = Query(
        default=str("coredb.media.json"),
        description="Path to the raw dataset file (json)"
    )
):
    try:
        return PreparationService.prepare_data(
            content_based_dir_path=content_based_dir_path,
            raw_dataset_name=raw_dataset_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing data: {str(e)}")

"""
Data Preprocessing
"""
@content_based_router.post("/data-preprocessing")
async def preprocess_data(
    content_based_dir_path: str = Query(
        default=str(content_based_dir_path),  
        description="Path to the dataset file"
    ),
    segment_size: int = Query(
        default=6000,
        description="Number of rows per segment (default is 6000)"
    )
):
    try:
        return PreprocessingService.preprocess_data(
            content_based_dir_path=content_based_dir_path,
            segment_size=segment_size
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing data: {str(e)}")

"""
Feature Engineering
"""
@content_based_router.post("/feature-engineering")
async def engineer_features(
    content_based_dir_path: str = Query(
        default=str(content_based_dir_path),  
        description="Folder to save preprocessed datasets"
    ),
):
    try:
        return EngineeringService.engineer_features(
            content_based_dir_path=content_based_dir_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing data: {str(e)}")

"""
Model Training
"""
@content_based_router.post("/model-training")
async def train_model(
    content_based_dir_path: str = Query(
        default=str(content_based_dir_path),
        description="Path to the folder containing feature-engineered datasets."
    )
):
    try:
        return TrainingService.train_model(
            content_based_dir_path=content_based_dir_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model training: {str(e)}")

"""
Recommendation
"""
@content_based_router.post("/similar")
async def recommendations(
    request: Request,
    recommendation_request: RecommendationRequest,
    content_based_dir_path: str = Query(
        default=str(content_based_dir_path), 
        description="Directory path to the content-based model"
        ),
    n_items: int = Query(default=20, ge=1, le=100, description="Number of recommendations to return (1-100)"),
):
    # body = await request.json()
    # logging.info(f"Raw request body: {body}")
    logging.info(f"Sending Request to Content Based Recommendation Service V2")
    
    try:
        return RecommendationService.recommendation_service(
            recommendation_request = recommendation_request,
            content_based_dir_path = content_based_dir_path,
            n_items = n_items
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding recommendation: {str(e)}")

"""
Discover media using custom query
"""
@content_based_router.post("/discover")
async def discover_media(
    request: Request,
    recommendation_request: RecommendationRequest,
    content_based_dir_path: str = Query(
        default=str(content_based_dir_path), 
        description="Features folder"
        ),
    n_items: int = Query(default=10, ge=1, le=100, description="Number of recommendations to return (1-100)"),
):
    # body = await request.json()
    # logging.info(f"Raw request body: {body}")
    logging.info(f"Sending Request to Service")
    
    try:
        return DiscoveryService.discover_media(
            recommendation_request = recommendation_request,
            content_based_dir_path = content_based_dir_path,
            n_items = n_items
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding recommendation: {str(e)}")

@content_based_router.post("/evaluate-index", response_model=EvaluationResponse)
async def evaluate_index(
    content_based_dir_path: str = Query(
        default=str(content_based_dir_path),
        description="Folder to index"
    ),
    num_test_queries: int = Query(default=100,
      description="Number of test queries"),
):

    try:
        return EvaluationService.evaluate_index(
            content_based_dir_path=content_based_dir_path,
            num_test_queries=num_test_queries
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating index: {str(e)}")
    
    
@content_based_router.post("/execute-pipeline")
async def execute_full_pipeline(
    content_based_dir_path: str = Query(
        default=str(content_based_dir_path),
        description="Path to the dataset file"
    ),
    raw_dataset_name: str = Query(
        default=str("coredb.media.json"),
        description="Path to the raw dataset file (json)"
    ),
    segment_size: int = Query(
        default=6000,
        description="Number of rows per segment"
    )
):
    try:
        result = PipelineService.execute_full_pipeline(
            content_based_dir_path=content_based_dir_path,
            raw_dataset_name=raw_dataset_name,
            segment_size=segment_size
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")
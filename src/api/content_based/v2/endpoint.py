import logging
from fastapi import APIRouter, HTTPException, Query, Request
from src.config.content_based_config import ContentBasedConfigV2
from src.models.content_based.v2.services.pipeline_service import PipelineService
from src.models.content_based.v2.services.preparation_service import PreparationService
from src.models.content_based.v2.services.preprocessing_service import PreprocessingService
from src.models.content_based.v2.services.engineering_service import EngineeringService
from src.models.content_based.v2.services.training_service import TrainingService
from src.models.content_based.v2.services.recommendation_service import RecommendationService
from src.models.content_based.v2.services.evaluation_service import EvaluationService
from src.schemas.content_based_schema import RecommendationRequest, EvaluationResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

content_based_router_v2 = APIRouter()

# Initialize path configuration
raw_dataset_path = ContentBasedConfigV2().RAW_DATA_PATH
prepared_folder_path = ContentBasedConfigV2().PREPARED_FOLDER_PATH
processed_folder_path = ContentBasedConfigV2().PROCESSED_FOLDER_PATH
features_folder_path = ContentBasedConfigV2().FEATURES_FOLDER_PATH
transformers_folder_path = ContentBasedConfigV2().TRANSFORMERS_FOLDER_PATH
model_folder_path = ContentBasedConfigV2().MODEL_FOLDER_PATH


@content_based_router_v2.post("/execute-pipeline")
async def execute_full_pipeline(
    raw_dataset_path: str = Query(
        default=str(raw_dataset_path),
        description="Path to the dataset file"
    ),
    prepared_folder_path: str = Query(
        default=str(prepared_folder_path),
        description="Folder to save prepare datasets"
    ),
    processed_folder_path: str = Query(
        default=str(processed_folder_path),
        description="Folder to save preprocessed datasets"
    ),
    features_folder_path: str = Query(
        default=str(features_folder_path),
        description="Folder to save feature engineered datasets"
    ),
    transformers_folder_path: str = Query(
        default=str(transformers_folder_path),  
        description="Folder to save preprocessed datasets"
    ),
    model_folder_path: str = Query(
        default=str(model_folder_path),
        description="Path to save trained model"
    ),
    segment_size: int = Query(
        default=6000,
        description="Number of rows per segment"
    ),
    metric: str = Query(
        default="L2",
        description="Similarity metric for FAISS (L2 or Inner Product)"
    )
):
    try:
        result = PipelineService.execute_full_pipeline(
            raw_dataset_path=raw_dataset_path,
            prepared_folder_path=prepared_folder_path,
            processed_folder_path=processed_folder_path,
            features_folder_path=features_folder_path,
            transformers_folder_path=transformers_folder_path,
            model_folder_path=model_folder_path,
            segment_size=segment_size,
            metric=metric
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")

"""
Data Preparation from raw data
"""
@content_based_router_v2.post("/data-preparation")
async def prepare_data(
    raw_dataset_path: str = Query(
        default=str(raw_dataset_path),  
        description="Path to the raw dataset file (json)"
    ),
    prepared_folder_path: str = Query(
        default=str(prepared_folder_path),  
        description="Folder to save prepared datasets"
    )
):
    try:
        return PreparationService.prepare_data(
            raw_dataset_path=raw_dataset_path,
            prepared_folder_path=prepared_folder_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing data: {str(e)}")

"""
Data Preprocessing
"""
@content_based_router_v2.post("/data-preprocessing")
async def preprocess_data(
    prepared_folder_path: str = Query(
        default=str(prepared_folder_path),  
        description="Path to the dataset file"
    ),
    processed_folder_path: str = Query(
        default=str(processed_folder_path),  
        description="Folder to save preprocessed datasets"
    ),
    segment_size: int = Query(
        default=6000,
        description="Number of rows per segment (default is 6000)"
    )
):
    try:
        return PreprocessingService.preprocess_data(
            prepared_folder_path=prepared_folder_path,
            processed_folder_path=processed_folder_path,
            segment_size=segment_size
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing data: {str(e)}")

"""
Feature Engineering
"""
@content_based_router_v2.post("/feature-engineering")
async def engineer_features(
    processed_folder_path: str = Query(
        default=str(processed_folder_path),  
        description="Folder to save preprocessed datasets"
    ),
    features_folder_path: str = Query(
        default=str(features_folder_path),
        description="Folder to save feature engineered datasets"
    ),
    transformers_folder_path: str = Query(
        default=str(transformers_folder_path),  
        description="Folder to save preprocessed datasets"
    ),
):
    try:
        return EngineeringService.engineer_features(
            processed_folder_path=processed_folder_path,
            features_folder_path=features_folder_path,
            transformers_folder_path=transformers_folder_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing data: {str(e)}")

"""
Model Training
"""
@content_based_router_v2.post("/model-training")
async def train_model(
    features_folder_path: str = Query(
        default=str(features_folder_path),
        description="Path to the folder containing feature-engineered datasets."
    ),
    model_folder_path: str = Query(
        default=str(model_folder_path),
        description="Path to the folder where the model will be saved."
    ),
    metric: str = Query(
        default="L2",
        description="Similarity metric to use for FAISS (e.g., L2 or Inner Product)."
    ),
):
    try:
        return TrainingService.train_model(
            features_folder_path=features_folder_path,
            model_folder_path=model_folder_path,
            metric=metric
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model training: {str(e)}")

"""
Recommendation
"""
@content_based_router_v2.post("/similar")
async def recommendations(
    request: Request,
    recommendation_request: RecommendationRequest,
    features_folder_path: str = Query(
        default=str(features_folder_path), 
        description="Features folder"
        ),
    model_folder_path: str = Query(
        default=str(model_folder_path), 
        description="Path to the model"
        ),
    processed_folder_path: str = Query(
        default=str(processed_folder_path),
        description="Folder to save preprocessed datasets"
    ),
    transformers_folder_path: str = Query(
        default=str(transformers_folder_path),  
        description="Folder to save preprocessed datasets"
    ),
    n_items: int = Query(default=10, ge=1, le=100, description="Number of recommendations to return (1-100)"),
):
    # body = await request.json()
    # logging.info(f"Raw request body: {body}")
    logging.info(f"Sending Request to Service")
    
    try:
        return RecommendationService.recommendation_service(
            recommendation_request = recommendation_request,
            features_folder_path = features_folder_path,
            model_folder_path = model_folder_path,
            processed_folder_path=processed_folder_path,
            transformers_folder_path=transformers_folder_path,
            n_items = n_items
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding recommendation: {str(e)}")
    

@content_based_router_v2.post("/evaluate-index", response_model=EvaluationResponse)
async def evaluate_index(
    model_folder_path: str = Query(
        default=str(model_folder_path),
        description="Folder to index"
    ),
    features_folder_path: str = Query(
        default=str(features_folder_path),  
        description="Folder to feature matrix"
    ),
    num_test_queries: int = Query(default=100,
      description="Number of test queries"),
):

    try:
        return EvaluationService.evaluate_index(
            model_folder_path=model_folder_path,
            features_folder_path=features_folder_path,
            num_test_queries=num_test_queries
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating index: {str(e)}")


from fastapi import APIRouter, HTTPException, Query
from src.config.content_based_config import ContentBasedConfigV1
from src.models.content_based.v1.services.pipeline_service import PipelineService
from src.models.content_based.v1.services.preprocessing_service import PreprocessingService
from src.models.content_based.v1.services.engineering_service import EngineeringService
from src.models.content_based.v1.services.training_service import TrainingService
from src.models.content_based.v1.services.recommendation_service import RecommendationService
from src.schemas.content_based_schema import RecommendationRequest

content_based_router_v1 = APIRouter()

# Initialize path configuration
dataset_path = ContentBasedConfigV1().RAW_DATA_PATH
processed_folder_path = ContentBasedConfigV1().PROCESSED_FOLDER_PATH
features_folder_path = ContentBasedConfigV1().FEATURES_FOLDER_PATH
model_folder_path = ContentBasedConfigV1().MODEL_FOLDER_PATH


@content_based_router_v1.post("/execute-pipeline")
async def execute_full_pipeline(
    dataset_path: str = Query(
        default=str(dataset_path),
        description="Path to the dataset file"
    ),
    processed_folder_path: str = Query(
        default=str(processed_folder_path),
        description="Folder to save preprocessed datasets"
    ),
    features_folder_path: str = Query(
        default=str(features_folder_path),
        description="Folder to save feature engineered datasets"
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
            dataset_path=dataset_path,
            processed_folder_path=processed_folder_path,
            features_folder_path=features_folder_path,
            model_folder_path=model_folder_path,
            segment_size=segment_size,
            metric=metric
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")

"""
Data Preprocessing
"""
@content_based_router_v1.post("/data-preprocessing")
async def preprocess_data(
    dataset_path: str = Query(
        default=str(dataset_path),  
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
            dataset_path=dataset_path,
            processed_folder_path=processed_folder_path,
            segment_size=segment_size
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing data: {str(e)}")

"""
Feature Engineering
"""
@content_based_router_v1.post("/feature-engineering")
async def engineer_feature(
    processed_folder_path: str = Query(
        default=str(processed_folder_path),  
        description="Folder to save preprocessed datasets"
    ),
    features_folder_path: str = Query(
        default=str(features_folder_path),
        description="Folder to save feature engineered datasets"
    )
):
    try:
        return EngineeringService.engineer_feature(
            processed_folder_path=processed_folder_path,
            features_folder_path=features_folder_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing data: {str(e)}")

"""
Model Training
"""
@content_based_router_v1.post("/model-training")
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
@content_based_router_v1.post("/similar")
async def recommendations(
    recommendation_request: RecommendationRequest,
    features_folder_path: str = Query(
        default=str(features_folder_path), 
        description="Features folder"
        ),
    model_folder_path: str = Query(
        default=str(model_folder_path), 
        description="Path to the model"
        ),
    n_items: int = Query(default=10, ge=1, le=100, description="Number of recommendations to return (1-100)"),
):
    try:
        return RecommendationService.recommendation_service(
            recommendation_request = recommendation_request,
            features_folder_path = features_folder_path,
            model_folder_path = model_folder_path,
            n_items = n_items
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding recommendation: {str(e)}")
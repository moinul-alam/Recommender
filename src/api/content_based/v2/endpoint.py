import logging
from fastapi import APIRouter, HTTPException, Query, Request
from src.config.config import BaseConfig
from src.models.content_based.v2.services.pipeline_service import PipelineService
from src.models.content_based.v2.services.data_preparation_service import DataPreparationService
from src.models.content_based.v2.services.data_preprocessing_service import DataPreprocessingService
from src.models.content_based.v2.services.feature_engineering_service import FeatureEngineeringService
from src.models.content_based.v2.services.indexing_service import IndexingService

from src.models.content_based.v2.services.recommendation_service import RecommendationService
from src.models.content_based.v2.services.discovery_service import DiscoveryService
from src.models.content_based.v2.services.evaluation_service import EvaluationService
from src.schemas.content_based_schema import RecommendationRequest, EvaluationResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

version = 2
config = BaseConfig()
content_based_dir_path = config.CONTENT_BASED_PATH / f"v{version}"
content_based_router_v2 = APIRouter()

# Define constants for dataset/file/model names

file_names = {
    "dataset_name": "1_coredb.media.json",
    "prepared_dataset_name": "2_prepared_dataset",
    "item_map_name": "2_item_map",
    "preprocessed_dataset_name": "3_preprocessed_dataset",
    "preprocessed_segment_name": "3_processed_segment_",
    "tfidf_overview": "4_tfidf_overview",
    "tfidf_keywords": "4_tfidf_keywords",
    "mlb_genres": "4_mlb_genres",
    "svd_overview": "4_svd_overview",
    "svd_keywords": "4_svd_keywords",
    "pca": "4_pca",
    "feature_matrix_name": "4_feature_matrix",
    "model_config_name": "4_model_config",
    "index_name": "5_similarity_index"
}

"""
Data Preparation from raw data
"""
@content_based_router_v2.post("/data-preparation")
async def prepare_data(
    content_based_dir_path: str = Query(
        default=str(content_based_dir_path),  
        description="Path to the diretory"
    )
):
    try:
        return DataPreparationService.prepare_data(
            content_based_dir_path, file_names
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing data: {str(e)}")

"""
Data Preprocessing
"""
@content_based_router_v2.post("/data-preprocessing")
async def preprocess_data(
    content_based_dir_path: str = Query(
        default=str(content_based_dir_path),  
        description="Path to the prepared dataset file"
    ),
    segment_size: int = Query(
        default=10000,
        description="Number of rows per segment (default is 10000)"
    )
):
    try:
        return DataPreprocessingService.preprocess_data(
            content_based_dir_path, segment_size, file_names
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing data: {str(e)}")

"""
Feature Engineering
"""
@content_based_router_v2.post("/feature-engineering")
async def engineer_features(
    content_based_dir_path: str = Query(
        default=str(content_based_dir_path),  
        description="Folder to save preprocessed datasets"
    )
):
    try:
        return FeatureEngineeringService.engineer_features(
            content_based_dir_path, file_names
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing data: {str(e)}")

"""
Model Training
"""
@content_based_router_v2.post("/model-training")
async def train_model(
    content_based_dir_path: str = Query(
        default=str(content_based_dir_path),
        description="Path to the folder containing feature-engineered datasets."
    )
):
    try:
        return IndexingService.create_index(
            content_based_dir_path, file_names
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
@content_based_router_v2.post("/discover")
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

# @content_based_router_v2.post("/evaluate-index", response_model=EvaluationResponse)
# async def evaluate_index(
#     content_based_dir_path: str = Query(
#         default=str(content_based_dir_path),
#         description="Folder to index"
#     ),
#     num_test_queries: int = Query(default=100,
#       description="Number of test queries"),
# ):

#     try:
#         return EvaluationService.evaluate_index(
#             content_based_dir_path=content_based_dir_path,
#             num_test_queries=num_test_queries
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error evaluating index: {str(e)}")

# @content_based_router_v2.post("/execute-pipeline")
# async def execute_full_pipeline(
#     content_based_dir_path: str = Query(
#         default=str(content_based_dir_path),
#         description="Path to the dataset file"
#     ),
#     dataset_name: str = Query(
#         default=str("1_coredb.media.json"),
#         description="Path to the raw dataset file (json)"
#     ),
#     segment_size: int = Query(
#         default=6000,
#         description="Number of rows per segment"
#     )
# ):
#     try:
#         logger.info(f"Using content-based directory: {content_based_dir_path}")
#         result = PipelineService.execute_full_pipeline(
#             content_based_dir_path=content_based_dir_path,
#             dataset_name=dataset_name,
#             segment_size=segment_size
#         )
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")

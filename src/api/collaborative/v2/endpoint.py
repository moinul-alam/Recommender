from typing import Dict
from fastapi import APIRouter, Body, HTTPException, Query

from src.config.config import BaseConfig

from src.models.collaborative.v2.services.pipeline_service import PipelineService
from src.models.collaborative.v2.services.data_preprocessing_service import DataPreprocessingService
from src.models.collaborative.v2.services.feature_extraction_service import FeatureExtractionService
from src.models.collaborative.v2.services.indexing_service import IndexingService
from src.models.collaborative.v2.services.recommendation_service import RecommendationService

from src.models.collaborative.v2.services.user_recommendation_service import UserRecommendationService
from src.models.collaborative.v2.services.item_recommendation_service import ItemRecommendationService
from src.models.collaborative.v2.services.evaluation_service import EvaluationService
from src.schemas.recommender_schema import RecommendationRequest
from src.models.common.logger import app_logger

logger = app_logger(__name__)

version = 2
config = BaseConfig()
collaborative_dir_path  = config.COLLABORATIVE_PATH / f"v{version}"
collaborative_router_v2 = APIRouter()

"""
Data Preprocessing
"""
@collaborative_router_v2.post("/data-preprocessing")
async def process_data(
    collaborative_dir_path: str = Query(
        default=str(collaborative_dir_path),
        description="Path to the directory containing dataset and model files"
    ),
):
    logger.info(f"Received data preprocessing request in the route Collaborative v{version}")

    try:
        result = DataPreprocessingService.process_data(
            directory_path = collaborative_dir_path
        )
        
        if result is None:
            logger.error("Data preprocessing failed")
            raise HTTPException(status_code=500, detail="Preprocessing failed due to an internal error.")
        
        logger.info(f"Data preprocessing completed successfully: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error during data preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


"""
Feature Engineering and Dimensionality Reduction
"""
@collaborative_router_v2.post("/feature-extraction")
def extract_features(
    collaborative_dir_path: str = Query(
        default=str(collaborative_dir_path),
        description="Path to the directory containing dataset and model files"
    )
):
    logger.info(
        f"Model training request received | "
        f"Directory Path: {collaborative_dir_path}"
    )

    try:
        result = FeatureExtractionService.extract_features(
            directory_path = collaborative_dir_path
        ) 
        
        if not result:
            logger.error("Feature Extraction failed due to missing or incorrect data.")
            raise HTTPException(status_code=400, detail="Feature Extraction failed. Check dataset and parameters.")
        
        logger.info(f"Feature Extraction completed successfully. Output: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Unexpected error during Feature Extraction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during Feature Extraction.")


"""
Index Creation
"""
@collaborative_router_v2.post("/create-index")
def create_index(
    collaborative_dir_path: str = Query(
        default=str(collaborative_dir_path),
        description="Path to the directory containing dataset and model files"
    )
):
    try:
        result = IndexingService.create_index(
            directory_path = collaborative_dir_path
        )
        
        if not result:
            logger.error("Index creation failed due to missing or incorrect data.")
            raise HTTPException(status_code=400, detail="Index creation failed. Check dataset and parameters.")
        
        logger.info(f"Index created successfully. Output: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Unexpected error during Index Creation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during Index Creation.")

"""
Recommendation Endpoints
"""
@collaborative_router_v2.post("/recommendations")
def get_recommendations(
    recommendation_request: RecommendationRequest,
    collaborative_dir_path: str = Query(
        default=str(collaborative_dir_path),
        description="Path to the directory"
    ),
):
    logger.info(f"Received recommendation request: {recommendation_request}")
    
    try:
        return RecommendationService.get_recommendations(
            recommendation_request,
            directory_path=collaborative_dir_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding recommendation: {str(e)}")

"""
Item based Recommendation Endpoints
"""
@collaborative_router_v2.post("/recommendations/item-based")
def get_item_recommendations(
    recommendation_request: RecommendationRequest,
    collaborative_dir_path: str = Query(
        default=str(collaborative_dir_path),
        description="Path to preprocessed dataset"
    )
):
    try:
        logger.info(f"Received item-based recommendation request...")
        
        recommendations = ItemRecommendationService.get_item_recommendations(
            recommendation_request,
            directory_path = collaborative_dir_path
        )

        if not recommendations:
            logger.info("No recommendations found for the given items")
            return {"message": "No recommendations found"}

        logger.info(f"Generated {len(recommendations)} item-based recommendations")
        return recommendations

    except Exception as e:
        logger.error(f"Error generating item-based recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@collaborative_router_v2.post("/recommendations/user-based")
def get_user_recommendations(
    user_ratings: Dict[str, float] = Body(
        ..., 
        description="Dictionary of {tmdb_id: rating} pairs for user-based recommendation"
    ),
    collaborative_dir_path: str = Query(
        default=str(collaborative_dir_path),
        description="Path to preprocessed dataset"
    ),
    n_recommendations: int = Query(
        default=20,
        ge=1,
        le=100
    ),
    similarity_metric: str = Query(
        default='cosine',
        description="Similarity calculation method (euclidean/cosine)"
    ),
    min_similarity: float = Query(
        default=0,
        ge=0.0,
        le=1.0
    )
):
    """Generate user-based recommendations using ratings given by a user."""
    try:
        logger.info(f'Generating user-based recommendations for {len(user_ratings)} rated items')

        recommendations = UserRecommendationService.get_user_recommendations(
            user_ratings,
            collaborative_dir_path,
            n_recommendations,
            similarity_metric,
            min_similarity
        )

        if not recommendations:
            logger.info("No recommendations found for the given user ratings")
            return {"message": "No recommendations found"}

        logger.info(f"Generated {len(recommendations)} user-based recommendations")
        return recommendations

    except Exception as e:
        logger.error(f"Error generating user-based recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@collaborative_router_v2.post("/evaluation")
def evaluate(
    collaborative_dir_path: str = Query(
        default=str(collaborative_dir_path),
        description="Path to preprocessed dataset"
    ),
    sample_test_size: int = Query(
        default=10
    ),
    k: int = Query(
        default=10,
        ge=1,
        le=100
    ),
):
    try:
        results = EvaluationService.evaluate_recommender(
            directory_path = collaborative_dir_path,
            sample_test_size = sample_test_size,
            k = k
        )
        return results
    except ValueError as ve:
        logger.error(f"Evaluation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

"""
Full Pipeline Execution
"""    
@collaborative_router_v2.post("/execute-pipeline")
def execute_full_pipeline(
    collaborative_dir_path: str = Query(
        default=str(collaborative_dir_path),
        description="Path to the dataset file"
    )
):
    try:
        result = PipelineService.execute_full_pipeline(
            directory_path=collaborative_dir_path
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")
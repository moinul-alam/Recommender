import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from src.config.config import BaseConfig
from src.models.collaborative.v3.services.pipeline_service import PipelineService
from src.models.collaborative.v3.services.preprocessing_service import PreprocessingService
from src.models.collaborative.v3.services.model_training_service import ModelTrainingService
from src.models.collaborative.v3.services.recommendation_service import RecommendationService
from src.models.collaborative.v3.services.model_evaluation_service import ModelEvaluationService

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize router and config paths
collaborative_router_v3 = APIRouter()
config = CollaborativeConfigV3()
dataset_dir_path = config.DATASET_DIR_PATH
processed_dir_path = config.PROCESSED_DIR_PATH
model_dir_path = config.MODEL_DIR_PATH

# Constants
DEFAULT_NUM_RECOMMENDATIONS = 10
DEFAULT_N_FACTORS = 50
DEFAULT_N_EPOCHS = 10

# Base Models
class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    tmdb_id: int
    score: float

class BaseRecommendationRequest(BaseModel):
    """Base class for recommendation requests"""
    num_recommendations: int = Field(default=DEFAULT_NUM_RECOMMENDATIONS, ge=1, le=100)

class UserRecommendationRequest(BaseRecommendationRequest):
    """Request model for user-based recommendations"""
    tmdb_ids: List[int] = Field(..., min_length=1)
    ratings: Optional[List[float]] = Field(None, min_length=1)

    @field_validator('ratings')
    def validate_ratings(cls, v, values):
        if v is not None:
            if len(v) != len(values.data['tmdb_ids']):
                raise ValueError('Ratings length must match tmdb_ids length')
            if not all(0.0 <= r <= 5.0 for r in v):
                raise ValueError('Ratings must be between 0 and 5')
        return v

class SimilarMoviesRequest(BaseRecommendationRequest):
    """Request model for similar movies recommendations"""
    tmdb_ids: List[int] = Field(..., min_length=1)

@collaborative_router_v3.post("/execute-pipeline")
async def execute_full_pipeline(
    dataset_dir_path: str = Query(default=str(dataset_dir_path), description="Path to the dataset file"),
    processed_dir_path: str = Query(default=str(processed_dir_path), description="Directory to save preprocessed dataset"),
    model_dir_path: str = Query(default=str(model_dir_path), description="Directory to save model components"),
    sparse_user_threshold: int = Query(5, description="Minimum ratings per user"),
    sparse_item_threshold: int = Query(1, description="Minimum ratings per item"),
    split_percent: float = Query(0.8, description="Train-test split ratio"),
    chunk_size: int = Query(10000, description="Chunk size for processing"),
    normalization: Optional[str] = Query(None, description="L1, L2, None"),
    n_neighbors: int = Query(50, description="Number of nearest neighbors to consider"),
    similarity_metric: str = Query("L2", description="Similarity calculation method (euclidean/cosine)"),
    batch_size: int = Query(10000, description="Batch size for similarity matrix computation"),
    min_similarity: float = Query(0.1, description="Minimum similarity threshold")
):
    """Execute the full recommendation pipeline."""
    try:
        result = PipelineService.execute_full_pipeline(
            dataset_dir_path=dataset_dir_path,
            processed_dir_path=processed_dir_path,
            model_dir_path=model_dir_path,
            sparse_user_threshold=sparse_user_threshold,
            sparse_item_threshold=sparse_item_threshold,
            split_percent=split_percent,
            chunk_size=chunk_size,
            normalization=normalization,
            n_neighbors=n_neighbors,
            similarity_metric=similarity_metric,
            batch_size=batch_size,
            min_similarity=min_similarity
        )
        
        if result is None:
            raise HTTPException(status_code=500, detail="Pipeline execution failed")
        
        return result
    except Exception as e:
        logger.error(f"Pipeline execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")

@collaborative_router_v3.post("/data-preprocessing")
async def process_data(
    dataset_dir_path: str = Query(default=str(dataset_dir_path), description="Path to the dataset file"),
    processed_dir_path: str = Query(default=str(processed_dir_path), description="Directory to save preprocessed dataset"),
    sparse_user_threshold: int = Query(5, description="Minimum ratings per user"),
    sparse_item_threshold: int = Query(1, description="Minimum ratings per item"),
    split_percent: float = Query(0.8, description="Train-test split ratio"),
    chunk_size: int = Query(10000, description="Chunk size for processing"),
    normalization: Optional[str] = Query("l2", description="L1, L2, None")
):
    """Preprocess the dataset for model training."""
    logger.info(
        f"Received data preprocessing request with "
        f"dataset_dir_path={dataset_dir_path}, "
        f"processed_dir_path={processed_dir_path}, "
        f"sparse_user_threshold={sparse_user_threshold}, "
        f"sparse_item_threshold={sparse_item_threshold}, "
        f"split_percent={split_percent}, "
        f"chunk_size={chunk_size}, "
        f"normalization={normalization}"
    )

    try:
        result = PreprocessingService.process_data(
            dataset_dir_path=dataset_dir_path,
            processed_dir_path=processed_dir_path,
            sparse_user_threshold=sparse_user_threshold,
            sparse_item_threshold=sparse_item_threshold,
            split_percent=split_percent,
            chunk_size=chunk_size,
            normalization=normalization
        )
        
        if result is None:
            logger.error("Data preprocessing failed")
            raise HTTPException(status_code=500, detail="Preprocessing failed due to an internal error.")
        
        logger.info(f"Data preprocessing completed successfully: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error during data preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@collaborative_router_v3.post("/model-training")
async def train_model(
    processed_dir_path: str = Query(default=str(processed_dir_path), description="Path to preprocessed dataset"),
    model_dir_path: str = Query(default=str(model_dir_path), description="Directory to save model components"),
    n_factors: int = Query(default=DEFAULT_N_FACTORS, ge=50, le=300, description="Number of latent factors"),
    n_epochs: int = Query(default=DEFAULT_N_EPOCHS, ge=1, le=100, description="Number of epochs")
):
    """Train the recommendation model."""
    logger.info(
        f"Received model training request: "
        f"processed_dir_path={processed_dir_path}, "
        f"model_dir_path={model_dir_path}, "
        f"n_factors={n_factors}, "
        f"n_epochs={n_epochs}"
    )

    try:
        result = ModelTrainingService.train_model(
            processed_dir_path=processed_dir_path,
            model_dir_path=model_dir_path,
            n_factors=n_factors,
            n_epochs=n_epochs
        )
        
        if result is None:
            raise HTTPException(status_code=400, detail="Model training failed")
        
        logger.info("Model training completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@collaborative_router_v3.post("/recommendations/user-based", response_model=List[RecommendationResponse])
async def get_recommendations(request: UserRecommendationRequest):
    """Get personalized recommendations based on user ratings."""
    try:
        service = RecommendationService(
            processed_dir_path=processed_dir_path,
            model_dir_path=model_dir_path
        )
        
        # Use default rating of 1.0 if no ratings provided
        ratings = request.ratings if request.ratings else [1.0] * len(request.tmdb_ids)
        
        recommendations = service.get_recommendations_for_user(
            tmdb_ids=request.tmdb_ids,
            ratings=ratings,
            num_recommendations=request.num_recommendations
        )
        
        return [
            RecommendationResponse(tmdb_id=tmdb_id, score=score)
            for tmdb_id, score in recommendations
        ]
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@collaborative_router_v3.post("/recommendations/item-based", response_model=List[RecommendationResponse])
async def get_similar_movies(request: SimilarMoviesRequest):
    """Get similar movie recommendations."""
    try:
        service = RecommendationService(
            processed_dir_path=processed_dir_path,
            model_dir_path=model_dir_path
        )
        
        # Convert list to tuple for caching optimization
        recommendations = service.get_similar_movies(
            tmdb_ids=tuple(request.tmdb_ids),
            num_recommendations=request.num_recommendations
        )
        
        return [
            RecommendationResponse(tmdb_id=tmdb_id, score=score)
            for tmdb_id, score in recommendations
        ]
    except Exception as e:
        logger.error(f"Error in get_similar_movies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@collaborative_router_v3.post("/evaluate-model")
async def evaluate_model(
    processed_dir_path: str = Query(default=str(processed_dir_path), description="Path to preprocessed dataset"),
    model_dir_path: str = Query(default=str(model_dir_path), description="Directory to save model components")
):
    """Evaluate the trained model's performance."""
    try:
        results = ModelEvaluationService.evaluate_model(
            processed_dir_path=processed_dir_path,
            model_dir_path=model_dir_path
        )
        
        if results is None:
            raise HTTPException(status_code=500, detail="Model evaluation failed")
            
        return results
    except ValueError as ve:
        logger.error(f"Evaluation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
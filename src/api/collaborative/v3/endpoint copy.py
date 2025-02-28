import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, conlist
from src.config.collaborative_config import CollaborativeConfigV3
from src.models.collaborative.v3.services.pipeline_service import PipelineService
from src.models.collaborative.v3.services.preprocessing_service import PreprocessingService
from src.models.collaborative.v3.services.model_training_service import ModelTrainingService
from src.models.collaborative.v3.services.recommendation_service import RecommendationService
from src.models.collaborative.v3.services.model_evaluation_service import ModelEvaluationService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

collaborative_router_v3 = APIRouter()

dataset_dir_path = CollaborativeConfigV3().DATASET_DIR_PATH
processed_dir_path = CollaborativeConfigV3().PROCESSED_DIR_PATH
model_dir_path = CollaborativeConfigV3().MODEL_DIR_PATH

@collaborative_router_v3.post("/execute-pipeline")
def execute_full_pipeline(
    dataset_dir_path: str = Query(
        default=str(dataset_dir_path),
        description="Path to the dataset file"
    ),
    processed_dir_path: str = Query(
        default=str(processed_dir_path),
        description="Directory to save preprocessed dataset"
    ),
    model_dir_path: str = Query(
        default=str(model_dir_path),
        description="Directory to save model components"
    ),
    sparse_user_threshold: int = Query(5, description="Minimum ratings per user"),
    sparse_item_threshold: int = Query(1, description="Minimum ratings per item"),
    split_percent: float = Query(0.8, description="Train-test split ratio"),
    chunk_size: int = Query(10000, description="Chunk size for processing"),
    normalization: Optional[str] = Query(None, description="L1, L2, None"),
    n_neighbors: Optional[int] = Query(
        default=50, 
        description="Number of nearest neighbors to consider"
    ),
    similarity_metric: str = Query(
        default='L2', 
        description="Similarity calculation method (euclidean/cosine)"
    ),
    batch_size: int = Query(
        default=10000, 
        description="Batch size for similarity matrix computation"
    ),
    min_similarity: float = Query(
        default=0.1, 
        description="Minimum similarity threshold"
    )
):
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
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")

@collaborative_router_v3.post("/data-preprocessing")
def process_data(
    dataset_dir_path: str = Query(
        default=str(dataset_dir_path),
        description="Path to the dataset file"
    ),
    processed_dir_path: str = Query(
        default=str(processed_dir_path),
        description="Directory to save preprocessed dataset"
    ),
    sparse_user_threshold: int = Query(5, description="Minimum ratings per user"),
    sparse_item_threshold: int = Query(1, description="Minimum ratings per item"),
    split_percent: float = Query(0.8, description="Train-test split ratio"),
    chunk_size: int = Query(10000, description="Chunk size for processing"),
    normalization: Optional[str] = Query(None, description="L1, L2, None")
):
    logger.info(
        f"Received data preprocessing request with "
        f"dataset_dir_path={dataset_dir_path}, "
        f"processed_dir_path={processed_dir_path}, "
        f"sparse_user_threshold={sparse_user_threshold}, "
        f"sparse_item_threshold={sparse_item_threshold}, "
        f"split_percent={split_percent}"
        f"chunk_size={chunk_size}"
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
def train_model(
    processed_dir_path: str = Query(
        default=str(processed_dir_path),
        description="Path to preprocessed dataset"
    ),
    model_dir_path: str = Query(
        default=str(model_dir_path),
        description="Directory to save model components"
    ),
    n_epochs: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Number of epochs"
    ),
):
    logger.info(
        f"Received model training request: "
        f"processed_dir_path={processed_dir_path}, "
        f"model_dir_path={model_dir_path}, "
    )

    try:
        result = ModelTrainingService.train_model(
            processed_dir_path=processed_dir_path,
            model_dir_path=model_dir_path,
            n_factors=100,
            n_epochs=n_epochs
        )
        
        if result is None:
            raise HTTPException(status_code=400, detail="Model training failed")
        
        logger.info(f"Model training completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class ItemRating(BaseModel):
    tmdb_id: int
    rating: float

class MovieRating(BaseModel):
    tmdb_id: int
    rating: float

class UserRecommendationRequest(BaseModel):
    tmdb_ids: List[int]
    ratings: Optional[List[float]] = None
    num_recommendations: Optional[int] = 10

class SimilarMoviesRequest(BaseModel):
    tmdb_ids: conlist(int, min_length=1)
    num_recommendations: Optional[int] = 10

class RecommendationResponse(BaseModel):
    tmdb_id: int
    score: float


@collaborative_router_v3.post("/recommend/user", response_model=List[RecommendationResponse])
async def get_recommendations(request: UserRecommendationRequest):
    try:
        service = RecommendationService(
            processed_dir_path=processed_dir_path,
            model_dir_path=model_dir_path
        )
        
        recommendations = service.get_recommendations_for_user(
            tmdb_ids=request.tmdb_ids,
            ratings=request.ratings if request.ratings else [1.0] * len(request.tmdb_ids),
            num_recommendations=request.num_recommendations
        )
        return [
            RecommendationResponse(tmdb_id=tmdb_id, score=score)
            for tmdb_id, score in recommendations
        ]
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@collaborative_router_v3.post("/recommend/similar", response_model=List[RecommendationResponse])
async def get_similar_movies(request: SimilarMoviesRequest):
    try:
        service = RecommendationService(
            processed_dir_path=processed_dir_path,
            model_dir_path=model_dir_path
        )
        
        recommendations = service.get_similar_movies(
            tmdb_ids=tuple(request.tmdb_ids),  # Convert to tuple for caching
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
def evaluate_model(
    processed_dir_path: str = Query(
        default=str(processed_dir_path),
        description="Path to preprocessed dataset"
    ),
    model_dir_path: str = Query(
        default=str(model_dir_path),
        description="Directory to save model components"
    )
):
    try:
        results = ModelEvaluationService.evaluate_model(
            processed_dir_path=processed_dir_path,
            model_dir_path=model_dir_path
            
        )
        return results
    except ValueError as ve:
        logger.error(f"Evaluation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
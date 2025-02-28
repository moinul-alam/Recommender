import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, Body, HTTPException, Query
from src.config.collaborative_config import CollaborativeConfigV2
from src.models.collaborative.v2.services.pipeline_service import PipelineService
from src.models.collaborative.v2.services.preprocessing_service import PreprocessingService
from src.models.collaborative.v2.services.model_training_service import ModelTrainingService
from src.models.collaborative.v2.services.user_recommendation_service import UserRecommendationService
from src.models.collaborative.v2.services.item_recommendation_service import ItemRecommendationService
from src.models.collaborative.v2.services.model_evaluation_service import ModelEvaluationService

# from src.models.collaborative.v2.services.svd_service import SVDService
  
# from src.models.collaborative.v2.services.recommendation_service import RecommendationService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

collaborative_router_v2 = APIRouter()

collaborative_dir_path = CollaborativeConfigV2().DIR_PATH

@collaborative_router_v2.post("/execute-pipeline")
def execute_full_pipeline(
    collaborative_dir_path: str = Query(
        default=str(collaborative_dir_path),
        description="Path to the dataset file"
    ),
    dataset_name: str = Query(
        default=str("1_movielens_dataset.csv"),
        description="Name of the dataset file"
    ),
    sparse_user_threshold: int = Query(5, description="Minimum ratings per user"),
    sparse_item_threshold: int = Query(1, description="Minimum ratings per item"),
    split_percent: float = Query(0.8, description="Train-test split ratio"),
    chunk_size: int = Query(10000, description="Chunk size for processing"),
    n_neighbors: Optional[int] = Query(
        default=100, 
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
            collaborative_dir_path=collaborative_dir_path,
            dataset_name=dataset_name,
            sparse_user_threshold=sparse_user_threshold,
            sparse_item_threshold=sparse_item_threshold,
            split_percent=split_percent,
            chunk_size=chunk_size,
            n_neighbors=n_neighbors,
            similarity_metric=similarity_metric,
            batch_size=batch_size,
            min_similarity=min_similarity
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")

@collaborative_router_v2.post("/data-preprocessing")
def process_data(
    collaborative_dir_path: str = Query(
        default=str(collaborative_dir_path),
        description="Path to the directory containing dataset and model files"
    ),
    dataset_name: str = Query(
        default=str("1_movielens_dataset.csv"),
        description="Name of the dataset file"
    ),
    sparse_user_threshold: int = Query(5, description="Minimum ratings per user"),
    sparse_item_threshold: int = Query(1, description="Minimum ratings per item"),
    split_percent: float = Query(0.8, description="Train-test split ratio"),
    chunk_size: int = Query(10000, description="Chunk size for processing")
):
    logger.info(
        f"Received data preprocessing request in the route Collaborative v2 "
    )

    try:
        result = PreprocessingService.process_data(
            collaborative_dir_path=collaborative_dir_path,
            dataset_name=dataset_name,
            sparse_user_threshold=sparse_user_threshold,
            sparse_item_threshold=sparse_item_threshold,
            split_percent=split_percent,
            chunk_size=chunk_size
        )
        
        if result is None:
            logger.error("Data preprocessing failed")
            raise HTTPException(status_code=500, detail="Preprocessing failed due to an internal error.")
        
        logger.info(f"Data preprocessing completed successfully: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error during data preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@collaborative_router_v2.post("/model-training")
def train_model(
    collaborative_dir_path: str = Query(
        default=str(collaborative_dir_path),
        description="Path to the directory containing dataset and model files"
    ),
    n_neighbors: Optional[int] = Query(
        default=100, 
        description="Number of nearest neighbors to consider"
    ),
    similarity_metric: str = Query(
        default='L2', 
        description="Similarity calculation method (euclidean/cosine)"
    ),
    min_similarity: float = Query(
        default=0.1, 
        description="Minimum similarity threshold"
    )
):
    logger.info(
        f"Model training request received | "
        f"Directory Path: {collaborative_dir_path}"
        f"Neighbors: {n_neighbors}, Similarity: {similarity_metric}"
    )

    try:
        result = ModelTrainingService.train_model(
            collaborative_dir_path=collaborative_dir_path,
            n_neighbors=n_neighbors,
            similarity_metric=similarity_metric,
            min_similarity=min_similarity
        ) 
        
        if not result:
            logger.error("Model training failed due to missing or incorrect data.")
            raise HTTPException(status_code=400, detail="Model training failed. Check dataset and parameters.")
        
        logger.info(f"Model training completed successfully. Output: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Unexpected error during model training: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during model training.")


@collaborative_router_v2.post("/recommendations/item-based")
def get_item_recommendations(
    items: List[int] = Body(
        ..., 
        description="List of TMDB IDs for item-based recommendation"
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
    min_similarity: float = Query(
        default=0.1,
        ge=0.0,
        le=1.0
    )
):
    """Generate item-based recommendations using only TMDB IDs."""
    try:
        logger.info(f'Generating item-based recommendations for {len(items)} items')

        recommendations = ItemRecommendationService.get_item_recommendations(
            items=items,
            collaborative_dir_path=collaborative_dir_path,
            n_recommendations=n_recommendations,
            min_similarity=min_similarity
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
            user_ratings=user_ratings,
            collaborative_dir_path=collaborative_dir_path,
            n_recommendations=n_recommendations,
            min_similarity=min_similarity
        )

        if not recommendations:
            logger.info("No recommendations found for the given user ratings")
            return {"message": "No recommendations found"}

        logger.info(f"Generated {len(recommendations)} user-based recommendations")
        return recommendations

    except Exception as e:
        logger.error(f"Error generating user-based recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@collaborative_router_v2.post("/evaluate-model")
def evaluate_model(
    collaborative_dir_path: str = Query(
        default=str(collaborative_dir_path),
        description="Path to preprocessed dataset"
    ),
    sample_size: int = Query(
        default=100,
    ),
):
    try:
        results = ModelEvaluationService.evaluate_model(
            collaborative_dir_path=collaborative_dir_path
            
        )
        return results
    except ValueError as ve:
        logger.error(f"Evaluation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
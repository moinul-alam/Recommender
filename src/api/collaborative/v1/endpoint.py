import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel
from src.config.collaborative_config import CollaborativeConfigV1
from src.models.collaborative.v1.services.preprocessing_service import PreprocessingService
from src.models.collaborative.v1.services.model_training_service import ModelTrainingService
from src.models.collaborative.v1.services.recommendation_service import RecommendationService
from src.models.collaborative.v1.services.model_evaluation_service import ModelEvaluationService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

collaborative_router_v1 = APIRouter()

class ItemRating(BaseModel):
    tmdb_id: int
    rating: float

dataset_dir_path = CollaborativeConfigV1().DATASET_DIR_PATH
processed_dir_path = CollaborativeConfigV1().PROCESSED_DIR_PATH
model_dir_path = CollaborativeConfigV1().MODEL_DIR_PATH

@collaborative_router_v1.post("/data-preprocessing")
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
):
    logger.info(
        f"Received data preprocessing request with "
        f"dataset_dir_path={dataset_dir_path}, "
        f"processed_dir_path={processed_dir_path}, "
        f"sparse_user_threshold={sparse_user_threshold}, "
        f"sparse_item_threshold={sparse_item_threshold}, "
        f"split_percent={split_percent}"
    )

    try:
        result = PreprocessingService.process_data(
            dataset_dir_path=dataset_dir_path,
            processed_dir_path=processed_dir_path,
            sparse_user_threshold=sparse_user_threshold,
            sparse_item_threshold=sparse_item_threshold,
            split_percent=split_percent
        )
        
        if result is None:
            raise HTTPException(status_code=400, detail="Preprocessing failed")
        
        logger.info(f"Data preprocessing completed successfully: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error during data preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@collaborative_router_v1.post("/model-training")
def train_model(
    processed_dir_path: str = Query(
        default=str(processed_dir_path),
        description="Path to preprocessed dataset"
    ),
    model_dir_path: str = Query(
        default=str(model_dir_path),
        description="Directory to save model components"
    ),
    n_neighbors: Optional[int] = Query(
        default=None, 
        description="Number of nearest neighbors to consider"
    ),
    similarity_metric: str = Query(
        default='cosine', 
        description="Similarity calculation method (cosine/euclidean)"
    )
):
    """
    Train item-item collaborative filtering model.
    
    Args:
        processed_dir_path: Path to preprocessed data
        model_dir_path: Directory to save model
        n_neighbors: Optional number of neighbors
        similarity_metric: Similarity calculation method
    
    Returns:
        Paths to saved model components
    """
    logger.info(
        f"Received model training request: "
        f"processed_dir_path={processed_dir_path}, "
        f"model_dir_path={model_dir_path}, "
        f"n_neighbors={n_neighbors}, "
        f"similarity_metric={similarity_metric}"
    )

    try:
        result = ModelTrainingService.train_model(
            processed_dir_path=processed_dir_path,
            model_dir_path=model_dir_path,
            model_components_path=model_dir_path,
            n_neighbors=n_neighbors,
            similarity_metric=similarity_metric
        )
        
        if result is None:
            raise HTTPException(status_code=400, detail="Model training failed")
        
        logger.info(f"Model training completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@collaborative_router_v1.post("/item-recommendations")
def get_item_recommendations(
    items: Dict[str, float] = Body(
        ...,
        description="Dictionary of {tmdb_id: rating} pairs"
    ),
    processed_dir_path: str = Query(
        default=str(processed_dir_path),
        description="Path to preprocessed dataset"
    ),
    model_dir_path: str = Query(
        default=str(model_dir_path),
        description="Directory containing model components"
    ),
    n_recommendations: int = Query(
        default=10,
        ge=1,  # Add validation
        le=100  # Add validation
    ),
    min_similarity: float = Query(
        default=0.1,
        ge=0.0,  # Add validation
        le=1.0   # Add validation
    )
):
    logger.info(
        f"Received item recommendation request: "
        f"n_items={len(items)}, "
        f"n_items={items},"
        f"model_dir_path={processed_dir_path}, "
        f"model_dir_path={model_dir_path}, "
        f"n_recommendations={n_recommendations}, "
        f"min_similarity={min_similarity}"
    )

    try:
        recommendations = RecommendationService.get_recommendations(
            items=items,
            processed_dir_path=processed_dir_path,
            model_dir_path=model_dir_path,
            n_recommendations=n_recommendations,
            min_similarity=min_similarity
        )
        
        if recommendations is None:
            raise HTTPException(
                status_code=404, 
                detail="Could not generate recommendations"
            )
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@collaborative_router_v1.post("/evaluate-model")
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
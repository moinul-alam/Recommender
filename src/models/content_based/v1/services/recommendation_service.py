import pandas as pd
import logging
from pathlib import Path
from fastapi import HTTPException
from src.models.content_based.v1.pipeline.Recommender import Recommender
from src.schemas.content_based_schema import Recommendation, RecommendationResponse, RecommendationRequest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RecommendationService:
    @staticmethod
    def recommendation_service(
        recommendation_request: RecommendationRequest, 
        features_folder_path: str, 
        model_folder_path: str, 
        n_items: int
    ) -> RecommendationResponse:
        try:
            tmdbId = recommendation_request.tmdbId
            metadata = recommendation_request.metadata

            features_folder = Path(features_folder_path)
            model_folder = Path(model_folder_path)

            # Validate features folder
            if not features_folder.is_dir():
                raise HTTPException(status_code=400, detail=f"Features folder not found: {features_folder_path}")
            
            features_file = features_folder / "engineered_features.feather"
            if not features_file.exists():
                raise HTTPException(status_code=400, detail=f"Features file not found: {features_file}")
            
            feature_matrix = pd.read_feather(features_file)
            if feature_matrix.empty:
                raise HTTPException(status_code=400, detail="Feature dataset is empty.")

            # Validate model folder
            if not model_folder.is_dir():
                raise HTTPException(status_code=400, detail=f"Model folder not found: {model_folder_path}")
            
            model_file = model_folder / "content_based_model.index"
            if not model_file.exists():
                raise HTTPException(status_code=400, detail=f"Model file not found: {model_file}")

            # Instantiate recommender and get recommendations
            recommender = Recommender(
                tmdbId=tmdbId,
                metadata=metadata.dict() if metadata else None,
                feature_matrix=feature_matrix,
                model_file=str(model_file),
                n_items=n_items
            )

            recommendations = recommender.get_recommendation()

            # Convert recommendations to Pydantic models
            recommendation_models = [
                Recommendation(tmdbId=rec["tmdbId"], similarity=rec["similarity"])
                for rec in recommendations
            ]

            return RecommendationResponse(
                status=f"Successfully retrieved {len(recommendation_models)} recommendations",
                queriedMedia=str(tmdbId),
                similarMedia=recommendation_models,
            )
        except Exception as e:
            logger.error(f"Error during recommendation retrieval: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in recommendation service: {str(e)}")

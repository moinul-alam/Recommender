from pathlib import Path
import pandas as pd
import numpy as np
import logging
from fastapi import HTTPException
from src.models.content_based.v2.pipeline.index_creation import IndexCreation
from src.schemas.content_based_schema import PipelineResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TrainingService:
    @staticmethod
    def train_model(content_based_dir_path: str) -> PipelineResponse:
        try:
            # Convert paths to Path objects
            content_based_dir_path = Path(content_based_dir_path)

            # Validate features folder
            if not content_based_dir_path.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"Features folder not found: {content_based_dir_path}"
                )

            # Validate combined features file
            features_file = content_based_dir_path / "3_engineered_features.feather"
            if not features_file.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Combined features dataset not found: {features_file}"
                )

            # Load the feature-engineered dataset
            feature_matrix = pd.read_feather(features_file)
            if feature_matrix.empty:
                raise HTTPException(status_code=400, detail="Feature dataset is empty.")

            # Ensure all columns are numeric
            if not all(feature_matrix.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                raise HTTPException(
                    status_code=400,
                    detail="Feature dataset must contain only numeric columns for FAISS compatibility."
                )

            # Initialize IndexCreation class and train the model
            model_path = content_based_dir_path / "4_content_based_model.index"
            model_trainer = IndexCreation(feature_matrix, str(model_path))
            saved_model_path = model_trainer.apply_model_training()

            logger.info(f"Model training successful. Model saved at: {saved_model_path}")

            return PipelineResponse(
                status="Model Training successful",
                output=1,
                output_path=saved_model_path
            )
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in model training: {str(e)}")

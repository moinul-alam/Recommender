import pandas as pd
import numpy as np
import logging
from pathlib import Path
import gc
from fastapi import HTTPException
from scipy import sparse
import joblib
from typing import Dict, List, Optional

from src.models.content_based.v2.pipeline.ModelTraining import ModelTraining
from src.schemas.content_based_schema import PipelineResponse

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ModelTrainingService:
    @staticmethod
    def train_model(content_based_dir_path: str) -> PipelineResponse:
        try:
            # Initialize paths
            content_based_dir_path = Path(content_based_dir_path)
            
            if not content_based_dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {content_based_dir_path}")
            
            # Load data
            feature_matrix = sparse.load_npz(content_based_dir_path / "3_final_feature_matrix.npz")
            item_ids = np.load(content_based_dir_path / "3_final_item_ids.npy")
            
            logger.info(f"Loaded feature matrix with shape: {feature_matrix.shape}")
            logger.info(f"Loaded item_ids with shape: {item_ids.shape}")
            
            # Initialize and run model training
            model_trainer = ModelTraining(
                feature_matrix=feature_matrix,
                item_ids=item_ids,
                n_components_svd=1000             
            )
            
            # Fit SVD and transform feature matrix
            transformed_features_with_ids = model_trainer.fit_transform_and_combine()
            
            # Save model and transformed features
            model_output_path = content_based_dir_path / "model_output"
            model_output_path.mkdir(exist_ok=True, parents=True)
            
            # Save SVD model
            model_trainer.save_transformers(str(model_output_path))
            
            # Save combined dataframe (item_ids with transformed features)
            combined_features_path = model_output_path / "transformed_features_with_ids.parquet"
            transformed_features_with_ids.to_parquet(str(combined_features_path), index=False)
            logger.info(f"Saved combined features and IDs to: {str(combined_features_path)}")
            
            return PipelineResponse(
                status="Model training completed successfully",
                output=transformed_features_with_ids.shape[0],
                output_path=str(model_output_path)
            )

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Error in model training: {str(e)}"
            )
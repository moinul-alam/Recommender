import pandas as pd
import numpy as np
import logging
from pathlib import Path
import gc
from fastapi import HTTPException
from scipy import sparse
import json
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
            
             
            model_trainer = ModelTraining(
                feature_matrix=feature_matrix,
                item_ids=item_ids,
                n_components_svd=200              
                )
            
            
            return PipelineResponse(
                status="Feature engineering completed successfully",
                output=len(segment_files),
                output_path=str(metadata_path)
            )

        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Error in feature engineering: {str(e)}"
            )

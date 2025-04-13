import os
from pathlib import Path
import faiss
import pandas as pd
import numpy as np
import logging
from fastapi import HTTPException
from src.models.content_based.v2.pipeline.index_creation import IndexCreation
from src.schemas.content_based_schema import PipelineResponse
from src.models.common.DataLoader import load_data
from src.models.common.DataSaver import save_data, save_multiple_dataframes

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class IndexingService:
    @staticmethod
    def create_index(content_based_dir_path: str, file_names: dict) -> PipelineResponse:
        try:
            content_based_dir_path = Path(content_based_dir_path)

            if not content_based_dir_path.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"Directory not found: {content_based_dir_path}"
                )
                
            feature_matrix_name = file_names["feature_matrix_name"]
            if not feature_matrix_name.endswith('.pkl'):
                feature_matrix_name += '.pkl'
            
            feature_matrix_path = os.path.join(content_based_dir_path,feature_matrix_path)
            logger.info(f"Feature matrix path: {feature_matrix_path}")
            
            if not os.path.isfile(feature_matrix_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature matrix file not found: {feature_matrix_path}"
                )
                
            feature_matrix = load_data(feature_matrix_path)
            logger.info(f"Feature matrix loaded from {feature_matrix_path}")
            if feature_matrix is None or feature_matrix.empty:
                raise HTTPException(status_code=400, detail="Feature matrix is empty or invalid")
            
            # Check if the feature matrix contains the required columns
            required_columns = {"item_id"}
            missing_columns = required_columns - set(feature_matrix.columns)
            if missing_columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required columns: {missing_columns}"
                )
            
            # Check for NaN values in feature columns
            feature_columns = [col for col in feature_matrix.columns if col != "item_id"]
            if feature_matrix[feature_columns].isnull().values.any():
                raise HTTPException(
                    status_code=400,
                    detail="Feature matrix contains NaN values. Ensure preprocessing removes missing data."
                )
                
            # Ensure all columns are numeric for FAISS compatibility
            if not all(feature_matrix.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                raise HTTPException(
                    status_code=400,
                    detail="Feature dataset must contain only numeric columns for FAISS compatibility."
                )
                
            IndexCreator = IndexCreation(feature_matrix)
            
            index = IndexCreator.model_training()
            
            # Save the index to a file
            index_name = file_names["faiss_index_name"]
            
            index_path = os.path.join(content_based_dir_path, index_name)
            
            if not os.path.exists(content_based_dir_path):
                os.makedirs(content_based_dir_path)
            
            faiss.write_index(index, index_path)
            logger.info(f"FAISS index saved to: {index_path}")

            return PipelineResponse(
                status="Success",
                message="Index created successfully",
                output=str(content_based_dir_path)
            )
        except Exception as e:
            logger.error(f"Error during index creation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in index creation: {str(e)}")

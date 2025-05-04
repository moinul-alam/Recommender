import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

from fastapi import HTTPException
from src.models.collaborative.v2.pipeline.feature_extraction import FeatureExtraction
from src.models.common.DataLoader import load_data
from src.models.common.DataSaver import save_data, save_objects

from src.schemas.content_based_schema import PipelineResponse


class FeatureExtractionService:
    @staticmethod
    def extract_features(
        collaborative_dir_path: str,
        file_names: dict,
        n_components_user: int = 200,
        n_components_item: int = 200,
        batch_size: int = 50000
    ) -> PipelineResponse:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        try:
            # Validate input paths
            collaborative_dir_path = Path(collaborative_dir_path)
            if not collaborative_dir_path.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid directory path: {collaborative_dir_path}"
                )

            # Load user-item matrix
            user_item_matrix_path = collaborative_dir_path / file_names["user_item_matrix"]            
            if not user_item_matrix_path.is_file():
                raise HTTPException(
                    status_code=400,
                    detail=f"User-item matrix file not found: {user_item_matrix_path}"
                )
                
            user_item_matrix = load_data(user_item_matrix_path)

            logger.info(f"Loaded user-item matrix with shape: {user_item_matrix.shape}")

            extractor = FeatureExtraction(
                n_components_user=n_components_user,
                n_components_item=n_components_item,
                batch_size=batch_size
            )

            # Train model
            model_results = extractor.extract(user_item_matrix)

            # Define output paths
            files_to_save = {
                file_names["user_matrix"]: model_results['user_matrix'],
                file_names["item_matrix"]: model_results['item_matrix'],
                file_names["svd_user_model"]: model_results['svd_user_model'],
                file_names["svd_item_model"]: model_results['svd_item_model'],
                file_names["model_info"]: model_results['model_info']
            }
            
            save_objects(
                directory_path=collaborative_dir_path,
                objects=files_to_save
            )
            
            logger.info(f"Model components saved successfully.")
            
            return PipelineResponse(
                status="success",
                message="Feature extraction completed successfully.",
                output=str(collaborative_dir_path),
            )

        except Exception as e:
            logger.error(f"Error during model training: {e}", exc_info=True)
            return None
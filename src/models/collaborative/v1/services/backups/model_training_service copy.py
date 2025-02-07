import logging
import os
import pickle
import pathlib
from typing import Dict, Optional
import pandas as pd
import numpy as np
import faiss
from src.models.collaborative.v1.pipeline.ModelTraining import ModelTraining

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ModelTrainingService:
    @staticmethod
    def train_model(processed_dir_path: str, model_dir_path: str, 
                    model_components_path: str) -> Optional[Dict[str, str]]:
        """
        Train SVD model and save components.
        
        Args:
            processed_dir_path (str): Directory with preprocessed data
            model_dir_path (str): Directory to save model
            model_components_path (str): Path to save model components
        
        Returns:
            Dictionary with paths to saved model components
        """
        try:
            # Load user-item matrix
            user_item_matrix_path = pathlib.Path(processed_dir_path) / "user_item_matrix.pkl"
            
            if not user_item_matrix_path.exists():
                logger.error(f"User-item matrix not found: {user_item_matrix_path}")
                return None

            with open(user_item_matrix_path, "rb") as f:
                user_item_matrix = pickle.load(f)

            logger.info(f"Loaded user-item matrix with shape: {user_item_matrix.shape}")

            # Ensure model directory exists
            model_path = pathlib.Path(model_dir_path)
            model_path.mkdir(parents=True, exist_ok=True)

            # Train model
            processor = ModelTraining()
            svd_components, faiss_index = processor.train(user_item_matrix)

            # Define output paths
            paths = {
                "svd_components_path": model_path / "svd_components.pkl",
                "faiss_index_path": model_path / "faiss_index.pkl"
            }

            # Save model components
            with open(paths["svd_components_path"], "wb") as f:
                pickle.dump(svd_components, f)

            with open(paths["faiss_index_path"], "wb") as f:
                pickle.dump(faiss_index, f)

            logger.info(f"Model training complete. Files saved in {model_path}")
            return {str(k): str(v) for k, v in paths.items()}

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return None
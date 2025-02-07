import logging
import os
import pickle
import pathlib
from typing import Dict, Optional
import pandas as pd
import faiss

from src.models.collaborative.v1.pipeline.ModelTraining import ModelTraining

class ModelTrainingService:
    @staticmethod
    def train_model(
        processed_dir_path: str, 
        model_dir_path: str, 
        model_components_path: str,
        n_neighbors: Optional[int] = None,
        similarity_metric: str = 'cosine'
    ) -> Optional[Dict[str, str]]:
        """
        Train item-item collaborative filtering model.
        
        Args:
            processed_dir_path (str): Directory with preprocessed data
            model_dir_path (str): Directory to save model
            model_components_path (str): Path to save model components
            n_neighbors (int, optional): Number of neighbors to consider
            similarity_metric (str): Similarity calculation method
        
        Returns:
            Dictionary with paths to saved model components
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        try:
            # Validate and prepare input paths
            processed_path = pathlib.Path(processed_dir_path)
            model_path = pathlib.Path(model_dir_path)
            
            # Ensure model directory exists
            model_path.mkdir(parents=True, exist_ok=True)

            # Load user-item matrix
            user_item_matrix_path = processed_path / "user_item_matrix.pkl"
            
            if not user_item_matrix_path.exists():
                logger.error(f"User-item matrix not found: {user_item_matrix_path}")
                return None

            with open(user_item_matrix_path, "rb") as f:
                user_item_matrix = pickle.load(f)

            logger.info(f"Loaded user-item matrix with shape: {user_item_matrix.shape}")

            # Train model
            processor = ModelTraining(
                n_neighbors=n_neighbors, 
                similarity_metric=similarity_metric
            )
            model_components, faiss_index = processor.train(user_item_matrix)

            # Define output paths
            paths = {
                "item_similarity_matrix": model_path / "item_similarity_matrix.pkl",
                "model_info": model_path / "model_info.pkl",
                "faiss_index": model_path / "faiss_index.pkl"
            }

            # Save model components
            def safe_pickle_dump(obj, path):
                with open(path, "wb") as f:
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

            safe_pickle_dump(model_components['item_similarity_matrix'], paths["item_similarity_matrix"])
            safe_pickle_dump(model_components['model_info'], paths["model_info"])
            
            with open(paths["faiss_index"], "wb") as f:
                faiss.write_index(faiss_index, str(paths["faiss_index"]))

            logger.info(f"Item-Item model training complete. Files saved in {model_path}")
            return {str(k): str(v) for k, v in paths.items()}

        except Exception as e:
            logger.error(f"Error during model training: {e}", exc_info=True)
            return None
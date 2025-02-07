import logging
import pathlib
import pickle
from typing import Dict, Optional
import numpy as np
import faiss
from src.models.collaborative.v2.pipeline.ModelTraining import ModelTraining

class ModelTrainingService:
    @staticmethod
    def train_model(
        processed_dir_path: str, 
        model_dir_path: str,
        n_neighbors: Optional[int] = 50,
        similarity_metric: str = 'L2',
        batch_size: int = 10000,
        min_similarity: float = 0.1
    ) -> Optional[Dict[str, str]]:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        try:
            # Validate and prepare input paths
            processed_path = pathlib.Path(processed_dir_path)
            model_path = pathlib.Path(model_dir_path)
            model_path.mkdir(parents=True, exist_ok=True)

            # Load user-item matrix
            user_item_matrix_path = processed_path / "user_item_matrix.pkl"
            if not user_item_matrix_path.exists():
                logger.error(f"User-item matrix not found: {user_item_matrix_path}")
                return None

            with open(user_item_matrix_path, "rb") as f:
                user_item_matrix = pickle.load(f)

            logger.info(f"Loaded user-item matrix with shape: {user_item_matrix.shape}")

            # Train model on the full dataset
            trainer = ModelTraining(
                n_neighbors=n_neighbors, 
                similarity_metric=similarity_metric,
                batch_size=batch_size,
                min_similarity=min_similarity,
                use_disk_index=True,
                index_path=model_path / "faiss_index.ivf"
            )

            model_components, faiss_index = trainer.train(user_item_matrix)

            # Define output paths
            paths = {
                "item_matrix": model_path / "item_matrix.pkl",
                "svd_model": model_path / "svd_model.pkl",
                "model_info": model_path / "model_info.pkl",
                "faiss_index": model_path / "faiss_index.ivf"
            }

            # Save model components
            def safe_pickle_dump(obj, path):
                with open(path, "wb") as f:
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

            safe_pickle_dump(model_components['item_matrix'], paths["item_matrix"])
            safe_pickle_dump(model_components['svd_model'], paths["svd_model"])
            safe_pickle_dump(model_components['model_info'], paths["model_info"])

            logger.info(f"Item-Item model training complete. Files saved in {model_path}")
            return {str(k): str(v) for k, v in paths.items()}

        except Exception as e:
            logger.error(f"Error during model training: {e}", exc_info=True)
            return None

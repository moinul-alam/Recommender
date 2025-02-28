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
        collaborative_dir_path: str,
        n_neighbors: Optional[int] = 50,
        similarity_metric: str = 'cosine',
        batch_size: int = 50000,
        min_similarity: float = 0.1
    ) -> Optional[Dict[str, str]]:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        try:
            # Validate input paths
            collaborative_dir_path = pathlib.Path(collaborative_dir_path)
            if not collaborative_dir_path.exists():
                logger.error(f"Collaborative directory not found: {collaborative_dir_path}")
                return None

            # Load user-item matrix
            user_item_matrix_path = collaborative_dir_path / "2_user_item_matrix.pkl"
            if not user_item_matrix_path.exists():
                logger.error(f"User-item matrix not found: {user_item_matrix_path}")
                return None

            with open(user_item_matrix_path, "rb") as f:
                user_item_matrix = pickle.load(f)

            logger.info(f"Loaded user-item matrix with shape: {user_item_matrix.shape}")

            # Initialize ModelTraining for FlatIP
            trainer = ModelTraining(
                n_neighbors=n_neighbors, 
                similarity_metric=similarity_metric,
                batch_size=batch_size,
                min_similarity=min_similarity,
                use_disk_index=True,
                n_components_user=300,
                n_components_item=300,
                user_index_path=str(collaborative_dir_path / "3_faiss_user_index.flat"),
                item_index_path=str(collaborative_dir_path / "3_faiss_item_index.flat")
            )

            # Train model
            model_results, faiss_indices = trainer.train(user_item_matrix)

            # Define output paths
            paths = {
                "user_matrix": collaborative_dir_path / "3_user_matrix.pkl",
                "item_matrix": collaborative_dir_path / "3_item_matrix.pkl",
                "svd_user_model": collaborative_dir_path / "3_svd_user_model.pkl",
                "svd_item_model": collaborative_dir_path / "3_svd_item_model.pkl",
                "model_info": collaborative_dir_path / "3_model_info.pkl",
                "faiss_user_index": collaborative_dir_path / "3_faiss_user_index.flat",
                "faiss_item_index": collaborative_dir_path / "3_faiss_item_index.flat"
            }

            # Save model components
            def safe_pickle_dump(obj, path):
                with open(path, "wb") as f:
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

            safe_pickle_dump(model_results['user_matrix'], paths["user_matrix"])
            safe_pickle_dump(model_results['item_matrix'], paths["item_matrix"])
            safe_pickle_dump(model_results['svd_user_model'], paths["svd_user_model"])
            safe_pickle_dump(model_results['svd_item_model'], paths["svd_item_model"])
            safe_pickle_dump(model_results['model_info'], paths["model_info"])

            logger.info(f"Model training complete. Files saved in {paths}")
            return {str(k): str(v) for k, v in paths.items()}

        except Exception as e:
            logger.error(f"Error during model training: {e}", exc_info=True)
            return None
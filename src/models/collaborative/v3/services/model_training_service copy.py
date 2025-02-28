import logging
import pathlib
import pickle
from typing import Dict, Optional
import numpy as np
from scipy import sparse
from src.models.collaborative.v3.pipeline.ModelTraining import ModelTraining

logger = logging.getLogger(__name__)

class ModelTrainingService:
    @staticmethod
    def validate_input_paths(processed_dir: str, model_dir: str) -> bool:
        processed_path = pathlib.Path(processed_dir)
        model_path = pathlib.Path(model_dir)
        
        required_files = [
            processed_path / "user_item_matrix.pkl",
            processed_path / "item_mapping.pkl",
            processed_path / "item_reverse_mapping.pkl"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"Required file not found: {file_path}")
                return False
        
        model_path.mkdir(parents=True, exist_ok=True)
        return True

    @classmethod
    def train_model(
        cls,
        processed_dir_path: str,
        model_dir_path: str,
        n_factors: int = 50,
        learning_rate: float = 0.001,
        regularization: float = 0.1,
        n_epochs: int = 20,
        batch_size: int = 512,
        early_stopping_patience: int = 3,
        validation_split: float = 0.1,
        chunk_size: int = 1000
    ) -> Optional[Dict[str, str]]:
        logger = logging.getLogger(cls.__name__)
        
        try:
            # Validate paths
            if not cls.validate_input_paths(processed_dir_path, model_dir_path):
                return None
            
            # Load preprocessed data
            processed_path = pathlib.Path(processed_dir_path)
            with open(processed_path / "user_item_matrix.pkl", "rb") as f:
                user_item_matrix = pickle.load(f)
            
            # Initialize and train model
            model = ModelTraining(
                n_factors=n_factors,
                learning_rate=learning_rate,
                regularization=regularization,
                n_epochs=n_epochs,
                batch_size=batch_size,
                early_stopping_patience=early_stopping_patience,
                validation_split=validation_split,
                chunk_size=chunk_size
            )

            
            # Train the model
            logger.info("Starting model training...")
            history = model.fit(user_item_matrix)
            
            # Save model and training history
            model_path = pathlib.Path(model_dir_path)
            paths = {
                "model_path": model_path / "matrix_factorization_model.pkl",
                "history_path": model_path / "training_history.pkl"
            }
            
            model.save(paths["model_path"])
            
            with open(paths["history_path"], "wb") as f:
                pickle.dump(history, f)
            
            logger.info(f"Model training complete. Files saved in {model_path}")
            
            return {str(k): str(v) for k, v in paths.items()}
            
        except Exception as e:
            logger.error(f"Error during model training: {e}", exc_info=True)
            return None
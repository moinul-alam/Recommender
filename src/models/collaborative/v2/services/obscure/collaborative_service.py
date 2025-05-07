import faiss
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
from src.models.collaborative.v1.pipeline.CollaborativeFilter import CollaborativeFilter
from src.config.collaborative_config import ModelConfig, CollaborativeConfigV1
from src.schemas.collaborative_schema import TrainingStatus

class CollaborativeService:
    def __init__(self, config: CollaborativeConfigV1):
        self.config = config
        self.model = None
        self.training_status = TrainingStatus.PENDING
        self._ensure_directories()
        self._try_load_model()

    def _ensure_directories(self) -> None:
        self.config.COLLABORATIVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.config.COLLABORATIVE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
    
    def _adjust_config_for_data_size(self, n_users: int) -> ModelConfig:
        """Adjusts model configuration dynamically to ensure compatibility with FAISS."""
        if n_users < 1000:
            n_components = 48  # Must be divisible by m=8
            nlist = 50
        elif n_users < 10000:
            n_components = 96  # Must be divisible by m=8
            nlist = 100
        else:
            n_components = 200  # Ensure divisibility
            nlist = 200

        m = 8  # Number of subquantizers (fixed)
        if n_components % m != 0:
            n_components = (n_components // m) * m  # Adjust to nearest valid value

        return ModelConfig(n_components=n_components, nlist=nlist, m=m, k_neighbors=10, similarity_threshold=0.3, time_weight_factor=0.1)

    def _try_load_model(self) -> None:
        try:
            self._load_model()
        except (FileNotFoundError, Exception) as e:
            print(f"Error loading model: {e}")
            self.model = None

    def _load_model(self) -> None:
        with open(self.config.MODEL_COMPONENTS_PATH, 'rb') as f:
            model_components = pickle.load(f)
        self.model = CollaborativeFilter(model_components['config'])
        self.model.faiss_index = faiss.read_index(str(self.config.FAISS_INDEX_PATH))
        for attr, value in model_components.items():
            if attr != 'config':
                setattr(self.model, attr, value)

    def save_training_data(self, file_content: bytes) -> bool:
        try:
            with open(self.config.TRAINING_DATA_PATH, 'wb') as f:
                f.write(file_content)
            return True
        except Exception as e:
            print(f"Error saving training data: {e}")
            return False
    
    def _save_model_components(self):
        try:
            model_components = {
                'config': self.model.config,
                'faiss_index': self.model.faiss_index,
                'scaler': self.model.scaler,
                'svd': self.model.svd,
                # Add any other necessary attributes
            }
            with open(self.config.MODEL_COMPONENTS_PATH, 'wb') as f:
                pickle.dump(model_components, f)
            print("Model components saved successfully.")
        except Exception as e:
            print(f"Error saving model components: {e}")

    def _evaluate_model(self):
        """
        Evaluate the performance of the collaborative filtering model.
        This method could be used to check metrics like precision, recall, and F1 score.
        """
        try:
            # Assuming you have a test dataset or validation data
            # Here you might compare the predictions to actual values or perform cross-validation
            predictions = self.model.predict()  # Example, update with your actual prediction method
            actual_values = self.test_data  # Your validation/test data

            # Calculate performance metrics
            precision = self.calculate_precision(predictions, actual_values)
            recall = self.calculate_recall(predictions, actual_values)
            f1_score = self.calculate_f1_score(precision, recall)

            # You could return a dictionary or any other format depending on your needs
            return {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
            }

        except Exception as e:
            print(f"Error evaluating the model: {e}")
            return None

    def train_model(self) -> Tuple[TrainingStatus, str, Optional[Dict[str, float]]]:
        try:
            self.training_status = TrainingStatus.IN_PROGRESS
            df = pd.read_csv(self.config.TRAINING_DATA_PATH)
            n_users = df['userId'].nunique()
            config = self._adjust_config_for_data_size(n_users)
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            self.model = CollaborativeFilter(config)
            self.model.train(train_df)
            self._save_model_components()
            metrics = self._evaluate_model(test_df)
            self.training_status = TrainingStatus.COMPLETED
            return self.training_status, "Model trained successfully", metrics
        except Exception as e:
            self.training_status = TrainingStatus.FAILED
            return self.training_status, f"Training failed: {str(e)}", None

    def get_recommendations(self, user_id: Optional[str] = None, ratings: Optional[Dict[int, float]] = None, n_recommendations: int = 10) -> Tuple[List[Tuple[int, float]], bool]:
        if not self.model:
            raise ValueError("Model not trained yet. Please train the model first.")
        is_guest = not user_id or user_id not in self.model.user_means
        recommendations = self.model.recommend_movies(user_id=user_id, n_recommendations=n_recommendations, guest_ratings=ratings or {})
        return recommendations, is_guest

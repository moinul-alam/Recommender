import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.models.collaborative.v2.services.recommendation_service import RecommendationService
from src.models.common.file_config import file_names
from src.models.common.DataLoader import load_data

logger = logging.getLogger(__name__)

class Evaluator:
    @staticmethod
    def evaluate_prediction(
        directory_path: str,
        test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        rmse_list = []
        mae_list = []
        
        training_data_path = directory_path / file_names["train_set"]
        if not training_data_path.exists():
            raise FileNotFoundError(f"Training data file not found at {training_data_path}")
        
        test_data_path = directory_path / file_names["test_set"]
        if not test_data_path.exists():
            raise FileNotFoundError(f"Test data file not found at {test_data_path}")
        
        training_data = load_data(training_data_path)
        if training_data.empty:
            raise ValueError("Training data is empty.")
        
        test_data = load_data(test_data_path)
        if test_data.empty:
            raise ValueError("Test data is empty.")
        
        
        # Iterate through each user in the test set
        for userId in test_data['userId'].unique():
            user_ratings = test_data[test_data['userId'] == userId].set_index('movieId')['rating'].to_dict()
            if not user_ratings:
                continue

            # Get recommendations for the user
            recommendations = RecommendationService.get_recommendations(
                user_ratings=user_ratings,
                directory_path=directory_path
            )
            
            if not recommendations:
                continue

            recommended_items = {rec['movieId']: rec['similarity'] for rec in recommendations}
            
            # Calculate RMSE and MAE for the recommended items
            y_true = np.array([user_ratings[item] for item in recommended_items.keys() if item in user_ratings])
            y_pred = np.array([recommended_items[item] for item in recommended_items.keys() if item in user_ratings])

            if len(y_true) > 0:
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                rmse_list.append(rmse)
                mae_list.append(mae)

        # Calculate average metrics
        avg_rmse = np.mean(rmse_list) if rmse_list else 0.0
        avg_mae = np.mean(mae_list) if mae_list else 0.0

        metrics = {
            'rmse': avg_rmse,
            'mae': avg_mae            
        }

        logger.info("\nEvaluation Summary:")
        logger.info(f"Total test samples: {len(test_data)}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"MAE: {metrics['mae']:.4f}")

        return metrics
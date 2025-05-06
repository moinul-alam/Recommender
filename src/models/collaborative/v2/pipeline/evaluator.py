import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.models.collaborative.v2.services.user_recommendation_service import UserRecommendationService

logger = logging.getLogger(__name__)

class Evaluator:
    @staticmethod
    def evaluate_prediction(
        collaborative_dir_path: str,
        file_names: dict,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        n_recommendations: int = 10,
        min_similarity: float = 0.1
    ) -> Dict[str, Any]:
        rmse_list = []
        mae_list = []

        # Iterate through each user in the test set
        for user_id in train_data['user_id'].unique():
            user_ratings = train_data[train_data['user_id'] == user_id].set_index('tmdb_id')['rating'].to_dict()
            if not user_ratings:
                continue

            # Get recommendations for the user
            recommendations = UserRecommendationService.get_user_recommendations(
                user_ratings=user_ratings,
                collaborative_dir_path=collaborative_dir_path,
                file_names=file_names,
                n_recommendations=n_recommendations,
                min_similarity=min_similarity
            )
            
            if not recommendations:
                continue

            recommended_items = {rec['tmdb_id']: rec['similarity'] for rec in recommendations}
            
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
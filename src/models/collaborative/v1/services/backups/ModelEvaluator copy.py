import os
import logging
import pathlib
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)

from src.models.collaborative.v1.pipeline.ItemRecommender import ItemRecommender

logger = logging.getLogger(__name__)

class ModelEvaluator:
    @staticmethod
    def compute_recommendation_metrics(
        recommender: ItemRecommender, 
        test_data: pd.DataFrame,
        n_recommendations: int = 1
    ) -> Dict[str, float]:
        """
        Compute recommendation performance metrics
        
        Args:
            recommender: Trained recommender
            test_data: Test dataset for evaluation
            n_recommendations: Number of recommendations to generate
        
        Returns:
            Dictionary of performance metrics
        """
        actual_ratings = []
        predicted_ratings = []
        
        for _, row in test_data.iterrows():
            try:
                user_items = {str(int(row['tmdb_id'])): row['rating']}
                recommendations = recommender.recommend(
                    items=user_items, 
                    n_recommendations=n_recommendations
                )
                
                if recommendations:
                    actual_ratings.append(row['rating'])
                    predicted_ratings.append(recommendations[0]['predicted_rating'])
            
            except Exception as e:
                logger.error(f"Recommendation error: {e}")
                continue
        
        return {
            'mse': mean_squared_error(actual_ratings, predicted_ratings),
            'rmse': np.sqrt(mean_squared_error(actual_ratings, predicted_ratings)),
            'mae': mean_absolute_error(actual_ratings, predicted_ratings),
            'r2_score': r2_score(actual_ratings, predicted_ratings),
            'sample_size': len(actual_ratings)
        }

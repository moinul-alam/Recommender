import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    precision_score,
    recall_score
)

logger = logging.getLogger(__name__)

class ModelEvaluator:
    @staticmethod
    def compute_recommendation_metrics(
        recommender, 
        test_data: pd.DataFrame,
        n_recommendations: int = 10,
        threshold: float = 3.5  # Rating threshold for positive recommendation
    ) -> Dict[str, Any]:
        """
        Comprehensive recommendation performance evaluation
        
        Args:
            recommender: Trained recommender
            test_data: Test dataset for evaluation
            n_recommendations: Number of recommendations to generate
            threshold: Rating threshold for positive recommendations
        
        Returns:
            Comprehensive metrics dictionary
        """
        actual_ratings = []
        predicted_ratings = []
        recommendation_hits = []
        ground_truth_relevance = []
        
        for _, row in test_data.iterrows():
            try:
                # Convert TMDB ID to string
                tmdb_id = str(int(row['tmdb_id']))
                
                # Get recommendations based on this item
                recommendations = recommender.recommend(
                    items={tmdb_id: row['rating']}, 
                    n_recommendations=n_recommendations
                )
                
                if recommendations:
                    # Prediction metrics
                    actual_ratings.append(row['rating'])
                    predicted_ratings.append(recommendations[0]['predicted_rating'])
                    
                    # Recommendation relevance metrics
                    recommended_ids = [rec['tmdb_id'] for rec in recommendations]
                    
                    # Check if recommended movies are relevant
                    is_hit = any(
                        float(rec['predicted_rating']) >= threshold 
                        for rec in recommendations
                    )
                    recommendation_hits.append(is_hit)
                    
                    # Ground truth relevance
                    ground_truth_relevance.append(row['rating'] >= threshold)
            
            except Exception as e:
                logger.error(f"Recommendation error for {tmdb_id}: {e}")
                continue
        
        # Comprehensive performance metrics
        metrics = {
            # Prediction accuracy metrics
            'mse': mean_squared_error(actual_ratings, predicted_ratings),
            'rmse': np.sqrt(mean_squared_error(actual_ratings, predicted_ratings)),
            'mae': mean_absolute_error(actual_ratings, predicted_ratings),
            'r2_score': r2_score(actual_ratings, predicted_ratings),
            
            # Recommendation quality metrics
            'precision': precision_score(ground_truth_relevance, recommendation_hits),
            'recall': recall_score(ground_truth_relevance, recommendation_hits),
            
            # Metadata
            'sample_size': len(actual_ratings),
            'total_test_samples': len(test_data),
            'recommendation_coverage': len(actual_ratings) / len(test_data) * 100
        }
        
        return metrics

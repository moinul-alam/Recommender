import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

class ModelEvaluator:
    @staticmethod
    def compute_recommendation_metrics(
        recommender, 
        test_data: pd.DataFrame,
        n_recommendations: int = 10,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Compute metrics with detailed prediction tracking
        """
        evaluation_records = []
        
        for _, row in test_data.iterrows():
            try:
                tmdb_id = str(int(row['tmdb_id']))
                actual_rating = float(row['rating'])
                
                # Log input data
                logger.debug(f"Processing test item: TMDB_ID={tmdb_id}, Rating={actual_rating}")
                
                # Get recommendations
                recommendations = recommender.recommend(
                    items={tmdb_id: actual_rating}, 
                    n_recommendations=n_recommendations
                )
                
                if recommendations:
                    predicted_rating = recommendations[0]['predicted_rating']
                    similarity_score = recommendations[0]['similarity_score']
                    
                    evaluation_records.append({
                        'tmdb_id': tmdb_id,
                        'actual_rating': actual_rating,
                        'predicted_rating': predicted_rating,
                        'similarity_score': similarity_score,
                        'error': abs(actual_rating - predicted_rating),
                        'recommendations': recommendations
                    })
                    
                    logger.debug(
                        f"Prediction details: "
                        f"Actual={actual_rating:.2f}, "
                        f"Predicted={predicted_rating:.2f}, "
                        f"Error={abs(actual_rating - predicted_rating):.2f}, "
                        f"Similarity={similarity_score:.2f}"
                    )
                else:
                    logger.warning(f"No recommendations generated for TMDB_ID={tmdb_id}")
            
            except Exception as e:
                logger.error(f"Error processing TMDB_ID={tmdb_id}: {e}")
                continue
        
        # Convert records to DataFrame for analysis
        evaluation_df = pd.DataFrame(evaluation_records)
        
        if evaluation_df.empty:
            logger.error("No valid predictions generated")
            return {}, pd.DataFrame()
        
        # Calculate metrics
        actual_ratings = evaluation_df['actual_rating'].values
        predicted_ratings = evaluation_df['predicted_rating'].values
        
        metrics = {
            'mse': mean_squared_error(actual_ratings, predicted_ratings),
            'rmse': np.sqrt(mean_squared_error(actual_ratings, predicted_ratings)),
            'mae': mean_absolute_error(actual_ratings, predicted_ratings),
            'r2_score': r2_score(actual_ratings, predicted_ratings),
            'sample_size': len(evaluation_df),
            'total_test_samples': len(test_data),
            'recommendation_coverage': (len(evaluation_df) / len(test_data)) * 100,
            'rating_range': {
                'actual': {
                    'min': float(actual_ratings.min()),
                    'max': float(actual_ratings.max()),
                    'mean': float(actual_ratings.mean())
                },
                'predicted': {
                    'min': float(predicted_ratings.min()),
                    'max': float(predicted_ratings.max()),
                    'mean': float(predicted_ratings.mean())
                }
            }
        }
        
        # Log detailed statistics
        logger.info("\nEvaluation Statistics:")
        logger.info(f"Total test samples: {len(test_data)}")
        logger.info(f"Valid predictions: {len(evaluation_df)}")
        logger.info("\nRating Distributions:")
        logger.info(f"Actual ratings: {evaluation_df['actual_rating'].describe()}")
        logger.info(f"Predicted ratings: {evaluation_df['predicted_rating'].describe()}")
        logger.info("\nError Distribution:")
        logger.info(f"Prediction errors: {evaluation_df['error'].describe()}")
        
        return metrics, evaluation_df
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.models.collaborative.v2.services.recommendation_service import RecommendationService

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Evaluates a recommendation model by comparing predicted ratings to actual user ratings.
    """

    @staticmethod
    def compute_recommendation_metrics(
        test_data: pd.DataFrame,
        processed_dir_path: str,
        model_dir_path: str,
        n_recommendations: int = 10,
        min_similarity: float = 0.1,
        default_predicted_rating: float = 3.0  # Default rating for items with no recommendations
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Compute recommendation performance metrics, including RMSE, MAE, and R2 score.
        """
        evaluation_records = []
        
        for _, row in test_data.iterrows():
            try:
                tmdb_id = str(int(row['tmdb_id']))
                actual_rating = float(row['rating'])

                # Log test sample details
                logger.debug(f"Processing TMDB_ID={tmdb_id}, Actual Rating={actual_rating}")

                # Simulate user history by including multiple items
                user_history = {tmdb_id: actual_rating}  # Add more items if available

                # Generate recommendations
                recommendations = RecommendationService.get_recommendations(
                    items=user_history, 
                    processed_dir_path=processed_dir_path,
                    model_dir_path=model_dir_path,
                    n_recommendations=n_recommendations,
                    min_similarity=min_similarity
                )

                if recommendations:
                    # Compute predicted rating as the average of top recommendations
                    predicted_rating = np.mean([rec['predicted_rating'] for rec in recommendations])
                    similarity_score = np.mean([rec['similarity'] for rec in recommendations])
                else:
                    # Use default predicted rating if no recommendations are generated
                    predicted_rating = default_predicted_rating
                    similarity_score = 0.0
                    logger.warning(f"No recommendations generated for TMDB_ID={tmdb_id}")

                evaluation_records.append({
                    'tmdb_id': tmdb_id,
                    'actual_rating': actual_rating,
                    'predicted_rating': predicted_rating,
                    'similarity_score': similarity_score,
                    'error': abs(actual_rating - predicted_rating),
                    'recommendations': recommendations
                })

                logger.debug(
                    f"TMDB_ID={tmdb_id}, "
                    f"Actual={actual_rating:.2f}, "
                    f"Predicted={predicted_rating:.2f}, "
                    f"Error={abs(actual_rating - predicted_rating):.2f}, "
                    f"Similarity={similarity_score:.2f}"
                )

            except Exception as e:
                logger.error(f"Error processing TMDB_ID={tmdb_id}: {e}", exc_info=True)
                continue
        
        # Convert to DataFrame for analysis
        evaluation_df = pd.DataFrame(evaluation_records)

        if evaluation_df.empty:
            logger.error("No valid predictions generated. Unable to compute metrics.")
            return {'error': 'No valid recommendations'}, pd.DataFrame()

        # Compute error metrics
        actual_ratings = evaluation_df['actual_rating'].values
        predicted_ratings = evaluation_df['predicted_rating'].values

        metrics = {
            'mse': mean_squared_error(actual_ratings, predicted_ratings),
            'rmse': np.sqrt(mean_squared_error(actual_ratings, predicted_ratings)),
            'mae': mean_absolute_error(actual_ratings, predicted_ratings),
            'r2_score': r2_score(actual_ratings, predicted_ratings),
            'sample_size': len(evaluation_df),
            'total_test_samples': len(test_data),
            'recommendation_coverage': (evaluation_df['tmdb_id'].nunique() / test_data['tmdb_id'].nunique()) * 100,
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

        # Log evaluation statistics
        logger.info("\nEvaluation Summary:")
        logger.info(f"Total test samples: {len(test_data)}")
        logger.info(f"Valid predictions: {len(evaluation_df)}")
        logger.info(f"Recommendation coverage: {metrics['recommendation_coverage']:.2f}%")
        logger.info("\nRating Distributions:")
        logger.info(f"Actual ratings:\n{evaluation_df['actual_rating'].describe()}")
        logger.info(f"Predicted ratings:\n{evaluation_df['predicted_rating'].describe()}")
        logger.info("\nError Distribution:")
        logger.info(f"Prediction errors:\n{evaluation_df['error'].describe()}")

        return metrics, evaluation_df
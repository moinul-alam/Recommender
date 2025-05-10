import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
from src.models.collaborative.v2.services.user_recommendation_service import UserRecommendationService

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Evaluates a recommendation model by comparing predicted ratings to actual user ratings.
    """

    @staticmethod
    def compute_recommendation_metrics(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        collaborative_dir_path: str,
        n_recommendations: int = 10,
        min_similarity: float = 0.1,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Compute recommendation performance metrics, including precision, recall, and F1-score.
        """
        evaluation_records = []
        users_evaluated = 0
        users_with_recommendations = 0

        for user_id in test_data['user_id'].unique():
            user_ratings = test_data[test_data['user_id'] == user_id].set_index('tmdb_id')['rating'].to_dict()
            if not user_ratings:
                continue

            recommendations = UserRecommendationService.get_user_recommendations(
                user_ratings=user_ratings,
                collaborative_dir_path=collaborative_dir_path,
                n_recommendations=n_recommendations,
                min_similarity=min_similarity
            )

            if not recommendations or "message" in recommendations:
                continue

            recommended_items = {rec['tmdb_id'] for rec in recommendations}
            actual_items = set(user_ratings.keys())

            if recommended_items:
                users_with_recommendations += 1

            true_positives = len(recommended_items & actual_items)
            false_positives = len(recommended_items - actual_items)
            false_negatives = len(actual_items - recommended_items)

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            evaluation_records.append({
                'user_id': user_id,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            users_evaluated += 1

        evaluation_df = pd.DataFrame(evaluation_records)

        overall_precision = evaluation_df['precision'].mean() if not evaluation_df.empty else 0
        overall_recall = evaluation_df['recall'].mean() if not evaluation_df.empty else 0
        overall_f1 = evaluation_df['f1_score'].mean() if not evaluation_df.empty else 0
        recommendation_coverage = (users_with_recommendations / users_evaluated) * 100 if users_evaluated > 0 else 0

        metrics = {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'recommendation_coverage': recommendation_coverage
        }
        logger.info("\nEvaluation Summary:")
        logger.info(f"Total test samples: {len(test_data)}")
        logger.info(f"Valid predictions: {len(evaluation_df)}")
        logger.info(f"Recommendation coverage: {metrics['recommendation_coverage']:.2f}%")
        logger.info("\nRating Distributions:")
        logger.info(f"Actual ratings:\n{test_data['rating'].describe()}")
        logger.info("\nError Distribution:")
        logger.info(f"Evaluation Metrics: {metrics}")

        return metrics, evaluation_df

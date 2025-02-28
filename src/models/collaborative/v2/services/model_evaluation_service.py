import os
import logging
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any
from src.models.collaborative.v2.pipeline.ModelEvaluator import ModelEvaluator

logger = logging.getLogger(__name__)

class ModelEvaluationService:
    """
    Service to evaluate recommendation model performance.
    """

    @classmethod
    def evaluate_model(
        cls, 
        collaborative_dir_path: str,
        sample_size: int = 100,
        n_recommendations: int = 10,
        min_similarity: float = 0.1
    ) -> Dict[str, Any]:
        """
        Evaluates the recommendation model by generating predictions and computing performance metrics.
        """
        try:
            # Load train dataset
            train_data_path = os.path.join(collaborative_dir_path, '2_train.feather')
            test_data_path = os.path.join(collaborative_dir_path, '2_test.feather')

            if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
                raise FileNotFoundError("Train or test dataset not found.")
            
            train_data = pd.read_feather(train_data_path)
            test_data = pd.read_feather(test_data_path)

            if train_data.empty or test_data.empty:
                raise ValueError("Train or test dataset is empty.")

            # Sample test data
            test_sample = test_data.sample(n=min(sample_size, len(test_data)), random_state=42)

            # Load mappings
            def load_pickle(file_name):
                path = os.path.join(collaborative_dir_path, file_name)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{file_name} not found at {path}")
                with open(path, 'rb') as f:
                    return pickle.load(f)

            user_reverse_mapping = load_pickle('2_user_reverse_mapping.pkl')
            item_reverse_mapping = load_pickle('2_item_reverse_mapping.pkl')

            # Convert internal IDs to TMDB IDs
            test_sample['user_id'] = test_sample['user_id'].map(user_reverse_mapping)
            test_sample['tmdb_id'] = test_sample['tmdb_id'].map(item_reverse_mapping)
            test_sample.dropna(subset=['user_id', 'tmdb_id'], inplace=True)

            if test_sample.empty:
                raise ValueError("No valid user or item IDs found after mapping.")

            # Compute evaluation metrics
            metrics, evaluation_df = ModelEvaluator.compute_recommendation_metrics(
                train_data=train_data,
                test_data=test_sample,
                collaborative_dir_path=collaborative_dir_path,
                n_recommendations=n_recommendations,
                min_similarity=min_similarity
            )

            # Save evaluation results
            results = {
                'metrics': metrics,
                'evaluation_details': evaluation_df.to_dict(orient='records')
            }

            results_path = os.path.join(collaborative_dir_path, '4_model_evaluation_results.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)

            logger.info(f"Model evaluation completed. Results saved at {results_path}")
            return metrics

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}", exc_info=True)
            raise

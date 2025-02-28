import os
import logging
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any
from src.models.collaborative.v2.services.recommendation_service import RecommendationService
from src.models.collaborative.v2.pipeline.ModelEvaluator import ModelEvaluator

logger = logging.getLogger(__name__)

class ModelEvaluationService:
    """
    Service to evaluate recommendation model performance.
    """

    @classmethod
    def evaluate_model(
        cls, 
        processed_dir_path: str,
        model_dir_path: str,
        sample_size: int = 100,
        n_recommendations: int = 10,
        min_similarity: float = 0.1
    ) -> Dict[str, Any]:
        """
        Evaluates the recommendation model by generating predictions and computing performance metrics.
        """
        try:
            # Load test dataset
            test_data_path = os.path.join(processed_dir_path, 'test.feather')
            if not os.path.exists(test_data_path):
                raise FileNotFoundError(f"Test dataset not found at {test_data_path}")

            test_data = pd.read_feather(test_data_path)
            if test_data.empty:
                raise ValueError("Test dataset is empty.")

            # Sample the test data
            test_sample = test_data.sample(n=min(sample_size, len(test_data)), random_state=42)

            logger.info(f"Test Data Statistics:")
            logger.info(f"Total test samples: {len(test_data)}")
            logger.info(f"Sample size for evaluation: {len(test_sample)}")
            logger.info(f"Rating distribution in sample:\n{test_sample['rating'].describe()}")

            # Load item reverse mapping
            mapping_path = os.path.join(processed_dir_path, 'item_reverse_mapping.pkl')
            if not os.path.exists(mapping_path):
                raise FileNotFoundError(f"Item reverse mapping file not found at {mapping_path}")

            with open(mapping_path, 'rb') as f:
                item_reverse_mapping = pickle.load(f)

            # Map internal IDs to TMDB IDs
            test_sample['tmdb_id'] = test_sample['tmdb_id'].map(item_reverse_mapping)
            test_sample = test_sample.dropna(subset=['tmdb_id'])  # Remove invalid mappings

            if test_sample.empty:
                raise ValueError("No valid TMDB IDs found in the test sample after mapping.")

            # Compute evaluation metrics
            metrics, evaluation_df = ModelEvaluator.compute_recommendation_metrics(
                test_data=test_sample,
                processed_dir_path=processed_dir_path,
                model_dir_path=model_dir_path,
                n_recommendations=n_recommendations,
                min_similarity=min_similarity
            )

            # Save evaluation results
            results = {
                'metrics': metrics,
                'evaluation_details': evaluation_df.to_dict(orient='records')
            }

            results_path = os.path.join(model_dir_path, 'model_evaluation_results.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)

            logger.info(f"Model evaluation completed successfully. Results saved at {results_path}")
            return metrics

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}", exc_info=True)
            raise


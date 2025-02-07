import os
import logging
import pickle
import pandas as pd
from typing import Dict, Any
from src.models.collaborative.v1.services.recommendation_service import RecommendationService
from src.models.collaborative.v1.pipeline.ModelEvaluator import ModelEvaluator

logger = logging.getLogger(__name__)


class ModelEvaluationService:
    @classmethod
    def evaluate_model(
        cls, 
        processed_dir_path: str,
        model_dir_path: str,
        sample_size: int = 50  # Use fixed sample size
    ) -> Dict[str, Any]:
        try:
            # Load and sample test data
            test_data = pd.read_feather(os.path.join(processed_dir_path, 'test.feather'))
            test_sample = test_data.sample(n=sample_size, random_state=42)
            
            logger.info(f"\nTest Data Sample Statistics:")
            logger.info(f"Original test set size: {len(test_data)}")
            logger.info(f"Sample size: {len(test_sample)}")
            logger.info(f"Rating distribution in sample:\n{test_sample['rating'].describe()}")
            
            # Load reverse mapping
            with open(os.path.join(processed_dir_path, 'item_reverse_mapping.pkl'), 'rb') as f:
                item_reverse_mapping = pickle.load(f)
            
            # Map IDs and verify
            test_sample['tmdb_id'] = test_sample['tmdb_id'].map(item_reverse_mapping)
            unmapped = test_sample['tmdb_id'].isna().sum()
            if unmapped > 0:
                logger.warning(f"Found {unmapped} unmapped TMDB IDs")
            
            # Initialize recommender
            first_item = test_sample.iloc[0]
            recommendation_result = RecommendationService.get_recommendations(
                items={str(int(first_item['tmdb_id'])): first_item['rating']},
                processed_dir_path=processed_dir_path,
                model_dir_path=model_dir_path
            )
            
            if not recommendation_result:
                raise ValueError("Failed to initialize recommender")
            
            recommender = recommendation_result['metadata'].get('recommender')
            
            # Compute metrics and get evaluation details
            metrics, evaluation_df = ModelEvaluator.compute_recommendation_metrics(
                recommender, 
                test_sample.dropna(subset=['tmdb_id'])
            )
            
            # Save detailed evaluation results
            results = {
                'metrics': metrics,
                'evaluation_details': evaluation_df.to_dict(orient='records')
            }
            
            with open(os.path.join(model_dir_path, 'model_evaluation_results.pkl'), 'wb') as f:
                pickle.dump(results, f)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
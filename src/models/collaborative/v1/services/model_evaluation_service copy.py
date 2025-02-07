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
        sample_fraction: float = 0.001,
        n_recommendations: int = 10
    ) -> Dict[str, Any]:
        """
        Robust model evaluation with dynamic sampling
        """
        try:
            # Load test data
            test_data_path = os.path.join(processed_dir_path, 'test.feather')
            test_data = pd.read_feather(test_data_path)
            
            # Dynamic sampling
            test_sample = test_data.sample(
                frac=sample_fraction, 
                random_state=42
            )
            
            # Restore original TMDB IDs
            reverse_mapping_path = os.path.join(processed_dir_path, 'item_reverse_mapping.pkl')
            with open(reverse_mapping_path, "rb") as f:
                item_reverse_mapping = pickle.load(f)
            
            test_sample['tmdb_id'] = test_sample['tmdb_id'].map(item_reverse_mapping)
            
            # Get recommendations service result
            recommendation_result = RecommendationService.get_recommendations(
                items={str(int(test_sample['tmdb_id'].iloc[0])): test_sample['rating'].iloc[0]},
                processed_dir_path=processed_dir_path,
                model_dir_path=model_dir_path
            )
            
            recommender = recommendation_result['metadata'].get('recommender')
            
            # Compute comprehensive metrics
            metrics = ModelEvaluator.compute_recommendation_metrics(
                recommender, 
                test_sample,
                n_recommendations=n_recommendations
            )
            
            # Persist metrics
            metrics_path = os.path.join(model_dir_path, 'model_metrics.pkl')
            with open(metrics_path, 'wb') as f:
                pickle.dump(metrics, f)
            
            logger.info(f"Model Evaluation Metrics: {metrics}")
            return metrics
        
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
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
        model_dir_path: str
    ) -> Dict[str, Any]:
        """
        Evaluate recommendation model performance
        
        Args:
            processed_dir_path: Path to processed data directory
            model_dir_path: Path to model directory
        
        Returns:
            Evaluation metrics dictionary
        """
        try:
            # Load test data
            test_data_path = os.path.join(processed_dir_path, 'test.feather')
            test_data = pd.read_feather(test_data_path)
            logger.info(f"Loaded test data with {len(test_data)} entries.")
            
            # Sample a small fraction of test data for debugging
            test_sample = test_data.sample(frac=0.001, random_state=42)  # 1% of data
            logger.info(f"Using {len(test_sample)} entries for evaluation.")

            # Load reverse mapping
            reverse_mapping_path = os.path.join(processed_dir_path, 'item_reverse_mapping.pkl')
            if not os.path.exists(reverse_mapping_path):
                raise ValueError("item_reverse_mapping.pkl not found in processed directory")
            
            with open(reverse_mapping_path, "rb") as f:
                item_reverse_mapping = pickle.load(f)
            logger.info(f"Loaded item reverse mapping with {len(item_reverse_mapping)} entries.")

            # Reconstruct original tmdb_id
            test_sample["tmdb_id"] = pd.to_numeric(test_sample["tmdb_id"], errors='coerce').map(item_reverse_mapping)
            logger.info("Reconstructed original tmdb_id values.")

            # Check if there are any missing mappings after the mapping operation
            missing_mappings = test_sample[test_sample["tmdb_id"].isna()]
            if not missing_mappings.empty:
                logger.warning(f"Found {len(missing_mappings)} missing tmdb_id mappings.")

            # Log first few rows after reconstruction
            logger.info(f"Reconstructed Test Data Sample:\n{test_sample.head(5)}")

            # Prepare input items: Select first valid tmdb_id with rating from the sample
            valid_sample = test_sample.dropna(subset=["tmdb_id", "rating"])
            if valid_sample.empty:
                raise ValueError("No valid test data available after tmdb_id mapping.")
            
            first_item = valid_sample.iloc[0]
            input_items = {str(int(first_item["tmdb_id"])): float(first_item["rating"])}
            logger.info(f"Testing recommender with input: {input_items}")

            # Get recommendations using the RecommendationService
            recommendation_result = RecommendationService.get_recommendations(
                items=input_items,
                processed_dir_path=processed_dir_path,
                model_dir_path=model_dir_path
            )

            logger.info(f"Recommendation result metadata: {recommendation_result.get('metadata')}")

            # Check if recommender initialized
            if not recommendation_result or "metadata" not in recommendation_result:
                raise ValueError("Failed to initialize recommender. Check logs for errors.")

            # Extract recommender from result
            recommender = recommendation_result['metadata'].get('recommender')

            # Compute metrics
            metrics = ModelEvaluator.compute_recommendation_metrics(
                recommender, 
                valid_sample  # Pass valid sample for evaluation
            )
            
            # Optional: Save metrics
            metrics_path = os.path.join(model_dir_path, 'model_metrics.pkl')
            with open(metrics_path, 'wb') as f:
                pickle.dump(metrics, f)

            logger.info(f"Model Evaluation Metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise

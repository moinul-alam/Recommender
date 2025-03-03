from typing import Optional
import numpy as np
import pandas as pd
import faiss
import logging
from pathlib import Path
from fastapi import HTTPException
import time

from src.models.content_based.v2.pipeline.IndexEvaluator import IndexEvaluator

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EvaluationService:
    @staticmethod
    def evaluate_index(
        model_folder_path: str,
        features_folder_path: str,
        num_test_queries: Optional[int] = 100
    ):
        try:
            model_folder_path = Path(model_folder_path)
            features_folder_path = Path(features_folder_path)

            # Load index
            if not model_folder_path.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"Model folder not found: {model_folder_path}"
                )
            
            index_path = model_folder_path / "content_based_model.index"
            index = faiss.read_index(str(index_path))
            
            # Load feature matrix
            if not features_folder_path.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"Features folder not found: {features_folder_path}"
                )
            
            feature_matrix_path = features_folder_path / "engineered_features.feather"
            feature_matrix = pd.read_feather(feature_matrix_path)
            ground_truth = feature_matrix.iloc[:, 1:].to_numpy().astype(np.float32)
            
            # Generate test queries
            num_test_queries = min(num_test_queries, len(ground_truth))
            test_queries = ground_truth[np.random.choice(ground_truth.shape[0], num_test_queries, replace=False)]
            
            # Initialize evaluator
            evaluator = IndexEvaluator(index, ground_truth)
            
            # Ensure ground_truth has valid shape before calling compression ratio
            feature_dim = ground_truth.shape[1] if len(ground_truth.shape) > 1 else 1
            
            # Measure latency
            start_time = time.time()
            evaluator.evaluate_recall(test_queries, k=10, n_ground_truth=10)  # Sample evaluation
            end_time = time.time()
            latency_avg = (end_time - start_time) / num_test_queries  # Average latency per query

            # Run evaluations
            results = {
                "stats": evaluator.get_index_stats(),
                "recall": evaluator.evaluate_recall(test_queries, k=10, n_ground_truth=10),
                "precision": evaluator.evaluate_precision(test_queries, k=10, n_ground_truth=10),
                "mAP": evaluator.evaluate_map(test_queries, k=10, n_ground_truth=10),
                "NDCG": evaluator.evaluate_ndcg(test_queries, k=10, n_ground_truth=10),
                "query_coverage": evaluator.evaluate_query_coverage(test_queries, k=10),
                "latency": (latency_avg, end_time),  # Tuple of latency and end_time
                "compression_ratio": evaluator.evaluate_compression_ratio(feature_dim),
                "memory_usage": evaluator.evaluate_memory()
            }
            
            # Return results
            return results
            
        except Exception as e:
            logger.error(f"Error during index evaluation: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error during index evaluation: {str(e)}"
            )

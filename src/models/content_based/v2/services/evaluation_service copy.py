from typing import Optional
import numpy as np
import pandas as pd
import faiss
import logging
from pathlib import Path
from fastapi import HTTPException

from src.schemas.content_based_schema import EvaluationResponse, IndexStats
from src.models.content_based.v2.pipeline.IndexEvaluator import IndexEvaluator

logger = logging.getLogger(__name__)

class EvaluationService:
    @staticmethod
    def evaluate_index(
        model_folder_path: str,
        features_folder_path: str,
        num_test_queries: Optional[int] = 100
    ) -> EvaluationResponse:
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
            
            feature_matrix_path= features_folder_path / "engineered_features.feather"
            feature_matrix = pd.read_feather(feature_matrix_path)
            features = feature_matrix.iloc[:, 1:].to_numpy().astype(np.float32)
            
            # Generate test queries
            num_test_queries = min(num_test_queries, len(features))
            test_queries = features[:num_test_queries]
            
            # Initialize evaluator
            evaluator = IndexEvaluator(index, features)
            
            # Run evaluations
            stats = evaluator.get_index_stats()
            recall = evaluator.evaluate_recall(test_queries, k=10, n_ground_truth=10)
            precision = evaluator.evaluate_precision(test_queries, k=10, n_ground_truth=10)
            latency = evaluator.evaluate_latency(test_queries, k=10)
            memory = evaluator.evaluate_memory()
            
            # Log results
            logger.info(f"Evaluation completed for index: {index_path}")
            logger.info(f"Recall@10: {recall:.3f}")
            logger.info(f"Precision@10: {precision:.3f}")
            logger.info(f"Latency: {latency[0]*1000:.2f}ms ± {latency[1]*1000:.2f}ms")
            
            return EvaluationResponse(
                index_stats=IndexStats(**stats),
                recall_at_10=recall,
                precision_at_10=precision,
                latency=latency,
                memory_usage_bytes=memory
            )
            
        except Exception as e:
            logger.error(f"Error during index evaluation: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error during index evaluation: {str(e)}"
            )

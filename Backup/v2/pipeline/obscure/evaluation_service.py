# import os
# import logging
# from pathlib import Path
# import pickle
# from fastapi import HTTPException
# import pandas as pd
# import numpy as np
# from typing import Dict, Any
# from src.models.collaborative.v2.pipeline.ModelEvaluator import ModelEvaluator
# from src.models.collaborative.v2.services.base_recommendation_service import BaseRecommendationService

# from src.models.common.DataLoader import load_data

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# class EvaluationService:
#     @staticmethod
#     def evaluate_recommender(
#         collaborative_dir_path: str,
#         file_names: dict,
#         sample_size: int = 100,
#         n_recommendations: int = 10,
#         min_similarity: float = 0.1
#     ) -> Dict[str, Any]:
#         try:
#             collaborative_dir_path = Path(collaborative_dir_path)
            
#             if not collaborative_dir_path.is_dir():
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"Invalid directory path: {collaborative_dir_path}"
#                 )
                
#             train_data_path = collaborative_dir_path / file_names["train_set"] 
#             if not train_data_path.exists():
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"Train data file not found at {train_data_path}"
#                 )
#             train_data = load_data(train_data_path)
#             if train_data.empty:
#                 raise HTTPException(
#                     status_code=400,
#                     detail="Train data is empty."
#                 )
                        
#             test_data_path = collaborative_dir_path / file_names["test_set"]
#             if not test_data_path.exists():
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"Test data file not found at {test_data_path}"
#                 )
#             test_data = load_data(test_data_path)
#             if test_data.empty:
#                 raise HTTPException(
#                     status_code=400,
#                     detail="Test data is empty."
#                 )

#             # Sample test data
#             test_sample = test_data.sample(n=min(sample_size, len(test_data)), random_state=42)
            
            
#             recommender_components = BaseRecommendationService.load_model_components(collaborative_dir_path, file_names)
            
#             # Load mappings
#             def load_pickle(file_name):
#                 path = os.path.join(collaborative_dir_path, file_name)
#                 if not os.path.exists(path):
#                     raise FileNotFoundError(f"{file_name} not found at {path}")
#                 with open(path, 'rb') as f:
#                     return pickle.load(f)

#             # Compute evaluation metrics
#             metrics, evaluation_df = ModelEvaluator.compute_recommendation_metrics(
#                 train_data=train_data,
#                 test_data=test_sample,
#                 collaborative_dir_path=collaborative_dir_path,
#                 n_recommendations=n_recommendations,
#                 min_similarity=min_similarity
#             )

#             # Save evaluation results
#             results = {
#                 'metrics': metrics,
#                 'evaluation_details': evaluation_df.to_dict(orient='records')
#             }

#             results_path = os.path.join(collaborative_dir_path, '4_model_evaluation_results.pkl')
#             with open(results_path, 'wb') as f:
#                 pickle.dump(results, f)

#             logger.info(f"Model evaluation completed. Results saved at {results_path}")
#             return metrics

#         except Exception as e:
#             logger.error(f"Model evaluation failed: {e}", exc_info=True)
#             raise

import logging
from pathlib import Path
from fastapi import HTTPException
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from typing import Dict, List, Any, Optional

from src.schemas.recommender_schema import RecommenderEvaluation, RecommendationRequest, Item
from src.models.common.file_config import file_names
from src.models.collaborative.v2.services.recommendation_service import RecommendationService
from src.models.common.DataLoader import load_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def precision_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Calculate precision at k."""
    if not recommended or k <= 0:
        return 0.0
    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / k


def recall_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Calculate recall at k."""
    if not relevant or not recommended or k <= 0:
        return 0.0
    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / len(relevant)


def f1_at_k(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def hit_rate_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Calculate hit rate at k (binary: 1 if any relevant item in top-k, 0 otherwise)."""
    if not recommended or not relevant or k <= 0:
        return 0.0
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


def ndcg_at_k(recommended: List[int], relevant_scores: Dict[int, float], k: int) -> float:
    """Calculate NDCG at k."""
    if not recommended or not relevant_scores or k <= 0:
        return 0.0
    
    # Create relevance scores for recommended items
    relevance = []
    for item in recommended[:k]:
        relevance.append(relevant_scores.get(item, 0.0))
    
    # Create ideal relevance scores (sorted in descending order)
    ideal_relevance = sorted(relevant_scores.values(), reverse=True)[:k]
    
    if not ideal_relevance or all(score == 0 for score in ideal_relevance):
        return 0.0
    
    # Pad with zeros if needed
    while len(relevance) < k:
        relevance.append(0.0)
    while len(ideal_relevance) < k:
        ideal_relevance.append(0.0)
    
    try:
        return ndcg_score([ideal_relevance], [relevance])
    except Exception as e:
        logger.warning(f"Error calculating NDCG: {e}")
        return 0.0


class EvaluationService:
    @staticmethod
    def evaluate_recommender(
        directory_path: str,
        sample_test_size: int = 10,
        k: int = 10,
        req_source: str = "movieId",
        recommendation_type: str = "both"  # "item_based", "user_based", or "both"
    ) -> RecommenderEvaluation:
        """
        Evaluate the recommender system using test data.
        
        Args:
            directory_path: Path to the model directory
            sample_test_size: Number of users to sample for evaluation (0 for all)
            k: Number of recommendations to evaluate
            req_source: Source type for recommendations ("movieId" or "tmdb")
            recommendation_type: Type of recommendations to evaluate
        """
        try:
            directory_path = Path(directory_path)

            if not directory_path.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid directory path: {directory_path}"
                )

            # Load training and test data
            training_data_path = directory_path / file_names["train_set"]
            if not training_data_path.exists():
                raise FileNotFoundError(f"Training data file not found at {training_data_path}")

            test_data_path = directory_path / file_names["test_set"]
            if not test_data_path.exists():
                raise FileNotFoundError(f"Test data file not found at {test_data_path}")

            training_data = load_data(training_data_path)
            if training_data.empty:
                raise ValueError("Training data is empty.")

            test_data = load_data(test_data_path)
            if test_data.empty:
                raise ValueError("Test data is empty.")

            logger.info(f"Loaded training data: {len(training_data)} rows")
            logger.info(f"Loaded test data: {len(test_data)} rows")

            # Sample test data if requested
            if sample_test_size > 0 and len(test_data) > sample_test_size:
                # Sample users, not individual ratings
                unique_users = test_data['userId'].unique()
                if len(unique_users) > sample_test_size:
                    sampled_users = np.random.choice(unique_users, size=sample_test_size, replace=False)
                    test_sample = test_data[test_data['userId'].isin(sampled_users)]
                else:
                    test_sample = test_data
            else:
                test_sample = test_data

            logger.info(f"Using {len(test_sample)} test samples from {test_sample['userId'].nunique()} users")

            # Get evaluation results
            results = EvaluationService.get_evaluation(
                test_data=test_sample,
                training_data=training_data,
                directory_path=str(directory_path),
                k=k,
                req_source=req_source,
                recommendation_type=recommendation_type
            )

            metrics = results["metrics"]
            evaluation_df = results["evaluation_df"]

            logger.info(f"Evaluation completed. Average metrics: {metrics}")

            return RecommenderEvaluation(
                status="success",
                message=f"Evaluation completed successfully for {len(evaluation_df)} users",
                precision=metrics.get("precision", 0.0),
                recall=metrics.get("recall", 0.0),
                f1_score=metrics.get("f1_score", 0.0),
                NDCG=metrics.get("NDCG", 0.0),
                HitRate=metrics.get("HitRate", 0.0)
            )

        except ValueError as ve:
            logger.error(f"Value error during evaluation: {ve}", exc_info=True)
            return RecommenderEvaluation(
                status="error",
                message=str(ve),
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                NDCG=0.0,
                HitRate=0.0
            )

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}", exc_info=True)
            return RecommenderEvaluation(
                status="error", 
                message=f"Evaluation failed: {str(e)}",
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                NDCG=0.0,
                HitRate=0.0
            )

    @staticmethod
    def get_evaluation(
        test_data: pd.DataFrame,
        training_data: pd.DataFrame,
        directory_path: str,
        k: int = 10,
        req_source: str = "movieId",
        recommendation_type: str = "both",
        min_user_ratings: int = 5
    ) -> Dict[str, Any]:
        """
        Perform the actual evaluation.
        
        Args:
            test_data: Test dataset
            training_data: Training dataset
            directory_path: Path to model directory
            k: Number of recommendations to evaluate
            req_source: Source type for recommendations
            recommendation_type: Type of recommendations to evaluate
            min_user_ratings: Minimum number of ratings a user must have in training data
        """
        
        all_metrics = {
            "precision": [],
            "recall": [],
            "f1_score": [],
            "NDCG": [],
            "HitRate": []
        }

        evaluation_records = []
        successful_evaluations = 0
        failed_evaluations = 0

        # Group test data by user
        test_users = test_data['userId'].unique()
        logger.info(f"Evaluating {len(test_users)} users")

        for user_id in test_users:
            try:
                # Get user's training data (for generating recommendations)
                user_training_data = training_data[training_data['userId'] == user_id]
                
                # Skip users with insufficient training data
                if len(user_training_data) < min_user_ratings:
                    logger.debug(f"Skipping user {user_id}: insufficient training data ({len(user_training_data)} ratings)")
                    continue

                # Get user's test data (ground truth)
                user_test_data = test_data[test_data['userId'] == user_id]
                if user_test_data.empty:
                    continue

                # Prepare recommendation request
                recommendation_items = []
                for _, row in user_training_data.iterrows():
                    item = Item(
                        movieId=int(row['movieId']) if req_source == "movieId" else None,
                        tmdbId=int(row['movieId']) if req_source == "tmdb" else None,  # Assuming movieId maps to tmdbId
                        rating=float(row['rating'])
                    )
                    recommendation_items.append(item)

                recommendation_request = RecommendationRequest(
                    items=recommendation_items,
                    n_recommendations=k * 2,  # Request more to ensure we get enough
                    req_source=req_source
                )

                # Get recommendations
                recommendation_response = RecommendationService.get_recommendations(
                    recommendation_request=recommendation_request,
                    directory_path=directory_path
                )

                if not recommendation_response or not recommendation_response.recommendations:
                    logger.warning(f"No recommendations received for user {user_id}")
                    failed_evaluations += 1
                    continue

                # Extract recommended items based on recommendation type
                recommended_items = []
                
                # Process different recommendation types
                recommendations_dict = recommendation_response.recommendations
                
                if recommendation_type == "item_based" and "item_based" in recommendations_dict:
                    recs = recommendations_dict["item_based"].recommendations
                    recommended_items = EvaluationService._extract_item_ids(recs, req_source)
                elif recommendation_type == "user_based" and "user_based" in recommendations_dict:
                    recs = recommendations_dict["user_based"].recommendations
                    recommended_items = EvaluationService._extract_item_ids(recs, req_source)
                elif recommendation_type == "both":
                    # Combine both types, prioritizing item-based
                    if "item_based" in recommendations_dict:
                        item_recs = recommendations_dict["item_based"].recommendations
                        recommended_items.extend(EvaluationService._extract_item_ids(item_recs, req_source))
                    
                    if "user_based" in recommendations_dict:
                        user_recs = recommendations_dict["user_based"].recommendations
                        user_items = EvaluationService._extract_item_ids(user_recs, req_source)
                        # Add user-based recommendations that aren't already in the list
                        for item in user_items:
                            if item not in recommended_items:
                                recommended_items.append(item)

                if not recommended_items:
                    logger.warning(f"No valid recommended items extracted for user {user_id}")
                    failed_evaluations += 1
                    continue

                # Prepare ground truth
                relevant_items = set(user_test_data['movieId'].astype(int).tolist())
                relevant_scores = dict(zip(
                    user_test_data['movieId'].astype(int), 
                    user_test_data['rating'].astype(float)
                ))

                # Calculate metrics
                precision = precision_at_k(recommended_items, relevant_items, k)
                recall = recall_at_k(recommended_items, relevant_items, k)
                f1 = f1_at_k(precision, recall)
                hitrate = hit_rate_at_k(recommended_items, relevant_items, k)
                ndcg = ndcg_at_k(recommended_items, relevant_scores, k)

                # Store metrics
                all_metrics["precision"].append(precision)
                all_metrics["recall"].append(recall)
                all_metrics["f1_score"].append(f1)
                all_metrics["HitRate"].append(hitrate)
                all_metrics["NDCG"].append(ndcg)

                evaluation_records.append({
                    "userId": user_id,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "HitRate": hitrate,
                    "NDCG": ndcg,
                    "num_recommendations": len(recommended_items),
                    "num_relevant": len(relevant_items),
                    "num_training_ratings": len(user_training_data)
                })

                successful_evaluations += 1

            except Exception as e:
                logger.warning(f"Error evaluating user {user_id}: {e}")
                failed_evaluations += 1
                continue

        logger.info(f"Evaluation completed: {successful_evaluations} successful, {failed_evaluations} failed")

        # Calculate average metrics
        avg_metrics = {}
        for metric, values in all_metrics.items():
            if values:
                avg_metrics[metric] = round(np.mean(values), 4)
                avg_metrics[f"{metric}_std"] = round(np.std(values), 4)
            else:
                avg_metrics[metric] = 0.0
                avg_metrics[f"{metric}_std"] = 0.0

        # Add summary statistics
        avg_metrics["num_evaluated_users"] = successful_evaluations
        avg_metrics["num_failed_users"] = failed_evaluations
        avg_metrics["success_rate"] = round(successful_evaluations / (successful_evaluations + failed_evaluations), 4) if (successful_evaluations + failed_evaluations) > 0 else 0.0

        return {
            "metrics": avg_metrics,
            "evaluation_df": pd.DataFrame(evaluation_records)
        }

    @staticmethod
    def _extract_item_ids(recommendations: List, req_source: str) -> List[int]:
        """Extract item IDs from recommendation objects."""
        item_ids = []
        for rec in recommendations:
            if req_source == "movieId" and hasattr(rec, 'movieId') and rec.movieId is not None:
                item_ids.append(int(rec.movieId))
            elif req_source == "tmdb" and hasattr(rec, 'tmdbId') and rec.tmdbId is not None:
                item_ids.append(int(rec.tmdbId))
        return item_ids
import logging
from pathlib import Path
from fastapi import HTTPException
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from typing import Dict, List, Any, Tuple

from src.schemas.recommender_schema import RecommenderEvaluation, RecommendationRequest, Item
from src.models.common.file_config import file_names
from src.models.collaborative.v2.services.recommendation_service import RecommendationService
from src.models.common.DataLoader import load_data
from src.models.common.logger import app_logger

logger = app_logger(__name__)

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
        k: int = 20,
        req_source: str = "movieId"
    ) -> Tuple[RecommenderEvaluation, RecommenderEvaluation]:
        """
        Evaluate both item-based and user-based recommender systems using test data.
        
        Args:
            directory_path: Path to the model directory
            sample_test_size: Number of users to sample for evaluation (0 for all)
            k: Number of recommendations to evaluate
            req_source: Source type for recommendations ("movieId" or "tmdb")
        
        Returns:
            Tuple of (item_based_evaluation, user_based_evaluation)
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

            # Evaluate item-based recommender
            logger.info("Starting item-based recommender evaluation...")
            item_based_results = EvaluationService.get_evaluation(
                test_data=test_sample,
                training_data=training_data,
                directory_path=str(directory_path),
                k=k,
                req_source=req_source,
                recommendation_type="item_based"
            )

            item_based_metrics = item_based_results["metrics"]
            item_based_evaluation_df = item_based_results["evaluation_df"]

            logger.info(f"Item-based evaluation completed. Average metrics: {item_based_metrics}")

            item_based_evaluation = RecommenderEvaluation(
                status="success",
                recommender_type="item_based",
                precision=item_based_metrics.get("precision", 0.0),
                recall=item_based_metrics.get("recall", 0.0),
                f1_score=item_based_metrics.get("f1_score", 0.0),
                NDCG=item_based_metrics.get("NDCG", 0.0),
                HitRate=item_based_metrics.get("HitRate", 0.0)
            )

            # Evaluate user-based recommender
            logger.info("Starting user-based recommender evaluation...")
            user_based_results = EvaluationService.get_evaluation(
                test_data=test_sample,
                training_data=training_data,
                directory_path=str(directory_path),
                k=k,
                req_source=req_source,
                recommendation_type="user_based"
            )

            user_based_metrics = user_based_results["metrics"]
            user_based_evaluation_df = user_based_results["evaluation_df"]

            logger.info(f"User-based evaluation completed. Average metrics: {user_based_metrics}")

            user_based_evaluation = RecommenderEvaluation(
                status="success",
                recommender_type="user_based",
                precision=user_based_metrics.get("precision", 0.0),
                recall=user_based_metrics.get("recall", 0.0),
                f1_score=user_based_metrics.get("f1_score", 0.0),
                NDCG=user_based_metrics.get("NDCG", 0.0),
                HitRate=user_based_metrics.get("HitRate", 0.0)
            )

            return item_based_evaluation, user_based_evaluation

        except ValueError as ve:
            logger.error(f"Value error during evaluation: {ve}", exc_info=True)
            error_evaluation = RecommenderEvaluation(
                status="error",
                recommender_type="unknown",
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                NDCG=0.0,
                HitRate=0.0
            )
            return error_evaluation, error_evaluation

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}", exc_info=True)
            error_evaluation = RecommenderEvaluation(
                status="error",
                recommender_type="unknown",
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                NDCG=0.0,
                HitRate=0.0
            )
            return error_evaluation, error_evaluation

    @staticmethod
    def get_evaluation(
        test_data: pd.DataFrame,
        training_data: pd.DataFrame,
        directory_path: str,
        k: int = 10,
        req_source: str = "movieId",
        recommendation_type: str = "item_based",
        min_user_ratings: int = 5
    ) -> Dict[str, Any]:
        """
        Perform the actual evaluation for a specific recommendation type.
        
        Args:
            test_data: Test dataset
            training_data: Training dataset
            directory_path: Path to model directory
            k: Number of recommendations to evaluate
            req_source: Source type for recommendations
            recommendation_type: Type of recommendations to evaluate ("item_based" or "user_based")
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
        logger.info(f"Evaluating {len(test_users)} users for {recommendation_type} recommender")

        for user_id in test_users:
            try:
                # Get user's training data (for generating recommendations)
                logger.info(f"Evaluating user {user_id} for {recommendation_type} recommender")
                
                user_training_data = training_data[training_data['userId'] == user_id]
                
                logger.info(f"User {user_id} has {len(user_training_data)} items in their training data.")
                
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

                # Extract recommended items for the specific recommendation type
                recommended_items = []
                recommendations_dict = recommendation_response.recommendations
                
                if recommendation_type in recommendations_dict:
                    recs = recommendations_dict[recommendation_type].recommendations
                    recommended_items = EvaluationService._extract_item_ids(recs, req_source)
                else:
                    logger.warning(f"No {recommendation_type} recommendations found for user {user_id}")
                    failed_evaluations += 1
                    continue

                if not recommended_items:
                    logger.warning(f"No valid recommended items extracted for user {user_id} from {recommendation_type}")
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
                    "recommendation_type": recommendation_type,
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
                logger.warning(f"Error evaluating user {user_id} for {recommendation_type}: {e}")
                failed_evaluations += 1
                continue

        logger.info(f"{recommendation_type} evaluation completed: {successful_evaluations} successful, {failed_evaluations} failed")

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
        avg_metrics["recommendation_type"] = recommendation_type

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
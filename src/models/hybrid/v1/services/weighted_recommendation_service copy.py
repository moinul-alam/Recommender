from collections import defaultdict
from typing import Any, Dict, List
import logging
from fastapi import HTTPException
from src.models.collaborative.v2.services.user_recommendation_service import UserRecommendationService
from src.models.collaborative.v2.services.item_recommendation_service import ItemRecommendationService
from src.models.content_based.v2.services.recommendation_service import RecommendationService
from src.schemas.content_based_schema import RecommendationRequest, RecommendationResponse

logger = logging.getLogger(__name__)

class WeighedRecommendationService:
    @staticmethod
    def get_user_recommendations(
        user_ratings: Dict[str, float],
        request_data: List[Dict[str, Any]],
        content_based_dir_path: str,
        collaborative_dir_path: str,
        n_recommendations: int,
        min_similarity: float
    ) -> List[Dict]:
        try:
            logger.info(f"User ratings received: {user_ratings}")

            # Convert user ratings to integer keys and float values
            try:
                user_ratings = {int(key): float(value) for key, value in user_ratings.items()}
                logger.info(f"Converted user ratings: {user_ratings}")
            except Exception as e:
                logger.error(f"Error in converting user ratings: {str(e)}")
                raise HTTPException(status_code=400, detail="Invalid rating format")

            # Process metadata
            metadata_dict = {item["tmdb_id"]: item["metadata"] for item in request_data}
            logger.info(f"Processed metadata for {len(metadata_dict)} movies")

            # Get Collaborative Recommendations (IndexFlatIP)
            collaborative_recommendations = UserRecommendationService.get_user_recommendations(
                user_ratings=user_ratings,
                collaborative_dir_path=collaborative_dir_path,
                n_recommendations=n_recommendations,
                min_similarity=min_similarity
            )

            collaborative_scores = {}
            for rec in collaborative_recommendations:
                if "tmdb_id" in rec and "predicted_rating" in rec:
                    # Use the predicted rating directly (already from IndexFlatIP)
                    collaborative_scores[rec["tmdb_id"]] = rec["predicted_rating"]
                else:
                    logger.warning(f"Skipping invalid collaborative recommendation: {rec}")

            # Get Content-Based Recommendations (IndexFlatIP)
            content_based_scores = defaultdict(float)

            for tmdb_id, metadata in metadata_dict.items():
                recommendation_request = RecommendationRequest(tmdb_id=tmdb_id, metadata=metadata)

                content_recommendations: RecommendationResponse = RecommendationService.recommendation_service(
                    recommendation_request=recommendation_request,
                    content_based_dir_path=content_based_dir_path,
                    n_items=n_recommendations
                )

                if not hasattr(content_recommendations, "similarMedia"):
                    logger.error(f"Invalid response format from content-based: {content_recommendations}")
                    continue  # Skip if invalid format

                for rec in content_recommendations.similarMedia:
                    rec_id = rec.tmdb_id
                    try:
                        # Use the similarity score directly (already from IndexFlatIP)
                        similarity = float(rec.similarity)  # No need to strip '%' or normalize
                        content_based_scores[rec_id] = max(content_based_scores[rec_id], similarity)  # Take max
                    except ValueError:
                        logger.warning(f"Skipping invalid similarity value: {rec.similarity}")

            # Normalize Scores (Z-score)
            def z_score_normalization(scores):
                values = list(scores.values())
                mean = sum(values) / len(values) if values else 0
                std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5 if values else 1
                return {k: (v - mean) / std if std != 0 else 0 for k, v in scores.items()}

            collaborative_scores = z_score_normalization(collaborative_scores)
            content_based_scores = z_score_normalization(content_based_scores)

            # Dynamic Weighting
            num_ratings = len(user_ratings)
            rating_threshold = 10

            weight_cf = min(num_ratings / rating_threshold, 1.0)  # Collaborative filtering weight
            weight_cb = 1 - weight_cf  # Content-based weight

            logger.info(f"Dynamic weights - CF: {weight_cf}, CB: {weight_cb}")

            # Compute Final Hybrid Score
            final_scores = defaultdict(float)
            for rec_id in set(collaborative_scores.keys()).union(content_based_scores.keys()):
                final_scores[rec_id] = (
                    weight_cf * collaborative_scores.get(rec_id, 0) +
                    weight_cb * content_based_scores.get(rec_id, 0)
                )

            # Rank and Return Recommendations
            ranked_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
            final_recommendations = [{"tmdb_id": rec_id, "score": score} for rec_id, score in ranked_recommendations]

            logger.info(f"Final hybrid recommendations: {final_recommendations}")
            return final_recommendations

        except Exception as e:
            logger.error(f"Error in hybrid recommendation: {str(e)}")
            raise HTTPException(status_code=500, detail="Hybrid recommendation failed")
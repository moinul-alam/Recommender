from typing import List, Dict, Optional
import logging
from fastapi import HTTPException
from src.models.collaborative.v2.pipeline.item_recommender import ItemRecommender
from src.models.collaborative.v2.services.base_recommendation_service import BaseRecommendationService
from src.schemas.recommender_schema import Recommendation, RecommendationResponse, RecommendationRequest, RecommendationRequestedItem


logger = logging.getLogger(__name__)

class ItemRecommendationService:
    @staticmethod
    def get_item_recommendations(
        recommendation_request: RecommendationRequest,
        collaborative_dir_path: str,
        similarity_metric: str,
        min_similarity: float
    ) -> RecommendationResponse:
        """
        Generate item-based recommendations based on a list of item IDs.
        
        Args:
            items: List of item IDs to base recommendations on
            collaborative_dir_path: Directory path containing model files
            file_names: Dictionary mapping component names to filenames
            n_recommendations: Number of recommendations to return
            min_similarity: Minimum similarity threshold for recommendations
            
        Returns:
            List of recommendation dictionaries with tmdb_id and similarity
        """
        try:
            logger.info(f"Received recommendation request at Service...")
            user_item_ratings = [(item.tmdb_id, item.rating) for item in recommendation_request.items]
            
            logger.info(f"User item ratings: {user_item_ratings}")
            
            rating_threshold = 3
            
            liked_items = [(int(tmdb_id), 
                            (rating)) for tmdb_id, rating in user_item_ratings if rating is None or rating >= rating_threshold]
            
            logger.info(f"Filtered liked items: {liked_items}")
            
            n_recommendations = recommendation_request.n_recommendations or 10
            
            logger.info(f"Items for recommendation: {liked_items}")

            # Load model components
            components = BaseRecommendationService.load_model_components(collaborative_dir_path)
            
            logger.info(f"Loaded model components: {components.keys()}")
            # Initialize recommender
            recommender = ItemRecommender(
                faiss_index=components["faiss_item_index"],
                embedding_matrix=components["item_matrix"],
                user_item_mappings=components["user_item_mappings"],
                similarity_metric= similarity_metric,
                min_similarity=min_similarity
            )

            recommendations = recommender.generate_recommendations(
                item_ids=liked_items,
                n_recommendations=n_recommendations
            )

            logger.info(f"Generated {len(recommendations)} item-based recommendations")
            return recommendations if recommendations else []

        except Exception as e:
            logger.error(f"Error in generating item-based recommendations: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")
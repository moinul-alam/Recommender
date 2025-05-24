import pathlib
from functools import lru_cache
from fastapi import HTTPException
from typing import Optional, List, Tuple, Dict
from src.models.collaborative.v2.services.base_recommendation_service import BaseRecommendationService
from src.schemas.recommender_schema import RecommendationRequest, Recommendation, RecommendationResponse, RecommendationCategory
from src.models.common.logger import app_logger
from src.models.common.DataLoader import load_data
from src.models.common.file_config import file_names
from src.models.collaborative.v2.pipeline.item_recommender import ItemRecommender
from src.models.collaborative.v2.pipeline.user_recommender import UserRecommender

logger = app_logger(__name__)

class RecommendationService:
    
    @staticmethod
    @lru_cache(maxsize=32)
    def _load_mapping_files(directory_path: str) -> Dict:
        """Cache mapping file loading to avoid repeated I/O operations."""
        tmdb_to_movie_path = pathlib.Path(directory_path) / file_names["tmdb_to_movie_mapping"]
        movie_to_tmdb_path = pathlib.Path(directory_path) / file_names["movie_to_tmdb_mapping"]
        
        if not tmdb_to_movie_path.is_file() or not movie_to_tmdb_path.is_file():
            raise HTTPException(status_code=400, detail="Mapping files not found")
        
        tmdb_to_movie_map = load_data(tmdb_to_movie_path)
        movie_to_tmdb_map = load_data(movie_to_tmdb_path)
        
        if not isinstance(tmdb_to_movie_map, dict) or not isinstance(movie_to_tmdb_map, dict):
            raise HTTPException(status_code=400, detail="Invalid mapping dictionaries")
        
        return {
            "tmdb_to_movie_map": tmdb_to_movie_map,
            "movie_to_tmdb_map": movie_to_tmdb_map
        }
    
    @staticmethod
    @lru_cache(maxsize=16)
    def _load_model_components(directory_path: str) -> Dict:
        """Cache model components loading to avoid repeated loading."""
        return BaseRecommendationService.load_model_components(directory_path)
    
    @staticmethod
    def _convert_tmdb_to_movie_ratings(
        recommendation_request: RecommendationRequest, 
        tmdb_to_movie_map: Dict
    ) -> List[Tuple[int, float]]:
        """Step 1: Convert TMDB IDs to movie IDs if needed."""
        if recommendation_request.req_source == "tmdb":
            # Filter valid TMDB IDs and convert to movie IDs
            tmdb_ids = [item.tmdbId for item in recommendation_request.items]
            valid_tmdb_ids = [tmdb_id for tmdb_id in tmdb_ids if str(tmdb_id) in tmdb_to_movie_map]
            
            if not valid_tmdb_ids:
                raise HTTPException(status_code=400, detail="No matching movieId found for provided tmdbId")
            
            return [(int(tmdb_to_movie_map[str(item.tmdbId)]), item.rating) 
                   for item in recommendation_request.items 
                   if str(item.tmdbId) in tmdb_to_movie_map]
        else:
            # Already movie IDs
            return [(int(item.movieId), item.rating) for item in recommendation_request.items]
    
    @staticmethod
    def _convert_recommendations_to_response_format(
        raw_recommendations: List[Dict], 
        req_source: str, 
        movie_to_tmdb_map: Dict = None
    ) -> List[Recommendation]:
        """Convert raw recommendations to response format based on request source."""
        formatted_recommendations = []
        
        for rec in raw_recommendations:
            recommendation = Recommendation(
                movie_source=req_source,
                similarity=rec.get("similarity"),
                predicted_rating=rec.get("predicted_rating")
            )
            
            if req_source == "tmdb":
                # Need to convert movieId back to tmdbId
                movie_id = rec.get("movieId")
                if movie_id and str(movie_id) in movie_to_tmdb_map:
                    recommendation.tmdbId = int(movie_to_tmdb_map[str(movie_id)])
                    recommendation.movieId = None
                else:
                    continue  # Skip if conversion fails
            else:
                # Use movieId directly
                recommendation.movieId = rec.get("movieId")
                recommendation.tmdbId = None
            
            formatted_recommendations.append(recommendation)
        
        return formatted_recommendations
    
    @staticmethod
    def get_recommendations(
        recommendation_request: RecommendationRequest,
        directory_path: str
    ) -> Optional[RecommendationResponse]:
        try:
            logger.info(f"Received recommendation request: {recommendation_request}")
            
            # Step 1: Load mappings only if needed (for TMDB requests)
            mappings = {}
            if recommendation_request.req_source == "tmdb":
                mappings = RecommendationService._load_mapping_files(directory_path)
            
            # Convert to movie ID ratings
            user_item_ratings = RecommendationService._convert_tmdb_to_movie_ratings(
                recommendation_request, 
                mappings.get("tmdb_to_movie_map", {})
            )
            
            if not user_item_ratings:
                raise HTTPException(status_code=400, detail="No valid user item ratings provided")
            
            logger.info(f"Processed user-item ratings: {user_item_ratings}")
            
            # Step 2: Load model components (cached)
            components = RecommendationService._load_model_components(directory_path)
            model_info = components["model_info"]
            similarity_metric = model_info["similarity_metric"]
            
            recommendations_dict = {}
            
            # Step 3: Generate item-based recommendations
            try:
                item_recommender = ItemRecommender(
                    faiss_index=components["faiss_item_index"],
                    embedding_matrix=components["item_matrix"],
                    user_item_matrix=components["user_item_matrix"],
                    user_item_mappings=components["user_item_mappings"],
                    user_item_means=components["user_item_means"],
                    similarity_metric=similarity_metric,
                    min_similarity=0.1
                )
                
                # Filter liked items (rating >= 3 or None)
                like_threshold = 3
                liked_items = [(movieId, rating) for movieId, rating in user_item_ratings 
                              if rating is None or rating >= like_threshold]
                
                item_based_raw = item_recommender.generate_recommendations(
                    item_ids=liked_items,
                    n_recommendations=recommendation_request.n_recommendations
                )
                
                logger.info(f"Generated {len(item_based_raw)} item-based recommendations")
                
                # Convert to response format
                item_based_recommendations = RecommendationService._convert_recommendations_to_response_format(
                    item_based_raw, 
                    recommendation_request.req_source, 
                    mappings.get("movie_to_tmdb_map", {})
                )
                
                recommendations_dict["item_based"] = RecommendationCategory(
                    recommendation_type="item_based",
                    recommendations=item_based_recommendations
                )
                
            except Exception as e:
                logger.error(f"Error in item-based recommendations: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to generate item-based recommendations: {str(e)}")
            
            # Step 4: Generate user-based recommendations
            try:
                # Validate required user components
                svd_user_model = components["svd_components"].get(file_names["svd_user_model"])
                if not svd_user_model:
                    logger.warning("SVD user model not found, skipping user-based recommendations")
                elif "faiss_user_index" not in components or "user_matrix" not in components:
                    logger.warning("Required user components missing, skipping user-based recommendations")
                else:
                    user_recommender = UserRecommender(
                        faiss_user_index=components["faiss_user_index"],
                        user_embedding_matrix=components["user_matrix"],
                        faiss_item_index=components["faiss_item_index"],
                        item_embedding_matrix=components["item_matrix"],
                        user_item_matrix=components["user_item_matrix"],
                        user_item_mappings=components["user_item_mappings"],
                        svd_user_model=svd_user_model,
                        user_item_means=components["user_item_means"],
                        similarity_metric=similarity_metric,
                        min_similarity=0.1,
                        n_neighbors=50
                    )
                    
                    user_based_raw = user_recommender.generate_recommendations(
                        user_ratings=user_item_ratings,
                        n_recommendations=recommendation_request.n_recommendations
                    )
                    
                    logger.info(f"Generated {len(user_based_raw)} user-based recommendations")
                    
                    if user_based_raw:
                        # Convert to response format
                        user_based_recommendations = RecommendationService._convert_recommendations_to_response_format(
                            user_based_raw, 
                            recommendation_request.req_source, 
                            mappings.get("movie_to_tmdb_map", {})
                        )
                        
                        recommendations_dict["user_based"] = RecommendationCategory(
                            recommendation_type="user_based",
                            recommendations=user_based_recommendations
                        )
                
            except Exception as e:
                logger.warning(f"User-based recommendations failed: {str(e)}")
                # Continue without user-based recommendations
            
            # Step 5: Return recommendations
            if recommendations_dict:
                status = "success" if "user_based" in recommendations_dict else "partial_success"
                return RecommendationResponse(
                    status=status,
                    recommendations=recommendations_dict
                )
            else:
                raise HTTPException(status_code=500, detail="No recommendations could be generated")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in generating recommendations: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")
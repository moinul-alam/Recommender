import pathlib
import logging
from fastapi import HTTPException
from typing import Dict, List, Optional
from src.models.collaborative.v2.services.base_recommendation_service import BaseRecommendationService
from src.schemas.recommender_schema import RecommendationRequest, Recommendation, RecommendationResponse, RecommendationCategory
from src.models.common.logger import app_logger
from src.models.common.DataLoader import load_data
from src.models.common.file_config import file_names
from src.models.collaborative.v2.pipeline.item_recommender import ItemRecommender
from src.models.collaborative.v2.pipeline.user_recommender import UserRecommender

logger = app_logger(__name__)

class RecommendationService:
    def get_recommendations(
        recommendation_request: RecommendationRequest,
        directory_path: str
    ) -> Optional[RecommendationResponse]:
        try:
            logger.info(f"Received recommendation request: {recommendation_request}")
            
            # Process items based on request source
            user_item_ratings = []
        
            # Load mapping dictionaries
            tmdb_to_movie_path = pathlib.Path(directory_path) / file_names["tmdb_to_movie_mapping"]
            movie_to_tmdb_path = pathlib.Path(directory_path) / file_names["movie_to_tmdb_mapping"]
            
            if not tmdb_to_movie_path.is_file() or not movie_to_tmdb_path.is_file():
                raise HTTPException(status_code=400, detail=f"Mapping files not found")
            
            # Load the pre-generated bidirectional mappings
            tmdb_to_movie_map = load_data(tmdb_to_movie_path)
            movie_to_tmdb_map = load_data(movie_to_tmdb_path)
            
            if not isinstance(tmdb_to_movie_map, dict) or not isinstance(movie_to_tmdb_map, dict):
                raise HTTPException(status_code=400, detail="Invalid mapping dictionaries")
            
            if recommendation_request.req_source == "tmdb":
                # Process tmdbId-based request
                tmdbIds = [item.tmdbId for item in recommendation_request.items]
                filtered_tmdbIds = [tmdbId for tmdbId in tmdbIds if str(tmdbId) in tmdb_to_movie_map]
                
                if not filtered_tmdbIds:
                    raise HTTPException(status_code=400, detail="No matching movieId found for provided tmdbId")
                
                user_item_ratings = [(int(tmdb_to_movie_map[str(item.tmdbId)]), item.rating) 
                                for item in recommendation_request.items 
                                if str(item.tmdbId) in tmdb_to_movie_map]
            else:  # movieId source
                user_item_ratings = [(int(item.movieId), item.rating) 
                                for item in recommendation_request.items]
                
            if not user_item_ratings:
                raise HTTPException(status_code=400, detail="No valid user item ratings provided")
            
            logger.info(f"Processed user-item ratings: {user_item_ratings}")
            
            # Load model components
            components = BaseRecommendationService.load_model_components(directory_path)
            
            model_info = components["model_info"]
            
            # Validate that SVD components are loaded correctly
            svd_components = components["svd_components"]
            svd_user_model = svd_components.get(file_names["svd_user_model"])
            
            logger.info(f"Loaded SVD user model: {svd_user_model}")
            if svd_user_model is None:
                raise HTTPException(status_code=500, detail="SVD user model not found")
            
            similarity_metric = model_info["similarity_metric"]
            # Generate item-based recommendations
            like_threshold = 3
            liked_items = [(movieId, rating) for movieId, rating in user_item_ratings 
                        if rating is None or rating >= like_threshold]
            
            recommendations_dict = {}
            
            # Generate item-based recommendations first
            try:
                item_recommender = ItemRecommender(
                    faiss_index=components["faiss_item_index"],
                    embedding_matrix=components["item_matrix"],
                    user_item_mappings=components["user_item_mappings"],
                    similarity_metric=similarity_metric,
                    min_similarity=0.1,
                    tmdb_to_movie_map=tmdb_to_movie_map,
                    movie_to_tmdb_map=movie_to_tmdb_map,
                    req_source=recommendation_request.req_source
                )
                
                item_based_raw_recommendations = item_recommender.generate_recommendations(
                    item_ids=liked_items,
                    n_recommendations=recommendation_request.n_recommendations
                )
                
                logger.info(f"Generated {len(item_based_raw_recommendations)} item-based recommendations")
                
                # Format item-based recommendations
                item_based_recommendations = []
                for rec in item_based_raw_recommendations:
                    # Add a default predicted_rating of None if it's missing
                    predicted_rating = rec.get("predicted_rating")
                    
                    item_based_recommendations.append(
                        Recommendation(
                            movie_source=recommendation_request.req_source,
                            movieId=rec.get("movieId") if recommendation_request.req_source == "movieId" else None,
                            tmdbId=rec.get("tmdbId") if recommendation_request.req_source == "tmdb" else None,
                            similarity=rec.get("similarity"),
                            predicted_rating=predicted_rating
                        )
                    )
                
                recommendations_dict["item_based"] = RecommendationCategory(
                    recommendation_type="item_based",
                    recommendations=item_based_recommendations
                )
            except Exception as e:
                logger.error(f"Error in item-based recommendations: {str(e)}")
                # If item recommendations fail, we can't continue
                raise HTTPException(status_code=500, detail=f"Failed to generate item-based recommendations: {str(e)}")
            
            # Generate user-based recommendations - if this fails, we still return item-based recommendations
            try:
                # Validate that required user components exist
                if "faiss_user_index" not in components or "user_matrix" not in components:
                    logger.warning("Required user components missing")
                    # Early return instead of raising - return what we have so far
                    return RecommendationResponse(
                        status="partial_success",
                        recommendations=recommendations_dict
                    )
                
                # Validate user mappings
                user_item_mappings = components["user_item_mappings"]
                if "user_mapping" not in user_item_mappings or "user_reverse_mapping" not in user_item_mappings:
                    logger.warning("User mappings missing, will attempt to continue with item mappings only")
                
                user_recommender = UserRecommender(
                    faiss_user_index=components["faiss_user_index"],
                    user_embedding_matrix=components["user_matrix"],
                    faiss_item_index=components["faiss_item_index"],
                    item_embedding_matrix=components["item_matrix"],
                    user_item_matrix=components["user_item_matrix"],
                    user_item_mappings=components["user_item_mappings"],
                    svd_user_model=svd_user_model,
                    similarity_metric=similarity_metric,
                    min_similarity=0.1,
                    n_neighbors=50,
                    tmdb_to_movie_map=tmdb_to_movie_map,
                    movie_to_tmdb_map=movie_to_tmdb_map,
                    req_source=recommendation_request.req_source
                )
                
                user_based_raw_recommendations = user_recommender.generate_recommendations(
                    user_ratings=user_item_ratings,
                    n_recommendations=recommendation_request.n_recommendations
                )
                
                logger.info(f"Generated {len(user_based_raw_recommendations)} user-based recommendations")
                
                # Only add user-based recommendations if there are any
                if user_based_raw_recommendations:
                    # Format user-based recommendations
                    user_based_recommendations = []
                    for rec in user_based_raw_recommendations:
                        user_based_recommendations.append(
                            Recommendation(
                                movie_source=recommendation_request.req_source,
                                movieId=rec.get("movieId") if recommendation_request.req_source == "movieId" else None,
                                tmdbId=rec.get("tmdbId") if recommendation_request.req_source == "tmdb" else None,
                                similarity=rec.get("similarity"),
                                predicted_rating=rec.get("predicted_rating")
                            )
                        )
                    
                    recommendations_dict["user_based"] = RecommendationCategory(
                        recommendation_type="user_based",
                        recommendations=user_based_recommendations
                    )
                else:
                    logger.warning("User-based recommender returned empty results, continuing with item-based only")
                
            except Exception as e:
                logger.error(f"Error in user-based recommendations: {str(e)}")
                # Don't raise an exception here, just continue without user-based recommendations
            
            # If we have at least some recommendations, return them with appropriate status
            if recommendations_dict:
                status = "success" if "user_based" in recommendations_dict else "partial_success"
                return RecommendationResponse(
                    status=status,
                    recommendations=recommendations_dict
                )
            else:
                # This should never happen since we'd have raised an exception if item-based failed
                raise HTTPException(status_code=500, detail="No recommendations could be generated")
            
        except HTTPException:
            # Re-raise HTTP exceptions as they are already properly formatted
            raise
        except Exception as e:
            logger.error(f"Error in generating recommendations: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")
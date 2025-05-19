import pathlib
from fastapi import HTTPException, Path
from typing import Dict, List, Optional, Union
import pickle
import faiss
import numpy as np
import pandas as pd
from scipy import sparse
import logging
from src.models.collaborative.v2.pipeline.recommender import BaseRecommender
from src.schemas.recommender_schema import RecommendationRequest, Recommendation, RecommendationResponse, RecommendationCategory
from src.models.common.logger import app_logger
from src.models.common.DataLoader import load_data
from src.models.common.file_config import file_names
from src.models.collaborative.v2.pipeline.item_recommender import ItemRecommender
from src.models.collaborative.v2.pipeline.user_recommender import UserRecommender


logger = app_logger(__name__)

class RecommendationService:

    @staticmethod
    def get_recommendations(
        recommendation_request: RecommendationRequest,
        directory_path: str
    ) -> RecommendationResponse:
        try:
            # Step 1: Log received items
            logger.info(f"Received recommendation request: {recommendation_request}")
            
            # Step 2: Process items based on request source
            user_item_ratings = []
            tmdb_to_movie_map = {}
            movie_to_tmdb_map = {}
            
            if recommendation_request.req_source == "tmdb":
                movieId_tmdbId_mapping_path = pathlib.Path(directory_path) / file_names["movieId_tmdbId_mapping"]
                if not movieId_tmdbId_mapping_path.is_file():
                    raise HTTPException(
                        status_code=400,
                        detail=f"MovieId to tmdbId mapping file not found: {movieId_tmdbId_mapping_path}"
                    )
                movieId_tmdbId_mapping = load_data(movieId_tmdbId_mapping_path)
                
                # Handle both DataFrame and Dictionary formats of the mapping data
                if isinstance(movieId_tmdbId_mapping, pd.DataFrame):
                    if movieId_tmdbId_mapping.empty:
                        raise HTTPException(
                            status_code=400,
                            detail="MovieId to tmdbId mapping is empty or invalid"
                        )
                    
                    # Filter mapping based on provided tmdbIds
                    filtered_mapping = movieId_tmdbId_mapping[movieId_tmdbId_mapping["tmdbId"].isin([item.tmdbId for item in recommendation_request.items])]
                    if filtered_mapping.empty:
                        raise HTTPException(
                            status_code=400,
                            detail="No matching movieId found for the provided tmdbId"
                        )
                    
                    # Create mapping dictionaries
                    tmdb_to_movie_map = filtered_mapping.set_index("tmdbId")["movieId"].to_dict()
                    movie_to_tmdb_map = filtered_mapping.set_index("movieId")["tmdbId"].to_dict()
                    
                elif isinstance(movieId_tmdbId_mapping, dict):
                    if not movieId_tmdbId_mapping:
                        raise HTTPException(
                            status_code=400,
                            detail="MovieId to tmdbId mapping is empty or invalid"
                        )
                    
                    # Check if the dictionary is structured as expected
                    # Assume structure is {tmdbId: movieId}
                    tmdb_ids = [item.tmdbId for item in recommendation_request.items]
                    filtered_tmdb_ids = [tmdb_id for tmdb_id in tmdb_ids if tmdb_id in movieId_tmdbId_mapping]
                    
                    if not filtered_tmdb_ids:
                        raise HTTPException(
                            status_code=400,
                            detail="No matching movieId found for the provided tmdbId"
                        )
                    
                    # Use the dictionary directly if it has the correct format
                    if all(isinstance(key, (int, str)) for key in movieId_tmdbId_mapping.keys()):
                        tmdb_to_movie_map = {tmdb_id: movieId_tmdbId_mapping[tmdb_id] for tmdb_id in filtered_tmdb_ids}
                        # Create the reverse mapping
                        movie_to_tmdb_map = {v: k for k, v in tmdb_to_movie_map.items()}
                    else:
                        # If dict has a different structure, create a proper mapping
                        # This assumes the dict has a nested structure or a different format
                        # Adjust this according to the actual structure of your dictionary
                        raise HTTPException(
                            status_code=500,
                            detail="Unsupported format for movieId to tmdbId mapping"
                        )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="MovieId to tmdbId mapping is in an unsupported format"
                    )
                
                # Create user-item ratings tuples
                user_item_ratings = [(int(tmdb_to_movie_map[item.tmdbId]), item.rating) for item in recommendation_request.items if item.tmdbId in tmdb_to_movie_map]
            else:  # movieId source
                # Get all movieIds from the request
                movieIds = [int(item.movieId) for item in recommendation_request.items]
                
                # Load movieId to tmdbId mapping to create reverse mapping
                movieId_tmdbId_mapping_path = pathlib.Path(directory_path) / file_names["movieId_tmdbId_mapping"]
                if movieId_tmdbId_mapping_path.is_file():
                    movieId_tmdbId_mapping = load_data(movieId_tmdbId_mapping_path)
                    
                    # Handle both DataFrame and Dictionary formats
                    if isinstance(movieId_tmdbId_mapping, pd.DataFrame):
                        if not movieId_tmdbId_mapping.empty:
                            # Filter mapping based on provided movieIds
                            filtered_mapping = movieId_tmdbId_mapping[movieId_tmdbId_mapping["movieId"].isin(movieIds)]
                            movie_to_tmdb_map = filtered_mapping.set_index("movieId")["tmdbId"].to_dict()
                            tmdb_to_movie_map = filtered_mapping.set_index("tmdbId")["movieId"].to_dict()
                    
                    elif isinstance(movieId_tmdbId_mapping, dict):
                        if movieId_tmdbId_mapping:
                            # Check the structure of the dictionary
                            # If it's {tmdbId: movieId}, we need to invert it
                            if all(isinstance(key, (int, str)) for key in movieId_tmdbId_mapping.keys()):
                                # Create inverse map assuming dict is {tmdbId: movieId}
                                tmdb_to_movie_map = movieId_tmdbId_mapping
                                # Filter and create the inverse mapping
                                movie_to_tmdb_map = {v: k for k, v in tmdb_to_movie_map.items() if v in movieIds}
                
                # Create user-item ratings tuples
                user_item_ratings = [(int(item.movieId), item.rating) for item in recommendation_request.items]
            
            # Step 3: Log the processed items
            logger.info(f"Processed user-item ratings: {user_item_ratings}")
            
            if not user_item_ratings:
                raise HTTPException(
                    status_code=400,
                    detail="No valid user item ratings provided"
                )
            
            # Step 4: Load model components
            components = BaseRecommender.load_model_components(directory_path)
            
            # Step 5: Generate item-based recommendations
            like_threshold = 3
            liked_items = [(movie_id, rating) for movie_id, rating in user_item_ratings if rating is None or rating >= like_threshold]
            
            item_recommender = ItemRecommender(
                faiss_index=components["faiss_item_index"],
                embedding_matrix=components["item_matrix"],
                user_item_mappings=components["user_item_mappings"],
                similarity_metric=components["model_info"].similarity_metric,
                min_similarity=0.1
            )
            
            item_based_raw_recommendations = item_recommender.generate_recommendations(
                item_ids=liked_items,
                n_recommendations=recommendation_request.n_recommendations
            )
            
            logger.info(f"Generated {len(item_based_raw_recommendations)} item-based recommendations")
            
            # Format item-based recommendations according to schema
            item_based_recommendations = []
            for rec in item_based_raw_recommendations:
                tmdb_id = rec.get("tmdb_id")
                
                # Determine which ID to include based on request source
                movieId = None
                if recommendation_request.req_source == "movieId":
                    # For movieId requests, include both movieId and tmdbId
                    movieId = tmdb_to_movie_map.get(tmdb_id)
                
                item_based_recommendations.append(Recommendation(
                    movie_source=recommendation_request.req_source,
                    movieId=movieId,
                    tmdb_id=tmdb_id,
                    similarity=rec.get("similarity"),
                    predicted_rating=rec.get("predicted_rating")
                ))
            
            # Create recommendations dictionary for the response
            recommendations_dict = {
                "item_based": RecommendationCategory(
                    recommendation_type="item_based",
                    recommendations=item_based_recommendations
                )
            }
            
            # Step 6: Try to generate user-based recommendations
            try:
                user_recommender = UserRecommender(
                    faiss_user_index=components["faiss_user_index"],
                    user_embedding_matrix=components["user_matrix"],
                    faiss_item_index=components["faiss_item_index"],
                    item_embedding_matrix=components["item_matrix"],
                    user_item_matrix=components["user_item_matrix"],
                    item_mapping=components["item_mapping"],
                    item_reverse_mapping=components["item_reverse_mapping"],
                    svd_user_model=components["svd_user_model"],
                    similarity_metric=components["model_info"].similarity_metric,
                    min_similarity=0.1,
                    n_neighbors=50
                )
                
                user_based_raw_recommendations = user_recommender.generate_recommendations(
                    user_ratings=user_item_ratings,
                    n_recommendations=recommendation_request.n_recommendations
                )
                
                logger.info(f"Generated {len(user_based_raw_recommendations)} user-based recommendations")
                
                # Format user-based recommendations according to schema
                user_based_recommendations = []
                for rec in user_based_raw_recommendations:
                    tmdb_id = rec.get("tmdb_id")
                    
                    # Determine which ID to include based on request source
                    movieId = None
                    if recommendation_request.req_source == "movieId":
                        # For movieId requests, include both movieId and tmdbId
                        movieId = tmdb_to_movie_map.get(tmdb_id)
                    
                    user_based_recommendations.append(Recommendation(
                        movie_source=recommendation_request.req_source,
                        movieId=movieId,
                        tmdb_id=tmdb_id,
                        similarity=rec.get("similarity"),
                        predicted_rating=rec.get("predicted_rating")
                    ))
                
                # Add user-based recommendations to the dictionary
                recommendations_dict["user_based"] = RecommendationCategory(
                    recommendation_type="user_based",
                    recommendations=user_based_recommendations
                )
                
            except Exception as e:
                logger.error(f"Error in generating user-based recommendations: {str(e)}")
                # Continue with item-based recommendations only
            
            # Step 7: Try to generate content-based recommendations (if applicable)
            # This is a placeholder for content-based recommendation logic
            # You would implement your content-based recommender here
            
            # Return the response with all available recommendation types
            return RecommendationResponse(
                status="success",
                recommendations=recommendations_dict
            )
            
        except Exception as e:
            logger.error(f"Error in generating recommendations: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")
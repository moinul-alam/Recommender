import numpy as np
import faiss
from typing import Dict, List, Tuple
import logging
from scipy import sparse
from src.models.common.logger import app_logger

logger = app_logger(__name__)

class UserRecommender:
    """Generates recommendations based on user similarity (collaborative filtering)."""
    
    def __init__(
        self,
        faiss_user_index: faiss.Index,
        user_embedding_matrix: np.ndarray,
        faiss_item_index: faiss.Index,
        item_embedding_matrix: np.ndarray,
        user_item_matrix: sparse.csr_matrix,
        user_item_mappings: dict,
        svd_user_model: dict,
        similarity_metric: str = 'cosine',
        min_similarity: float = 0.1,
        n_neighbors: int = 50,
        tmdb_to_movie_map: Dict = None,
        movie_to_tmdb_map: Dict = None,
        req_source: str = "movieId"
    ):
        self.faiss_user_index = faiss_user_index
        self.user_embedding_matrix = user_embedding_matrix
        self.faiss_item_index = faiss_item_index
        self.item_embedding_matrix = item_embedding_matrix
        self.user_item_matrix = user_item_matrix
        self.user_item_mappings = user_item_mappings
        self.svd_user_model = svd_user_model
        self.similarity_metric = similarity_metric
        self.min_similarity = min_similarity
        self.n_neighbors = n_neighbors
        self.tmdb_to_movie_map = tmdb_to_movie_map
        self.movie_to_tmdb_map = movie_to_tmdb_map
        self.req_source = req_source
        
            
        # Get both item and user mappings
        self.item_mapping = self.user_item_mappings.get("item_mapping", {})
        self.item_reverse_mapping = self.user_item_mappings.get("item_reverse_mapping", {})
        self.user_mapping = self.user_item_mappings.get("user_mapping", {})
        self.user_reverse_mapping = self.user_item_mappings.get("user_reverse_mapping", {})
        
        # Validate that necessary mappings exist
        if not self.item_mapping or not self.item_reverse_mapping:
            logger.error("Item mappings are missing or empty")
            raise ValueError("Item mappings are required but not provided")


    def convert_to_similarity(self, inner_product: float) -> float:
        """
        Convert FAISS inner product to interpretable similarity based on the metric.
        For cosine: With L2 normalized vectors, inner product equals cosine similarity (range 0-1)
        For inner_product: Return as-is (or scale empirically if you know bounds).
        """
        if self.similarity_metric == 'cosine':
            return max(0.0, min(1.0, inner_product))
        else:
            return inner_product

    def _filter_and_map_items(self, user_ratings: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Maps movieIds to internal indices and filters out invalid ones."""
        return [
            (self.item_mapping.get(movieId), rating)
            for movieId, rating in user_ratings
            if self.item_mapping.get(movieId) is not None
        ]

    def generate_recommendations(
        self, 
        user_ratings: List[Tuple[int, float]], 
        n_recommendations: int
    ) -> List[Dict]:
        """Generates movie recommendations for a user based on similar users."""
        try:
            # Map movieIds to internal indices and filter invalid ones
            valid_indices = self._filter_and_map_items(user_ratings)
            
            if not valid_indices:
                logger.warning("No valid movie indices found after mapping")
                return []

            # Create a sparse vector for the active user
            n_items = self.user_item_matrix.shape[1]
            active_user_vector = np.zeros(n_items)
            for idx, rating in valid_indices:
                active_user_vector[idx] = rating

            # Calculate the user's mean rating
            rated_indices = [idx for idx, _ in valid_indices]
            if not rated_indices:
                return []  # Return empty list if no valid ratings

            ratings = [rating for _, rating in valid_indices]
            user_mean = np.mean(ratings) if ratings else 0
            
            logger.info(f"User mean rating: {user_mean}")

            # Apply mean normalization (consistent with preprocessing)
            normalized_user_vector = active_user_vector.copy()
            for idx in rated_indices:
                normalized_user_vector[idx] = active_user_vector[idx] - user_mean

            # Create a sparse matrix from the normalized vector
            active_user_sparse = sparse.csr_matrix(normalized_user_vector.reshape(1, -1))
            
            # Ensure SVD user model exists before transformation
            if self.svd_user_model is None:
                logger.error("SVD user model is None, cannot transform user vector")
                raise ValueError("SVD user model is None")
                
            # Transform to user embedding space using SVD model
            try:
                user_vector = self.svd_user_model.transform(active_user_sparse)
            except Exception as e:
                logger.error(f"Error transforming user vector with SVD: {str(e)}")
                raise
            
            user_vector = user_vector.reshape(1, -1)  # Reshape to 2D for FAISS
            
            user_vector = user_vector.astype(np.float32) 
            
            # Find similar users
            search_size = min(self.n_neighbors, self.user_embedding_matrix.shape[0])
            D, I = self.faiss_user_index.search(user_vector, search_size)

            # Create a set of items already rated by the active user
            rated_items = set(idx for idx, _ in valid_indices)

            # Collect recommendations from similar users
            item_scores = {}
            item_similarity_sums = {}
            neighbor_count = 0

            for i, user_idx in enumerate(I[0]):
                if user_idx < 0:  # Skip invalid indices
                    continue

                # Convert distance to similarity score (now in 0-1 range)
                user_similarity = self.convert_to_similarity(D[0][i])
                
                logger.debug(f"User {user_idx} similarity: {user_similarity}")

                # Only consider users above the similarity threshold
                if user_similarity < self.min_similarity:
                    continue

                neighbor_count += 1

                # Get items rated by this similar user
                user_ratings = self.user_item_matrix[user_idx].toarray().flatten()

                for item_idx, norm_rating in enumerate(user_ratings):
                    # Skip zero ratings and items the active user has already rated
                    if norm_rating == 0 or item_idx in rated_items:
                        continue

                    # Initialize scores dictionary for this item if needed
                    if item_idx not in item_scores:
                        item_scores[item_idx] = 0
                        item_similarity_sums[item_idx] = 0

                    # Add this user's contribution weighted by similarity
                    # The similarity is now properly in 0-1 range
                    item_scores[item_idx] += norm_rating * user_similarity
                    item_similarity_sums[item_idx] += user_similarity

            # If no neighbors above the threshold, return empty list
            if neighbor_count == 0:
                logger.warning("No neighbors found above similarity threshold")
                return []

            # Calculate final scores and convert to recommendations
            recommendations = []
            for item_idx, score in item_scores.items():
                if item_similarity_sums[item_idx] > 0:
                    # Normalize by sum of similarities
                    normalized_score = score / item_similarity_sums[item_idx]
                    
                    # Add back user's mean to get back to original rating scale (1-5 for MovieLens)
                    predicted_rating = normalized_score + user_mean
                    
                    # Add bounds check to prevent extreme values
                    predicted_rating = max(1.0, min(5.0, predicted_rating))

                    # Get the original TMDB ID using item_reverse_mapping
                    tmdbId = self.item_reverse_mapping.get(item_idx)

                    if tmdbId is not None:
                        # Create recommendation with IDs based on request source
                        recommendation = {
                            "similarity": float(item_similarity_sums[item_idx]),
                            "predicted_rating": float(round(predicted_rating, 2))
                        }

                        # Add the appropriate ID based on request source
                        if self.req_source == "tmdb":
                            recommendation["tmdbId"] = int(tmdbId)
                        else:  # movieId source
                            if str(tmdbId) in self.tmdb_to_movie_map:
                                recommendation["movieId"] = int(self.tmdb_to_movie_map[str(tmdbId)])
                            else:
                                # Skip items that can't be mapped
                                continue
                        
                        recommendations.append(recommendation)

            # Sort by similarity and return top N
            recommendations.sort(key=lambda x: x["similarity"], reverse=True)
            return recommendations[:n_recommendations]

        except Exception as e:
            logger.error(f"Error in generating user recommendations: {str(e)}")
            raise RuntimeError(f"Error in generating user recommendations: {str(e)}")
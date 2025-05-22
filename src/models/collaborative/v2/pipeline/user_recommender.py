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
        user_item_matrix: sparse.csr_matrix,  # Already mean-centered matrix
        user_item_mappings: dict,
        svd_user_model: dict,
        user_item_means: dict,  # Contains 'user_means' and 'item_means' arrays
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
        self.user_item_matrix = user_item_matrix  # This is already mean-centered
        self.user_item_mappings = user_item_mappings
        self.svd_user_model = svd_user_model
        self.user_item_means = user_item_means
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
        
        # Extract original means from the provided data
        self.user_means = user_item_means.get('user_means', np.array([]))
        self.item_means = user_item_means.get('item_means', np.array([]))
        self.global_mean = np.mean(self.user_means) if len(self.user_means) > 0 else 3.5
        
        logger.info(f"Loaded original means - Global: {self.global_mean:.2f}, "
                   f"Users: {len(self.user_means)}, Items: {len(self.item_means)}")

    def convert_to_similarity(self, inner_product: float) -> float:
        """Convert FAISS inner product to interpretable similarity based on the metric."""
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

    def _predict_rating_for_item(
        self, 
        item_idx: int, 
        similar_users: List[Tuple[int, float]], 
        active_user_mean: float
    ) -> Tuple[float, float]:
        """
        Predict rating for a specific item using user-based CF.
        Returns: (predicted_rating, confidence_score)
        """
        try:
            # Get item mean
            item_mean = (self.item_means[item_idx] 
                        if item_idx < len(self.item_means) 
                        else self.global_mean)
            
            # Collect ratings from similar users who have rated this item
            weighted_sum = 0.0
            similarity_sum = 0.0
            neighbor_count = 0
            
            for user_idx, similarity in similar_users:
                # Get this user's rating for the item (from mean-centered matrix)
                centered_rating = self.user_item_matrix[user_idx, item_idx]
                
                if centered_rating != 0:  # User has rated this item
                    # Get the original user mean for proper denormalization
                    neighbor_user_mean = (self.user_means[user_idx] 
                                         if user_idx < len(self.user_means) 
                                         else self.global_mean)
                    
                    # Convert centered rating back to original rating
                    original_rating = centered_rating + neighbor_user_mean
                    
                    # Calculate user's deviation from their own mean
                    user_deviation = original_rating - neighbor_user_mean
                    
                    # Weight by similarity
                    weighted_sum += similarity * user_deviation
                    similarity_sum += abs(similarity)
                    neighbor_count += 1
            
            if neighbor_count == 0 or similarity_sum == 0:
                # No similar users have rated this item - fallback to baseline
                predicted_rating = active_user_mean + (item_mean - self.global_mean)
                confidence = 0.0
            else:
                # Calculate weighted average deviation
                avg_deviation = weighted_sum / similarity_sum
                
                # User-based CF prediction: user_mean + weighted_avg_deviation
                predicted_rating = active_user_mean + avg_deviation
                
                # Calculate confidence based on number of neighbors and similarity strength
                avg_similarity = similarity_sum / neighbor_count
                neighbor_factor = min(neighbor_count / 5.0, 1.0)  # More neighbors = higher confidence
                confidence = avg_similarity * neighbor_factor
            
            # Apply confidence-based regression for low-confidence predictions
            if confidence < 0.3:
                # Regress towards a baseline prediction
                baseline_prediction = active_user_mean + (item_mean - self.global_mean)
                regression_factor = 1.0 - confidence
                predicted_rating = (predicted_rating * confidence + 
                                  baseline_prediction * regression_factor)
            
            # Ensure rating is within valid bounds
            predicted_rating = max(1.0, min(5.0, predicted_rating))
            
            return predicted_rating, confidence
            
        except Exception as e:
            logger.error(f"Error predicting rating for item {item_idx}: {str(e)}")
            # Fallback to baseline
            baseline = active_user_mean + (item_mean - self.global_mean) if item_mean else active_user_mean
            return max(1.0, min(5.0, baseline)), 0.0

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

            # Calculate the user's mean rating
            ratings = [rating for _, rating in valid_indices]
            active_user_mean = np.mean(ratings) if ratings else self.global_mean
            
            logger.info(f"Active user mean rating: {active_user_mean:.2f}")

            # Create mean-centered vector for the active user
            n_items = self.user_item_matrix.shape[1]
            active_user_vector = np.zeros(n_items)
            
            # Fill with mean-centered ratings
            for idx, rating in valid_indices:
                active_user_vector[idx] = rating - active_user_mean

            # Create a sparse matrix from the mean-centered vector
            active_user_sparse = sparse.csr_matrix(active_user_vector.reshape(1, -1))
            
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
            
            user_vector = user_vector.reshape(1, -1).astype(np.float32)
            
            # Find similar users
            search_size = min(self.n_neighbors * 2, self.user_embedding_matrix.shape[0])  # Cast wider net
            D, I = self.faiss_user_index.search(user_vector, search_size)

            # Filter similar users by similarity threshold
            similar_users = []
            for i, user_idx in enumerate(I[0]):
                if user_idx < 0:  # Skip invalid indices
                    continue

                user_similarity = self.convert_to_similarity(D[0][i])
                
                if user_similarity >= self.min_similarity:
                    similar_users.append((user_idx, user_similarity))

            if not similar_users:
                logger.warning("No neighbors found above similarity threshold")
                return []

            logger.info(f"Found {len(similar_users)} similar users above threshold")

            # Create a set of items already rated by the active user
            rated_items = set(idx for idx, _ in valid_indices)

            # Get all items that similar users have rated (excluding already rated ones)
            candidate_items = set()
            for user_idx, _ in similar_users:
                user_ratings_row = self.user_item_matrix[user_idx].toarray().flatten()
                for item_idx, centered_rating in enumerate(user_ratings_row):
                    if centered_rating != 0 and item_idx not in rated_items:
                        candidate_items.add(item_idx)

            logger.info(f"Found {len(candidate_items)} candidate items")

            # Generate predictions for all candidate items
            recommendations = []
            for item_idx in candidate_items:
                predicted_rating, confidence = self._predict_rating_for_item(
                    item_idx=item_idx,
                    similar_users=similar_users,
                    active_user_mean=active_user_mean
                )

                # Get the original TMDB ID using item_reverse_mapping
                tmdbId = self.item_reverse_mapping.get(item_idx)

                if tmdbId is not None:
                    # Create recommendation with IDs based on request source
                    recommendation = {
                        "similarity": float(round(confidence, 3)),  # Use confidence as similarity measure
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

            # Sort by predicted rating (primary) and confidence (secondary)
            recommendations.sort(key=lambda x: (x["predicted_rating"], x["similarity"]), reverse=True)
            
            logger.info(f"Generated {len(recommendations)} recommendations, returning top {n_recommendations}")
            
            return recommendations[:n_recommendations]

        except Exception as e:
            logger.error(f"Error in generating user recommendations: {str(e)}")
            raise RuntimeError(f"Error in generating user recommendations: {str(e)}")
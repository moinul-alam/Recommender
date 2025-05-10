import numpy as np
import faiss
from typing import Dict, List, Optional, Tuple
import logging
from scipy import sparse
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

class BaseRecommender:
    """Base class for recommenders to share common methods."""

    def __init__(
        self,
        faiss_index: faiss.Index,
        embedding_matrix: np.ndarray,
        item_mapping: Dict[int, int],
        item_reverse_mapping: Dict[int, int],
        min_similarity: float
    ):
        self.faiss_index = faiss_index
        self.embedding_matrix = embedding_matrix
        self.item_mapping = item_mapping
        self.item_reverse_mapping = item_reverse_mapping
        self.min_similarity = min_similarity

    def _filter_and_map_items(self, items: Dict[int, float]) -> List[Tuple[int, float]]:
        """Maps TMDB IDs to internal indices and filters out invalid ones."""
        return [
            (self.item_mapping.get(tmdb_id), rating)
            for tmdb_id, rating in items.items()
            if self.item_mapping.get(tmdb_id) is not None
        ]

    def _convert_distance_to_similarity(self, inner_product: float) -> float:
        """Converts FAISS inner product to a similarity percentage."""
        return max(0, min(100, (inner_product + 1) * 50))

class UserRecommender(BaseRecommender):
    """Generates recommendations based on user similarity (collaborative filtering)."""
    
    def __init__(
        self,
        faiss_user_index: faiss.Index,
        user_embedding_matrix: np.ndarray,
        faiss_item_index: faiss.Index,
        item_embedding_matrix: np.ndarray,
        user_item_matrix: sparse.csr_matrix,
        item_mapping: Dict[int, int],
        item_reverse_mapping: Dict[int, int],
        svd_user_model,
        min_similarity: float,
        n_neighbors: int = 50,
        use_pearson: bool = True
    ):
        super().__init__(
            faiss_index=faiss_item_index,
            embedding_matrix=item_embedding_matrix,
            item_mapping=item_mapping,
            item_reverse_mapping=item_reverse_mapping,
            min_similarity=min_similarity
        )
        self.faiss_user_index = faiss_user_index
        self.user_embedding_matrix = user_embedding_matrix
        self.user_item_matrix = user_item_matrix
        self.svd_user_model = svd_user_model
        self.n_neighbors = n_neighbors
        self.use_pearson = use_pearson
        
        # Calculate global mean and item biases during initialization
        self._calculate_global_stats()
        
    def _calculate_global_stats(self):
        """Calculate global rating mean and item biases."""
        # Convert sparse matrix to array for calculations
        matrix_array = self.user_item_matrix.toarray()
        
        # Calculate global mean of all non-zero ratings
        non_zero_mask = matrix_array != 0
        if np.any(non_zero_mask):
            self.global_mean = np.sum(matrix_array[non_zero_mask]) / np.sum(non_zero_mask)
        else:
            self.global_mean = 0
            
        # Calculate item biases (difference between item mean and global mean)
        self.item_biases = np.zeros(matrix_array.shape[1])
        
        for item_idx in range(matrix_array.shape[1]):
            item_ratings = matrix_array[:, item_idx]
            non_zero_ratings = item_ratings[item_ratings != 0]
            
            if len(non_zero_ratings) > 0:
                item_mean = np.mean(non_zero_ratings)
                self.item_biases[item_idx] = item_mean - self.global_mean
    
    def _calculate_pearson_similarity(self, active_user_vector, user_idx):
        """Calculate Pearson correlation between active user and another user."""
        # Get other user's ratings
        other_user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        
        # Find items rated by both users
        common_items = []
        common_ratings_active = []
        common_ratings_other = []
        
        for i, (active_rating, other_rating) in enumerate(zip(active_user_vector, other_user_ratings)):
            if active_rating > 0 and other_rating > 0:  # Both users have rated this item
                common_items.append(i)
                common_ratings_active.append(active_rating)
                common_ratings_other.append(other_rating)
        
        # If fewer than 3 items in common, return 0 similarity
        if len(common_items) < 3:
            return 0
            
        # Calculate Pearson correlation
        try:
            correlation, _ = pearsonr(common_ratings_active, common_ratings_other)
            # Convert to 0-100 scale and handle NaN
            if np.isnan(correlation):
                return 0
            else:
                # Map from [-1, 1] to [0, 100]
                return max(0, min(100, (correlation + 1) * 50))
        except:
            return 0  # Return 0 similarity if calculation fails

    def generate_recommendations(self, items: Dict[int, float], n_recommendations: int) -> List[Dict]:
        """Generates movie recommendations for a user based on similar users."""
        try:
            # Map TMDB IDs to internal indices and filter invalid ones
            valid_indices = self._filter_and_map_items(items)
            if not valid_indices:
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

            # Apply dual normalization (subtract user mean AND item bias)
            normalized_user_vector = active_user_vector.copy()
            for idx in rated_indices:
                # Normalize using both user mean and item bias
                normalized_user_vector[idx] = active_user_vector[idx] - user_mean - self.item_biases[idx]

            # Create a sparse matrix from the normalized vector
            active_user_sparse = sparse.csr_matrix(normalized_user_vector.reshape(1, -1))

            # Transform to user embedding space using SVD model
            user_vector = self.svd_user_model.transform(active_user_sparse)

            # Find similar users
            search_size = min(self.n_neighbors * 2, self.user_embedding_matrix.shape[0])
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

                # Calculate similarity score
                if self.use_pearson:
                    # Use Pearson correlation for similarity calculation
                    user_similarity = self._calculate_pearson_similarity(active_user_vector, user_idx)
                else:
                    # Use FAISS distance converted to similarity
                    user_similarity = self._convert_distance_to_similarity(D[0][i])

                # Only consider users above the similarity threshold
                if user_similarity < self.min_similarity:
                    continue

                neighbor_count += 1
                if neighbor_count > self.n_neighbors:
                    break  # Stop after we've found enough valid neighbors

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

                    # We need to normalize this rating by subtracting user mean and item bias
                    # (assuming user_ratings still has user means)
                    adjusted_rating = norm_rating - self.item_biases[item_idx]
                    
                    # Add this user's contribution weighted by similarity
                    item_scores[item_idx] += adjusted_rating * user_similarity
                    item_similarity_sums[item_idx] += user_similarity

            # If no neighbors above the threshold, return empty list
            if neighbor_count == 0:
                return []

            # Calculate final scores and convert to recommendations
            recommendations = []
            for item_idx, score in item_scores.items():
                if item_similarity_sums[item_idx] > 0:
                    # Normalize by sum of similarities
                    normalized_score = score / item_similarity_sums[item_idx]

                    # Add back user's mean AND item bias to get back to original rating scale
                    predicted_rating = normalized_score + user_mean + self.item_biases[item_idx]

                    # Get the original TMDB ID
                    tmdb_id = self.item_reverse_mapping.get(item_idx)

                    if tmdb_id is not None:
                        recommendations.append({
                            "tmdb_id": int(tmdb_id),
                            "similarity": float(item_similarity_sums[item_idx]),
                            "predicted_rating": float(round(predicted_rating, 2)),
                            "confidence": min(100, float(item_similarity_sums[item_idx] / 10))  # Simple confidence metric
                        })

            # Sort by predicted rating and return top N
            recommendations.sort(key=lambda x: x["predicted_rating"], reverse=True)
            return recommendations[:n_recommendations]

        except Exception as e:
            logger.error(f"Error in generating user recommendations: {str(e)}")
            raise RuntimeError(f"Error in generating user recommendations: {str(e)}")

class ItemRecommender(BaseRecommender):
    """Generates recommendations based on item similarity (content-based filtering)."""

    def generate_recommendations(self, item_ids: List[int], n_recommendations: int) -> List[Dict]:
        """Finds similar items for multiple movies (content-based filtering)."""
        try:
            # Map TMDB IDs to internal indices
            item_indices = [self.item_mapping.get(tmdb_id) for tmdb_id in item_ids if tmdb_id in self.item_mapping]
            if not item_indices:
                return []

            # Aggregate item vectors
            item_vectors = self.embedding_matrix[item_indices]
            query_vector = np.mean(item_vectors, axis=0)

            # Search similar items
            search_size = min(2 * n_recommendations, self.embedding_matrix.shape[0])
            D, I = self.faiss_index.search(query_vector.reshape(1, -1), search_size)

            # Filter out query items and items below similarity threshold
            query_items_set = set(item_indices)
            recommendations = []

            for idx, dist in zip(I[0], D[0]):
                if idx >= 0 and idx not in query_items_set:
                    similar_tmdb_id = self.item_reverse_mapping.get(idx)
                    if similar_tmdb_id is None:
                        continue
                        
                    similarity_score = self._convert_distance_to_similarity(dist)
                    
                    # Skip items below similarity threshold
                    if similarity_score < self.min_similarity:
                        continue

                    recommendations.append({
                        "tmdb_id": int(similar_tmdb_id),
                        "similarity": float(similarity_score)
                    })

                    if len(recommendations) == n_recommendations:
                        break

            return sorted(recommendations, key=lambda x: x["similarity"], reverse=True)

        except Exception as e:
            logger.error(f"Error in generating item recommendations: {str(e)}")
            raise RuntimeError(f"Error in generating item recommendations: {str(e)}")
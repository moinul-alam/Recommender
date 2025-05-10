import numpy as np
import faiss
from typing import Dict, List, Tuple
import logging
from scipy import sparse

logger = logging.getLogger(__name__)


class BaseRecommender:
    """Base class for recommenders to share common methods."""

    def __init__(
        self,
        faiss_index: faiss.Index,
        embedding_matrix: np.ndarray,
        item_mapping: Dict[int, int],
        item_reverse_mapping: Dict[int, int],
        similarity_metric: str,
        min_similarity: float
    ):
        self.faiss_index = faiss_index
        self.embedding_matrix = embedding_matrix
        self.item_mapping = item_mapping
        self.item_reverse_mapping = item_reverse_mapping
        self.similarity_metric = similarity_metric
        self.min_similarity = min_similarity

    def _filter_and_map_items(self, items: Dict[int, float]) -> List[Tuple[int, float]]:
        """Maps TMDB IDs to internal indices and filters out invalid ones."""
        return [
            (self.item_mapping.get(tmdb_id), rating)
            for tmdb_id, rating in items.items()
            if self.item_mapping.get(tmdb_id) is not None
        ]

    def convert_to_similarity(self, inner_product: float) -> float:
        """
        Convert FAISS inner product to interpretable similarity based on the metric.
        For cosine: With L2 normalized vectors, inner product equals cosine similarity (range 0-1)
        For inner_product: Return as-is (or scale empirically if you know bounds).
        """
        if self.similarity_metric == 'cosine':
            # For L2 normalized vectors, inner product is already cosine similarity
            # Just ensure it's in the [0,1] range (might be slightly outside due to floating point)
            return max(0.0, min(1.0, inner_product))
        else:
            return inner_product


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
        similarity_metric: str,
        min_similarity: float,
        n_neighbors: int = 50
    ):
        super().__init__(
            faiss_index=faiss_item_index,
            embedding_matrix=item_embedding_matrix,
            item_mapping=item_mapping,
            item_reverse_mapping=item_reverse_mapping,
            similarity_metric=similarity_metric,
            min_similarity=min_similarity
        )
        self.faiss_user_index = faiss_user_index
        self.user_embedding_matrix = user_embedding_matrix
        self.user_item_matrix = user_item_matrix
        self.svd_user_model = svd_user_model
        self.n_neighbors = n_neighbors

    def generate_recommendations(self, user_ratings: Dict[int, float], n_recommendations: int) -> List[Dict]:
        """Generates movie recommendations for a user based on similar users."""
        try:
            # Map TMDB IDs to internal indices and filter invalid ones
            valid_indices = self._filter_and_map_items(user_ratings)
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
            
            logger.info(f"User mean rating: {user_mean}")

            # Apply mean normalization (consistent with preprocessing)
            normalized_user_vector = active_user_vector.copy()
            for idx in rated_indices:
                normalized_user_vector[idx] = active_user_vector[idx] - user_mean

            # Create a sparse matrix from the normalized vector
            active_user_sparse = sparse.csr_matrix(normalized_user_vector.reshape(1, -1))
            
            # Transform to user embedding space using SVD model
            user_vector = self.svd_user_model.transform(active_user_sparse)
            
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
                
                logger.info(f"User {user_idx} similarity: {user_similarity}")

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

                    # Get the original TMDB ID
                    tmdb_id = self.item_reverse_mapping.get(item_idx)

                    if tmdb_id is not None:
                        recommendations.append({
                            "tmdb_id": int(tmdb_id),
                            "similarity": float(item_similarity_sums[item_idx]),
                            "predicted_rating": float(round(predicted_rating, 2))
                        })

            # Sort by similarity and return top N
            recommendations.sort(key=lambda x: x["similarity"], reverse=True)
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
            
            # Normalize the query vector if using cosine similarity
            if self.similarity_metric == 'cosine':
                norm = np.linalg.norm(query_vector)
                if norm > 0:
                    query_vector = query_vector / norm
            
            # Convert to proper format for FAISS
            query_vector = query_vector.reshape(1, -1).astype(np.float32)

            # Search similar items
            search_size = min(2 * n_recommendations, self.embedding_matrix.shape[0])
            D, I = self.faiss_index.search(query_vector, search_size)

            # Filter out query items and items below similarity threshold
            query_items_set = set(item_indices)
            recommendations = []

            for idx, dist in zip(I[0], D[0]):
                if idx >= 0 and idx not in query_items_set:
                    similar_tmdb_id = self.item_reverse_mapping.get(idx)
                    if similar_tmdb_id is None:
                        continue
                        
                    # Convert to similarity in 0-1 range
                    similarity_score = self.convert_to_similarity(dist)
                    
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
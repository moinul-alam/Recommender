import numpy as np
import faiss
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class ItemRecommender:
    """Generates recommendations based on item similarity (content-based filtering)."""

    def __init__(
        self,
        faiss_index: faiss.Index,
        embedding_matrix: np.ndarray,
        user_item_mappings: dict,
        similarity_metric: str = 'cosine',
        min_similarity: float = 0.1
    ):
        self.faiss_index = faiss_index
        self.embedding_matrix = embedding_matrix
        self.user_item_mappings = user_item_mappings
        self.similarity_metric = similarity_metric
        self.min_similarity = min_similarity
        
        # Extract item mappings
        self.item_mapping = self.user_item_mappings.get("item_mapping", {})
        self.item_reverse_mapping = self.user_item_mappings.get("item_reverse_mapping", {})

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
        
    def generate_recommendations(
        self,
        item_ids: List[Tuple[int, float]],
        n_recommendations: int
    ) -> List[Dict]:
        """Finds similar items for multiple movies (content-based filtering)."""
        try:
            item_ids = [int(movie_id) for movie_id, _ in item_ids]
            
            if not item_ids:
                return []
            
            # Map movie IDs to internal indices
            item_indices = [self.item_mapping.get(movie_id) for movie_id in item_ids if movie_id in self.item_mapping]
            
            if not item_indices:
                logger.warning(f"None of the provided movie IDs {item_ids} could be mapped to internal indices")
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
            distances, indices = self.faiss_index.search(query_vector, search_size)

            # Filter out query items and items below similarity threshold
            query_items_set = set(item_indices)
            recommendations = []

            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0 and idx not in query_items_set:
                    # Convert internal index back to tmdb_id
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
                        "similarity": float(similarity_score),
                        "predicted_rating": None  # Item-based doesn't predict rating
                    })

                    if len(recommendations) == n_recommendations:
                        break

            return sorted(recommendations, key=lambda x: x["similarity"], reverse=True)

        except Exception as e:
            logger.error(f"Error in generating item recommendations: {str(e)}")
            raise RuntimeError(f"Error in generating item recommendations: {str(e)}")
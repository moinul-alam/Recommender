import numpy as np
import faiss
from typing import Dict, List, Tuple
import logging
from scipy import sparse
from src.models.common.logger import app_logger

logger = app_logger(__name__)


class BaseRecommender:
    """Base class for recommenders to share common methods."""

    @staticmethod
    def load_model_components(directory_path: str) -> Dict:
        """Loads all necessary model components from disk."""
        import pathlib
        import pickle
        from src.models.common.file_config import file_names
        
        try:
            base_path = pathlib.Path(directory_path)
            components = {}
            
            # Load model info
            model_info_path = base_path / file_names["model_info"]
            with open(model_info_path, 'rb') as f:
                components["model_info"] = pickle.load(f)
                
            # Load FAISS indices
            faiss_item_path = base_path / file_names["faiss_item_index"]
            components["faiss_item_index"] = faiss.read_index(str(faiss_item_path))
            
            faiss_user_path = base_path / file_names["faiss_user_index"]
            components["faiss_user_index"] = faiss.read_index(str(faiss_user_path))
            
            # Load embedding matrices
            item_matrix_path = base_path / file_names["item_matrix"]
            with open(item_matrix_path, 'rb') as f:
                components["item_matrix"] = pickle.load(f)
                
            user_matrix_path = base_path / file_names["user_matrix"]
            with open(user_matrix_path, 'rb') as f:
                components["user_matrix"] = pickle.load(f)
                
            # Load user-item matrix
            user_item_matrix_path = base_path / file_names["user_item_matrix"]
            with open(user_item_matrix_path, 'rb') as f:
                components["user_item_matrix"] = pickle.load(f)
                
            # Load SVD user model
            svd_user_model_path = base_path / file_names["svd_user_model"]
            with open(svd_user_model_path, 'rb') as f:
                components["svd_user_model"] = pickle.load(f)
                
            # Load mappings
            mappings_path = base_path / file_names["user_item_mappings"]
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
                components["user_item_mappings"] = mappings
                components["item_mapping"] = mappings.get("item_mapping", {})
                components["item_reverse_mapping"] = mappings.get("item_reverse_mapping", {})
                components["user_mapping"] = mappings.get("user_mapping", {})
                components["user_reverse_mapping"] = mappings.get("user_reverse_mapping", {})
            
            return components
            
        except Exception as e:
            logger.error(f"Error loading model components: {str(e)}")
            raise RuntimeError(f"Error loading model components: {str(e)}")

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


class UserRecommender:
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
        similarity_metric: str = 'cosine',
        min_similarity: float = 0.1,
        n_neighbors: int = 50
    ):
        self.faiss_user_index = faiss_user_index
        self.user_embedding_matrix = user_embedding_matrix
        self.faiss_item_index = faiss_item_index
        self.item_embedding_matrix = item_embedding_matrix
        self.user_item_matrix = user_item_matrix
        self.item_mapping = item_mapping
        self.item_reverse_mapping = item_reverse_mapping
        self.svd_user_model = svd_user_model
        self.similarity_metric = similarity_metric
        self.min_similarity = min_similarity
        self.n_neighbors = n_neighbors

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
            (self.item_mapping.get(movie_id), rating)
            for movie_id, rating in user_ratings
            if self.item_mapping.get(movie_id) is not None
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
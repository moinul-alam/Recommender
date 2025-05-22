import numpy as np
import faiss
from typing import Dict, List, Tuple
import logging
from scipy import sparse

logger = logging.getLogger(__name__)

class ItemRecommender:
    """Generates recommendations based on item similarity (content-based filtering)."""

    def __init__(
        self,
        faiss_index: faiss.Index,
        embedding_matrix: np.ndarray,
        user_item_matrix: sparse.csr_matrix,  # Already mean-centered matrix
        user_item_mappings: dict,
        user_item_means: dict,  # Contains 'user_means' and 'item_means' arrays
        similarity_metric: str = 'cosine',
        min_similarity: float = 0.1,
        tmdb_to_movie_map: Dict = None,
        movie_to_tmdb_map: Dict = None,
        req_source: str = "movieId"
    ):
        self.faiss_index = faiss_index
        self.embedding_matrix = embedding_matrix
        self.user_item_matrix = user_item_matrix  # This is already mean-centered
        self.user_item_mappings = user_item_mappings
        self.user_item_means = user_item_means
        self.similarity_metric = similarity_metric
        self.min_similarity = min_similarity
        self.tmdb_to_movie_map = tmdb_to_movie_map
        self.movie_to_tmdb_map = movie_to_tmdb_map
        self.req_source = req_source
        
        # Extract item mappings
        self.item_mapping = self.user_item_mappings.get("item_mapping", {})
        self.item_reverse_mapping = self.user_item_mappings.get("item_reverse_mapping", {})
        
        # Extract original means from the provided data
        self.user_means = user_item_means.get('user_means', np.array([]))
        self.item_means = user_item_means.get('item_means', np.array([]))
        self.global_mean = np.mean(self.user_means) if len(self.user_means) > 0 else 3.5
        
        # Calculate item popularity for confidence adjustment
        self.item_rating_counts = np.array((self.user_item_matrix != 0).sum(axis=0)).flatten()
        self.min_ratings_threshold = 5
        
        logger.info(f"Loaded original means - Global: {self.global_mean:.2f}, "
                   f"Users: {len(self.user_means)}, Items: {len(self.item_means)}")

    def convert_to_similarity(self, inner_product: float) -> float:
        """Convert FAISS inner product to interpretable similarity based on the metric."""
        if self.similarity_metric == 'cosine':
            return max(0.0, min(1.0, inner_product))
        else:
            return inner_product

    def _predict_rating(
        self, 
        target_item_idx: int, 
        user_rated_items: List[Tuple[int, float]], 
        all_similar_items: List[Tuple[int, float]]
    ) -> Tuple[float, float]:
        """
        Predict rating using item-based CF with proper similarity weighting.
        Returns: (predicted_rating, confidence_score)
        """
        try:
            # Create lookup dict for user ratings for faster access
            user_ratings_dict = {idx: rating for idx, rating in user_rated_items}
            
            # Filter similar items to only those the user has actually rated
            relevant_similar_items = [
                (sim_idx, similarity) for sim_idx, similarity in all_similar_items
                if sim_idx in user_ratings_dict and similarity >= self.min_similarity
            ]
            
            if not relevant_similar_items:
                # Fallback: use item mean with low confidence
                target_item_mean = (self.item_means[target_item_idx] 
                                  if target_item_idx < len(self.item_means) 
                                  else self.global_mean)
                return max(1.0, min(5.0, target_item_mean)), 0.0
            
            # Get target item's mean
            target_item_mean = (self.item_means[target_item_idx] 
                              if target_item_idx < len(self.item_means) 
                              else self.global_mean)
            
            # Calculate weighted sum using item-based CF formula
            numerator = 0.0
            denominator = 0.0
            
            for similar_idx, similarity in relevant_similar_items:
                user_rating = user_ratings_dict[similar_idx]
                similar_item_mean = (self.item_means[similar_idx] 
                                   if similar_idx < len(self.item_means) 
                                   else self.global_mean)
                
                # User's deviation from the similar item's mean
                deviation = user_rating - similar_item_mean
                
                # Weight by similarity
                numerator += similarity * deviation
                denominator += abs(similarity)  # Use absolute similarity
            
            if denominator > 0:
                # Item-based prediction: target_item_mean + weighted_average_deviation
                weighted_deviation = numerator / denominator
                predicted_rating = target_item_mean + weighted_deviation
                
                # Calculate confidence based on similarity strength and number of neighbors
                avg_similarity = denominator / len(relevant_similar_items)
                neighbor_factor = min(len(relevant_similar_items) / 5.0, 1.0)  # More neighbors = higher confidence
                confidence = avg_similarity * neighbor_factor
                
            else:
                predicted_rating = target_item_mean
                confidence = 0.0
            
            # Apply confidence-based regression to item mean for low-confidence predictions
            if confidence < 0.5:
                regression_factor = 1.0 - confidence
                predicted_rating = (predicted_rating * confidence + 
                                  target_item_mean * regression_factor)
            
            # Ensure rating is within valid range
            predicted_rating = max(1.0, min(5.0, predicted_rating))
            
            logger.debug(f"Item {target_item_idx}: predicted={predicted_rating:.2f}, "
                        f"item_mean={target_item_mean:.2f}, confidence={confidence:.2f}, "
                        f"neighbors={len(relevant_similar_items)}")
            
            return predicted_rating, confidence
            
        except Exception as e:
            logger.error(f"Error predicting rating for item {target_item_idx}: {str(e)}")
            # Fallback to item mean
            target_item_mean = (self.item_means[target_item_idx] 
                              if target_item_idx < len(self.item_means) 
                              else self.global_mean)
            return max(1.0, min(5.0, target_item_mean)), 0.0

    def generate_recommendations(
        self,
        item_ids: List[Tuple[int, float]],
        n_recommendations: int
    ) -> List[Dict]:
        """Finds similar items for multiple movies (content-based filtering) with predicted ratings."""
        try:
            if not item_ids:
                return []
            
            # Convert movieIds to internal indices
            user_rated_items = []
            item_indices = []
            
            for movieId, rating in item_ids:
                movieId = int(movieId)
                internal_idx = self.item_mapping.get(movieId)
                if internal_idx is not None:
                    user_rated_items.append((internal_idx, float(rating)))
                    item_indices.append(internal_idx)
            
            if not item_indices:
                logger.warning(f"None of the provided movie IDs could be mapped to internal indices")
                return []

            logger.info(f"Mapped {len(item_indices)} movies to internal indices")

            # Get all candidate items from similarity search
            all_candidates = set()
            
            # For each rated item, find similar items
            for item_idx in item_indices:
                if item_idx >= self.embedding_matrix.shape[0]:
                    continue
                    
                # Get embedding and normalize if needed
                item_vector = self.embedding_matrix[item_idx:item_idx+1]
                if self.similarity_metric == 'cosine':
                    norm = np.linalg.norm(item_vector)
                    if norm > 0:
                        item_vector = item_vector / norm
                
                # Search for similar items
                search_size = min(100, self.embedding_matrix.shape[0])  # Cast wider net
                distances, indices = self.faiss_index.search(item_vector.astype(np.float32), search_size)
                
                # Add candidates (excluding the query item itself)
                for idx, dist in zip(indices[0], distances[0]):
                    if idx >= 0 and idx != item_idx:
                        similarity_score = self.convert_to_similarity(dist)
                        if similarity_score >= self.min_similarity:
                            all_candidates.add(idx)

            logger.info(f"Found {len(all_candidates)} candidate items")

            # Generate recommendations for each candidate
            recommendations = []
            query_items_set = set(item_indices)
            
            for candidate_idx in all_candidates:
                if candidate_idx in query_items_set:
                    continue
                
                # For this candidate, find its similar items that the user has rated
                if candidate_idx >= self.embedding_matrix.shape[0]:
                    continue
                    
                candidate_vector = self.embedding_matrix[candidate_idx:candidate_idx+1]
                if self.similarity_metric == 'cosine':
                    norm = np.linalg.norm(candidate_vector)
                    if norm > 0:
                        candidate_vector = candidate_vector / norm
                
                # Search for items similar to this candidate
                search_size = min(50, self.embedding_matrix.shape[0])
                distances, indices = self.faiss_index.search(candidate_vector.astype(np.float32), search_size)
                
                # Build list of similar items with their similarities
                similar_items = []
                for idx, dist in zip(indices[0], distances[0]):
                    if idx >= 0 and idx != candidate_idx:
                        similarity_score = self.convert_to_similarity(dist)
                        if similarity_score >= self.min_similarity:
                            similar_items.append((idx, similarity_score))
                
                # Predict rating for this candidate
                predicted_rating, confidence = self._predict_rating(
                    target_item_idx=candidate_idx,
                    user_rated_items=user_rated_items,
                    all_similar_items=similar_items
                )

                # Convert internal index back to tmdbId
                similar_tmdbId = self.item_reverse_mapping.get(candidate_idx)
                if similar_tmdbId is None:
                    continue

                # Format recommendation based on request source
                recommendation = {
                    "similarity": float(round(confidence, 3)),  # Use confidence as similarity measure
                    "predicted_rating": float(round(predicted_rating, 2))
                }

                # Handle ID mapping based on request source
                if self.req_source == "tmdb":
                    recommendation["tmdbId"] = int(similar_tmdbId)
                else:  # movieId request
                    # Convert tmdbId to movieId
                    if str(similar_tmdbId) in self.tmdb_to_movie_map:
                        recommendation["movieId"] = int(self.tmdb_to_movie_map[str(similar_tmdbId)])
                    else:
                        # Skip items that can't be mapped
                        continue

                recommendations.append(recommendation)

            # Sort by predicted rating (primary) and confidence (secondary)
            recommendations.sort(key=lambda x: (x["predicted_rating"], x["similarity"]), reverse=True)
            
            logger.info(f"Generated {len(recommendations)} recommendations, returning top {n_recommendations}")
            
            return recommendations[:n_recommendations]

        except Exception as e:
            logger.error(f"Error in generating item recommendations: {str(e)}")
            raise RuntimeError(f"Error in generating item recommendations: {str(e)}")
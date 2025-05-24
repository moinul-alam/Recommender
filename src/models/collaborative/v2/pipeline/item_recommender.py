import numpy as np
import faiss
from typing import Dict, List, Tuple
import logging
from scipy import sparse
from functools import lru_cache

logger = logging.getLogger(__name__)

class ItemRecommender:
    """Generates recommendations based on item similarity (content-based filtering)."""

    def __init__(
        self,
        faiss_index: faiss.Index,
        embedding_matrix: np.ndarray,
        user_item_matrix: sparse.csr_matrix,
        user_item_mappings: dict,
        user_item_means: dict,
        similarity_metric: str = 'cosine',
        min_similarity: float = 0.1
    ):
        self.faiss_index = faiss_index
        self.embedding_matrix = embedding_matrix
        self.user_item_matrix = user_item_matrix
        self.user_item_mappings = user_item_mappings
        self.user_item_means = user_item_means
        self.similarity_metric = similarity_metric
        self.min_similarity = min_similarity
        
        # Extract mappings
        self.item_mapping = self.user_item_mappings.get("item_mapping", {})
        self.item_reverse_mapping = self.user_item_mappings.get("item_reverse_mapping", {})
        
        # Extract means
        self.user_means = user_item_means.get('user_means', np.array([]))
        self.item_means = user_item_means.get('item_means', np.array([]))
        self.global_mean = np.mean(self.user_means) if len(self.user_means) > 0 else 3.5
        
        # Precompute item popularity for better recommendations
        self.item_rating_counts = np.array((self.user_item_matrix != 0).sum(axis=0)).flatten()
        self.min_ratings_threshold = 5
        
        logger.info(f"ItemRecommender initialized - Global mean: {self.global_mean:.2f}")

    @lru_cache(maxsize=1000)
    def _get_similar_items(self, item_idx: int, k: int = 100) -> List[Tuple[int, float]]:
        """Cache similar items lookup for better performance."""
        if item_idx >= self.embedding_matrix.shape[0]:
            return []
            
        item_vector = self.embedding_matrix[item_idx:item_idx+1]
        if self.similarity_metric == 'cosine':
            norm = np.linalg.norm(item_vector)
            if norm > 0:
                item_vector = item_vector / norm
        
        search_size = min(k, self.embedding_matrix.shape[0])
        distances, indices = self.faiss_index.search(item_vector.astype(np.float32), search_size)
        
        similar_items = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0 and idx != item_idx:
                similarity_score = self._convert_to_similarity(dist)
                if similarity_score >= self.min_similarity:
                    similar_items.append((idx, similarity_score))
        
        return similar_items

    def _convert_to_similarity(self, inner_product: float) -> float:
        """Convert FAISS inner product to interpretable similarity."""
        if self.similarity_metric == 'cosine':
            return max(0.0, min(1.0, inner_product))
        return inner_product

    def _predict_rating(
        self, 
        target_item_idx: int, 
        user_rated_items: List[Tuple[int, float]]
    ) -> Tuple[float, float]:
        """Predict rating using improved item-based collaborative filtering."""
        try:
            # Get similar items for the target item
            similar_items = self._get_similar_items(target_item_idx, k=50)
            
            if not similar_items:
                # Fallback to item mean
                target_item_mean = (self.item_means[target_item_idx] 
                                  if target_item_idx < len(self.item_means) 
                                  else self.global_mean)
                return max(1.0, min(5.0, target_item_mean)), 0.0
            
            # Create lookup for user ratings
            user_ratings_dict = {idx: rating for idx, rating in user_rated_items}
            
            # Filter similar items to only those rated by the user
            relevant_similar = [
                (sim_idx, similarity) for sim_idx, similarity in similar_items
                if sim_idx in user_ratings_dict
            ]
            
            if not relevant_similar:
                # Fallback to item mean
                target_item_mean = (self.item_means[target_item_idx] 
                                  if target_item_idx < len(self.item_means) 
                                  else self.global_mean)
                return max(1.0, min(5.0, target_item_mean)), 0.0
            
            # Get target item's mean
            target_item_mean = (self.item_means[target_item_idx] 
                              if target_item_idx < len(self.item_means) 
                              else self.global_mean)
            
            # Calculate weighted prediction using item-based CF
            numerator = 0.0
            denominator = 0.0
            
            for similar_idx, similarity in relevant_similar:
                user_rating = user_ratings_dict[similar_idx]
                similar_item_mean = (self.item_means[similar_idx] 
                                   if similar_idx < len(self.item_means) 
                                   else self.global_mean)
                
                # User's deviation from similar item's mean
                deviation = user_rating - similar_item_mean
                
                # Weight by similarity (with popularity adjustment)
                popularity_weight = min(self.item_rating_counts[similar_idx] / self.min_ratings_threshold, 2.0)
                adjusted_similarity = similarity * popularity_weight
                
                numerator += adjusted_similarity * deviation
                denominator += abs(adjusted_similarity)
            
            if denominator > 0:
                weighted_deviation = numerator / denominator
                predicted_rating = target_item_mean + weighted_deviation
                
                # Calculate confidence
                avg_similarity = denominator / len(relevant_similar)
                neighbor_factor = min(len(relevant_similar) / 5.0, 1.0)
                confidence = avg_similarity * neighbor_factor
            else:
                predicted_rating = target_item_mean
                confidence = 0.0
            
            # Apply confidence-based regression for low confidence predictions
            if confidence < 0.4:
                regression_factor = 1.0 - confidence
                predicted_rating = (predicted_rating * confidence + 
                                  target_item_mean * regression_factor)
            
            # Ensure valid rating range
            predicted_rating = max(1.0, min(5.0, predicted_rating))
            
            return predicted_rating, confidence
            
        except Exception as e:
            logger.error(f"Error predicting rating for item {target_item_idx}: {str(e)}")
            # Fallback
            target_item_mean = (self.item_means[target_item_idx] 
                              if target_item_idx < len(self.item_means) 
                              else self.global_mean)
            return max(1.0, min(5.0, target_item_mean)), 0.0

    def generate_recommendations(
        self,
        item_ids: List[Tuple[int, float]],
        n_recommendations: int
    ) -> List[Dict]:
        """Generate item-based recommendations returning movieIds."""
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
                logger.warning("No valid movie indices found after mapping")
                return []

            logger.info(f"Mapped {len(item_indices)} movies to internal indices")

            # Collect all candidate items from similarity searches
            all_candidates = set()
            query_items_set = set(item_indices)
            
            for item_idx in item_indices:
                similar_items = self._get_similar_items(item_idx, k=100)
                for candidate_idx, _ in similar_items:
                    if candidate_idx not in query_items_set:
                        all_candidates.add(candidate_idx)

            logger.info(f"Found {len(all_candidates)} candidate items")

            # Generate predictions for all candidates
            recommendations = []
            
            for candidate_idx in all_candidates:
                predicted_rating, confidence = self._predict_rating(
                    target_item_idx=candidate_idx,
                    user_rated_items=user_rated_items
                )

                # Convert back to movieId
                movieId = self.item_reverse_mapping.get(candidate_idx)
                if movieId is None:
                    continue

                recommendation = {
                    "movieId": int(movieId),
                    "similarity": float(round(confidence, 3)),
                    "predicted_rating": float(round(predicted_rating, 2))
                }
                
                recommendations.append(recommendation)

            # Sort by predicted rating (primary) and confidence (secondary)
            recommendations.sort(key=lambda x: (x["predicted_rating"], x["similarity"]), reverse=True)
            
            # Apply diversity filter to avoid too similar items
            diverse_recommendations = self._apply_diversity_filter(recommendations, diversity_threshold=0.8)
            
            logger.info(f"Generated {len(diverse_recommendations)} diverse recommendations, returning top {n_recommendations}")
            
            return diverse_recommendations[:n_recommendations]

        except Exception as e:
            logger.error(f"Error in generating item recommendations: {str(e)}")
            raise RuntimeError(f"Error in generating item recommendations: {str(e)}")
    
    def _apply_diversity_filter(self, recommendations: List[Dict], diversity_threshold: float = 0.8) -> List[Dict]:
        """Apply diversity filter to avoid recommending too similar items."""
        if len(recommendations) <= 1:
            return recommendations
            
        diverse_recs = [recommendations[0]]  # Always include the top recommendation
        
        for rec in recommendations[1:]:
            rec_movieId = rec["movieId"]
            rec_idx = self.item_mapping.get(rec_movieId)
            
            if rec_idx is None:
                continue
                
            # Check similarity with already selected items
            is_diverse = True
            for selected_rec in diverse_recs:
                selected_movieId = selected_rec["movieId"]
                selected_idx = self.item_mapping.get(selected_movieId)
                
                if selected_idx is None:
                    continue
                    
                # Get similarity between items
                similar_items = self._get_similar_items(rec_idx, k=20)
                similarity_with_selected = 0.0
                
                for sim_idx, sim_score in similar_items:
                    if sim_idx == selected_idx:
                        similarity_with_selected = sim_score
                        break
                
                if similarity_with_selected > diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_recs.append(rec)
                
        return diverse_recs
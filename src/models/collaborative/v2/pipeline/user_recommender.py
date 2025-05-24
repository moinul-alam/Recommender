import numpy as np
import faiss
from typing import Dict, List, Tuple
import logging
from scipy import sparse
from functools import lru_cache
from src.models.common.logger import app_logger
from src.models.common.file_config import file_names


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
        user_item_means: dict,
        similarity_metric: str = 'cosine',
        min_similarity: float = 0.1,
        n_neighbors: int = 50
    ):
        self.faiss_user_index = faiss_user_index
        self.user_embedding_matrix = user_embedding_matrix
        self.faiss_item_index = faiss_item_index
        self.item_embedding_matrix = item_embedding_matrix
        self.user_item_matrix = user_item_matrix
        self.user_item_mappings = user_item_mappings
        self.svd_user_model = svd_user_model
        self.user_item_means = user_item_means
        self.similarity_metric = similarity_metric
        self.min_similarity = min_similarity
        self.n_neighbors = n_neighbors
        
        # Extract mappings
        self.item_mapping = self.user_item_mappings.get("item_mapping", {})
        self.item_reverse_mapping = self.user_item_mappings.get("item_reverse_mapping", {})
        self.user_mapping = self.user_item_mappings.get("user_mapping", {})
        self.user_reverse_mapping = self.user_item_mappings.get("user_reverse_mapping", {})
        
        # Validate mappings
        if not self.item_mapping or not self.item_reverse_mapping:
            logger.error("Item mappings are missing or empty")
            raise ValueError("Item mappings are required but not provided")
        
        # Extract means
        # self.user_means = user_item_means.get('user_means', np.array([]))
        # self.item_means = user_item_means.get('item_means', np.array([]))
        # self.global_mean = np.mean(self.user_means) if len(self.user_means) > 0 else 3.5
        # Extract means
        user_means = file_names['user_means']
        logger.info(f"Loading user means from {user_means}")
        item_means = file_names['item_means']
        logger.info(f"Loading item means from {item_means}")
        self.user_means = user_item_means.get(user_means, np.array([]))
        logger.info(f"User means loaded with shape: {self.user_means.shape}")
        self.item_means = user_item_means.get(item_means, np.array([]))
        logger.info(f"Item means loaded with shape: {self.item_means.shape}")
        self.global_mean = np.mean(self.user_means) if len(self.user_means) > 0 else 3.5
        
        # Precompute item statistics for better recommendations
        self.item_rating_counts = np.array((self.user_item_matrix != 0).sum(axis=0)).flatten()
        self.item_avg_ratings = np.array(self.user_item_matrix.sum(axis=0)).flatten()
        
        # Avoid division by zero
        nonzero_counts = self.item_rating_counts.copy()
        nonzero_counts[nonzero_counts == 0] = 1
        self.item_avg_ratings = self.item_avg_ratings / nonzero_counts
        
        logger.info(f"UserRecommender initialized - Global mean: {self.global_mean:.2f}")

    def _convert_to_similarity(self, inner_product: float) -> float:
        """Convert FAISS inner product to interpretable similarity."""
        if self.similarity_metric == 'cosine':
            return max(0.0, min(1.0, inner_product))
        return inner_product

    def _map_movie_ids_to_indices(self, user_ratings: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Map movieIds to internal indices and filter invalid ones."""
        return [
            (self.item_mapping.get(movieId), rating)
            for movieId, rating in user_ratings
            if self.item_mapping.get(movieId) is not None
        ]

    @lru_cache(maxsize=1000)
    def _get_item_statistics(self, item_idx: int) -> Tuple[float, int]:
        """Cache item statistics for better performance."""
        item_mean = (self.item_means[item_idx] 
                    if item_idx < len(self.item_means) 
                    else self.global_mean)
        item_count = self.item_rating_counts[item_idx] if item_idx < len(self.item_rating_counts) else 0
        return item_mean, item_count

    def _predict_rating_for_item(
        self, 
        item_idx: int, 
        similar_users: List[Tuple[int, float]], 
        active_user_mean: float
    ) -> Tuple[float, float]:
        """Predict rating for a specific item using improved user-based CF."""
        try:
            item_mean, item_count = self._get_item_statistics(item_idx)
            
            # Collect ratings from similar users
            weighted_sum = 0.0
            similarity_sum = 0.0
            neighbor_count = 0
            
            for user_idx, similarity in similar_users:
                # Get rating from mean-centered matrix
                centered_rating = self.user_item_matrix[user_idx, item_idx]
                
                if centered_rating != 0:  # User has rated this item
                    # Get neighbor's mean
                    neighbor_mean = (self.user_means[user_idx] 
                                   if user_idx < len(self.user_means) 
                                   else self.global_mean)
                    
                    # Convert to original rating
                    original_rating = centered_rating + neighbor_mean
                    
                    # Calculate deviation from neighbor's mean
                    deviation = original_rating - neighbor_mean
                    
                    # Apply popularity weighting for more robust predictions
                    popularity_weight = min(item_count / 10.0, 2.0) if item_count > 0 else 0.5
                    adjusted_similarity = similarity * popularity_weight
                    
                    weighted_sum += adjusted_similarity * deviation
                    similarity_sum += abs(adjusted_similarity)
                    neighbor_count += 1
            
            if neighbor_count == 0 or similarity_sum == 0:
                # Fallback to baseline prediction
                baseline = active_user_mean + (item_mean - self.global_mean)
                predicted_rating = max(1.0, min(5.0, baseline))
                confidence = 0.0
            else:
                # User-based CF prediction
                avg_deviation = weighted_sum / similarity_sum
                predicted_rating = active_user_mean + avg_deviation
                
                # Calculate confidence with popularity adjustment
                avg_similarity = similarity_sum / neighbor_count
                neighbor_factor = min(neighbor_count / 8.0, 1.0)
                popularity_factor = min(item_count / 20.0, 1.0) if item_count > 0 else 0.3
                confidence = avg_similarity * neighbor_factor * popularity_factor
            
            # Apply confidence-based regression for low-confidence predictions
            if confidence < 0.3:
                baseline = active_user_mean + (item_mean - self.global_mean)
                regression_factor = 1.0 - confidence
                predicted_rating = (predicted_rating * confidence + 
                                  baseline * regression_factor)
            
            # Ensure valid rating range
            predicted_rating = max(1.0, min(5.0, predicted_rating))
            
            return predicted_rating, confidence
            
        except Exception as e:
            logger.error(f"Error predicting rating for item {item_idx}: {str(e)}")
            # Fallback
            baseline = active_user_mean + (item_mean - self.global_mean) if item_mean else active_user_mean
            return max(1.0, min(5.0, baseline)), 0.0

    def _create_user_embedding(self, valid_indices: List[Tuple[int, float]], active_user_mean: float) -> np.ndarray:
        """Create user embedding using SVD components or weighted item embeddings."""
        try:
            # Method 1: Use SVD model if it's a TruncatedSVD object
            if hasattr(self.svd_user_model, 'components_'):
                # Create user profile vector
                n_items = len(self.item_mapping)
                user_profile = np.zeros(n_items)
                
                for item_idx, rating in valid_indices:
                    if item_idx < n_items:
                        # Center the rating
                        item_mean = (self.item_means[item_idx] 
                                   if item_idx < len(self.item_means) 
                                   else self.global_mean)
                        centered_rating = rating - item_mean
                        user_profile[item_idx] = centered_rating
                
                # Transform using SVD components
                user_embedding = self.svd_user_model.transform(user_profile.reshape(1, -1))[0]
                return user_embedding.astype(np.float32)
            
            # Method 2: Check if it's a dictionary with components
            elif isinstance(self.svd_user_model, dict) and 'V_T' in self.svd_user_model:
                V_T = self.svd_user_model['V_T']
                sigma = self.svd_user_model.get('sigma', 1.0)
                
                # Create user profile vector
                n_items = len(self.item_mapping)
                user_profile = np.zeros(n_items)
                
                for item_idx, rating in valid_indices:
                    if item_idx < n_items:
                        # Center the rating
                        item_mean = (self.item_means[item_idx] 
                                   if item_idx < len(self.item_means) 
                                   else self.global_mean)
                        centered_rating = rating - item_mean
                        user_profile[item_idx] = centered_rating
                
                # Project using SVD
                if isinstance(sigma, np.ndarray):
                    user_embedding = np.dot(user_profile, V_T.T) / sigma
                else:
                    user_embedding = np.dot(user_profile, V_T.T) / sigma
                return user_embedding.astype(np.float32)
            
            # Method 3: Weighted average of item embeddings (fallback)
            else:
                embeddings = []
                weights = []
                
                for item_idx, rating in valid_indices:
                    if item_idx < self.item_embedding_matrix.shape[0]:
                        item_embedding = self.item_embedding_matrix[item_idx]
                        # Weight by deviation from item mean
                        item_mean = (self.item_means[item_idx] 
                                   if item_idx < len(self.item_means) 
                                   else self.global_mean)
                        weight = abs(rating - item_mean) + 0.1  # Add small constant to avoid zero weights
                        
                        embeddings.append(item_embedding)
                        weights.append(weight)
                
                if embeddings:
                    embeddings = np.array(embeddings)
                    weights = np.array(weights)
                    weights = weights / np.sum(weights)  # Normalize weights
                    
                    user_embedding = np.average(embeddings, axis=0, weights=weights)
                    return user_embedding.astype(np.float32)
                
                # Fallback: return mean embedding
                return np.mean(self.item_embedding_matrix, axis=0).astype(np.float32)
                
        except Exception as e:
            logger.error(f"Error creating user embedding: {str(e)}")
            # Return mean embedding as fallback
            return np.mean(self.item_embedding_matrix, axis=0).astype(np.float32)

    @lru_cache(maxsize=100)
    def _find_similar_users(self, user_embedding_hash: int) -> List[Tuple[int, float]]:
        """Find similar users using cached lookup."""
        try:
            # Reconstruct embedding from hash (simplified approach)
            # In practice, you'd need a more sophisticated caching mechanism
            user_embedding = self._get_embedding_from_hash(user_embedding_hash)
            
            if user_embedding is None:
                return []
            
            # Normalize for cosine similarity
            if self.similarity_metric == 'cosine':
                norm = np.linalg.norm(user_embedding)
                if norm > 0:
                    user_embedding = user_embedding / norm
            
            # Search for similar users
            search_size = min(self.n_neighbors * 2, self.user_embedding_matrix.shape[0])
            distances, indices = self.faiss_user_index.search(
                user_embedding.reshape(1, -1), search_size
            )
            
            similar_users = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0:
                    similarity_score = self._convert_to_similarity(dist)
                    if similarity_score >= self.min_similarity:
                        similar_users.append((idx, similarity_score))
            
            return similar_users[:self.n_neighbors]
            
        except Exception as e:
            logger.error(f"Error finding similar users: {str(e)}")
            return []

    def _get_embedding_from_hash(self, embedding_hash: int) -> np.ndarray:
        """Placeholder for embedding reconstruction from hash."""
        # This is a simplified approach - in practice, you'd implement proper caching
        return None

    def _get_candidate_items(self, similar_users: List[Tuple[int, float]], user_rated_items: set) -> List[int]:
        """Get candidate items from similar users with popularity filtering."""
        item_scores = {}
        
        for user_idx, similarity in similar_users:
            # Get items rated by this similar user
            user_items = self.user_item_matrix[user_idx].nonzero()[1]
            
            for item_idx in user_items:
                if item_idx not in user_rated_items:
                    # Get rating and item popularity
                    rating = self.user_item_matrix[user_idx, item_idx]
                    item_popularity = self.item_rating_counts[item_idx] if item_idx < len(self.item_rating_counts) else 1
                    
                    # Score based on rating, similarity, and popularity
                    score = rating * similarity * min(np.log(item_popularity + 1), 3.0)
                    
                    if item_idx in item_scores:
                        item_scores[item_idx] += score
                    else:
                        item_scores[item_idx] = score
        
        # Sort by score and return top candidates
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return [item_idx for item_idx, _ in sorted_items[:500]]  # Limit candidates

    def generate_recommendations(
        self,
        user_ratings: List[Tuple[int, float]],
        n_recommendations: int
    ) -> List[Dict]:
        """Generate user-based recommendations returning movieIds."""
        try:
            if not user_ratings:
                return []
            
            # Map movieIds to internal indices
            valid_indices = self._map_movie_ids_to_indices(user_ratings)
            
            if not valid_indices:
                logger.warning("No valid movie indices found after mapping")
                return []
            
            logger.info(f"Mapped {len(valid_indices)} movies to internal indices")
            
            # Calculate active user's mean rating
            ratings = [rating for _, rating in valid_indices]
            active_user_mean = np.mean(ratings)
            
            # Create user embedding
            user_embedding = self._create_user_embedding(valid_indices, active_user_mean)
            
            # Find similar users
            similar_users = []
            try:
                # Normalize for cosine similarity
                if self.similarity_metric == 'cosine':
                    norm = np.linalg.norm(user_embedding)
                    if norm > 0:
                        user_embedding = user_embedding / norm
                
                # Search for similar users
                search_size = min(self.n_neighbors * 2, self.user_embedding_matrix.shape[0])
                distances, indices = self.faiss_user_index.search(
                    user_embedding.reshape(1, -1), search_size
                )
                
                for idx, dist in zip(indices[0], distances[0]):
                    if idx >= 0:
                        similarity_score = self._convert_to_similarity(dist)
                        if similarity_score >= self.min_similarity:
                            similar_users.append((idx, similarity_score))
                
                similar_users = similar_users[:self.n_neighbors]
                
            except Exception as e:
                logger.error(f"Error finding similar users: {str(e)}")
                return []
            
            if not similar_users:
                logger.warning("No similar users found")
                return []
            
            logger.info(f"Found {len(similar_users)} similar users")
            
            # Get candidate items
            user_rated_items = {item_idx for item_idx, _ in valid_indices}
            candidate_items = self._get_candidate_items(similar_users, user_rated_items)
            
            if not candidate_items:
                logger.warning("No candidate items found")
                return []
            
            logger.info(f"Found {len(candidate_items)} candidate items")
            
            # Generate predictions for candidate items
            recommendations = []
            
            for item_idx in candidate_items:
                predicted_rating, confidence = self._predict_rating_for_item(
                    item_idx, similar_users, active_user_mean
                )
                
                # Convert back to movieId
                movieId = self.item_reverse_mapping.get(item_idx)
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
            
            # Apply diversity filter
            diverse_recommendations = self._apply_diversity_filter(recommendations)
            
            logger.info(f"Generated {len(diverse_recommendations)} diverse recommendations, returning top {n_recommendations}")
            
            return diverse_recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error in generating user recommendations: {str(e)}")
            raise RuntimeError(f"Error in generating user recommendations: {str(e)}")

    def _apply_diversity_filter(self, recommendations: List[Dict], diversity_threshold: float = 0.7) -> List[Dict]:
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
                
                # Calculate item similarity using embeddings
                if (rec_idx < self.item_embedding_matrix.shape[0] and 
                    selected_idx < self.item_embedding_matrix.shape[0]):
                    
                    rec_embedding = self.item_embedding_matrix[rec_idx]
                    selected_embedding = self.item_embedding_matrix[selected_idx]
                    
                    # Cosine similarity
                    rec_norm = np.linalg.norm(rec_embedding)
                    selected_norm = np.linalg.norm(selected_embedding)
                    
                    if rec_norm > 0 and selected_norm > 0:
                        similarity = np.dot(rec_embedding, selected_embedding) / (rec_norm * selected_norm)
                        
                        if similarity > diversity_threshold:
                            is_diverse = False
                            break
            
            if is_diverse:
                diverse_recs.append(rec)
        
        return diverse_recs
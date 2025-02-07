import logging
from typing import Any, Dict, List
import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class ItemRecommender:
    def __init__(
        self,
        item_similarity_matrix: np.ndarray,
        item_mapping: dict,
        item_reverse_mapping: dict,
        faiss_index: faiss.Index,
        model_info: Dict[str, Any]
    ):
        if not isinstance(faiss_index, faiss.Index):
            raise TypeError("faiss_index must be a FAISS Index instance")
        
        # Convert all keys to integers in the mappings
        self.item_mapping = {int(k): int(v) for k, v in item_mapping.items()}
        self.item_reverse_mapping = {int(k): int(v) for k, v in item_reverse_mapping.items()}
        
        self.item_similarity_matrix = item_similarity_matrix
        self.faiss_index = faiss_index
        self.model_info = model_info
        
        logger.info(f"ItemRecommender initialized with {len(item_mapping)} items")
        # Debug log some sample mappings
        sample_items = list(self.item_mapping.items())[:5]
        logger.info(f"Sample item mappings (TMDB_ID -> Matrix_Index): {sample_items}")

    def _get_predicted_rating(self, target_idx: int, input_items: dict, similarity_vector: np.ndarray) -> float:
        """Calculate predicted rating using weighted average of input item ratings."""
        try:
            # Debug input
            logger.debug(f"Input items: {input_items}")
            logger.debug(f"Target idx: {target_idx}")
            
            # Convert input item IDs to integers and filter valid ones
            input_items_int = {int(k): float(v) for k, v in input_items.items()}
            
            # Get matrix indices for valid input items
            input_matrix_indices = []
            input_ratings = []
            
            for tmdb_id, rating in input_items_int.items():
                if tmdb_id in self.item_mapping:
                    matrix_idx = self.item_mapping[tmdb_id]
                    input_matrix_indices.append(matrix_idx)
                    input_ratings.append(rating)
                else:
                    logger.debug(f"TMDB ID {tmdb_id} not found in mapping")
            
            if not input_matrix_indices:
                raise ValueError(f"No valid input items found in mapping. Available mappings: {list(self.item_mapping.keys())[:5]}")
            
            # Extract similarities for valid input items
            input_similarities = similarity_vector[input_matrix_indices]
            
            # Apply softmax to similarities
            exp_sim = np.exp(input_similarities)
            softmax_weights = exp_sim / np.sum(exp_sim)
            
            # Calculate weighted average
            predicted = float(np.dot(softmax_weights, input_ratings))
            logger.debug(f"Predicted rating: {predicted}")
            
            return predicted
            
        except Exception as e:
            logger.error(f"Error in _get_predicted_rating: {str(e)}")
            raise

    def recommend(self, items: dict, n_recommendations: int = 10, min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """Generate recommendations based on input items."""
        if not items:
            logger.warning("No input items provided")
            return []
        
        try:
            # Debug log input items
            logger.info(f"Processing recommendations for items: {items}")
            
            # Convert input TMDB IDs to matrix indices
            input_indices = []
            for tmdb_id in items.keys():
                try:
                    idx = self.item_mapping[int(tmdb_id)]
                    input_indices.append(idx)
                except (KeyError, ValueError) as e:
                    logger.debug(f"Could not map TMDB ID {tmdb_id}: {e}")
            
            if not input_indices:
                logger.warning("No valid input items found in mapping")
                return []
                
            logger.info(f"Found {len(input_indices)} valid input items")
            
            # Get item vectors from FAISS
            item_vectors = np.array([
                self.faiss_index.reconstruct(int(idx))
                for idx in input_indices
            ], dtype=np.float32)
            
            # Compute query vector
            query_vector = np.mean(item_vectors, axis=0).reshape(1, -1).astype(np.float32)
            
            # Find similar items
            k = min(n_recommendations + len(items), self.faiss_index.ntotal)
            scores, indices = self.faiss_index.search(query_vector, k)
            
            recommendations = []
            seen_items = {int(tmdb_id) for tmdb_id in items.keys()}
            
            for score, idx in zip(scores[0], indices[0]):
                idx = int(idx)
                tmdb_id = self.item_reverse_mapping[idx]
                
                if tmdb_id in seen_items or score < min_similarity:
                    continue
                
                # Get reconstructed vector and compute similarities
                reconstructed_vector = self.faiss_index.reconstruct(idx)
                similarity_vector = cosine_similarity(
                    [reconstructed_vector],
                    np.array([
                        self.faiss_index.reconstruct(int(i))
                        for i in range(self.faiss_index.ntotal)
                    ], dtype=np.float32)
                )[0]
                
                try:
                    predicted_rating = self._get_predicted_rating(idx, items, similarity_vector)
                    
                    recommendations.append({
                        'tmdb_id': str(tmdb_id),
                        'predicted_rating': round(predicted_rating, 2),
                        'similarity_score': float(score),
                        'rank': len(recommendations) + 1
                    })
                    
                except Exception as e:
                    logger.error(f"Error calculating predicted rating for item {tmdb_id}: {str(e)}")
                    continue
                
                if len(recommendations) >= n_recommendations:
                    break
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in recommendation process: {str(e)}")
            raise
        
        
        

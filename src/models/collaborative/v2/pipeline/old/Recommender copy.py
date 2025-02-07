import numpy as np
import faiss
from typing import Dict, List

class Recommender:
    def __init__(
        self,
        faiss_index: faiss.Index,
        item_matrix: np.ndarray,
        svd_model: object,
        item_mapping: Dict[int, int],
        item_reverse_mapping: Dict[int, int],
        model_info: Dict,
        min_similarity: float
    ):
        self.faiss_index = faiss_index
        self.item_matrix = item_matrix
        self.svd_model = svd_model
        self.item_mapping = item_mapping
        self.item_reverse_mapping = item_reverse_mapping
        self.model_info = model_info
        self.min_similarity = min_similarity

    def generate_recommendations(
        self, 
        items: Dict[int, float],  
        n_recommendations: int
    ) -> List[Dict]:
        try:
            # Filter and map valid input items
            valid_indices = [
                (self.item_mapping.get(tmdb_id), rating) 
                for tmdb_id, rating in items.items() 
                if self.item_mapping.get(tmdb_id) is not None
            ]

            if not valid_indices:
                return []
            
            item_indices, item_ratings = zip(*valid_indices)
            item_ratings = np.array(item_ratings)
            
            # Compute user preference vector using mean rating and normalized vectors
            item_vectors = self.item_matrix[list(item_indices)]
            user_vector = np.mean(item_vectors, axis=0)

            # Search for similar items, excluding input items
            D, I = self.faiss_index.search(user_vector.reshape(1, -1), n_recommendations + len(item_indices))

            # Process results and apply filters
            query_items_set = set(item_indices)
            recommendations = []
            for idx, dist in zip(I[0], D[0]):
                if idx >= 0 and idx not in query_items_set and dist >= self.min_similarity:
                    tmdb_id = self.item_reverse_mapping.get(idx)

                    # Convert FAISS distance to similarity percentage
                    similarity_score = max(0, min(100, (1 - dist) * 100))

                    # Compute predicted rating using cosine similarity with input items
                    item_vector = self.item_matrix[idx]
                    similarities = np.dot(item_vectors, item_vector) / (
                        np.linalg.norm(item_vectors, axis=1) * np.linalg.norm(item_vector)
                    )
                    weighted_sum = np.dot(similarities, item_ratings)
                    similarity_sum = np.sum(similarities)
                    predicted_rating = float(
                        weighted_sum / similarity_sum if similarity_sum > 0 
                        else np.mean(item_ratings)
                    )

                    recommendations.append({
                        "tmdb_id": tmdb_id,
                        "similarity": similarity_score,
                        "predicted_rating": round(predicted_rating, 2)
                    })

                    if len(recommendations) == n_recommendations:
                        break

            # Sort recommendations by similarity score
            return sorted(recommendations, key=lambda x: x["similarity"], reverse=True)

        except Exception as e:
            raise RuntimeError(f"Error in generating recommendations: {str(e)}")
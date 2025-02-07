import logging
import numpy as np
import faiss

logger = logging.getLogger(__name__)

class ItemRecommender:
    def __init__(self, svd_components, faiss_index, item_mapping, item_reverse_mapping):
        if not all([svd_components, faiss_index, item_mapping, item_reverse_mapping]):
            raise ValueError("One or more model components are missing")

        self.svd_matrix = svd_components.get("svd_matrix")
        self.faiss_index = faiss_index
        self.item_mapping = item_mapping
        self.item_reverse_mapping = item_reverse_mapping

        if self.svd_matrix is None:
            raise ValueError("SVD matrix is missing in svd_components")

    def recommend(self, input_items, top_n=10):
        input_vectors = []
        input_ratings = []
        input_tmdb_ids = set()

        for item in input_items:
            tmdb_id = item.get("tmdb_id")
            rating = item.get("rating")

            if tmdb_id is None or rating is None:
                logger.error(f"Invalid input item: {item}")
                continue

            item_index = self.item_mapping.get(tmdb_id)
            if item_index is None:
                logger.error(f"tmdb_id {tmdb_id} not found in item_mapping")
                continue

            if item_index >= self.svd_matrix.shape[0]:
                logger.error(f"Invalid index {item_index} for tmdb_id {tmdb_id}, exceeds {self.svd_matrix.shape[0]}")
                continue

            input_vectors.append(self.svd_matrix[item_index])
            input_ratings.append(rating)
            input_tmdb_ids.add(tmdb_id)

        if not input_vectors:
            raise ValueError("No valid input items found")

        query_vector = np.average(input_vectors, weights=input_ratings, axis=0)

        distances, indices = self.faiss_index.search(
            query_vector.reshape(1, -1), 
            top_n + len(input_items)
        )

        recommendations = []
        for idx, similarity in zip(indices[0], distances[0]):
            if idx >= self.svd_matrix.shape[0]:
                logger.error(f"Faiss returned out-of-bounds index {idx}")
                continue

            tmdb_id = self.item_reverse_mapping.get(idx)
            if tmdb_id is None:
                logger.error(f"No tmdb_id found for index {idx} in item_reverse_mapping")
                continue

            if tmdb_id in input_tmdb_ids:
                continue

            # Compute predicted rating using weighted sum formula
            weighted_sum = sum(float(r) * float(similarity) for r in input_ratings)
            similarity_sum = sum(float(similarity) for _ in input_ratings)
            predicted_rating = weighted_sum / similarity_sum if similarity_sum != 0 else 0

            recommendations.append({
                "tmdb_id": tmdb_id, 
                "predicted_rating": round(predicted_rating, 2),  
                "similarity_score": round(float(similarity), 4)
            })

            if len(recommendations) >= top_n:
                break

        return recommendations

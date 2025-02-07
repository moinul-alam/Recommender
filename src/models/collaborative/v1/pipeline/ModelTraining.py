import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class ModelTraining:
    def __init__(
        self,
        n_neighbors: int = None, 
        similarity_metric: str = 'euclidean',
        random_state: int = 42
    ):
        if similarity_metric not in ['cosine', 'euclidean']:
            raise ValueError("Similarity metric must be 'cosine' or 'euclidean'")
        
        self.n_neighbors = n_neighbors
        self.similarity_metric = similarity_metric
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.item_similarity_matrix = None

    def _determine_neighbors(self, matrix: pd.DataFrame) -> int:
        n_items = matrix.shape[1]
        recommended_neighbors = int(np.sqrt(n_items))
        recommended_neighbors = max(10, min(recommended_neighbors, 100))
        
        logger.info(f"Neighbor selection: Total items={n_items}, Selected neighbors={recommended_neighbors}")
        return recommended_neighbors

    def train(self, user_item_matrix: pd.DataFrame) -> Tuple[Dict, faiss.Index]:
        if user_item_matrix.empty:
            raise ValueError("Input matrix is empty")
        
        if self.n_neighbors is None:
            self.n_neighbors = self._determine_neighbors(user_item_matrix)
        
        # Ensure matrix is item-item by transposing
        matrix_dense = user_item_matrix.values.astype(np.float32)  # Ensure float32 type
        matrix_scaled = self.scaler.fit_transform(matrix_dense.T)  # (num_items, num_users)

        # Compute item-item similarity
        if self.similarity_metric == 'cosine':
            item_similarity = cosine_similarity(matrix_scaled)
        else:
            item_similarity = 1 / (1 + np.linalg.norm(
                matrix_scaled[:, np.newaxis] - matrix_scaled, 
                axis=2
            ))

        self.item_similarity_matrix = item_similarity
        
        try:
            # Initialize FAISS index with the correct dimension
            faiss_index = faiss.IndexFlatL2(matrix_scaled.shape[1])
            
            # Ensure data is contiguous and float32
            matrix_scaled = np.ascontiguousarray(matrix_scaled, dtype=np.float32)
            
            # Add vectors to the index
            faiss_index.add(matrix_scaled)
            
            logger.info(f"Faiss index created with {faiss_index.ntotal} item vectors")
        except Exception as e:
            logger.error(f"Faiss index creation failed: {e}")
            raise
        
        model_info = {
            'n_items': matrix_scaled.shape[0],
            'n_neighbors': self.n_neighbors,
            'similarity_metric': self.similarity_metric,
            'top_similarity_scores': self._get_top_similarity_stats()
        }
        
        logger.info(f"Model trained successfully: {model_info}")
        
        return {
            'item_similarity_matrix': item_similarity,
            'model_info': model_info
        }, faiss_index

    def _get_top_similarity_stats(self, top_k: int = 5) -> list:
        if self.item_similarity_matrix is None:
            return []
        
        # Create a copy to avoid modifying the original
        sim_matrix = self.item_similarity_matrix.copy()
        np.fill_diagonal(sim_matrix, 0)
        top_similarities = np.sort(sim_matrix.flatten())[-top_k:]
        
        return top_similarities.tolist()

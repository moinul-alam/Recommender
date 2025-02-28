import logging
from typing import Dict, Tuple, Optional
import numpy as np
from scipy import sparse
import faiss
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)

class ModelTraining:
    def __init__(
        self,
        n_neighbors: int = 100,
        similarity_metric: str = 'cosine',
        batch_size: int = 20000,
        min_similarity: float = 0.1,
        random_state: int = 42,
        n_components_item: int = 300,
        n_components_user: int = 300,
        use_disk_index: bool = False, 
        item_index_path: str = "3_faiss_item_index.flat",
        user_index_path: str = "3_faiss_user_index.flat"
    ):
        """
        Initialize the ModelTraining class with configurable parameters.
        Uses IndexFlatIP for exact similarity computation.
        
        Args:
            n_neighbors: Number of neighbors to consider for recommendations
            similarity_metric: Metric used for similarity ('cosine' or 'L2')
            batch_size: Size of batches for processing large matrices
            min_similarity: Minimum similarity threshold for recommendations
            random_state: Random seed for reproducibility
            n_components_item: Number of SVD components for item embeddings
            n_components_user: Number of SVD components for user embeddings
            use_disk_index: Whether to save FAISS indices to disk
            item_index_path: Path to save item FAISS index
            user_index_path: Path to save user FAISS index
        """
        self.n_neighbors = n_neighbors
        self.similarity_metric = similarity_metric
        self.batch_size = batch_size
        self.min_similarity = min_similarity
        self.random_state = random_state
        self.n_components_item = n_components_item
        self.n_components_user = n_components_user
        self.use_disk_index = use_disk_index
        self.item_index_path = item_index_path
        self.user_index_path = user_index_path
        np.random.seed(random_state)

    def train(self, user_item_matrix: sparse.csr_matrix) -> Tuple[Dict, Dict]:
        """
        Train the model using a user-item interaction matrix.
        
        Args:
            user_item_matrix: A sparse CSR matrix of user-item interactions
            
        Returns:
            Tuple containing model components and FAISS indices
        """
        if user_item_matrix.nnz == 0:
            raise ValueError("Input matrix is empty (no non-zero elements)")

        n_users, n_items = user_item_matrix.shape
        logger.info(f"Training for {n_users} users and {n_items} items")

        # Validate n_components doesn't exceed matrix dimensions
        self.n_components_user = min(self.n_components_user, min(n_users - 1, n_items))
        self.n_components_item = min(self.n_components_item, min(n_items - 1, n_users))

        # **User-User Matrix Training**
        logger.info(f"Reducing user dimensions to {self.n_components_user} using Truncated SVD")
        svd_user = TruncatedSVD(n_components=self.n_components_user, random_state=self.random_state)
        user_matrix = svd_user.fit_transform(user_item_matrix)

        # **Item-Item Matrix Training**
        item_user_matrix = user_item_matrix.T.tocsr()
        logger.info(f"Reducing item dimensions to {self.n_components_item} using Truncated SVD")
        svd_item = TruncatedSVD(n_components=self.n_components_item, random_state=self.random_state)
        item_matrix = svd_item.fit_transform(item_user_matrix)

        # **Create FAISS Indices**
        user_index = self._create_faiss_index(
            user_matrix, 
            index_type="user", 
            index_path=self.user_index_path
        )
        
        item_index = self._create_faiss_index(
            item_matrix, 
            index_type="item", 
            index_path=self.item_index_path
        )
        
        model_info = {
            'n_items': n_items,
            'n_users': n_users,
            'n_neighbors': self.n_neighbors,
            'similarity_metric': self.similarity_metric,
            'sparsity': user_item_matrix.nnz / (n_users * n_items),
            'n_components_item': self.n_components_item,
            'n_components_user': self.n_components_user,
            'index_type': 'IndexFlatIP'
        }

        return {
            'user_matrix': user_matrix,
            'item_matrix': item_matrix,
            'svd_user_model': svd_user,
            'svd_item_model': svd_item,
            'model_info': model_info
        }, {
            'user_index': user_index,
            'item_index': item_index
        }

    def _create_faiss_index(
        self, 
        matrix: np.ndarray, 
        index_type: str, 
        index_path: str
    ) -> faiss.Index:
        """
        Create a FAISS IndexFlatIP for the given matrix.
        
        Args:
            matrix: Matrix of vectors to index
            index_type: Type of index ('user' or 'item')
            index_path: Path to save the index if use_disk_index is True
            
        Returns:
            FAISS IndexFlatIP
        """
        dimension = matrix.shape[1]
        logger.info(f"Creating FAISS IndexFlatIP for {index_type} index with {dimension} dimensions")
        
        # Prepare matrix copy for potential normalization
        index_matrix = matrix.copy().astype(np.float32)

        # Apply normalization for cosine similarity
        if self.similarity_metric == 'cosine':
            logger.info(f"Using cosine similarity (normalizing vectors)")
            faiss.normalize_L2(index_matrix)
        
        # Create IndexFlatIP for inner product similarity
        index = faiss.IndexFlatIP(dimension)
        
        # Add vectors in batches
        logger.info(f"Adding {index_matrix.shape[0]} {index_type} embeddings to IndexFlatIP")
        for start_idx in range(0, index_matrix.shape[0], self.batch_size):
            end_idx = min(start_idx + self.batch_size, index_matrix.shape[0])
            batch = index_matrix[start_idx:end_idx]
            index.add(batch)
            if (end_idx - start_idx) > 100000:  # Only log for large batches
                logger.info(f"Added batch {start_idx}-{end_idx} to FAISS {index_type} index")

        # Save FAISS Index
        if self.use_disk_index:
            faiss.write_index(index, str(index_path))
            logger.info(f"Saved FAISS {index_type} index to {index_path}")

        return index
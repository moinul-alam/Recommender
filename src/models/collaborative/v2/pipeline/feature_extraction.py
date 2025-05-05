import logging
import numpy as np
from typing import Dict, Tuple
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)

class FeatureExtraction:
    def __init__(
        self,
        n_components_item: int = 300,
        n_components_user: int = 300,
        batch_size: int = 20000,
    ):
        self.n_components_item = n_components_item
        self.n_components_user = n_components_user
        self.batch_size = batch_size
        self.random_state = 42

    def extract(self, user_item_matrix: sparse.csr_matrix) -> Tuple[Dict]:
        if user_item_matrix.nnz == 0:
            raise ValueError("Input matrix is empty (no non-zero elements)")

        n_users, n_items = user_item_matrix.shape
        logger.info(f"Training for {n_users} users and {n_items} items")

        # Validate matrix doesn't have empty rows or columns
        item_counts = np.diff(user_item_matrix.tocsc().indptr)
        user_counts = np.diff(user_item_matrix.indptr)
        
        if np.any(item_counts == 0):
            logger.warning(f"Found {np.sum(item_counts == 0)} items with no interactions")
        if np.any(user_counts == 0):
            logger.warning(f"Found {np.sum(user_counts == 0)} users with no interactions")

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

        # Identify zero vectors in item_matrix
        zero_vector_indices = np.where(np.all(item_matrix == 0, axis=1))[0]
        if len(zero_vector_indices) > 0:
            logger.warning(f"Found {len(zero_vector_indices)} zero vectors which cannot be normalized")
            logger.info(f"Zero vector indices: {zero_vector_indices}")
            
            # Check the original data for these items
            for idx in zero_vector_indices:
                original_item_interactions = item_user_matrix[idx].nnz
                logger.info(f"Item {idx} has {original_item_interactions} interactions in original data")
            
            # Fix zero vectors with a small random noise to avoid normalization issues
            for idx in zero_vector_indices:
                # Generate small random values (but deterministic based on index)
                np.random.seed(self.random_state + idx)
                item_matrix[idx] = np.random.uniform(1e-6, 1e-5, size=item_matrix.shape[1])
                logger.info(f"Replaced zero vector for item {idx} with small random values")

        model_info = {
            'n_items': n_items,
            'n_users': n_users,
            'sparsity': user_item_matrix.nnz / (n_users * n_items),
            'n_components_item': self.n_components_item,
            'n_components_user': self.n_components_user,
            'zero_vector_indices': zero_vector_indices.tolist() if len(zero_vector_indices) > 0 else []
        }

        return {
            'user_matrix': user_matrix,
            'item_matrix': item_matrix,
            'svd_user_model': svd_user,
            'svd_item_model': svd_item,
            'model_info': model_info
        }
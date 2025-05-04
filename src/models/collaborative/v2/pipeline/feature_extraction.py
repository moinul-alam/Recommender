import logging
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

        
        model_info = {
            'n_items': n_items,
            'n_users': n_users,
            'sparsity': user_item_matrix.nnz / (n_users * n_items),
            'n_components_item': self.n_components_item,
            'n_components_user': self.n_components_user,
        }

        return {
            'user_matrix': user_matrix,
            'item_matrix': item_matrix,
            'svd_user_model': svd_user,
            'svd_item_model': svd_item,
            'model_info': model_info
        }
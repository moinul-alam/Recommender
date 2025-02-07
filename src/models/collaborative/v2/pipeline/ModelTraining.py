import logging
from typing import Dict, Tuple
import numpy as np
from scipy import sparse
import faiss
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)

class ModelTraining:
    def __init__(
        self,
        n_neighbors: int = None, 
        similarity_metric: str = 'L2',
        batch_size: int = 10000,
        min_similarity: float = 0.2,
        random_state: int = 42,
        n_components: int = 500,  # SVD reduced dimensions
        nlist: int = 100,  # FAISS clusters
        use_disk_index: bool = False,  # Store FAISS index on disk
        index_path: str = "faiss_index.ivf"
    ):
        self.n_neighbors = n_neighbors
        self.similarity_metric = similarity_metric
        self.batch_size = batch_size
        self.min_similarity = min_similarity
        self.random_state = random_state
        self.n_components = n_components
        self.nlist = nlist
        self.use_disk_index = use_disk_index
        self.index_path = index_path
        np.random.seed(random_state)

    def train(self, user_item_matrix: sparse.csr_matrix) -> Tuple[Dict, faiss.Index]:
        if user_item_matrix.nnz == 0:
            raise ValueError("Input matrix is empty (no non-zero elements)")

        item_user_matrix = user_item_matrix.T.tocsr()
        n_items = item_user_matrix.shape[0]
        
        # Process SVD in batches
        logger.info(f"Reducing dimensions to {self.n_components} using Truncated SVD with batch size {self.batch_size}")
        svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        
        # Fit SVD on the full matrix first
        svd.fit(item_user_matrix)
        
        # Transform in batches
        item_matrix = np.zeros((n_items, self.n_components), dtype=np.float32)
        for start_idx in range(0, n_items, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_items)
            batch = item_user_matrix[start_idx:end_idx]
            item_matrix[start_idx:end_idx] = svd.transform(batch)
            logger.info(f"Processed SVD batch {start_idx}-{end_idx} of {n_items} items")

        logger.info(f"Creating FAISS index with {self.nlist} clusters")
        quantizer = faiss.IndexFlatL2(self.n_components)

        if self.similarity_metric == 'cosine':
            logger.info("Using cosine similarity for FAISS index")
            faiss.normalize_L2(item_matrix)  # Normalize to unit vectors
            index = faiss.IndexIVFFlat(quantizer, self.n_components, self.nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            logger.info("Using L2 similarity for FAISS index")
            index = faiss.IndexIVFFlat(quantizer, self.n_components, self.nlist, faiss.METRIC_L2)

        # Train the index
        index.train(item_matrix)

        # Add items to index in batches
        logger.info("Adding items to FAISS index in batches")
        for start_idx in range(0, n_items, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_items)
            batch = item_matrix[start_idx:end_idx]
            index.add(batch)
            logger.info(f"Added batch {start_idx}-{end_idx} of {n_items} items to FAISS index")

        if self.use_disk_index:
            faiss.write_index(index, str(self.index_path))
            logger.info(f"FAISS index saved to {self.index_path}")

        model_info = {
            'n_items': n_items,
            'n_users': user_item_matrix.shape[0],
            'n_neighbors': self.n_neighbors,
            'similarity_metric': self.similarity_metric,
            'sparsity': user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]),
            'n_components': self.n_components
        }

        return {
            'item_matrix': item_matrix,
            'svd_model': svd,  # Save the trained SVD model
            'model_info': model_info
        }, index
import logging
from typing import Optional
import numpy as np
from scipy import sparse
import faiss
from pathlib import Path

class IndexCreation:
    """
    Class for creating FAISS indexes from embedding matrices.
    Supports cosine similarity and inner product similarity.
    """

    def __init__(
        self,
        similarity_metric: str = 'cosine',
        batch_size: int = 20000,
    ):
        """
        Initialize IndexCreation object.

        Args:
            similarity_metric: Similarity metric to use ('cosine' or 'inner_product')
            batch_size: Batch size for adding vectors to FAISS index
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        valid_metrics = ['cosine', 'inner_product']
        if similarity_metric not in valid_metrics:
            self.logger.warning(f"Invalid similarity_metric '{similarity_metric}'. Using 'cosine' instead.")
            self.similarity_metric = 'cosine'
        else:
            self.similarity_metric = similarity_metric

        self.batch_size = max(1000, batch_size)

    def create_faiss_index(
        self,
        matrix: np.ndarray,
        index_type: str,
        index_path: str
    ) -> faiss.Index:
        """
        Create and save a FAISS index for the given matrix.

        Args:
            matrix: Matrix of embeddings (numpy array)
            index_type: Type of index ('user' or 'item') for logging
            index_path: Path where to save the FAISS index

        Returns:
            FAISS index object
        """
        if matrix is None or matrix.size == 0:
            raise ValueError(f"Empty or None matrix provided for {index_type} index")

        if not isinstance(matrix, np.ndarray):
            self.logger.warning(f"Converting {index_type} matrix to numpy array")
            if sparse.issparse(matrix):
                matrix = matrix.toarray()
            else:
                matrix = np.array(matrix)

        num_vectors, dimension = matrix.shape
        self.logger.info(f"Creating FAISS index for {num_vectors} {index_type} vectors with {dimension} dimensions")

        index_matrix = matrix.astype(np.float32)

        if self.similarity_metric == 'cosine':
            self.logger.info("Using cosine similarity - applying L2 normalization")
            norms = np.linalg.norm(index_matrix, axis=1, keepdims=True)
            non_zero_mask = norms.squeeze() > 0

            if not np.all(non_zero_mask):
                zero_vectors = np.sum(~non_zero_mask)
                self.logger.warning(f"Found {zero_vectors} zero vectors that cannot be normalized")

            index_matrix[non_zero_mask] /= norms[non_zero_mask]
        else:
            self.logger.info("Using inner product similarity - skipping normalization")

        index = faiss.IndexFlatIP(dimension)

        self.logger.info(f"Adding {num_vectors} {index_type} embeddings to FAISS index")

        total_added = 0
        for start_idx in range(0, num_vectors, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_vectors)
            batch = index_matrix[start_idx:end_idx]
            index.add(batch)
            total_added += (end_idx - start_idx)

            if self.batch_size >= 10000 or end_idx == num_vectors:
                self.logger.info(f"Added {total_added}/{num_vectors} vectors to {index_type} index")

        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            faiss.write_index(index, str(index_path))
            self.logger.info(f"Saved FAISS {index_type} index to {index_path}")
        except Exception as e:
            self.logger.error(f"Failed to save FAISS {index_type} index: {str(e)}")
            raise

        return index

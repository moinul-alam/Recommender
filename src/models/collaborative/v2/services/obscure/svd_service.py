import pickle
import numpy as np
from sklearn.decomposition import TruncatedSVD
import os
import scipy.sparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SVDService:
    def __init__(self, processed_dir_path: str, n_components: int = 1000, variance_threshold: float = 0.90):
        self.processed_dir_path = processed_dir_path
        self.n_components = n_components
        self.variance_threshold = variance_threshold

    def find_optimal_n_components(self, user_item_matrix, max_components=1000):
        """
        Find the optimal number of components for Truncated SVD based on explained variance.

        Args:
            user_item_matrix: The user-item interaction matrix (sparse or dense).
            max_components: Maximum number of components to consider.

        Returns:
            int: Optimal number of components.
        """
        if scipy.sparse.issparse(user_item_matrix):
            logger.info("Sparse matrix detected. Converting to CSR format.")
            user_item_matrix = user_item_matrix.tocsr()

        # Log matrix size
        logger.info(f"User-item matrix shape: {user_item_matrix.shape}")

        # Determine max possible components (cannot exceed min(n_users, n_items))
        max_possible_components = min(max_components, min(user_item_matrix.shape))

        if max_possible_components < 2:
            logger.warning("Matrix is too small for SVD. Returning 1 component.")
            return 1

        # Apply SVD
        svd = TruncatedSVD(n_components=max_possible_components, random_state=42)
        svd.fit(user_item_matrix)

        # Calculate cumulative explained variance
        explained_variance = svd.explained_variance_ratio_.cumsum()
        logger.info(f"Explained variance (first 10 components): {explained_variance[:10]}")

        # Find the number of components that meet the variance threshold
        n_components = np.argmax(explained_variance >= self.variance_threshold) + 1

        # Handle edge case: If threshold is never reached, return best available
        if n_components == 1 and explained_variance[-1] < self.variance_threshold:
            logger.warning("Threshold not reached. Using the best available components.")
            return int(np.argmax(explained_variance) + 1)

        logger.info(f"Optimal number of components: {n_components}")
        return n_components

    def find_svd_components(self):
        """
        Load the user-item matrix and find the optimal number of components for SVD.

        Returns:
            int: Optimal number of components.
        """
        file_path = os.path.join(self.processed_dir_path, 'user_item_matrix.pkl')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")

        # Load the user-item matrix
        with open(file_path, 'rb') as file:
            user_item_matrix = pickle.load(file)

        # Check if the matrix is empty
        if user_item_matrix.shape[0] == 0 or user_item_matrix.shape[1] == 0:
            raise ValueError("The user-item matrix is empty.")

        # Find the optimal number of components
        n_components = self.find_optimal_n_components(user_item_matrix, self.n_components)
        return int(n_components)  # Ensure it's a standard int

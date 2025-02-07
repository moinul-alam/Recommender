import logging
import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ModelTraining:
    def __init__(self, n_components: int = None):
        """
        Initialize SVD training processor.
        
        Args:
            n_components (int, optional): Number of SVD components
        """
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.svd = None
        
    def _determine_components(self, matrix: pd.DataFrame) -> int:
        """
        Dynamically determine optimal SVD components.
        
        Args:
            matrix (pd.DataFrame): User-item matrix
        
        Returns:
            int: Recommended number of components
        """
        matrix_size = matrix.shape[1]
        
        # Heuristic: Use sqrt of matrix columns, bounded between 20-100
        recommended_components = int(np.sqrt(matrix_size))
        recommended_components = max(50, min(recommended_components, 200))
        
        logger.info(f"Dynamically selected {recommended_components} SVD components")
        return recommended_components

    def train(self, user_item_matrix: pd.DataFrame):
        """
        Train SVD model and create Faiss index.
        
        Args:
            user_item_matrix (pd.DataFrame): User-item interaction matrix
        
        Returns:
            Tuple of SVD components and Faiss index
        """
        # Determine components if not set
        if self.n_components is None:
            self.n_components = self._determine_components(user_item_matrix)

        # Convert to numpy for processing
        matrix_dense = user_item_matrix.values

        # Scale matrix
        matrix_scaled = self.scaler.fit_transform(matrix_dense)
        
        # Perform SVD
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        svd_matrix = self.svd.fit_transform(matrix_scaled)
        
        logger.info(f"SVD completed. Variance explained: {self.svd.explained_variance_ratio_.sum():.2%}")

        # Create Faiss index (L2 distance)
        faiss_index = faiss.IndexFlatL2(self.n_components)
        faiss_index.add(svd_matrix.astype('float32'))
        
        logger.info(f"Faiss index created with {faiss_index.ntotal} vectors")

        return {
            'svd_components': self.svd,
            'svd_matrix': svd_matrix,
            'variance_explained': self.svd.explained_variance_ratio_
        }, faiss_index
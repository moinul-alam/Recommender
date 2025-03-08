import numpy as np
import pandas as pd
import gc
import logging
import memory_profiler
from pathlib import Path
import psutil
from sklearn.decomposition import TruncatedSVD
import joblib
from scipy import sparse
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTraining:
    def __init__(
        self,
        feature_matrix: sparse.csr_matrix,
        item_ids: np.ndarray,
        n_components_svd: int = 300,
        random_state: Optional[int] = 42
    ):
        """Initialize model training with configurable parameters."""
        logger.info("Initializing ModelTraining")
        
        # Log memory usage at initialization
        logger.info(f"Memory usage at initialization: {memory_profiler.memory_usage()[0]} MB")
        
        self.random_state = random_state
        self.feature_matrix = feature_matrix
        self.item_ids = item_ids
        self.n_components_svd = n_components_svd
        
        self.svd = TruncatedSVD(
            n_components=self.n_components_svd, 
            random_state=self.random_state
        )
        
        self.is_fitted = False
        
        logger.info("Model Training initialized successfully")
        logger.info(f"Memory usage after initialization: {memory_profiler.memory_usage()[0]} MB")

    def fit(self) -> None:
        """Fit SVD model on the feature matrix."""
        logger.info("Fitting SVD model...")
        
        try:
            self.svd.fit(self.feature_matrix)
            self.is_fitted = True
            
            # Log explained variance
            explained_variance = np.sum(self.svd.explained_variance_ratio_)
            logger.info(f"SVD fitting complete. Explained variance: {explained_variance:.4f}")
            
            # Log memory usage after fitting
            logger.info(f"Memory usage after fitting SVD: {memory_profiler.memory_usage()[0]} MB")
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error fitting SVD model: {str(e)}", exc_info=True)
            raise

    def transform(self) -> np.ndarray:
        """Transform features using fitted SVD."""
        if not self.is_fitted:
            raise ValueError("SVD model must be fitted before transformation")
            
        try:
            logger.info("Transforming feature matrix using SVD...")
            transformed_features = self.svd.transform(self.feature_matrix)
            
            logger.info(f"Transformation complete. Shape of transformed features: {transformed_features.shape}")
            logger.info(f"Memory usage after transformation: {memory_profiler.memory_usage()[0]} MB")
            
            # Force garbage collection
            gc.collect()
            
            return transformed_features

        except Exception as e:
            logger.error(f"Error transforming features: {str(e)}", exc_info=True)
            raise

    def fit_transform_and_combine(self) -> pd.DataFrame:
        """Fit SVD model, transform feature matrix, and combine with item IDs."""
        logger.info("Performing fit, transform, and combining with item IDs...")
        
        # First fit the model
        self.fit()
        
        # Then transform the features
        transformed_features = self.transform()
        
        # Create column names for the transformed features
        feature_columns = [f"feature_{i}" for i in range(self.n_components_svd)]
        
        # Combine item IDs with transformed features in a DataFrame
        try:
            # First create DataFrame from transformed features
            df_transformed = pd.DataFrame(
                data=transformed_features,
                columns=feature_columns
            )
            
            # Add item IDs as the first column
            df_transformed.insert(0, 'item_id', self.item_ids)
            
            logger.info(f"Successfully combined item IDs with transformed features. DataFrame shape: {df_transformed.shape}")
            
            # Force garbage collection
            gc.collect()
            
            return df_transformed
            
        except Exception as e:
            logger.error(f"Error combining item IDs with transformed features: {str(e)}", exc_info=True)
            raise

    def save_transformers(self, path: str) -> None:
        """Save SVD model to disk."""
        if not self.is_fitted:
            raise ValueError("SVD model must be fitted before saving")
            
        logger.info(f"Saving SVD model to: {path}")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save SVD model
            joblib.dump(self.svd, path / "svd_model.pkl")
            
            # Also save model metadata
            metadata = {
                "n_components": self.n_components_svd,
                "explained_variance_ratio": self.svd.explained_variance_ratio_.tolist(),
                "explained_variance": self.svd.explained_variance_.tolist(),
                "singular_values": self.svd.singular_values_.tolist(),
                "n_features": self.svd.n_features_in_,
                "random_state": self.random_state
            }
            
            with open(path / "svd_metadata.json", 'w') as f:
                import json
                json.dump(metadata, f, indent=4)
            
            logger.info("SVD model and metadata saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving SVD model: {str(e)}", exc_info=True)
            raise

    def load_svd_model(self, path: str) -> None:
        """Load SVD model from disk."""
        logger.info(f"Loading SVD model from: {path}")
        path = Path(path)
        
        try:
            self.svd = joblib.load(path / "svd_model.pkl")
            self.is_fitted = True
            
            logger.info("SVD model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading SVD model: {str(e)}", exc_info=True)
            raise
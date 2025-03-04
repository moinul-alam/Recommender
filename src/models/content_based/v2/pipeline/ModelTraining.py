import numpy as np
import pandas as pd
import gc
import logging
import memory_profiler
from pathlib import Path
import psutil
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, IncrementalPCA, PCA
import joblib
from scipy import sparse
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_engineering.log'),
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
        """Initialize feature engineering with configurable parameters."""
        logger.info("Initializing FeatureEngineering")
        
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
        
        logger.info("Model Training initialized successfully")
        logger.info(f"Memory usage after initialization: {memory_profiler.memory_usage()[0]} MB")


    def train_model(self) -> pd.DataFrame:
        """Transform features using fitted transformers."""
        required_columns = ['tmdb_id', 'overview', 'genres', 'cast', 'director', 'keywords']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        if not self.is_fitted:
            raise ValueError("Transformers must be fitted before transformation")
            
            
            df = self.svd.transform(df)
            
            final_features.insert(0, 'tmdb_id', df['tmdb_id'])
            
            del pca_features, combined_features
            gc.collect()
            
            return final_features

        except Exception as e:
            logger.error(f"Error transforming features: {str(e)}", exc_info=True)
            raise

    def save_transformers(self, path: str) -> None:
        """Save all transformers to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted transformers")
            
        logger.info(f"Saving transformers to: {path}")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Only save stateful transformers
        transformers = {
            '3_svd': self.svd,
        }
        

    def load_svd_model(self, path: str) -> None:
        """Load all transformers from disk."""
        logger.info(f"Loading transformers from: {path}")
        path = Path(path)
        
        try:
            svd_model = joblib.load(path / "3_svd.pkl")
            self.svd = svd_model
            
            logger.info("Transformers loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading transformers: {str(e)}", exc_info=True)
            raise
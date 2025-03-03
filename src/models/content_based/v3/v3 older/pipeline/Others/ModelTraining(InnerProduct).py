import os
import faiss
import pandas as pd
import gc
import logging
import numpy as np
from sklearn.preprocessing import normalize
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ModelTraining:
    def __init__(self, feature_matrix: pd.DataFrame, model_path: str):
        """
        Initialize model training with feature matrix and configuration.
        Uses Inner Product (cosine similarity) for content-based recommendations.
        
        Args:
            feature_matrix: DataFrame containing tmdb_id and feature columns
            model_path: Path to save the FAISS index
        """
        self.feature_matrix = feature_matrix
        self.model_path = model_path
        
    def validate_feature_matrix(self) -> List[str]:
        """
        Ensure the feature matrix contains required columns and validate data quality.
        
        Returns:
            List of feature column names
        """
        required_columns = {"tmdb_id"}
        feature_columns = set(self.feature_matrix.columns) - required_columns
        
        missing_columns = required_columns - set(self.feature_matrix.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if not feature_columns:
            raise ValueError("No feature columns found besides tmdb_id")
            
        feature_cols_list = list(feature_columns)
        
        # Additional validations
        if self.feature_matrix['tmdb_id'].duplicated().any():
            raise ValueError("Duplicate tmdb_id values found")
            
        if self.feature_matrix[feature_cols_list].isnull().values.any():
            raise ValueError("Feature matrix contains NaN values. Ensure preprocessing removes missing data.")
            
        return feature_cols_list

    def model_training(self) -> str:
        """
        Train the FAISS model using Inner Product (cosine similarity) and save the index.
        
        Returns:
            Path to the saved model
        """
        try:
            feature_columns = self.validate_feature_matrix()
            
            # Prepare feature matrix
            media_features = self.feature_matrix[feature_columns].astype("float32")
            dimension = media_features.shape[1]
            
            logger.info(f"Feature matrix shape: {media_features.shape}")
            logger.info(f"Number of dimensions: {dimension}")

            # Convert to numpy array and check for inf/nan
            feature_array = media_features.to_numpy()
            if not np.isfinite(feature_array).all():
                raise ValueError("Feature matrix contains infinite or NaN values")

            # Normalize vectors for cosine similarity
            logger.info("Normalizing feature vectors for cosine similarity")
            feature_array = normalize(feature_array, norm='l2', axis=1)
            
            # Verify normalization
            norms = np.linalg.norm(feature_array, axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-5):
                logger.warning("Not all vectors are properly normalized")
                logger.debug(f"Norm range: min={norms.min():.6f}, max={norms.max():.6f}")

            # Initialize FAISS index with Inner Product
            logger.info("Creating FAISS IndexFlatIP for cosine similarity")
            faiss_index = faiss.IndexFlatIP(dimension)

            # Add features to the FAISS index
            logger.info("Adding normalized feature vectors to FAISS index")
            faiss_index.add(feature_array)
            
            # Verify index size
            logger.info(f"Number of vectors in index: {faiss_index.ntotal}")
            if faiss_index.ntotal != len(self.feature_matrix):
                raise ValueError("Number of vectors in index doesn't match feature matrix length")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save the FAISS index
            faiss.write_index(faiss_index, self.model_path)
            logger.info(f"FAISS model saved to: {self.model_path}")
            
            return self.model_path
                    
        except Exception as e:
            logger.error(f"Error during FAISS model training: {str(e)}", exc_info=True)
            raise

        finally:
            # Clean up memory
            gc.collect()

    def apply_model_training(self) -> str:
        """
        Apply FAISS model training.
        
        Returns:
            Path to the saved model
        """
        return self.model_training()
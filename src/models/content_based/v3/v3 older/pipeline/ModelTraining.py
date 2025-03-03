import os
import faiss
import pandas as pd
import numpy as np
import logging
from typing import List
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

class ModelTraining:
    def __init__(self, feature_matrix: pd.DataFrame, model_path: str):
        """
        Initialize model training with feature matrix and path.
        
        Args:
            feature_matrix: DataFrame containing tmdb_id and engineered feature columns
            model_path: Path to save the FAISS index
        """
        self.feature_matrix = feature_matrix
        self.model_path = model_path
        
    def validate_feature_matrix(self) -> List[str]:
        """
        Validate feature matrix structure and content.
        
        Returns:
            List of feature column names
        """
        # Verify tmdb_id column exists
        if 'tmdb_id' not in self.feature_matrix.columns:
            raise ValueError("Feature matrix must contain 'tmdb_id' column")
            
        # Get feature columns (all except tmdb_id)
        feature_columns = [col for col in self.feature_matrix.columns if col != 'tmdb_id']
        
        if not feature_columns:
            raise ValueError("No feature columns found besides tmdb_id")
            
        # Check for duplicate tmdb_ids
        if self.feature_matrix['tmdb_id'].duplicated().any():
            raise ValueError("Duplicate tmdb_id values found")
            
        # Verify all feature columns are numeric
        non_numeric_cols = [
            col for col in feature_columns 
            if not np.issubdtype(self.feature_matrix[col].dtype, np.number)
        ]
        if non_numeric_cols:
            raise ValueError(f"Non-numeric columns found: {non_numeric_cols}")
            
        # Check for NaN or infinite values
        if self.feature_matrix[feature_columns].isna().any().any():
            raise ValueError("Feature matrix contains NaN values")
            
        if np.isinf(self.feature_matrix[feature_columns].values).any():
            raise ValueError("Feature matrix contains infinite values")
            
        return feature_columns

    def model_training(self) -> str:
        """
        Train FAISS model using normalized feature vectors.
        
        Returns:
            Path to saved model
        """
        try:
            # Validate features
            feature_columns = self.validate_feature_matrix()
            
            # Extract features and convert to float32
            features = self.feature_matrix[feature_columns].values.astype('float32')
            
            # Normalize features for cosine similarity
            features = normalize(features, norm='l2', axis=1)
            
            # Create FAISS index (using inner product which is equivalent to cosine similarity 
            # for normalized vectors)
            dimension = len(feature_columns)
            index = faiss.IndexFlatIP(dimension)
            
            # Add vectors to index
            index.add(features)
            
            # Verify index size
            if index.ntotal != len(self.feature_matrix):
                raise ValueError("Number of vectors in index doesn't match feature matrix length")
            
            # Create directory if needed
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save index
            faiss.write_index(index, self.model_path)
            logger.info(f"FAISS model saved to: {self.model_path}")
            
            return self.model_path
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
            
    def apply_model_training(self) -> str:
        """
        Apply model training process.
        
        Returns:
            Path to saved model
        """
        return self.model_training()
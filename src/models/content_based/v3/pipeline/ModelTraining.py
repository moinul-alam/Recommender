import os
import faiss
import pandas as pd
import gc
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ModelTraining:
    def __init__(self, feature_matrix: pd.DataFrame, model_path: str):
        self.feature_matrix = feature_matrix
        self.model_path = model_path
        
    def validate_feature_matrix(self):
        """Ensure the feature matrix contains required columns and no NaN values."""
        required_columns = {"item_id"}
        missing_columns = required_columns - set(self.feature_matrix.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for NaN values in feature columns
        feature_columns = [col for col in self.feature_matrix.columns if col != "item_id"]
        if self.feature_matrix[feature_columns].isnull().values.any():
            raise ValueError("Feature matrix contains NaN values. Ensure preprocessing removes missing data.")


    def model_training(self):
        """Train the FAISS model and save the index."""
        try:
            # Validate the input feature matrix
            self.validate_feature_matrix()
            
            # Prepare feature matrix
            media_features = self.feature_matrix.drop("item_id", axis=1).astype("float32")
            dimension = media_features.shape[1]

            logger.info("Initializing FAISS index and Adding feature vectors")
            faiss_index = faiss.IndexFlatIP(dimension)
            faiss_index.add(media_features.values)
            
            logger.info("Saving the FAISS Index")
            faiss.write_index(faiss_index, self.model_path)
            logger.info(f"FAISS model saved to: {self.model_path}")
                    
        except Exception as e:
            logger.error(f"Error during FAISS model training: {str(e)}", exc_info=True)
            raise

        finally:
            # Clean up memory
            gc.collect()

    def apply_model_training(self):
        """Apply FAISS model training."""
        self.model_training()
        return self.model_path

import os
import faiss
import pandas as pd
import gc
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ModelTraining:
    def __init__(self, feature_matrix: pd.DataFrame, model_path: str, metric: str = "L2"):
        self.feature_matrix = feature_matrix
        self.model_path = model_path
        self.metric = metric
        
    def validate_feature_matrix(self):
        """Ensure the feature matrix contains required columns."""
        required_columns = {"tmdb_id"}
        feature_columns = set(self.feature_matrix.columns) - required_columns
        
        missing_columns = required_columns - set(self.feature_matrix.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert feature_columns set to list for indexing
        if self.feature_matrix[list(feature_columns)].isnull().values.any():
            raise ValueError("Feature matrix contains NaN values. Ensure preprocessing removes missing data.")


    def model_training(self):
        """Train the FAISS model and save the index."""
        try:
            # Validate the input feature matrix
            self.validate_feature_matrix()
            
            # Prepare feature matrix
            media_features = self.feature_matrix.drop("tmdb_id", axis=1).astype("float32")
            dimension = media_features.shape[1]

            # Initialize FAISS index based on the metric
            if self.metric == "L2":
                faiss_index = faiss.IndexFlatL2(dimension)
            elif self.metric == "Inner Product":
                faiss_index = faiss.IndexFlatIP(dimension)
            else:
                raise ValueError(f"Unsupported FAISS metric: {self.metric}")
            
            # Add features to the FAISS index
            logger.info("Adding feature vectors to FAISS index")
            faiss_index.add(media_features.values)
            
            # Save the FAISS index
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

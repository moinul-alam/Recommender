import os
import faiss
import pandas as pd
import gc
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class IndexCreation:
    def __init__(self, feature_matrix: pd.DataFrame):
        self.feature_matrix = feature_matrix

    def model_training(self):
        """Train the FAISS model and save the index."""
        try:          
            item_features = self.feature_matrix.drop("item_id", axis=1).astype("float32")
            dimension = item_features.shape[1]

            logger.info("Initializing FAISS index and Adding feature vectors")
            faiss_index = faiss.IndexFlatIP(dimension)
            faiss_index.add(item_features.values)
            
            return faiss_index
                    
        except Exception as e:
            logger.error(f"Error during FAISS model training: {str(e)}", exc_info=True)
            raise

        finally:
            gc.collect()
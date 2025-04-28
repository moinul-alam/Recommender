
# IndexCreation with optimized FAISS for maximum accuracy
import os
import faiss
import pandas as pd
import numpy as np
import gc
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class IndexCreation:
    def __init__(self, feature_matrix: pd.DataFrame):
        self.feature_matrix = feature_matrix

    def model_training(self):
        """Train a high-accuracy FAISS model using HNSW algorithm and save the index."""
        try:          
            # Extract item IDs and feature vectors
            item_ids = self.feature_matrix["item_id"]
            item_features = self.feature_matrix.drop("item_id", axis=1).astype("float32").values
            dimension = item_features.shape[1]

            logger.info(f"Initializing FAISS HNSW index with dimension {dimension}")
            
            # Normalize vectors for cosine similarity
            # This step is crucial for accuracy when using inner product
            faiss.normalize_L2(item_features)
            
            # Use HNSW algorithm in FAISS for high accuracy and good query performance
            # M: number of connections per layer (higher = better accuracy, more memory)
            # efConstruction: build-time accuracy parameter
            # efSearch: query-time accuracy parameter (can be adjusted later)
            M = 32
            efConstruction = 200
            
            # Create HNSW index with inner product (for cosine similarity with normalized vectors)
            faiss_index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
            faiss_index.hnsw.efConstruction = efConstruction
            faiss_index.hnsw.efSearch = 100  # High accuracy for queries
            
            logger.info("Adding feature vectors to FAISS HNSW index")
            faiss_index.add(item_features)
            
            logger.info("FAISS HNSW index created successfully")
            
            # Optional: Create ID mapping if needed later
            # item_id_map = dict(zip(range(len(item_ids)), item_ids))
            
            return faiss_index
                    
        except Exception as e:
            logger.error(f"Error during FAISS model training: {str(e)}", exc_info=True)
            raise

        finally:
            gc.collect()
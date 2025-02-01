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

    def model_training(self):
        # Check for required columns
        required_columns = {"tmdbId"}
        missing_columns = required_columns - set(self.feature_matrix.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Prepare feature matrix
        media_features = self.feature_matrix.drop("tmdbId", axis=1).astype("float32")
        dimension = media_features.shape[1]

        # Initialize FAISS index based on the metric
        if self.metric == "L2":
            faiss_index = faiss.IndexFlatL2(dimension)
        elif self.metric == "Inner Product":
            faiss_index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unsupported FAISS metric: {self.metric}")

        # Add features to the FAISS index
        faiss_index.add(media_features.values)

        # Save the FAISS index
        faiss.write_index(faiss_index, self.model_path)
        logger.info(f"FAISS index saved to {self.model_path}")

        # Clean up memory
        gc.collect()

    def apply_model_training(self):
        self.model_training()
        return self.model_path

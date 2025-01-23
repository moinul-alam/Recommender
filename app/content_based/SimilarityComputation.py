import os
from pathlib import Path
import faiss
import pandas as pd
import numpy as np
import gc
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SimilarityComputation:
    def __init__(self, save_path: str, metric: str = "L2"):
        self.save_path = Path(save_path)
        self.metric = metric

    def build_index(self, data: pd.DataFrame):
        required_columns = {"tmdbId"}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        media_features = data.drop("tmdbId", axis=1).astype("float32")
        dimension = media_features.shape[1]

        if self.metric == "L2":
            faiss_index = faiss.IndexFlatL2(dimension)
        elif self.metric == "Inner Product":
            faiss_index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unsupported FAISS metric: {self.metric}")

        faiss_index.add(media_features.values)

        faiss.write_index(faiss_index, str(self.save_path))
        logger.info(f"FAISS index saved to {self.save_path}")

        gc.collect()

        return str(self.save_path)


import os
from typing import Dict, List, Optional
import faiss
import numpy as np
import pandas as pd
import logging
from fastapi import HTTPException
from app.content_based.DataPreprocessing import DataPreprocessing

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    def __init__(self, tmdbId: int, metadata: Optional[Dict], index_path: str, features_path: str, n_items: int = 10):
        self.tmdbId = tmdbId
        self.metadata = metadata
        self.index_path = index_path
        self.features_path = features_path
        self.n_items = n_items

    def load_data(self) -> pd.DataFrame:
        try:
            if not os.path.isfile(self.features_path):
                raise FileNotFoundError(f"Features file not found: {self.features_path}")
            logger.info("Loading feature data...")
            return pd.read_feather(self.features_path)
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")

    def get_recommendation(self) -> List[Dict]:
        try:
            feature_matrix = self.load_data()
            tmdbId_all = feature_matrix.iloc[:, 0].to_numpy()
            features_all = feature_matrix.iloc[:, 1:].to_numpy().astype(np.float32)

            if self.tmdbId in tmdbId_all:
                return self.recommendation_for_existing(tmdbId_all, features_all)
            else:
                return self.recommendation_for_new(tmdbId_all, features_all)
        except Exception as e:
            logger.error(f"Error in get_recommendation: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error generating recommendations: {str(e)}"
            )

    def recommendation_for_existing(self, tmdbId_all: np.ndarray, features_all: np.ndarray) -> List[Dict]:
        try:
            query_idx = np.where(tmdbId_all == self.tmdbId)[0][0]
            query_vector = features_all[query_idx].reshape(1, -1)

            logger.debug(f"Loading FAISS index from {self.index_path}")
            if not os.path.isfile(self.index_path):
                raise FileNotFoundError(f"FAISS index file not found: {self.index_path}")
            index = faiss.read_index(self.index_path)

            distances, indices = index.search(query_vector, self.n_items + 1)

            similar_media = []
            for i, idx in enumerate(indices[0]):
                similar_media_id = int(tmdbId_all[idx])
                if similar_media_id == self.tmdbId:
                    continue
                similarity_score = round(1 / (1 + distances[0][i]), 4)
                percentage_score = f"{similarity_score * 100:.2f}%"
                similar_media.append({'tmdbId': similar_media_id, 'similarity': percentage_score})

            logger.info(f"Found {len(similar_media)} similar media items")
            return similar_media[:self.n_items]
        except IndexError as e:
            logger.error(f"Index error: {str(e)}", exc_info=True)
            raise RuntimeError(f"tmdbId {self.tmdbId} not found in the dataset")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error in getting recommendations: {str(e)}")

    def recommendation_for_new(self, tmdbId_all: np.ndarray, features_all: np.ndarray) -> List[Dict]:
        try:
            logger.info("Generating recommendations for a new item using metadata...")

            similar_media = [
                {'tmdbId': self.tmdbId, 'similarity': '100%'}
            ]
            return similar_media
        except Exception as e:
            logger.error(f"Error in recommendation_for_new: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error generating recommendations for new item: {str(e)}")

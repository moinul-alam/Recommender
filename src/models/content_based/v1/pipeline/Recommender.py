import os
from typing import Dict, List, Optional
import faiss
import numpy as np
import pandas as pd
import logging
from fastapi import HTTPException

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Recommender:
    def __init__(self, tmdbId: int, metadata: Optional[Dict], feature_matrix: pd.DataFrame, model_file: str, n_items: int):
        self.tmdbId = tmdbId
        self.metadata = metadata
        self.feature_matrix = feature_matrix
        self.model_file = model_file
        self.n_items = n_items

    def get_recommendation(self) -> List[Dict]:
        try:
            tmdbId_all = self.feature_matrix.iloc[:, 0].to_numpy()
            features_all = self.feature_matrix.iloc[:, 1:].to_numpy().astype(np.float32)

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

            if not os.path.isfile(self.model_file):
                raise FileNotFoundError(f"FAISS index file not found: {self.model_file}")
            index = faiss.read_index(self.model_file)

            distances, indices = index.search(query_vector, self.n_items + 1)

            similar_media = []
            for i, idx in enumerate(indices[0]):
                similar_media_id = int(tmdbId_all[idx])
                if similar_media_id == self.tmdbId:
                    continue
                similarity_score = round(1 / (1 + distances[0][i]), 4)
                similar_media.append({
                    'tmdbId': similar_media_id, 
                    'similarity': f"{similarity_score * 100:.2f}%"
                })

            logger.info(f"Found {len(similar_media)} similar media items")
            return similar_media[:self.n_items]
        except IndexError:
            logger.error(f"tmdbId {self.tmdbId} not found in the dataset", exc_info=True)
            raise HTTPException(status_code=404, detail=f"tmdbId {self.tmdbId} not found")
        except Exception as e:
            logger.error(f"Error in recommendation_for_existing: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error generating recommendations: {str(e)}")

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

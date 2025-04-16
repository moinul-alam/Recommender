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
    def __init__(self, 
                 item_id: int,
                 item_map: Dict[int, int],
                 feature_matrix: pd.DataFrame,
                 index:faiss.Index,
                 n_recommendations: int):
        self.item_id = item_id
        self.item_map = item_map
        self.feature_matrix = feature_matrix
        self.index = index
        self.n_recommendations = n_recommendations

    def _process_recommendations(self, distances, indices, item_ids) -> List[Dict]:
        """Process FAISS results into a recommendation list with L2 distance scores."""
        similar_items = []
        
        for i, idx in enumerate(indices):
            try:
                idx = int(idx)
                similar_item_id = int(item_ids[idx])
                
                if similar_item_id == self.item_id:
                    continue
   
                similarity_score = distances[i]
                similar_items.append({
                    'item_id': similar_item_id,
                    'similarity': float(similarity_score)
                })
                
            except (IndexError, ValueError) as e:
                logger.warning(f"Invalid index encountered: {idx} - {str(e)}")
                continue

            logger.debug(f"Similar Items: {similar_items}")
        
        logger.info(f"Found {len(similar_items)} similar media items")
        
        return similar_items[:self.n_recommendations]

    def get_recommendation_for_existing(self) -> List[Dict]:
        """Retrieve recommendations for an existing movie in the dataset."""
        try:
            item_ids = self.feature_matrix.iloc[:, 0].to_numpy()
            feature_matrix = self.feature_matrix.iloc[:, 1:].to_numpy().astype(np.float32)

            query_idx = np.where(item_ids == self.item_id)[0][0]
            query_vector = feature_matrix[query_idx].reshape(1, -1)

            k = min(self.n_recommendations + 1, len(item_ids))
            distances, indices = self.index.search(query_vector, k)

            return self._process_recommendations(distances[0], indices[0], item_ids)
        
        except IndexError:
            logger.error(f"item_id {self.item_id} not found in the dataset", exc_info=True)
            raise HTTPException(status_code=404, detail=f"item_id {self.item_id} not found")
        except Exception as e:
            logger.error(f"Error in recommendation_for_existing: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error generating recommendations: {str(e)}")

    def get_recommendation_for_new(self, new_features: pd.DataFrame) -> List[Dict]:
        """Retrieve recommendations for a new movie not in the dataset."""
        try:
            item_ids = self.feature_matrix.iloc[:, 0].to_numpy()

            query_vector = new_features.iloc[:, 1:].to_numpy().astype(np.float32)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

            distances, indices = self.index.search(query_vector, self.n_recommendations)

            indices = [idx.item() for idx in indices[0] if idx < len(item_ids)]
            distances = distances[0][:len(indices)]

            recommendations = self._process_recommendations(
                distances, 
                indices, 
                item_ids
            )

            self._update_feature_matrix_and_index(new_features)

            return recommendations

        except Exception as e:
            logger.error(f"Error in recommendation_for_new: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error generating recommendations: {str(e)}")

    def _update_feature_matrix_and_index(self, new_features: pd.DataFrame):
        """Update the feature matrix and FAISS index with new movie data."""
        try:
            if self.is_custom_query:
                logger.info('Custom Features adding skipped')
                return
            
            logger.info('Updating new feature in progress')
            new_item_id = new_features.iloc[0, 0]
            new_features = new_features.loc[:, ~new_features.columns.str.contains('^Unnamed')]
            new_features = new_features.reset_index(drop=True)

            logger.debug(f"New features columns: {new_features.columns.tolist()}")
            
            if new_item_id not in self.feature_matrix.iloc[:, 0].values:
                self.feature_matrix = pd.concat([self.feature_matrix, new_features], 
                                            ignore_index=True)
                self.feature_matrix.reset_index(drop=True, inplace=True)
                self.feature_matrix.to_feather(self.features_file)

                new_vector = new_features.iloc[:, 1:].to_numpy().astype(np.float32)
                if new_vector.ndim == 1:
                    new_vector = new_vector.reshape(1, -1)

                if new_vector.size == 0 or np.isnan(new_vector).any():
                    raise ValueError(f"Invalid new vector for item_id {new_item_id}")

                if self.index is None:
                    raise RuntimeError("FAISS index is not initialized.")

                self.index.add(new_vector)
                faiss.write_index(self.index, self.model_file)

                logger.info(f"Successfully added new vector for item_id {new_item_id} to index")
            else:
                logger.info(f"item_id {new_item_id} already exists in index")

        except Exception as e:
            logger.error(f"Error updating feature matrix and index: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error updating index: {str(e)}")
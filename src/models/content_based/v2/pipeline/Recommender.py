import os
from pathlib import Path
from typing import Dict, List, Optional
import faiss
import numpy as np
import pandas as pd
import logging
from fastapi import HTTPException

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Recommender:
    def __init__(self, tmdb_id: int, metadata: Optional[Dict], features_file: str, model_file: str, n_items: int):
        self.tmdb_id = tmdb_id
        self.metadata = metadata
        self.features_file = features_file
        self.model_file = model_file
        self.n_items = n_items
        self.index = None
        self._load_index()
        self._load_feature_matrix()

    def _load_index(self):
        """Load FAISS index from file."""
        if not os.path.isfile(self.model_file):
            raise FileNotFoundError(f"FAISS index file not found: {self.model_file}")
        
        try:
            self.index = faiss.read_index(self.model_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {str(e)}")
    
    def _load_feature_matrix(self):
        """Load feature matrix from feather file."""
        if not os.path.isfile(self.features_file):
            raise FileNotFoundError(f"Feature Matrix file not found: {self.features_file}")
        
        try:
            # Load and explicitly reset index
            self.feature_matrix = pd.read_feather(self.features_file)
            self.feature_matrix = self.feature_matrix.reset_index(drop=True)
            
            # Verify the structure
            if self.feature_matrix.empty:
                raise ValueError("Loaded feature matrix is empty")
                
            # Add logging to debug column names
            logger.debug(f"Feature matrix columns: {self.feature_matrix.columns.tolist()}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Feature Matrix: {str(e)}")


    def get_recommendation_for_existing(self) -> List[Dict]:
        """Retrieve recommendations for an existing movie in the dataset."""
        try:
            tmdb_id_all = self.feature_matrix.iloc[:, 0].to_numpy()
            features_all = self.feature_matrix.iloc[:, 1:].to_numpy().astype(np.float32)

            query_idx = np.where(tmdb_id_all == self.tmdb_id)[0][0]
            query_vector = features_all[query_idx].reshape(1, -1)

            distances, indices = self.index.search(query_vector, self.n_items + 1)

            return self._process_recommendations_safe(distances[0], indices[0], tmdb_id_all)

        except IndexError:
            logger.error(f"tmdb_id {self.tmdb_id} not found in the dataset", exc_info=True)
            raise HTTPException(status_code=404, detail=f"tmdb_id {self.tmdb_id} not found")
        except Exception as e:
            logger.error(f"Error in recommendation_for_existing: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error generating recommendations: {str(e)}")

    def get_recommendation_for_new(self, new_features: pd.DataFrame) -> List[Dict]:
        """Retrieve recommendations for a new movie not in the dataset."""
        try:
            tmdb_id_all = self.feature_matrix.iloc[:, 0].to_numpy()
            features_all = self.feature_matrix.iloc[:, 1:].to_numpy().astype(np.float32)

            query_vector = new_features.iloc[:, 1:].to_numpy().astype(np.float32)
            
            if query_vector.ndim == 1:  # Ensure query is 2D
                query_vector = query_vector.reshape(1, -1)

            distances, indices = self.index.search(query_vector, self.n_items)

            safe_indices = [idx.item() for idx in indices[0] if idx < len(tmdb_id_all)]
            safe_distances = distances[0][:len(safe_indices)]

            recommendations = self._process_recommendations_safe(
                safe_distances, 
                safe_indices, 
                tmdb_id_all
            )

            self._update_feature_matrix_and_index(new_features)

            return recommendations

        except Exception as e:
            logger.error(f"Error in recommendation_for_new: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error generating recommendations: {str(e)}")

    def _process_recommendations_safe(self, distances, indices, tmdb_id_all) -> List[Dict]:
        """Process FAISS results into a recommendation list with safety checks."""
        similar_media = []
        for i, idx in enumerate(indices):
            try:
                idx = int(idx)  # Ensure it's an integer
                similar_media_id = int(tmdb_id_all[idx])

                if similar_media_id == self.tmdb_id:
                    continue

                similarity_score = round(1 / (1 + distances[i]), 4)
                similar_media.append({
                    'tmdb_id': similar_media_id, 
                    'similarity': f"{similarity_score * 100:.2f}%"
                })
            except (IndexError, ValueError) as e:
                logger.warning(f"Invalid index encountered: {idx} - {str(e)}")
                continue

        return similar_media[:self.n_items]

    def _update_feature_matrix_and_index(self, new_features: pd.DataFrame):
        """Update the feature matrix and FAISS index with new movie data."""
     
        try:
            new_tmdb_id = new_features.iloc[0, 0]
            new_features = new_features.loc[:, ~new_features.columns.str.contains('^Unnamed')]
            # Ensure new_features has the same structure
            new_features = new_features.reset_index(drop=True)

            logger.debug(f"New features columns: {new_features.columns.tolist()}")
            
            if new_tmdb_id not in self.feature_matrix.iloc[:, 0].values:
                # Append new features
                self.feature_matrix = pd.concat([self.feature_matrix, new_features], 
                                            ignore_index=True)  # Use ignore_index=True
                self.feature_matrix.reset_index(drop=True, inplace=True)
                self.feature_matrix.to_feather(self.features_file)

                # Convert the new vector
                new_vector = new_features.iloc[:, 1:].to_numpy().astype(np.float32)

                if new_vector.ndim == 1:
                    new_vector = new_vector.reshape(1, -1)  # Ensure 2D shape

                if new_vector.size == 0 or np.isnan(new_vector).any():
                    raise ValueError(f"Invalid new vector for tmdb_id {new_tmdb_id}")

                if self.index is None:
                    raise RuntimeError("FAISS index is not initialized.")

                # Add the new vector to FAISS
                self.index.add(new_vector)

                # Save the updated FAISS index
                faiss.write_index(self.index, self.model_file)

                logger.info(f"Successfully added new vector for tmdb_id {new_tmdb_id} to index")
            else:
                logger.info(f"tmdb_id {new_tmdb_id} already exists in index")

        except Exception as e:
            logger.error(f"Error updating feature matrix and index: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error updating index: {str(e)}")

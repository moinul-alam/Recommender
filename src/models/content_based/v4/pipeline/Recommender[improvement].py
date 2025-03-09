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
    def __init__(self, item_id: int, 
                 metadata: Optional[Dict], 
                 features_file: str, 
                 model_file: str, 
                 n_items: int,
                 is_custom_query: Optional[bool] = False):
        self.item_id = item_id
        self.metadata = metadata
        self.features_file = features_file
        self.model_file = model_file
        self.n_items = n_items
        self.is_custom_query = is_custom_query
        self.index = None
        self._load_index()
        self._load_feature_matrix()
        
        # Determine index type (IP or L2)
        self.is_ip_index = isinstance(self.index, faiss.IndexFlatIP) or "IP" in self.index.__class__.__name__
        logger.info(f"Index type: {'Inner Product' if self.is_ip_index else 'L2'}")

    def _load_index(self):
        """Load FAISS index from file."""
        if not os.path.isfile(self.model_file):
            raise FileNotFoundError(f"FAISS index file not found: {self.model_file}")
        
        try:
            logger.info(f"Loading FAISS index from file: {self.model_file}")
            self.index = faiss.read_index(self.model_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {str(e)}")
    
    def _load_feature_matrix(self):
        """Load feature matrix from feather file."""
        if not os.path.isfile(self.features_file):
            raise FileNotFoundError(f"Feature Matrix file not found: {self.features_file}")
        
        try:
            logger.info(f"Loading Feature Matrix from file: {self.features_file}")
            self.feature_matrix = pd.read_feather(self.features_file)
            self.feature_matrix = self.feature_matrix.reset_index(drop=True)
            
            if self.feature_matrix.empty:
                raise ValueError("Loaded feature matrix is empty")
                
            logger.debug(f"Feature matrix columns: {self.feature_matrix.columns.tolist()}")
            logger.debug(f"Feature matrix shape: {self.feature_matrix.shape}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Feature Matrix: {str(e)}")

    def _normalize_vectors(self, vectors):
        """Normalize vectors for IP similarity if needed."""
        if self.is_ip_index:
            # Compute L2 norm of each vector
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            # Replace zero norms with 1 to avoid division by zero
            norms[norms == 0] = 1.0
            # Normalize vectors
            return vectors / norms
        return vectors

    def _convert_distance_to_similarity(self, distances):
        """Convert distances to similarity scores based on index type."""
        if self.is_ip_index:
            # For IP index, higher is better (already similarity)
            # Ensure values are between 0 and 1
            return np.clip(distances, 0, 1)
        else:
            # For L2 index, lower is better (distance)
            # Convert to similarity: exp(-distance)
            return np.exp(-distances)

    def _process_recommendations(self, distances, indices, item_id_all) -> List[Dict]:
        """Process FAISS results into a recommendation list with similarity scores."""
        similar_media = []
        
        # Convert distances to similarity scores
        similarity_scores = self._convert_distance_to_similarity(distances)
        
        for i, idx in enumerate(indices):
            try:
                idx = int(idx)
                if idx < 0 or idx >= len(item_id_all):
                    logger.warning(f"Index out of bounds: {idx}, max allowed: {len(item_id_all)-1}")
                    continue
                    
                similar_media_id = int(item_id_all[idx])
                
                if similar_media_id == self.item_id:
                    continue
   
                similarity_score = similarity_scores[i]
                similar_media.append({
                    'item_id': similar_media_id,
                    'similarity': f"{similarity_score * 100:.2f}%",
                    'raw_score': float(similarity_score)
                })
            except (IndexError, ValueError) as e:
                logger.warning(f"Invalid index encountered: {idx} - {str(e)}")
                continue

        logger.info(f"Found {len(similar_media)} similar media items")
        
        # Sort by similarity score (descending)
        similar_media.sort(key=lambda x: x['raw_score'], reverse=True)
        
        return similar_media[:self.n_items]

    def get_recommendation_for_existing(self) -> List[Dict]:
        """Retrieve recommendations for an existing movie in the dataset."""
        try:
            item_id_all = self.feature_matrix.iloc[:, 0].to_numpy()
            features_all = self.feature_matrix.iloc[:, 1:].to_numpy().astype(np.float32)
            
            # Find the index of the query item
            query_indices = np.where(item_id_all == self.item_id)[0]
            if len(query_indices) == 0:
                raise IndexError(f"item_id {self.item_id} not found in the dataset")
            
            query_idx = query_indices[0]
            query_vector = features_all[query_idx].reshape(1, -1)
            
            # Normalize vectors if using IP index
            if self.is_ip_index and not isinstance(self.index, faiss.IndexPreTransform):
                query_vector = self._normalize_vectors(query_vector)

            # Search for similar items
            k = min(self.n_items + 1, len(item_id_all))
            distances, indices = self.index.search(query_vector, k)

            return self._process_recommendations(distances[0], indices[0], item_id_all)
        
        except IndexError as e:
            logger.error(f"item_id {self.item_id} not found in the dataset: {str(e)}", exc_info=True)
            raise HTTPException(status_code=404, detail=f"item_id {self.item_id} not found")
        except Exception as e:
            logger.error(f"Error in recommendation_for_existing: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error generating recommendations: {str(e)}")

    def get_recommendation_for_new(self, new_features: pd.DataFrame) -> List[Dict]:
        """Retrieve recommendations for a new movie not in the dataset."""
        try:
            item_id_all = self.feature_matrix.iloc[:, 0].to_numpy()

            # Extract the feature vector
            query_vector = new_features.iloc[:, 1:].to_numpy().astype(np.float32)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
                
            # Check vector dimensions match the index
            if query_vector.shape[1] != self.index.d:
                raise ValueError(f"Feature vector dimension {query_vector.shape[1]} does not match index dimension {self.index.d}")
            
            # Normalize vectors if using IP index
            if self.is_ip_index and not isinstance(self.index, faiss.IndexPreTransform):
                query_vector = self._normalize_vectors(query_vector)

            # Search for similar items
            distances, indices = self.index.search(query_vector, self.n_items)

            # Filter out invalid indices
            safe_indices = [idx for idx in indices[0] if 0 <= idx < len(item_id_all)]
            safe_distances = distances[0][:len(safe_indices)]

            recommendations = self._process_recommendations(
                safe_distances, 
                safe_indices, 
                item_id_all
            )

            # Update feature matrix and index with new item
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
            
            # Clean up column names and reset index
            new_features = new_features.loc[:, ~new_features.columns.str.contains('^Unnamed')]
            new_features = new_features.reset_index(drop=True)

            logger.debug(f"New features columns: {new_features.columns.tolist()}")
            
            if new_item_id not in self.feature_matrix.iloc[:, 0].values:
                # Verify column alignment
                if list(new_features.columns) != list(self.feature_matrix.columns):
                    logger.warning(f"Column mismatch. Feature matrix: {self.feature_matrix.columns.tolist()}, New features: {new_features.columns.tolist()}")
                    
                    # Attempt to align columns
                    shared_columns = [col for col in self.feature_matrix.columns if col in new_features.columns]
                    if shared_columns:
                        new_features = new_features[shared_columns]
                        logger.info(f"Aligned to shared columns: {shared_columns}")
                
                # Add to feature matrix
                self.feature_matrix = pd.concat([self.feature_matrix, new_features], 
                                            ignore_index=True)
                self.feature_matrix.reset_index(drop=True, inplace=True)
                self.feature_matrix.to_feather(self.features_file)

                # Prepare vector for FAISS
                new_vector = new_features.iloc[:, 1:].to_numpy().astype(np.float32)
                if new_vector.ndim == 1:
                    new_vector = new_vector.reshape(1, -1)

                # Validate vector
                if new_vector.size == 0:
                    raise ValueError(f"Empty vector for item_id {new_item_id}")
                if np.isnan(new_vector).any():
                    logger.warning(f"Vector contains NaN values for item_id {new_item_id}")
                    # Replace NaN with zeros
                    new_vector = np.nan_to_num(new_vector, nan=0.0)

                if self.index is None:
                    raise RuntimeError("FAISS index is not initialized.")
                    
                # Normalize vector if using IP index
                if self.is_ip_index and not isinstance(self.index, faiss.IndexPreTransform):
                    new_vector = self._normalize_vectors(new_vector)

                # Add to index
                self.index.add(new_vector)
                faiss.write_index(self.index, self.model_file)

                logger.info(f"Successfully added new vector for item_id {new_item_id} to index")
            else:
                logger.info(f"item_id {new_item_id} already exists in index")

        except Exception as e:
            logger.error(f"Error updating feature matrix and index: {str(e)}", exc_info=True)
            # Log error but don't raise - this is non-critical functionality
            logger.warning(f"Failed to update index with new item. This won't affect current recommendations.")
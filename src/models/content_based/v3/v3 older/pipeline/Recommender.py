import os
from pathlib import Path
from typing import Dict, List, Optional
import faiss
import numpy as np
import pandas as pd
import logging
from fastapi import HTTPException
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Recommender:
    def __init__(self, 
                 tmdb_id: int, 
                 metadata: Optional[Dict], 
                 features_file: str, 
                 model_file: str, 
                 n_items: int,
                 is_custom_query: Optional[bool] = False):
        """
        Initialize recommender with model and feature data.
        
        Args:
            tmdb_id: ID of the movie to get recommendations for
            metadata: Optional metadata for the movie
            features_file: Path to engineered features feather file
            model_file: Path to FAISS index file
            n_items: Number of recommendations to return
            is_custom_query: Whether this is a custom query for a new movie
        """
        self.tmdb_id = tmdb_id
        self.metadata = metadata
        self.features_file = Path(features_file)
        self.model_file = Path(model_file)
        self.n_items = n_items
        self.is_custom_query = is_custom_query
        
        # Load data
        self._load_data()

    def _load_data(self) -> None:
        """Load FAISS index and feature matrix."""
        try:
            # Load FAISS index
            if not self.model_file.is_file():
                raise FileNotFoundError(f"FAISS index not found: {self.model_file}")
            self.index = faiss.read_index(str(self.model_file))
            
            # Load feature matrix
            if not self.features_file.is_file():
                raise FileNotFoundError(f"Feature matrix not found: {self.features_file}")
            self.feature_matrix = pd.read_feather(self.features_file)
            self.feature_matrix = self.feature_matrix.reset_index(drop=True)
            
            if self.feature_matrix.empty:
                raise ValueError("Feature matrix is empty")
                
            # Validate feature matrix structure
            if 'tmdb_id' not in self.feature_matrix.columns:
                raise ValueError("Feature matrix must contain tmdb_id column")
                
            logger.debug(f"Feature matrix columns: {self.feature_matrix.columns.tolist()}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise RuntimeError(f"Failed to load recommendation data: {str(e)}")

    def _get_feature_vector(self, features: pd.DataFrame) -> np.ndarray:
        """
        Process feature vector for querying.
        
        Args:
            features: DataFrame containing feature vector
            
        Returns:
            Normalized feature vector as numpy array
        """
        # Extract feature columns (excluding tmdb_id)
        feature_cols = [col for col in features.columns if col != 'tmdb_id']
        
        # Convert to numpy and ensure float32
        vector = features[feature_cols].values.astype(np.float32)
        
        # Reshape if needed
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
            
        # Normalize for cosine similarity
        return normalize(vector, norm='l2', axis=1)

    def _convert_distances_to_similarities(self, distances: np.ndarray) -> np.ndarray:
        """
        Convert cosine distances to similarity percentages.
        
        Args:
            distances: Array of cosine distances from FAISS
            
        Returns:
            Array of similarity scores
        """
        # For cosine similarity, convert inner product to percentage
        return (distances + 1) / 2 * 100

    def _process_recommendations(self, 
                               distances: np.ndarray, 
                               indices: np.ndarray, 
                               tmdb_ids: np.ndarray) -> List[Dict]:
        """
        Process FAISS results into recommendation format.
        
        Args:
            distances: Array of distances from FAISS
            indices: Array of indices from FAISS
            tmdb_ids: Array of all tmdb_ids
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        similarities = self._convert_distances_to_similarities(distances)
        
        for i, idx in enumerate(indices):
            try:
                idx = int(idx)
                similar_id = int(tmdb_ids[idx])
                
                # Skip if it's the query movie itself
                if similar_id == self.tmdb_id:
                    continue
                    
                recommendations.append({
                    'tmdb_id': similar_id,
                    'similarity': f"{similarities[i]:.2f}%",
                    'raw_score': float(distances[i])
                })
            except (IndexError, ValueError) as e:
                logger.warning(f"Invalid index encountered: {idx} - {str(e)}")
                continue
                
        return recommendations[:self.n_items]

    def get_recommendation_for_existing(self) -> List[Dict]:
        """Get recommendations for existing movie."""
        try:
            # Get tmdb_ids and feature vectors
            tmdb_ids = self.feature_matrix['tmdb_id'].values
            features = self.feature_matrix[
                [col for col in self.feature_matrix.columns if col != 'tmdb_id']
            ]
            
            # Find query movie index
            query_idx = np.where(tmdb_ids == self.tmdb_id)[0]
            if len(query_idx) == 0:
                raise HTTPException(
                    status_code=404, 
                    detail=f"tmdb_id {self.tmdb_id} not found"
                )
                
            # Get and normalize query vector
            query_vector = self._get_feature_vector(features.iloc[query_idx])
            
            # Search index
            distances, indices = self.index.search(query_vector, self.n_items + 1)
            
            return self._process_recommendations(distances[0], indices[0], tmdb_ids)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise RuntimeError(f"Failed to generate recommendations: {str(e)}")

    def get_recommendation_for_new(self, new_features: pd.DataFrame) -> List[Dict]:
        """Get recommendations for new movie."""
        try:
            # Get all tmdb_ids
            tmdb_ids = self.feature_matrix['tmdb_id'].values
            
            # Process query vector
            query_vector = self._get_feature_vector(new_features)
            
            # Search index
            distances, indices = self.index.search(query_vector, self.n_items)
            
            # Filter valid indices
            valid_mask = indices[0] < len(tmdb_ids)
            valid_indices = indices[0][valid_mask]
            valid_distances = distances[0][valid_mask]
            
            # Get recommendations
            recommendations = self._process_recommendations(
                valid_distances, 
                valid_indices, 
                tmdb_ids
            )
            
            # Update feature matrix and index if not custom query
            if not self.is_custom_query:
                self._update_feature_matrix_and_index(new_features)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise RuntimeError(f"Failed to generate recommendations: {str(e)}")

    def _update_feature_matrix_and_index(self, new_features: pd.DataFrame) -> None:
        """Update feature matrix and index with new movie."""
        try:
            new_tmdb_id = new_features.iloc[0, 0]
            
            # Skip if movie already exists
            if new_tmdb_id in self.feature_matrix['tmdb_id'].values:
                logger.info(f"tmdb_id {new_tmdb_id} already exists in index")
                return
                
            # Clean up new features
            new_features = new_features.loc[:, ~new_features.columns.str.contains('^Unnamed')]
            new_features = new_features.reset_index(drop=True)
            
            # Update feature matrix
            self.feature_matrix = pd.concat(
                [self.feature_matrix, new_features], 
                ignore_index=True
            )
            self.feature_matrix.to_feather(self.features_file)
            
            # Update index
            new_vector = self._get_feature_vector(new_features)
            self.index.add(new_vector)
            faiss.write_index(self.index, str(self.model_file))
            
            logger.info(f"Successfully added new movie {new_tmdb_id}")
            
        except Exception as e:
            logger.error(f"Error updating index: {str(e)}")
            raise RuntimeError(f"Failed to update index: {str(e)}")
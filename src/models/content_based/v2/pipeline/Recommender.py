from typing import Dict, List, Optional, Tuple
import faiss
import pandas as pd
import numpy as np
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class Recommender:
    def __init__(self, 
                 item_id: Optional[int],
                 item_map: pd.DataFrame,
                 feature_matrix: pd.DataFrame,
                 index: faiss.Index,
                 n_recommendations: int = 10):
        self.item_id = item_id
        self.item_map = item_map
        self.feature_matrix = feature_matrix
        self.index = index
        self.n_recommendations = n_recommendations
    
    def get_recommendation_for_existing(self) -> List[Dict]:
        """Retrieve recommendations for an existing item in the dataset."""
        try:
            if self.item_id is None:
                raise ValueError("Item ID is required for existing item recommendations")
                
            # Extract item IDs and feature vectors
            query_vector = self.feature_matrix.iloc[self.item_id].values.reshape(1, -1)
            
            # Search using FAISS index
            k = min(self.n_recommendations + 1, len(self.feature_matrix))
            distances, indices = self.index.search(query_vector, k)
            
            return self._process_search_results(distances[0], indices[0])
        
        except IndexError:
            logger.error(f"item_id {self.item_id} not found in the dataset", exc_info=True)
            raise HTTPException(status_code=404, detail=f"item_id {self.item_id} not found")
        except Exception as e:
            logger.error(f"Error in recommendation_for_existing: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

    def get_recommendation_for_new(self, new_features: pd.DataFrame) -> List[Dict]:
        """Retrieve recommendations for a new item not in the dataset."""
        try:
            # Ensure feature vector is properly shaped
            query_vector = new_features.values
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
                
            # Search for similar items
            distances, indices = self.index.search(query_vector, self.n_recommendations + 1)
            
            return self._process_search_results(distances[0], indices[0])

        except Exception as e:
            logger.error(f"Error in recommendation_for_new: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
    
    def _process_search_results(self, distances: np.ndarray, indices: np.ndarray) -> List[Dict]:
        """Process FAISS search results into a recommendation list with similarity scores."""
        similar_items = []
        
        for i, idx in enumerate(indices):
            try:
                idx = int(idx)
                
                # Skip if this is the query item itself
                if self.item_id is not None and idx == self.item_id:
                    continue
                
                # Convert distance to similarity score (1.0 means identical, 0.0 means completely different)
                # Assuming L2 distance with max value of 2.0
                similarity_score = float(1.0 - distances[i] / 2.0)
                
                similar_items.append({
                    'item_id': idx,
                    'similarity': similarity_score
                })
                
            except (IndexError, ValueError) as e:
                logger.warning(f"Invalid index encountered: {idx} - {str(e)}")
                continue
        
        logger.info(f"Found {len(similar_items)} similar media items")
        return similar_items[:self.n_recommendations]
    
    def get_recommendations_from_features(self, features: np.ndarray) -> List[Dict]:
        """Get recommendations directly from feature vector using Faiss."""
        try:
            # Ensure proper shape
            if features.ndim == 1:
                features = features.reshape(1, -1)
                
            # Search the index
            distances, indices = self.index.search(features, self.n_recommendations + 1)
            
            return self._process_search_results(distances[0], indices[0])
            
        except Exception as e:
            logger.error(f"Error getting recommendations from features: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
            
    def combine_recommendations(self, recommendation_sets: List[List[Dict]], weights: List[float]) -> List[Dict]:
        """Combine multiple recommendation sets with weights."""
        if len(recommendation_sets) != len(weights):
            raise ValueError("Number of recommendation sets must match number of weights")
            
        # Combine and weight recommendations
        weighted_recs = {}
        
        for i, rec_set in enumerate(recommendation_sets):
            weight = weights[i]
            for rec in rec_set:
                item_id = rec['item_id']
                if item_id not in weighted_recs:
                    weighted_recs[item_id] = rec['similarity'] * weight
                else:
                    weighted_recs[item_id] += rec['similarity'] * weight
        
        # Convert to list format
        combined_recommendations = [
            {'item_id': item_id, 'similarity': similarity}
            for item_id, similarity in weighted_recs.items()
        ]
        
        # Sort by weighted similarity score
        combined_recommendations.sort(key=lambda x: x['similarity'], reverse=True)
        
        return combined_recommendations[:self.n_recommendations]
        
    def filter_recommendations(
        self,
        recommendations: List[Dict], 
        excluded_tmdb_id: Optional[int] = None,
        media_type: Optional[str] = None,
        spoken_languages: Optional[List[str]] = None
    ) -> List[Dict]:
        """Filter recommendations based on criteria."""
        filtered_recommendations = []
        
        for rec in recommendations:
            item_id = rec['item_id']
            rec_row = self.item_map.loc[self.item_map['item_id'] == item_id]
            
            if rec_row.empty:
                logger.warning(f"Skipping item_id {item_id} - Not found in item map.")
                continue

            # Extract attributes for filtering
            rec_tmdb_id = rec_row['tmdb_id'].values[0]
            rec_media_type = rec_row['media_type'].values[0] if 'media_type' in rec_row.columns else None
            
            # Handle spoken languages if available
            rec_languages = []
            if 'spoken_languages' in rec_row.columns:
                rec_languages_str = rec_row['spoken_languages'].values[0]
                if isinstance(rec_languages_str, str):
                    rec_languages = [lang.strip() for lang in rec_languages_str.split(",")]

            # Skip if it's the excluded item
            if excluded_tmdb_id is not None and rec_tmdb_id == excluded_tmdb_id:
                continue

            # Apply media type filter if specified
            if media_type and rec_media_type != media_type:
                continue

            # Apply language filter if specified
            if spoken_languages and rec_languages:
                language_match = any(lang in rec_languages for lang in spoken_languages)
                if not language_match:
                    continue

            # If it passes all filters, add it
            filtered_recommendations.append(rec)
            
        return filtered_recommendations
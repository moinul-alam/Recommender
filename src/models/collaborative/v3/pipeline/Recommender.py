import numpy as np
from scipy import sparse
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path
import h5py

logger = logging.getLogger(__name__)

class Recommender:
    def __init__(
        self,
        model_path: Path,
        user_item_matrix: sparse.csr_matrix,
        item_mapping: Dict[int, int],
        item_reverse_mapping: Dict[int, int]
    ):
        """
        Initialize the recommender with model and mapping data.
        
        Args:
            model_path: Path to the trained model file (.h5 format)
            user_item_matrix: Sparse matrix of user-item interactions
            item_mapping: Dictionary mapping tmdb_ids to internal indices
            item_reverse_mapping: Dictionary mapping internal indices to tmdb_ids
        """
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Load model parameters from HDF5 file
            with h5py.File(model_path, 'r') as f:
                # Load model configuration
                self.n_factors = f.attrs['n_factors']
                self.global_bias = f.attrs['global_bias']
                
                # Load model parameters
                self.item_factors = f['item_factors'][:]
                self.user_factors = f['user_factors'][:]
                self.item_biases = f['item_biases'][:]
                self.user_biases = f['user_biases'][:]
            
            logger.info("Model loaded successfully")
            logger.info(f"Loaded item factors with shape: {self.item_factors.shape}")
            
            self.user_item_matrix = user_item_matrix
            self.item_mapping = item_mapping
            self.item_reverse_mapping = item_reverse_mapping
                
        except FileNotFoundError:
            logger.error(f"Model file not found at {model_path}")
            raise
        except Exception as e:
            logger.error(f"Error initializing Recommender: {str(e)}")
            raise
    
    def get_recommendations(
        self,
        tmdb_ids: List[int],
        ratings: List[float],
        n_recommendations: int = 10
    ) -> List[Tuple[int, float]]:
        """Generate personalized recommendations based on user ratings."""
        try:
            item_indices = [self.item_mapping[tid] for tid in tmdb_ids if tid in self.item_mapping]
            
            if not item_indices:
                return "Sorry, movie is not found."
                
            
            # Get user factors
            user_bias = np.sum(ratings) / len(ratings)
            user_factors = np.zeros(self.n_factors)
            
            for idx, rating in zip(item_indices, ratings):
                user_factors += (rating - self.global_bias - user_bias - 
                               self.item_biases[idx]) * self.item_factors[idx]
            user_factors /= len(ratings)
            
            # Generate predictions for all items
            all_predictions = (
                self.global_bias +
                user_bias +
                self.item_biases +
                np.dot(user_factors, self.item_factors.T)
            )
            
            # Mask out items the user has already rated
            all_predictions[item_indices] = -np.inf
            
            # Get top N recommendations
            top_indices = np.argsort(all_predictions)[-n_recommendations:][::-1]
            recommendations = [
                (self.item_reverse_mapping[idx], float(all_predictions[idx]))
                for idx in top_indices
            ]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise
    
    def get_similar_items(
        self,
        tmdb_ids: List[int],
        n_recommendations: int = 10
    ) -> List[Tuple[int, float]]:
        """Find similar items based on their latent factors."""
        try:
            # Convert tmdb_ids to internal indices
            item_indices = [self.item_mapping[tid] for tid in tmdb_ids if tid in self.item_mapping]

            if not item_indices:
                return "Sorry, movie is not found."
            
            # Calculate average item factors if multiple items provided
            query_factors = np.mean(self.item_factors[item_indices], axis=0)
            
            # Calculate similarities using dot product
            similarities = np.dot(self.item_factors, query_factors)
            
            # Mask out the query items
            similarities[item_indices] = -np.inf
            
            # Get top N similar items
            top_indices = np.argsort(similarities)[-n_recommendations:][::-1]
            similar_items = [
                (self.item_reverse_mapping[idx], float(similarities[idx]))
                for idx in top_indices
            ]
            
            return similar_items
            
        except Exception as e:
            logger.error(f"Error finding similar items: {str(e)}")
            raise
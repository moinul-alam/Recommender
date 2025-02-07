import faiss
import pandas as pd
import logging
import pickle
import pathlib
from typing import Any, Dict, Optional
from src.models.collaborative.v1.pipeline.ItemRecommender import ItemRecommender

logger = logging.getLogger(__name__)

class RecommendationService:
    @staticmethod
    def get_recommendations(
        items: dict,
        processed_dir_path: str,
        model_dir_path: str,
        n_recommendations: int = 10,
        min_similarity: float = 0.1
    ) -> Optional[Dict[str, Any]]:
        """
        Get recommendations for input items.
        
        Args:
            items: Dict of {tmdb_id: rating} pairs
            processed_dir_path: Path to processed data
            model_dir_path: Path to model files
            n_recommendations: Number of recommendations
            min_similarity: Minimum similarity threshold
        """
        try:
            # Convert string TMDB IDs to integers
            items_int = {int(tmdb_id): float(rating) for tmdb_id, rating in items.items()}
            
            # Setup paths
            processed_path = pathlib.Path(processed_dir_path)
            model_path = pathlib.Path(model_dir_path)
            
            # Load mappings
            item_mapping_path = processed_path / "item_mapping.pkl"
            item_reverse_mapping_path = processed_path / "item_reverse_mapping.pkl"
            
            if not all(p.exists() for p in [item_mapping_path, item_reverse_mapping_path]):
                logger.error("Missing required mapping files")
                return None
            
            with open(item_mapping_path, "rb") as f:
                item_mapping = pickle.load(f)
            with open(item_reverse_mapping_path, "rb") as f:
                item_reverse_mapping = pickle.load(f)
            
            logger.info("Loaded item mappings")
            
            # Load model components
            required_components = [
                'item_similarity_matrix',
                'model_info',
                'faiss_index'
            ]
            
            components = {}
            for component in required_components:
                path = model_path / f"{component}.pkl"
                if not path.exists():
                    logger.error(f"Missing model component: {component}")
                    return None
                
                with open(path, "rb") as f:
                    if component == 'faiss_index':
                        components[component] = faiss.read_index(str(path))
                    else:
                        components[component] = pickle.load(f)
            
            logger.info("Loaded model components")
            
            # Initialize recommender
            recommender = ItemRecommender(
                item_similarity_matrix=components['item_similarity_matrix'],
                item_mapping=item_mapping,
                item_reverse_mapping=item_reverse_mapping,
                faiss_index=components['faiss_index'],
                model_info=components['model_info']
            )
            
            # Generate recommendations
            recommendations = recommender.recommend(
                items=items_int,
                n_recommendations=n_recommendations,
                min_similarity=min_similarity
            )
            
            if not recommendations:
                logger.warning("No recommendations generated")
                return None
            
            return {
                'input_items': items,
                'recommendations': recommendations,
                'metadata': {
                    'model_info': components['model_info'],
                    'recommender': recommender,
                    'timestamp': str(pd.Timestamp.now())
                }
            }
            
        except Exception as e:
            logger.error(f"Error in recommendation service: {e}", exc_info=True)
            return None
import logging
import pickle
import pathlib
from typing import List, Dict
from src.models.collaborative.v1.pipeline.ItemRecommender import ItemRecommender

logger = logging.getLogger(__name__)

class RecommendationService:
    @staticmethod
    def generate_recommendations(
        input_items: List[dict], 
        model_dir_path: str, 
        processed_dir_path: str,
        top_n: int = 10
    ) -> Dict:
        try:
            # Load model components
            svd_path = pathlib.Path(model_dir_path) / "svd_components.pkl"
            index_path = pathlib.Path(model_dir_path) / "faiss_index.pkl"
            item_mapping_path = pathlib.Path(processed_dir_path) / "item_mapping.pkl"
            item_reverse_mapping_path = pathlib.Path(processed_dir_path) / "item_reverse_mapping.pkl"

            # Load model files
            try:
                with open(svd_path, "rb") as f:
                    svd_components = pickle.load(f)
                
                with open(index_path, "rb") as f:
                    faiss_index = pickle.load(f)

                with open(item_mapping_path, "rb") as f:
                    item_mapping = pickle.load(f)
                
                with open(item_reverse_mapping_path, "rb") as f:
                    item_reverse_mapping = pickle.load(f)
            except FileNotFoundError as fe:
                logger.error(f"Model file missing: {fe}")
                raise ValueError(f"Model file missing: {fe}")
            except pickle.UnpicklingError as pe:
                logger.error(f"Error loading pickle file: {pe}")
                raise ValueError(f"Error loading pickle file: {pe}")

            # Create recommender instance
            recommender = ItemRecommender(svd_components, faiss_index, item_mapping, item_reverse_mapping)
            
            # Convert input_items (Pydantic models) to dictionary
            input_items_dicts = [item.dict() if hasattr(item, "dict") else item for item in input_items]
            
            recommendations = recommender.recommend(input_items_dicts, top_n)

            logger.info(f"Generated {top_n} recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Recommendation generation error: {e}", exc_info=True)
            raise ValueError("Recommendation generation failed") from e

import pathlib
import pickle
import faiss
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class BaseRecommendationService:
    """Base class for recommendation services to handle shared loading logic."""

    @staticmethod
    def load_pickle(file_path):
        """Helper function to load pickle files."""
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load {file_path.name}")

    @staticmethod
    def load_model_components(collaborative_dir_path):
        """Loads all required model components and mappings."""
        collaborative_dir_path = pathlib.Path(collaborative_dir_path)
        
        user_item_matrix = BaseRecommendationService.load_pickle(collaborative_dir_path / "2_user_item_matrix.pkl")
        user_mapping = BaseRecommendationService.load_pickle(collaborative_dir_path / "2_user_mapping.pkl")
        user_reverse_mapping = BaseRecommendationService.load_pickle(collaborative_dir_path / "2_user_reverse_mapping.pkl")
        item_mapping = BaseRecommendationService.load_pickle(collaborative_dir_path / "2_item_mapping.pkl")
        item_reverse_mapping = BaseRecommendationService.load_pickle(collaborative_dir_path / "2_item_reverse_mapping.pkl")
        user_matrix = BaseRecommendationService.load_pickle(collaborative_dir_path / "3_user_matrix.pkl")
        item_matrix = BaseRecommendationService.load_pickle(collaborative_dir_path / "3_item_matrix.pkl")
        svd_user_model = BaseRecommendationService.load_pickle(collaborative_dir_path / "3_svd_user_model.pkl")
        svd_item_model = BaseRecommendationService.load_pickle(collaborative_dir_path / "3_svd_item_model.pkl")
        model_info = BaseRecommendationService.load_pickle(collaborative_dir_path / "3_model_info.pkl")

        try:
            faiss_user_index = faiss.read_index(str(collaborative_dir_path / "3_faiss_user_index.flat"))
            faiss_item_index = faiss.read_index(str(collaborative_dir_path / "3_faiss_item_index.flat"))
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to load FAISS index")

        return user_item_matrix, user_mapping, user_reverse_mapping, user_matrix, item_mapping, item_reverse_mapping, item_matrix, svd_user_model, svd_item_model, model_info, faiss_user_index, faiss_item_index
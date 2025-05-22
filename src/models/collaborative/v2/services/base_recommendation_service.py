import faiss
import logging
from fastapi import HTTPException
from pathlib import Path
from src.models.common.DataLoader import load_multiple
from src.models.common.file_config import file_names

logger = logging.getLogger(__name__)

class BaseRecommendationService:
    REQUIRED_KEYS = [
        "user_item_matrix", "user_item_mappings",
        "user_matrix", "item_matrix",
        "faiss_user_index", "faiss_item_index",
        "svd_components", "model_info", "user_item_means"
    ]

    @staticmethod
    def load_model_components(collaborative_dir_path: str) -> tuple:
        """
        Loads all required model components and FAISS indices.
        
        Args:
            collaborative_dir_path: Directory path containing model files
            file_names: Dictionary mapping component names to filenames
            
        Returns:
            Tuple of model components needed for recommendations
        """
        # Validate input
        BaseRecommendationService._validate_file_keys(file_names)

        collaborative_dir_path = Path(collaborative_dir_path)

        # Load standard model components
        components = BaseRecommendationService._load_required_components(
            path = collaborative_dir_path, 
            file_names = file_names
        )

        # Load FAISS indices
        faiss_user_index, faiss_item_index = BaseRecommendationService._load_faiss_indices(
            path = collaborative_dir_path, 
            file_names = file_names
        )

        return {
            "user_item_matrix": components["user_item_matrix"],
            "user_item_mappings": components["user_item_mappings"],
            "user_matrix": components["user_matrix"],
            "item_matrix": components["item_matrix"],
            "faiss_user_index": faiss_user_index,
            "faiss_item_index": faiss_item_index,
            "svd_components": components["svd_components"],
            "model_info": components["model_info"],
            "user_item_means": components["user_item_means"]
        }

    @staticmethod
    def _validate_file_keys(file_names: dict):
        """Validates that all required file keys are present in the provided dictionary."""
        missing_keys = [
            key for key in BaseRecommendationService.REQUIRED_KEYS if key not in file_names
        ]
        if missing_keys:
            logger.error(f"Missing file names for: {missing_keys}")
            raise HTTPException(status_code=400, detail=f"Missing file names: {missing_keys}")

    @staticmethod
    def _load_required_components(path: Path, file_names: dict) -> dict:
        """Loads model data components from files."""
        files_to_load = {
            "user_item_matrix": file_names["user_item_matrix"],
            "user_item_mappings": file_names["user_item_mappings"],
            "user_matrix": file_names["user_matrix"],
            "item_matrix": file_names["item_matrix"],
            "svd_components": file_names["svd_components"],
            "model_info" : file_names["model_info"],
            "user_item_means": file_names["user_item_means"]
        }

        try:
            components = load_multiple(path, files_to_load)
            return components
        except Exception as e:
            logger.error(f"Error loading model components from {path}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to load model components")

    @staticmethod
    def _load_faiss_indices(path: Path, file_names: dict) -> tuple:
        """Loads FAISS indices from files."""
        faiss_user_index_path = path / file_names["faiss_user_index"]
        faiss_item_index_path = path / file_names["faiss_item_index"]

        if not faiss_user_index_path.exists() or not faiss_item_index_path.exists():
            logger.error(f"FAISS index files not found in path: {path}")
            raise HTTPException(status_code=404, detail="FAISS index files not found")

        try:
            faiss_user_index = faiss.read_index(str(faiss_user_index_path))
            faiss_item_index = faiss.read_index(str(faiss_item_index_path))
            return faiss_user_index, faiss_item_index
        except Exception as e:
            logger.error(f"Error loading FAISS indices from {path}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to load FAISS indices")
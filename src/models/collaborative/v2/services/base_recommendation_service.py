import faiss
import logging
from fastapi import HTTPException
from pathlib import Path
from src.models.common.DataLoader import load_multiple

logger = logging.getLogger(__name__)

class BaseRecommendationService:
    REQUIRED_KEYS = [
        "user_item_matrix", "user_mapping", "user_reverse_mapping",
        "item_mapping", "item_reverse_mapping",
        "user_matrix", "item_matrix",
        "faiss_user_index", "faiss_item_index",
        "svd_user_model"
    ]

    @staticmethod
    def load_model_components(collaborative_dir_path: str, file_names: dict) -> tuple:
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
        components = BaseRecommendationService._load_data_components(
            collaborative_dir_path, file_names
        )

        (
            user_item_matrix,
            user_mapping,
            user_reverse_mapping,
            item_mapping,
            item_reverse_mapping,
            user_matrix,
            item_matrix,
            svd_user_model
        ) = (
            components["user_item_matrix"],
            components["user_mapping"],
            components["user_reverse_mapping"],
            components["item_mapping"],
            components["item_reverse_mapping"],
            components["user_matrix"],
            components["item_matrix"],
            components["svd_user_model"]
        )

        # Load FAISS indices
        faiss_user_index, faiss_item_index = BaseRecommendationService._load_faiss_indices(
            collaborative_dir_path, file_names
        )

        return (
            user_item_matrix,
            user_mapping,
            user_reverse_mapping,
            user_matrix,
            item_mapping,
            item_reverse_mapping,
            item_matrix,
            faiss_user_index,
            faiss_item_index,
            svd_user_model
        )

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
    def _load_data_components(path: Path, file_names: dict) -> dict:
        """Loads model data components from files."""
        files_to_load = {
            "user_item_matrix": file_names["user_item_matrix"],
            "user_mapping": file_names["user_mapping"],
            "user_reverse_mapping": file_names["user_reverse_mapping"],
            "item_mapping": file_names["item_mapping"],
            "item_reverse_mapping": file_names["item_reverse_mapping"],
            "user_matrix": file_names["user_matrix"],
            "item_matrix": file_names["item_matrix"],
            "svd_user_model": file_names["svd_user_model"]
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
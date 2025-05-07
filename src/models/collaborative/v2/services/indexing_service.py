import logging
from pathlib import Path
from typing import Dict, Optional
from fastapi import HTTPException
from src.models.collaborative.v2.pipeline.index_creation import IndexCreation
from src.models.common.file_config import file_names
from src.models.common.DataLoader import load_data
from src.models.common.DataSaver import save_data, save_objects

from src.schemas.content_based_schema import PipelineResponse


class IndexingService:
    @staticmethod
    def create_index(
        collaborative_dir_path: str,
        similarity_metric: str = "cosine",
        batch_size: int = 20000
    ) -> Optional[PipelineResponse]:
        """
        Create and save FAISS indexes for user and item matrices.
        
        Args:
            collaborative_dir_path: Directory containing matrices and where indexes will be saved
            file_names: Dictionary with keys for user_matrix, item_matrix, faiss_user_index, faiss_item_index
            similarity_metric: Similarity metric to use (cosine or inner_product)
            batch_size: Batch size for adding vectors to FAISS index
            
        Returns:
            PipelineResponse object with status and message, or None if error occurs
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        try:
            # Convert path to Path object and validate
            collaborative_dir_path = Path(collaborative_dir_path)
            if not collaborative_dir_path.exists() or not collaborative_dir_path.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid directory path: {collaborative_dir_path}"
                )
            
            # Validate required file names
            required_keys = ["user_matrix", "item_matrix", "faiss_user_index", "faiss_item_index"]
            missing_keys = [key for key in required_keys if key not in file_names]
            if missing_keys:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required keys in file_names: {missing_keys}"
                )
            
            # Construct and validate input file paths
            user_matrix_path = collaborative_dir_path / file_names["user_matrix"]
            item_matrix_path = collaborative_dir_path / file_names["item_matrix"]
            
            if not user_matrix_path.is_file():
                raise HTTPException(
                    status_code=400,
                    detail=f"User matrix file not found: {user_matrix_path}"
                )
                
            if not item_matrix_path.is_file():
                raise HTTPException(
                    status_code=400,
                    detail=f"Item matrix file not found: {item_matrix_path}"
                )
            
            # Load matrices
            logger.info(f"Loading user matrix from {user_matrix_path}")
            user_matrix = load_data(user_matrix_path)
            if user_matrix is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to load user matrix: {user_matrix_path}"
                )
                
            logger.info(f"Loading item matrix from {item_matrix_path}")
            item_matrix = load_data(item_matrix_path)
            if item_matrix is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to load item matrix: {item_matrix_path}"
                )
            
            if similarity_metric not in ["cosine", "inner_product"]:
                logger.warning(f"Unrecognized similarity metric: {similarity_metric}, defaulting to cosine")
                similarity_metric = "cosine"
            
            indexer = IndexCreation(
                similarity_metric=similarity_metric,
                batch_size=batch_size
            )
            
            user_index_path = collaborative_dir_path / file_names["faiss_user_index"]
            item_index_path = collaborative_dir_path / file_names["faiss_item_index"]
            
            logger.info(f"Creating user index at {user_index_path}")
            user_index = indexer.create_faiss_index(
                matrix=user_matrix,
                index_type="user",
                index_path=str(user_index_path)
            )
            logger.info(f"✅ Index created: {user_index.__class__.__name__} | Dimensionality: {user_index.d} | Total vectors: {user_index.ntotal}")
            
            logger.info(f"Creating item index at {item_index_path}")
            item_index = indexer.create_faiss_index(
                matrix=item_matrix,
                index_type="item",
                index_path=str(item_index_path)
            )
            logger.info(f" ✅ Index created: {item_index.__class__.__name__} | Dimensionality: {item_index.d} | Total vectors: {item_index.ntotal}")
                        
            return PipelineResponse(
                status="success",
                message="FAISS index creation completed successfully.",
                output=str(collaborative_dir_path),
            )

        except HTTPException as http_err:
            logger.error(f"HTTP error during index creation: {http_err.detail}")
            raise http_err
        except Exception as e:
            logger.error(f"Error during index creation: {str(e)}", exc_info=True)
            return None
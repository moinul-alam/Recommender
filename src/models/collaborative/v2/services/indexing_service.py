from pathlib import Path
from typing import Optional
from fastapi import HTTPException
from src.models.collaborative.v2.pipeline.index_creation import IndexCreation
from src.models.common.file_config import file_names
from src.models.common.DataLoader import load_data
from src.models.common.DataSaver import save_object
from src.models.common.logger import app_logger
from src.schemas.content_based_schema import PipelineResponse

logger = app_logger(__name__)

class IndexingService:
    @staticmethod
    def create_index(
        directory_path: str
    ) -> Optional[PipelineResponse]:
        """
        Create and save FAISS indexes for user and item matrices, and update model_info with similarity metric.
        
        Args:
            directory_path: Directory containing matrices and where indexes will be saved
            
        Returns:
            PipelineResponse object with status and message, or None if error occurs
        """
        # Default values
        similarity_metric: str = "cosine"
        batch_size: int = 20000
        
        try:
            # Convert path to Path object and validate
            directory_path = Path(directory_path)
            if not directory_path.exists() or not directory_path.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid directory path: {directory_path}"
                )
            
            # Validate required file names
            required_keys = ["user_matrix", "item_matrix", "faiss_user_index", "faiss_item_index", "model_info"]
            missing_keys = [key for key in required_keys if key not in file_names]
            if missing_keys:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required keys in file_names: {missing_keys}"
                )
            
            # Construct and validate input file paths
            user_matrix_path = directory_path / file_names["user_matrix"]
            item_matrix_path = directory_path / file_names["item_matrix"]
            model_info_path = directory_path / file_names["model_info"]  # Should not include .pkl extension
            
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
            
            # Load and update model_info
            model_info = {}
            if model_info_path.is_file():
                logger.info(f"Loading model_info from {model_info_path}")
                model_info = load_data(model_info_path)
                if model_info is None:
                    logger.warning(f"Failed to load model_info, creating new dictionary")
                    model_info = {}
            else:
                logger.info(f"model_info file not found at {model_info_path}, creating new dictionary")
            
            # Update model_info with similarity_metric
            model_info["similarity_metric"] = similarity_metric
            logger.info(f"Updated model_info with similarity_metric: {similarity_metric}")
            
            # Save updated model_info using save_object
            logger.info(f"Saving updated model_info to {model_info_path}")
            try:
                saved_path = save_object(
                    directory_path=directory_path,
                    obj=model_info,
                    file_name=file_names["model_info"],
                    protocol=4,
                    compress=3
                )
                logger.info(f"Successfully saved model_info to {saved_path}")
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save updated model_info to {model_info_path}: {str(e)}"
                )
            
            indexer = IndexCreation(
                similarity_metric=similarity_metric,
                batch_size=batch_size
            )
            
            user_index_path = directory_path / file_names["faiss_user_index"]
            item_index_path = directory_path / file_names["faiss_item_index"]
            
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
                message="FAISS index creation and model_info update completed successfully.",
                output=str(directory_path),
            )

        except HTTPException as http_err:
            logger.error(f"HTTP error during index creation: {http_err.detail}")
            raise http_err
        except Exception as e:
            logger.error(f"Error during index creation: {str(e)}", exc_info=True)
            return None
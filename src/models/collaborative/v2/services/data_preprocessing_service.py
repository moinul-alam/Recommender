import logging
from pathlib import Path
from fastapi import HTTPException
from src.models.common.DataLoader import load_data
from src.models.common.DataSaver import save_data, save_objects
from src.models.collaborative.v2.pipeline.data_preprocessing import DataPreprocessing
from src.schemas.content_based_schema import PipelineResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DataPreprocessingService:    
    @staticmethod
    def process_data(
        collaborative_dir_path: str,
        file_names: dict,
        sparse_user_threshold: int = 5,
        sparse_item_threshold: int = 5,
        split_percent: float = 0.8,
        segment_size: int = 10000,
    ) -> PipelineResponse:
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        try:
            # Validate input and output paths
            collaborative_dir_path = Path(collaborative_dir_path)
            
            if not collaborative_dir_path.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid directory path: {collaborative_dir_path}"
                )
            
            # Locate dataset file
            dataset_path = collaborative_dir_path / file_names["dataset_name"]
            
            if not dataset_path.is_file():
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset file not found: {dataset_path}"
                )
                       
            # Load dataset
            dataset = load_data(dataset_path)
            logger.info(f"Dataset loaded from {dataset_path}")
            
            if dataset is None or dataset.empty:
                raise HTTPException(status_code=400, detail="Dataset is empty or invalid")
                     
            # Initialize preprocessor
            preprocessor = DataPreprocessing(
                sparse_user_threshold,
                sparse_item_threshold,
                split_percent,
                segment_size
            )
            
            # Process data
            train, test, user_mapping, user_reverse_mapping, \
            item_mapping, item_reverse_mapping, \
            user_item_matrix = preprocessor.process(df=dataset)
            
            logger.info(f"Preprocessing complete. Train set size: {len(train)}, Test set size: {len(test)}")
            logger.info(f"User mapping size: {len(user_mapping)}, Item mapping size: {len(item_mapping)}")
            
            # Sample output for debugging
            if len(item_mapping) > 0:
                logger.info(f"Sample item mappings (first 5): {list(item_mapping.items())[:5]}")
                logger.info(f"Sample item reverse mappings (first 5): {list(item_reverse_mapping.items())[:5]}")
            
            files_to_save = {
                file_names["train_set"]: train,
                file_names["test_set"]: test,
                file_names["user_mapping"]: user_mapping,
                file_names["user_reverse_mapping"]: user_reverse_mapping,
                file_names["item_mapping"]: item_mapping,
                file_names["item_reverse_mapping"]: item_reverse_mapping,
                file_names["user_item_matrix"]: user_item_matrix
            }
                
            save_objects(
                directory_path=collaborative_dir_path,
                objects=files_to_save,
                compress=3
            )
            
            logger.info(f"Data saved successfully in {collaborative_dir_path}")
            
            return PipelineResponse(
                status="success",
                message="Data preprocessing completed successfully",
                output=str(collaborative_dir_path)
            )
        except HTTPException:
            raise            
        except Exception as e:
            logger.error(f"An error occurred during data preprocessing: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred during data preprocessing: {e}"
            )
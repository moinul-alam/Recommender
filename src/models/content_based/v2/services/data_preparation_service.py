import logging
import os
from fastapi import HTTPException
from pathlib import Path
from src.models.content_based.v2.pipeline.data_preparation import DataPreparation
from src.models.common.DataLoader import load_data
from src.models.common.DataSaver import save_multiple_dataframes

from src.schemas.content_based_schema import PipelineResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreparationService:
    @staticmethod
    def prepare_data(content_based_dir_path: str, file_names: dict) -> PipelineResponse:
        try:
            # Validate dataset path
            content_based_dir_path = Path(content_based_dir_path)
            if not content_based_dir_path.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid directory path: {content_based_dir_path}"
                )
            
            dataset_path = content_based_dir_path / file_names["dataset_name"]
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
            
            # Initialize data preprocessing
            data_preparer = DataPreparation(dataset)
            prepared_dataset, item_map = data_preparer.apply_data_preparation()
            
            # Save prepared dataset and item mapping
            prepared_dataset_name = file_names["prepared_dataset_name"]
            item_map_name = file_names["item_map_name"]
            dataframes = {
                prepared_dataset_name: prepared_dataset,
                item_map_name: item_map
            }
            
            save_multiple_dataframes(
                directory_path=content_based_dir_path,
                dataframes=dataframes,
                file_type="csv",
                index=False
            )
            
            logger.info(f"Dataset and item mapping saved successfully")

            # Return response
            return PipelineResponse(
                status="success",
                message="Data preparation completed successfully",
                output= str(content_based_dir_path)
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error preparing data: {str(e)}")

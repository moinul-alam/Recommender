import logging
import os
from pathlib import Path
from fastapi import HTTPException
from src.models.content_based.v2.pipeline.data_preprocessing import DataPreprocessing
from src.schemas.content_based_schema import PipelineResponse
from src.models.common.DataLoader import load_data
from src.models.common.DataSaver import save_data, save_multiple_dataframes

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataPreprocessingService:
    @staticmethod
    def preprocess_data(content_based_dir_path: str, segment_size: int, file_names: dict) -> PipelineResponse:
        try:
            content_based_dir_path=Path(content_based_dir_path)

            if not content_based_dir_path.is_dir():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Directoryu not found: {content_based_dir_path}"
                )

            dataset_path = content_based_dir_path / file_names["prepared_dataset_name"]
            logger.info(f"Dataset path: {dataset_path}")
    
            if not os.path.isfile(dataset_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset file not found: {dataset_path}"
                )
                
            dataset = load_data(dataset_path)
            logger.info(f"Dataset loaded from {dataset_path}")
            if dataset is None or dataset.empty:
                raise HTTPException(status_code=400, detail="Dataset is empty or invalid")
            

            # Initialize data preprocessing
            data_preprocessor = DataPreprocessing(dataset, segment_size)
            full_processed_dataset, processed_segments = data_preprocessor.apply_data_preprocessing()

            # Save processed segments
            preprocessed_dataset_name = file_names["preprocessed_dataset_name"]
            save_data(
                directory_path=content_based_dir_path,
                df=full_processed_dataset,
                file_name=preprocessed_dataset_name,
                file_type="csv",
                index=False
            )
            
            for i, segment in enumerate(processed_segments):
                segment_file_name = f"3_processed_segment_{i + 1}"
                save_data(
                    directory_path=content_based_dir_path,
                    df=segment,
                    file_name=segment_file_name,
                    file_type="csv",
                    index=False
                )

            return PipelineResponse(
                status= "success",
                message="Data preprocessed and saved successfully",
                output=str(content_based_dir_path)
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error preprocessing data: {str(e)}")

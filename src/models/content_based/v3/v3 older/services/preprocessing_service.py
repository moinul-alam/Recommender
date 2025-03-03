import os
from pathlib import Path
from fastapi import HTTPException
from src.models.content_based.v2.pipeline.DataPreprocessing import DataPreprocessing
from src.schemas.content_based_schema import PipelineResponse

class PreprocessingService:
    @staticmethod
    def preprocess_data(segment_size: int, prepared_folder_path: str, processed_folder_path: str) -> PipelineResponse:
        try:
            # Validate dataset path
            prepared_folder_path=Path(prepared_folder_path)
            processed_folder_path = Path(processed_folder_path)

            if not prepared_folder_path.is_dir():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Processed folder not found: {prepared_folder_path}"
                )
        
            # Validate or create processed folder path
            if not os.path.isdir(processed_folder_path):
                os.makedirs(processed_folder_path, exist_ok=True)

            dataset_path = prepared_folder_path / "prepared_dataset.csv"
            if not dataset_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Combined features dataset not found: {dataset_path}"
                )

            # Initialize data preprocessing
            data_preprocessor = DataPreprocessing(dataset_path, segment_size)
            full_processed_dataset, processed_segments = data_preprocessor.apply_data_preprocessing()

            # Save processed segments
            save_full_processed_dataset = os.path.join(processed_folder_path, f"full_processed_dataset.csv")
            full_processed_dataset.to_csv(save_full_processed_dataset, index=False)

            for i, segment in enumerate(processed_segments):
                segment_file = os.path.join(processed_folder_path, f"processed_segment_{i + 1}.csv")
                segment.to_csv(segment_file, index=False)

            # Return response
            return PipelineResponse(
                status="Data preprocessed and saved successfully",
                output=len(processed_segments),
                output_path=str(processed_folder_path)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error preprocessing data: {str(e)}")

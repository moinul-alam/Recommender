import os
from fastapi import HTTPException
from src.models.content_based.v1.pipeline.DataPreprocessing import DataPreprocessing
from src.schemas.content_based_schema import PipelineResponse

class PreprocessingService:
    @staticmethod
    def preprocess_data(dataset_path: str, processed_folder_path: str, segment_size: int) -> PipelineResponse:
        try:
            # Validate dataset path
            if not os.path.isfile(dataset_path):
                raise HTTPException(status_code=400, detail=f"Dataset file not found: {dataset_path}")

            # Validate or create processed folder path
            if not os.path.isdir(processed_folder_path):
                os.makedirs(processed_folder_path, exist_ok=True)

            # Initialize data preprocessing
            data_preprocessor = DataPreprocessing(dataset_path, segment_size)
            processed_segments = data_preprocessor.apply_data_preprocessing()

            # Save processed segments
            for i, segment in enumerate(processed_segments):
                segment_file = os.path.join(processed_folder_path, f"processed_segment_{i + 1}.csv")
                segment.to_csv(segment_file, index=False)

            # Return response
            return PipelineResponse(
                status="Data preprocessed and saved successfully",
                output=len(processed_segments),
                output_path=processed_folder_path
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error preprocessing data: {str(e)}")

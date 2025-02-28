import os
from pathlib import Path
from fastapi import HTTPException
from src.models.content_based.v2.pipeline.DataPreprocessing import DataPreprocessing
from src.schemas.content_based_schema import PipelineResponse

class PreprocessingService:
    @staticmethod
    def preprocess_data(content_based_dir_path: str, segment_size: int) -> PipelineResponse:
        try:
            # Validate dataset path
            content_based_dir_path=Path(content_based_dir_path)

            if not content_based_dir_path.is_dir():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Processed folder not found: {content_based_dir_path}"
                )

            dataset_path = content_based_dir_path / "1_prepared_dataset.csv"
            if not dataset_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Combined features dataset not found: {dataset_path}"
                )

            # Initialize data preprocessing
            data_preprocessor = DataPreprocessing(dataset_path, segment_size)
            full_processed_dataset, processed_segments = data_preprocessor.apply_data_preprocessing()

            # Save processed segments
            save_full_processed_dataset = os.path.join(content_based_dir_path, f"2_full_processed_dataset.csv")
            full_processed_dataset.to_csv(save_full_processed_dataset, index=False)

            for i, segment in enumerate(processed_segments):
                segment_file = os.path.join(content_based_dir_path, f"2_processed_segment_{i + 1}.csv")
                segment.to_csv(segment_file, index=False)

            # Return response
            return PipelineResponse(
                status="Data preprocessed and saved successfully",
                output=len(processed_segments),
                output_path=str(content_based_dir_path)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error preprocessing data: {str(e)}")

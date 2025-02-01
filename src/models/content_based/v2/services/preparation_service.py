import os
from fastapi import HTTPException
from src.models.content_based.v2.pipeline.DataPreparation import DataPreparation
from src.schemas.content_based_schema import PipelineResponse

class PreparationService:
    @staticmethod
    def prepare_data(raw_dataset_path: str, prepared_folder_path: str) -> PipelineResponse:
        try:
            # Validate dataset path
            if not os.path.isfile(raw_dataset_path):
                raise HTTPException(status_code=400, detail=f"Dataset file not found: {raw_dataset_path}")

            # Validate or create processed folder path
            if not os.path.isdir(prepared_folder_path):
                os.makedirs(prepared_folder_path, exist_ok=True)

            # Initialize data preprocessing
            data_preparer = DataPreparation(raw_dataset_path)
            prepared_dataset = data_preparer.apply_data_preparation()

            # Save processed dataset
            save_prepared_dataset = os.path.join(prepared_folder_path, "prepared_dataset.csv")
            prepared_dataset.to_csv(save_prepared_dataset, index=False)

            # Return response
            return PipelineResponse(
                status="Data prepared and saved successfully",
                output=1,
                output_path=save_prepared_dataset  # Provide the full path of the saved file
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error preparing data: {str(e)}")

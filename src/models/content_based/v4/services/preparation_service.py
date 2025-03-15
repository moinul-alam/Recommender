import os
from fastapi import HTTPException
from src.models.content_based.v4.pipeline.DataPreparation import DataPreparation
from src.schemas.content_based_schema import PipelineResponse

class PreparationService:
    @staticmethod
    def prepare_data(content_based_dir_path: str, raw_dataset_name: str) -> PipelineResponse:
        """
        Service layer method responsible for file operations and orchestrating the data preparation process.
        """
        try:
            # Validate dataset path
            raw_dataset_path = os.path.join(content_based_dir_path, raw_dataset_name)
            if not os.path.isfile(raw_dataset_path):
                raise HTTPException(status_code=400, detail=f"Dataset file not found: {raw_dataset_path}")

            # Process data using the DataPreparation class
            data_preparer = DataPreparation(raw_dataset_path)
            prepared_dataset, item_mapping = data_preparer.prepare()

            # Save processed dataset
            save_prepared_dataset = os.path.join(content_based_dir_path, "2_prepared_dataset.csv")
            prepared_dataset.to_csv(save_prepared_dataset, index=False)
            
            save_item_mapping = os.path.join(content_based_dir_path, "2_item_mapping.csv")
            item_mapping.to_csv(save_item_mapping, index=False)

            # Return response
            return PipelineResponse(
                status="Data prepared and saved successfully",
                output=1,
                output_path=save_prepared_dataset
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error preparing data: {str(e)}")
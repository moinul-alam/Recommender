import logging
from pathlib import Path

from fastapi import HTTPException
from src.models.content_based.v2.services.data_preparation_service import DataPreparationService
from src.models.content_based.v2.services.data_preprocessing_service import DataPreprocessingService
from src.models.content_based.v2.services.feature_engineering_service import FeatureEngineeringService
from src.models.content_based.v2.services.indexing_service import IndexingService
from src.schemas.content_based_schema import PipelineResponse

logger = logging.getLogger(__name__)

class PipelineService:
    @staticmethod
    def execute_full_pipeline(
        content_based_dir_path: str,
        file_names: dict,
        segment_size: int = 10000
    ) -> PipelineResponse :
        try:
            content_based_dir_path = Path(content_based_dir_path)
            if not content_based_dir_path.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid directory path: {content_based_dir_path}"
                )
            
            
            # Step 1: Data Preparation
            DataPreparationService.prepare_data(
                content_based_dir_path, file_names
            )

            # Step 2: Data Preprocessing
            DataPreprocessingService.preprocess_data(
                content_based_dir_path, segment_size, file_names
            )

            # Step 3: Feature Engineering
            FeatureEngineeringService.engineer_features(
                content_based_dir_path, file_names
            )

            # Step 4: Model Training
            IndexingService.create_index(
                content_based_dir_path, file_names
            )       
            
            return PipelineResponse(
                            status="Success",
                            message="Pipeline completed successfully",
                            output=str(content_based_dir_path)
                        )

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
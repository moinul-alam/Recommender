import logging
from pathlib import Path
from fastapi import HTTPException
from src.models.collaborative.v2.services.data_preprocessing_service import DataPreprocessingService
from src.models.collaborative.v2.services.feature_extraction_service import FeatureExtractionService
from src.models.collaborative.v2.services.indexing_service import IndexingService
from src.models.common.logger import app_logger

logger = app_logger(__name__)

class PipelineService:
    @staticmethod
    def execute_full_pipeline(
        directory_path: str
    ):
        try:
            # Data Preprocessing
            preprocessing_result = DataPreprocessingService.process_data(
                directory_path
            )
            
            if not preprocessing_result:
                return {"status": "Preprocessing failed"}
            
            # Feature Extraction
            feature_extraction_result = FeatureExtractionService.extract_features(
                directory_path
            )
            
            if not feature_extraction_result:
                return {"status": "Feature extraction failed"}
            
            # Index Creation
            indexing_result = IndexingService.create_index(
                directory_path
            )
            
            if not indexing_result:
                return {"status": "Index creation failed"}
            
            return {"status": "Pipeline executed successfully"}
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")
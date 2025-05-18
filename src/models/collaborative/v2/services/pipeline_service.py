import logging
from pathlib import Path
from fastapi import HTTPException
from src.models.collaborative.v2.services.data_preprocessing_service import DataPreprocessingService
from src.models.collaborative.v2.services.feature_extraction_service import FeatureExtractionService
from src.models.collaborative.v2.services.indexing_service import IndexingService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PipelineService:
    @staticmethod
    def execute_full_pipeline(
        collaborative_dir_path: str,
        n_components_item: int = 300,
        n_components_user: int = 300,
        similarity_metric: str = "cosine",
        batch_size: int = 10000
    ):
        try:
            # Data Preprocessing
            preprocessing_result = DataPreprocessingService.process_data(
                collaborative_dir_path
            )
            
            if not preprocessing_result:
                return {"status": "Preprocessing failed"}
            
            # Feature Extraction
            feature_extraction_result = FeatureExtractionService.extract_features(
                collaborative_dir_path,
                n_components_item,
                n_components_user,
                batch_size
            )
            
            if not feature_extraction_result:
                return {"status": "Feature extraction failed"}
            
            # Index Creation
            indexing_result = IndexingService.create_index(
                collaborative_dir_path,
                similarity_metric,
                batch_size
            )
            
            if not indexing_result:
                return {"status": "Index creation failed"}
            
            return {"status": "Pipeline executed successfully"}
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")
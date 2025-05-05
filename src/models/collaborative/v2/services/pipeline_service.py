import os
import logging
import pathlib
from fastapi import HTTPException
from models.collaborative.v2.services.data_preprocessing_service import PreprocessingService
from src.models.collaborative.v2.services.feature_extraction_service import FeatureExtractionService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PipelineService:
    @staticmethod
    def execute_full_pipeline(
        collaborative_dir_path: str,
        dataset_name: str,
        sparse_user_threshold: int = 5,
        sparse_item_threshold: int = 1,
        split_percent: float = 0.8,
        chunk_size: int = 10000,
        n_neighbors: int = 50,
        similarity_metric: str = "L2",
        batch_size: int = 1000,
        min_similarity: float = 0.1
    ):
        try:
            collaborative_dir_path = pathlib.Path(collaborative_dir_path)
            
            if not collaborative_dir_path.exists():
                logger.error(f"Collaborative directory not found: {collaborative_dir_path}")
                raise HTTPException(status_code=404, detail=f"Collaborative directory not found: {collaborative_dir_path}")       
                    

            preprocessing_result = PreprocessingService.process_data(
                collaborative_dir_path=collaborative_dir_path,
                dataset_name=dataset_name,
                sparse_user_threshold=sparse_user_threshold,
                sparse_item_threshold=sparse_item_threshold,
                split_percent=split_percent,
                chunk_size=chunk_size
            )

            if not preprocessing_result:
                return {"status": "Preprocessing failed"}

            training_result = FeatureExtractionService.extract_features(
                collaborative_dir_path=collaborative_dir_path,
                n_neighbors=n_neighbors,
                similarity_metric=similarity_metric,
                batch_size=batch_size,
                min_similarity=min_similarity
            )

            if not training_result:
                return {"status": "Model training failed"}

            return {"status": "Model Training Successful"}

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")

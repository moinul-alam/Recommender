import os
import logging
from fastapi import HTTPException
from src.models.collaborative.v2.services.preprocessing_service import PreprocessingService
from src.models.collaborative.v2.services.model_training_service import ModelTrainingService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PipelineService:
    @staticmethod
    def execute_full_pipeline(
        dataset_dir_path: str,
        processed_dir_path: str,
        model_dir_path: str,
        sparse_user_threshold: int = 5,
        sparse_item_threshold: int = 1,
        split_percent: float = 0.8,
        chunk_size: int = 10000,
        n_neighbors: int = 50,
        normalization: str = None,
        similarity_metric: str = "L2",
        batch_size: int = 1000,
        min_similarity: float = 0.1
    ):
        try:
            for path in [dataset_dir_path, processed_dir_path, model_dir_path]:
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                elif not os.path.isdir(path):
                    raise ValueError(f"Provided path '{path}' is not a directory")

            preprocessing_result = PreprocessingService.process_data(
                dataset_dir_path=dataset_dir_path,
                processed_dir_path=processed_dir_path,
                sparse_user_threshold=sparse_user_threshold,
                sparse_item_threshold=sparse_item_threshold,
                split_percent=split_percent,
                chunk_size=chunk_size,
                normalization=normalization
            )

            if not preprocessing_result:
                return {"status": "Preprocessing failed"}

            training_result = ModelTrainingService.train_model(
                processed_dir_path=processed_dir_path,
                model_dir_path=model_dir_path,
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

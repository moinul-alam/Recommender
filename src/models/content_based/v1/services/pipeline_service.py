import os
import logging
from src.models.content_based.v1.services.preprocessing_service import PreprocessingService
from src.models.content_based.v1.services.engineering_service import EngineeringService
from src.models.content_based.v1.services.training_service import TrainingService
from src.schemas.content_based_schema import PipelineResponse

logger = logging.getLogger(__name__)

class PipelineService:
    @staticmethod
    def execute_full_pipeline(
        dataset_path: str,
        processed_folder_path: str,
        features_folder_path: str,
        model_folder_path: str,
        segment_size: int = 6000,
        metric: str = "L2"
    ) -> PipelineResponse :
        """
        Execute the full content-based recommendation pipeline.
        
        Args:
            dataset_path (str): Path to the raw dataset
            processed_folder_path (str): Path for storing processed data
            features_folder_path (str): Path for storing feature-engineered data
            model_folder_path (str): Path for storing trained model
            segment_size (int, optional): Size of data segments. Defaults to 6000.
            metric (str, optional): Similarity metric for FAISS. Defaults to "L2".
        
        Returns:
            dict: Pipeline execution results
        """
        try:
            # Ensure output directories exist
            os.makedirs(processed_folder_path, exist_ok=True)
            os.makedirs(features_folder_path, exist_ok=True)
            os.makedirs(model_folder_path, exist_ok=True)

            # Step 1: Data Preprocessing
            preprocessing_result = PreprocessingService.preprocess_data(
                dataset_path=dataset_path,
                processed_folder_path=processed_folder_path,
                segment_size=segment_size
            )

            # Step 2: Feature Engineering
            feature_engineering_result = EngineeringService.engineer_feature(
                processed_folder_path=processed_folder_path,
                features_folder_path=features_folder_path
            )

            # Step 3: Model Training
            model_training_result = TrainingService.train_model(
                features_folder_path=features_folder_path,
                model_folder_path=model_folder_path,
                metric=metric
            )

            return {
                "preprocessing": preprocessing_result,
                "feature_engineering": feature_engineering_result,
                "model_training": model_training_result
            }
            

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise

# return PipelineResponse(
#                 status="Model Training successful",
#                 output=1,
#                 output_path=None
#             )
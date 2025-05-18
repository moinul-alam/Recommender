import logging
from pathlib import Path
from fastapi import HTTPException
from src.models.common.DataLoader import load_data
from src.models.common.file_config import file_names
from src.models.common.DataSaver import save_objects
from src.models.collaborative.v2.pipeline.data_preprocessing import DataPreprocessing
from src.schemas.content_based_schema import PipelineResponse
from src.models.common.logger import app_logger

logger = app_logger(__name__)

class DataPreprocessingService:
    """Service for preprocessing data for collaborative filtering pipelines."""
    
    @staticmethod
    def process_data(directory_path: str) -> PipelineResponse:
        """
        Preprocess dataset for collaborative filtering and save results.

        Args:
            directory_path (str): Path to the directory containing the dataset and where outputs will be saved.

        Returns:
            PipelineResponse: Response indicating the status and output directory.

        Raises:
            HTTPException: If the directory path is invalid, dataset is missing/empty, or preprocessing fails.
        """
        logger.info("Starting data preprocessing...")

        # Set default values for parameters
        sparse_user_threshold: int = 10
        sparse_item_threshold: int = 10
        split_percent: float = 0.8
        segment_size: int = 10000

        try:
            # Validate input and output paths
            directory_path = Path(directory_path)
            if not directory_path.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid directory path: {directory_path}"
                )

            # Locate dataset file
            dataset_path = directory_path / file_names["dataset_name"]
            if not dataset_path.is_file():
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset file not found: {dataset_path}"
                )

            # Load dataset
            dataset = load_data(dataset_path)
            logger.info(f"Dataset loaded from {dataset_path}")
            if dataset is None or dataset.empty:
                raise HTTPException(status_code=400, detail="Dataset is empty or invalid")

            # Initialize preprocessor
            preprocessor = DataPreprocessing(
                sparse_user_threshold,
                sparse_item_threshold,
                split_percent,
                segment_size
            )

            # Process data
            results = preprocessor.process(df=dataset)
            if results is None or not results:
                raise HTTPException(status_code=400, detail="Preprocessed data is empty or invalid")

            # Log preprocessing results
            logger.info(f"Preprocessing complete. Train set size: {len(results['train'])}, "
                        f"Test set size: {len(results['test'])}")
            logger.info(f"User mapping size: {len(results['user_mapping'])}, "
                        f"Item mapping size: {len(results['item_mapping'])}")
            logger.info(f"User-item matrix shape: {results['user_item_matrix'].shape}")

            # Sample output for debugging
            if results["item_mapping"]:
                logger.info(f"Sample item mappings (first 5): {list(results['item_mapping'].items())[:5]}")
                logger.info(f"Sample item reverse mappings (first 5): {list(results['item_reverse_mapping'].items())[:5]}")

            # Prepare mappings for saving
            user_item_mappings = {
                "user_mapping": results["user_mapping"],
                "user_reverse_mapping": results["user_reverse_mapping"],
                "item_mapping": results["item_mapping"],
                "item_reverse_mapping": results["item_reverse_mapping"]
            }

            # Files to save
            files_to_save = {
                file_names["train_set"]: results["train"],
                file_names["test_set"]: results["test"],
                file_names["user_item_matrix"]: results["user_item_matrix"],
                file_names["user_item_mappings"]: user_item_mappings
            }

            # Save processed data
            save_objects(
                directory_path=directory_path,
                objects=files_to_save,
                compress=3
            )

            # Test the saved user-item matrix
            logger.info("Testing the saved user-item matrix...")
            test_user_idx = 0
            user_item_matrix = results['user_item_matrix']
            if user_item_matrix.shape[0] > test_user_idx:
                test_item_indices = user_item_matrix[test_user_idx].indices
                ratings = user_item_matrix[test_user_idx].data

                num_items_rated = len(test_item_indices)
                logger.info(f"User {test_user_idx} has rated {num_items_rated} items.")

                max_items_to_show = 5
                if num_items_rated > max_items_to_show:
                    logger.info(f"Showing ratings for the first {max_items_to_show} items:")
                else:
                    logger.info(f"Showing ratings for all {num_items_rated} items:")

                for item, rating in zip(test_item_indices[:max_items_to_show], ratings[:max_items_to_show]):
                    logger.info(f"User {test_user_idx} -> Item {item} with rating {rating}")
            else:
                logger.warning(f"No ratings available for user {test_user_idx} in the user-item matrix.")

            # Test user-item mappings
            logger.info(f"Testing user item mappings: {list(user_item_mappings.keys())}")
            logger.info(f"Sample user_mapping: {list(user_item_mappings['user_mapping'].items())[:5]}")

            logger.info(f"Data saved successfully in {directory_path}")

            return PipelineResponse(
                status="success",
                message="Data preprocessing completed successfully",
                output=str(directory_path)
            )

        except Exception as e:
            logger.error(f"An error occurred during data preprocessing: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred during data preprocessing: {e}"
            )
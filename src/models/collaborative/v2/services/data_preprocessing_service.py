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


            # Files to save
            files_to_save = {
                file_names["train_set"]: results["train"],
                file_names["test_set"]: results["test"],
                file_names["user_item_matrix"]: results["user_item_matrix"],
                file_names["user_item_mappings"]: results["user_item_mappings"],
                file_names["movieId_tmdbId_mapping"]: results["movieId_tmdbId_mapping"]
            }

            # Save processed data
            save_objects(
                directory_path=directory_path,
                objects=files_to_save,
                compress=3
            )

            # Test the saved user-item matrix with multiple users
            logger.info("Testing the saved user-item matrix...")
            user_item_matrix = results['user_item_matrix']
            
            # Get statistics on the matrix
            nnz_per_user = [user_item_matrix[i].nnz for i in range(user_item_matrix.shape[0])]
            users_with_ratings = sum(1 for x in nnz_per_user if x > 0)
            total_users = user_item_matrix.shape[0]
            
            logger.info(f"Matrix statistics: {users_with_ratings}/{total_users} users have ratings")
            
            # Check density distribution
            if users_with_ratings > 0:
                avg_items_per_user = sum(nnz_per_user) / users_with_ratings
                logger.info(f"Average items per user with ratings: {avg_items_per_user:.2f}")
            
            # Sample a few users to check (first, middle, and last with ratings)
            users_to_check = []
            
            # Add first user with ratings
            for i in range(total_users):
                if user_item_matrix[i].nnz > 0:
                    users_to_check.append(i)
                    break
            
            # Add a user from the middle with ratings if available
            mid_user = total_users // 2
            search_range = min(100, total_users // 4)  # Look in the vicinity of the middle
            for i in range(mid_user - search_range, min(mid_user + search_range, total_users)):
                if 0 <= i < total_users and user_item_matrix[i].nnz > 0 and i not in users_to_check:
                    users_to_check.append(i)
                    break
            
            # Add last user with ratings
            for i in range(total_users - 1, -1, -1):
                if user_item_matrix[i].nnz > 0 and i not in users_to_check:
                    users_to_check.append(i)
                    break
            
            # Display ratings for selected users
            for test_user_idx in users_to_check:
                test_item_indices = user_item_matrix[test_user_idx].indices
                ratings = user_item_matrix[test_user_idx].data

                num_items_rated = len(test_item_indices)
                logger.info(f"User {test_user_idx} has rated {num_items_rated} items.")

                max_items_to_show = 5
                if num_items_rated > 0:
                    show_items = min(max_items_to_show, num_items_rated)
                    logger.info(f"Showing ratings for {show_items} items:")
                    for item, rating in zip(test_item_indices[:show_items], ratings[:show_items]):
                        logger.info(f"User {test_user_idx} -> Item {item} with rating {rating}")
                else:
                    logger.warning(f"No ratings available for user {test_user_idx} in the user-item matrix.")
            
            # Verify matrix corresponds to training data
            train_df = results["train"]
            sample_size = min(5, len(train_df))
            if sample_size > 0:
                logger.info("Verifying matrix entries against training data:")
                sample = train_df.sample(sample_size)
                
                for _, row in sample.iterrows():
                    user_id = int(row["userId"])
                    item_id = int(row["movieId"])
                    expected_rating = float(row["rating"])
                    
                    if user_id < user_item_matrix.shape[0] and item_id < user_item_matrix.shape[1]:
                        actual_rating = user_item_matrix[user_id, item_id]
                        if actual_rating == expected_rating:
                            logger.info(f"Verified: User {user_id} -> Item {item_id} = {actual_rating}")
                        else:
                            logger.warning(f"Mismatch: User {user_id} -> Item {item_id}, expected {expected_rating}, got {actual_rating}")
                    else:
                        logger.warning(f"Out of bounds: User {user_id} -> Item {item_id} outside matrix dimensions {user_item_matrix.shape}")

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
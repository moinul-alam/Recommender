import logging
import os
import json
import pathlib
import pickle
from typing import Dict, Optional
import pandas as pd
from src.models.collaborative.v1.pipeline.DataPreprocessing import DataPreprocessing

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PreprocessingService:
    @staticmethod
    def process_data(dataset_dir_path: str, processed_dir_path: str, 
                     sparse_threshold: int = 5, split_percent: float = 0.8) -> Optional[Dict[str, str]]:
        
        try:
            # Use pathlib for more robust path handling
            input_file = pathlib.Path(dataset_dir_path) / "movielens_small.csv"
            
            if not input_file.exists():
                logger.error(f"Dataset file not found: {input_file}")
                return None

            logger.info(f"Loading dataset from {input_file}")
            df = pd.read_csv(input_file)

            processor = DataPreprocessing(sparse_threshold, split_percent)
            train, test, item_mapping, user_mapping, item_reverse_mapping, user_reverse_mapping, user_item_matrix = processor.process(df)

            # Ensure processed directory exists
            processed_path = pathlib.Path(processed_dir_path)
            processed_path.mkdir(parents=True, exist_ok=True)

            # Define output paths
            paths = {
                "train_path": processed_path / "train.feather",
                "test_path": processed_path / "test.feather",
                "item_mapping_path": processed_path / "item_mapping.pkl",
                "user_mapping_path": processed_path / "user_mapping.pkl",
                "item_reverse_mapping": processed_path / "item_reverse_mapping.pkl",
                "user_reverse_mapping": processed_path / "user_reverse_mapping.pkl",
                "user_item_matrix_path": processed_path / "user_item_matrix.pkl"
            }

            # Save processed files
            train.to_feather(paths["train_path"])
            test.to_feather(paths["test_path"])

            with open(paths["item_mapping_path"], "wb") as f:
                pickle.dump(item_mapping, f)

            with open(paths["user_mapping_path"], "wb") as f:
                pickle.dump(user_mapping, f)

            with open(paths["item_reverse_mapping"], "wb") as f:
                pickle.dump(item_reverse_mapping, f)
            
            with open(paths["user_reverse_mapping"], "wb") as f:
                pickle.dump(user_reverse_mapping, f)

            with open(paths["user_item_matrix_path"], "wb") as f:
                pickle.dump(user_item_matrix, f)

            logger.info(f"Data processing complete. Files saved in {processed_path}")
            return {str(k): str(v) for k, v in paths.items()}

        except Exception as e:
            logger.error(f"Error during data processing: {e}")
            return None
import logging
import os
import json
import pathlib
import pickle
from typing import Dict, Optional, Tuple
import pandas as pd
from src.models.collaborative.v1.pipeline.DataPreprocessing import DataPreprocessing

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PreprocessingService:    
    @staticmethod
    def validate_input_paths(dataset_dir: str, processed_dir: str) -> bool:
        dataset_path = pathlib.Path(dataset_dir)
        processed_path = pathlib.Path(processed_dir)
        
        if not dataset_path.is_dir():
            logging.error(f"Invalid dataset directory: {dataset_dir}")
            return False
        
        processed_path.mkdir(parents=True, exist_ok=True)
        return True

    @classmethod
    def process_data(
        cls, 
        dataset_dir_path: str, 
        processed_dir_path: str, 
        sparse_user_threshold: int = 5,
        sparse_item_threshold: int = 5,
        split_percent: float = 0.8
    ) -> Optional[Dict[str, str]]:
        logger = logging.getLogger(cls.__name__)
        logger.setLevel(logging.INFO)
        
        try:
            # Validate paths
            if not cls.validate_input_paths(dataset_dir_path, processed_dir_path):
                return None
            
            # Locate input file
            input_file = pathlib.Path(dataset_dir_path) / "movielens_dataset.csv"
            if not input_file.exists():
                logger.error(f"Dataset file not found: {input_file}")
                return None
            
            # Load dataset
            logger.info(f"Loading dataset from {input_file}")
            df = pd.read_csv(input_file)
            
            # Initialize processor
            processor = DataPreprocessing(
                sparse_user_threshold=sparse_user_threshold,
                sparse_item_threshold=sparse_item_threshold,
                split_percent=split_percent
            )
            
            # Process data
            train, test, item_mapping, \
            item_reverse_mapping, \
            user_item_matrix = processor.process(df)
            
            print(list(item_mapping.items())[:5])
            print(list(item_reverse_mapping.items())[:5])
            
            # Prepare output paths
            processed_path = pathlib.Path(processed_dir_path)
            paths = {
                "train_path": processed_path / "train.feather",
                "test_path": processed_path / "test.feather",
                "item_mapping_path": processed_path / "item_mapping.pkl",
                "item_reverse_mapping_path": processed_path / "item_reverse_mapping.pkl",
                "user_item_matrix_path": processed_path / "user_item_matrix.pkl"
            }
            
            # Save processed files
            train.to_feather(paths["train_path"])
            test.to_feather(paths["test_path"])
            
            # Serialization utility for mappings
            def safe_pickle_dump(obj, path):
                with open(path, "wb") as f:
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            safe_pickle_dump(item_mapping, paths["item_mapping_path"])
            safe_pickle_dump(item_reverse_mapping, paths["item_reverse_mapping_path"])
            safe_pickle_dump(user_item_matrix, paths["user_item_matrix_path"])
            
            logger.info(f"Data preprocessing complete. Files saved in {processed_path}")
            
            return {str(k): str(v) for k, v in paths.items()}
        
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}", exc_info=True)
            return None
import logging
import os
import json
import pathlib
import pickle
from typing import Dict, Optional, Tuple
import pandas as pd
from src.models.collaborative.v3.pipeline.DataPreprocessing import DataPreprocessing

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PreprocessingService:    
    @staticmethod
    def validate_input_paths(dataset_dir: str, processed_dir: str) -> bool:
        dataset_path = pathlib.Path(dataset_dir)
        processed_path = pathlib.Path(processed_dir)

        if not dataset_path.is_dir():
            logger.error(f"Invalid dataset directory: {dataset_dir}")
            return False
        
        dataset_file = dataset_path / "movielens_dataset.csv"
        if not dataset_file.exists():
            logger.error(f"Dataset file not found: {dataset_file}")
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
        split_percent: float = 0.8,
        chunk_size: int = 10000,
        normalization: Optional[str] = None
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
                split_percent=split_percent,
                chunk_size = chunk_size,
                normalization = normalization
            )
            
            # Process data
            train, test, user_mapping, user_reverse_mapping, \
            item_mapping, item_reverse_mapping, \
            user_item_matrix = processor.process(df)
            
            print(list(item_mapping.items())[:5])
            print(list(item_reverse_mapping.items())[:5])
            
            # Prepare output paths
            processed_path = pathlib.Path(processed_dir_path)
            paths = {
                "train_path": processed_path / "train.feather",
                "test_path": processed_path / "test.feather",
                "user_mapping_path": processed_path / "user_mapping.pkl",
                "user_reverse_mapping_path": processed_path / "user_reverse_mapping.pkl",
                "item_mapping_path": processed_path / "item_mapping.pkl",
                "item_reverse_mapping_path": processed_path / "item_reverse_mapping.pkl",
                "user_item_matrix_path": processed_path / "user_item_matrix.pkl"
            }
            
            # Save processed files
            train.reset_index(drop=True).to_feather(paths["train_path"])
            test.reset_index(drop=True).to_feather(paths["test_path"])
            
            # Serialization utility for mappings
            def safe_pickle_dump(obj, path):
                try:
                    with open(path, "wb") as f:
                        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    logger.error(f"Failed to save {path}: {e}", exc_info=True)

            safe_pickle_dump(user_mapping, paths["user_mapping_path"])
            safe_pickle_dump(user_reverse_mapping, paths["user_reverse_mapping_path"])
            safe_pickle_dump(item_mapping, paths["item_mapping_path"])
            safe_pickle_dump(item_reverse_mapping, paths["item_reverse_mapping_path"])
            safe_pickle_dump(user_item_matrix, paths["user_item_matrix_path"])
            
            logger.info(f"Data preprocessing complete. Files saved in {processed_path}")
            
            return {str(k): str(v) for k, v in paths.items()}
        
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}", exc_info=True)
            return None
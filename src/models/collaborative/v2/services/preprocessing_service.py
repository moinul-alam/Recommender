import logging
import os
import json
from pathlib import Path
import pickle
from typing import Dict, Optional
from fastapi import HTTPException
import pandas as pd
from models.common.DataLoader import load_data
from models.common.DataSaver import save_data
from models.collaborative.v2.pipeline.data_preprocessing import DataPreprocessing

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PreprocessingService:    
    @staticmethod
    def process_data(
        collaborative_dir_path: str,
        file_names: dict,
        sparse_user_threshold: int = 5,
        sparse_item_threshold: int = 5,
        split_percent: float = 0.8,
        segment_size: int = 10000,
    ) -> Optional[Dict[str, str]]:
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        try:
            # Validate input and output paths
            collaborative_dir_path = Path(collaborative_dir_path)
            
            if not collaborative_dir_path.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid directory path: {collaborative_dir_path}"
                )
            
            # Locate datset file
            dataset_path = collaborative_dir_path / file_names["dataset_name"]
            
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
            train, test, user_mapping, user_reverse_mapping, \
            item_mapping, item_reverse_mapping, \
            user_item_matrix = preprocessor.process(df = dataset)
            
            print(list(item_mapping.items())[:5])
            print(list(item_reverse_mapping.items())[:5])
            
            # Prepare output paths
            paths = {
                "train_path": collaborative_dir_path / "2_train.feather",
                "test_path": collaborative_dir_path / "2_test.feather",
                "user_mapping_path": collaborative_dir_path / "2_user_mapping.pkl",
                "user_reverse_mapping_path": collaborative_dir_path / "2_user_reverse_mapping.pkl",
                "item_mapping_path": collaborative_dir_path / "2_item_mapping.pkl",
                "item_reverse_mapping_path": collaborative_dir_path / "2_item_reverse_mapping.pkl",
                "user_item_matrix_path": collaborative_dir_path / "2_user_item_matrix.pkl"
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
            
            logger.info(f"Data preprocessing complete. Files saved in {collaborative_dir_path}")
            
            return {str(k): str(v) for k, v in paths.items()}
        
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}", exc_info=True)
            return None
import logging
import os
from typing import Dict, Tuple, Optional, Union
import pathlib
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DataPreprocessing:
    def __init__(
        self, 
        sparse_user_threshold: int = 5, 
        sparse_item_threshold: int = 5,
        split_percent: float = 0.8
    ):
        """
        Initialize preprocessing parameters.
        
        Args:
            sparse_user_threshold (int): Minimum number of ratings for a user to be retained
            sparse_item_threshold (int): Minimum number of ratings for an item to be retained
            split_percent (float): Percentage of data to use for training
        """
        self._validate_parameters(sparse_user_threshold, sparse_item_threshold, split_percent)
        
        self.sparse_user_threshold = sparse_user_threshold
        self.sparse_item_threshold = sparse_item_threshold
        self.split_percent = split_percent
        
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _validate_parameters(
        self, 
        sparse_user_threshold: int, 
        sparse_item_threshold: int, 
        split_percent: float
    ) -> None:
        if sparse_user_threshold < 1:
            raise ValueError(f"User sparse threshold must be ≥ 1, got {sparse_user_threshold}")
        
        if sparse_item_threshold < 1:
            raise ValueError(f"Item sparse threshold must be ≥ 1, got {sparse_item_threshold}")
        
        if not 0 < split_percent < 1:
            raise ValueError(f"Split percentage must be between 0 and 1, got {split_percent}")

    def drop_sparse_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop users with few ratings
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= self.sparse_user_threshold].index
        
        # Drop items with few ratings
        item_counts = df["tmdb_id"].value_counts()
        valid_items = item_counts[item_counts >= self.sparse_item_threshold].index
        
        filtered_df = df[
            df["user_id"].isin(valid_users) & 
            df["tmdb_id"].isin(valid_items)
        ]
        
        self.logger.info(
            f"Filtering: "
            f"Original users: {len(user_counts)}, Remaining: {len(valid_users)} | "
            f"Original items: {len(item_counts)}, Remaining: {len(valid_items)}"
        )
        
        return filtered_df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_count = df["rating"].isnull().sum()
        if missing_count > 0:
            # Global mean imputation
            mean_rating = df["rating"].mean()
            df["rating"].fillna(mean_rating, inplace=True)
            
            self.logger.info(
                f"Imputed {missing_count} missing ratings "
                f"with global mean: {mean_rating:.2f}"
            )
        
        return df

    def create_mappings(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict]:
        # Create mappings
        unique_items = sorted(df["tmdb_id"].unique())
        
        # Ensure all IDs are integers
        item_mapping = {int(old_id): int(new_id) for new_id, old_id in enumerate(unique_items)}
        item_reverse_mapping = {int(v): int(k) for k, v in item_mapping.items()}
        
        # Convert tmdb_id to integer type before mapping
        df["tmdb_id"] = df["tmdb_id"].astype(int)
        df["tmdb_id"] = df["tmdb_id"].map(item_mapping)
        
        self.logger.info(f"Created mappings for {len(item_mapping)} items")
        
        return df, item_mapping, item_reverse_mapping

    def create_user_item_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        rating_column = "rating"
        
        # Ensure tmdb_id is int type
        df["tmdb_id"] = df["tmdb_id"].astype(int)
        
        # Aggregate potential duplicate entries
        df_grouped = df.groupby(['user_id', 'tmdb_id'])[rating_column].mean().reset_index()
        
        user_item_matrix = df_grouped.pivot(
            index='user_id', 
            columns='tmdb_id', 
            values=rating_column
        ).fillna(0)
        
        self.logger.info(f"User-Item matrix created: {user_item_matrix.shape}")
        return user_item_matrix

    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train, test = train_test_split(
            df, 
            test_size=(1 - self.split_percent), 
            random_state=42,
            stratify=df["user_id"] if len(df["user_id"].unique()) > 1 else None
        )
        
        # Ensure test set only contains users from training set
        test = test[test["user_id"].isin(train["user_id"])]
        
        self.logger.info(
            f"Train-test split: "
            f"Train size: {len(train)}, "
            f"Test size: {len(test)}"
        )
        
        return train, test

    def process(self, df: pd.DataFrame) -> Tuple[
        pd.DataFrame, 
        pd.DataFrame, 
        dict, 
        dict,  
        pd.DataFrame
    ]:
        # Validate input dataframe
        required_columns = ["user_id", "tmdb_id", "rating", "timestamp"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Input dataframe must contain columns: {required_columns}")
        
        # Convert tmdb_id to int if it isn't already
        df["tmdb_id"] = df["tmdb_id"].astype(int)
        
        # Drop timestamp for processing
        df = df.drop(columns=['timestamp'])
        
        # Preprocessing pipeline
        df = self.drop_sparse_entities(df)
        df = self.handle_missing_values(df)
        
        df, item_mapping, item_reverse_mapping = self.create_mappings(df)
        
        user_item_matrix = self.create_user_item_matrix(df)
        train, test = self.split_dataset(df)
        
        return (
            train, test, 
            item_mapping, 
            item_reverse_mapping,
            user_item_matrix
        )

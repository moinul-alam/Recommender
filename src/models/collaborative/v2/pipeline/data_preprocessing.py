import logging
import os
from typing import Dict, Tuple, Optional, Union
import pathlib
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from scipy import sparse
from sklearn.preprocessing import normalize
import gc

# Configure global logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataPreprocessing:
    def __init__(
        self, 
        sparse_user_threshold: int = 5, 
        sparse_item_threshold: int = 5,
        split_percent: float = 0.8,
        segment_size: int = 10000
    ):
        self._validate_parameters(sparse_user_threshold, sparse_item_threshold, split_percent)
        
        self.sparse_user_threshold = sparse_user_threshold
        self.sparse_item_threshold = sparse_item_threshold
        self.split_percent = split_percent
        self.segment_size = segment_size
        
        self.logger = logging.getLogger(self.__class__.__name__)

    def _validate_parameters(self, sparse_user_threshold: int, sparse_item_threshold: int, split_percent: float) -> None:
        if sparse_user_threshold < 1:
            raise ValueError(f"User sparse threshold must be ≥ 1, got {sparse_user_threshold}")
        if sparse_item_threshold < 1:
            raise ValueError(f"Item sparse threshold must be ≥ 1, got {sparse_item_threshold}")
        if not 0 < split_percent < 1:
            raise ValueError(f"Split percentage must be between 0 and 1, got {split_percent}")

    def drop_sparse_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Dropping sparse users and items...")
        
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= self.sparse_user_threshold].index
        
        item_counts = df["tmdb_id"].value_counts()
        valid_items = item_counts[item_counts >= self.sparse_item_threshold].index
        
        filtered_df = df[df["user_id"].isin(valid_users) & df["tmdb_id"].isin(valid_items)]
        
        logger.info(f"Filtered dataset: {len(valid_users)} users, {len(valid_items)} items remain.")
        return filtered_df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_count = df["rating"].isnull().sum()
        if missing_count > 0:
            mean_rating = df["rating"].mean()
            df["rating"].fillna(mean_rating, inplace=True)
            logger.info(f"Filled {missing_count} missing ratings with mean rating {mean_rating:.2f}")
        return df

    def create_mappings(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict, dict, dict]:
        logger.info("Creating item and user mappings...")
        
        unique_users = sorted(df["user_id"].unique())
        unique_items = sorted(df["tmdb_id"].unique())

        user_mapping = {int(old_id): int(new_id) for new_id, old_id in enumerate(unique_users)}
        user_reverse_mapping = {int(v): int(k) for k, v in user_mapping.items()}

        item_mapping = {int(old_id): int(new_id) for new_id, old_id in enumerate(unique_items)}
        item_reverse_mapping = {int(v): int(k) for k, v in item_mapping.items()}

        df["user_id"] = df["user_id"].map(user_mapping)
        df["tmdb_id"] = df["tmdb_id"].map(item_mapping)

        logger.info(f"Generated {len(user_mapping)} user mappings and {len(item_mapping)} item mappings.")
        
        return df, user_mapping, user_reverse_mapping, item_mapping, item_reverse_mapping

    def create_user_item_matrix(self, df: pd.DataFrame) -> sparse.csr_matrix:
        logger.info("Creating user-item matrix...")

        n_users = df["user_id"].max() + 1
        n_items = df["tmdb_id"].max() + 1
        
        rows, cols, data = [], [], []
        
        for start in range(0, len(df), self.segment_size):
            end = start + self.segment_size
            chunk = df.iloc[start:end]
            
            rows.extend(chunk["user_id"].values)
            cols.extend(chunk["tmdb_id"].values)
            data.extend(chunk["rating"].values)
            
            del chunk
            gc.collect()
        
        user_item_matrix = sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_items)
        ).tocsr()
        
        del rows, cols, data
        gc.collect()

        user_means = np.array(user_item_matrix.sum(axis=1)).flatten() / (
            np.maximum(user_item_matrix.getnnz(axis=1), 1)
        )

        for i in range(n_users):
            user_ratings = user_item_matrix[i].data
            if len(user_ratings) > 0:
                user_item_matrix[i].data = user_ratings - user_means[i]

        logger.info(
            f"Created sparse matrix: {user_item_matrix.shape}, "
            f"density: {user_item_matrix.nnz / (n_users * n_items):.4%}"
        )
        return user_item_matrix

    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Performing time-aware train-test split...")

        df = df.sort_values(by=["user_id", "timestamp"])

        train_list, test_list = [], []

        for user, user_df in df.groupby("user_id"):
            n_ratings = len(user_df)

            if n_ratings < 2:  
                train_list.append(user_df)  # Keep all in train if only one rating
            else:
                split_idx = int(n_ratings * self.split_percent)
                train_list.append(user_df.iloc[:split_idx])
                test_list.append(user_df.iloc[split_idx:])

        train = pd.concat(train_list)
        test = pd.concat(test_list)

        test = test[test["user_id"].isin(train["user_id"])]  # Ensure all test users exist in train

        logger.info(f"Train-test split complete: Train size {len(train)}, Test size {len(test)}")
        return train.drop(columns=["timestamp"]), test.drop(columns=["timestamp"])

    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict, dict, sparse.csr_matrix]:
        logger.info("Starting data preprocessing pipeline...")

        required_columns = ["user_id", "tmdb_id", "rating", "timestamp"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Input dataframe must contain columns: {required_columns}")

        df = self.drop_sparse_entities(df)
        # df = self.handle_missing_values(df)

        # Split before dropping timestamp**
        train, test = self.split_dataset(df)

        # Create mappings after splitting
        train, user_mapping, user_reverse_mapping, item_mapping, item_reverse_mapping = self.create_mappings(train)
        test["user_id"] = test["user_id"].map(user_mapping)
        test["tmdb_id"] = test["tmdb_id"].map(item_mapping)
        test = test.dropna()

        user_item_matrix = self.create_user_item_matrix(train)

        logger.info("Data preprocessing completed successfully.")
        return train, test, user_mapping, user_reverse_mapping, item_mapping, item_reverse_mapping, user_item_matrix

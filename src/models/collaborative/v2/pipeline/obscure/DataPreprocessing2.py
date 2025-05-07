import logging
import os
from typing import Dict, Tuple, Optional, Union
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from scipy import sparse
from sklearn.preprocessing import normalize
import gc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataPreprocessing:
    def __init__(
        self,
        sparse_user_threshold: int = 5,
        sparse_item_threshold: int = 5,
        split_percent: float = 0.8,
        chunk_size: int = 10000,
        normalization: Optional[str] = None
    ):
        """
        Initialize preprocessing parameters.
        """
        self._validate_parameters(sparse_user_threshold, sparse_item_threshold, split_percent, normalization)

        self.sparse_user_threshold = sparse_user_threshold
        self.sparse_item_threshold = sparse_item_threshold
        self.split_percent = split_percent
        self.chunk_size = chunk_size
        self.normalization = normalization

        self.logger = logging.getLogger(self.__class__.__name__)

    def _validate_parameters(self, sparse_user_threshold, sparse_item_threshold, split_percent, normalization):
        if sparse_user_threshold < 1:
            raise ValueError(f"User sparse threshold must be ≥ 1, got {sparse_user_threshold}")
        if sparse_item_threshold < 1:
            raise ValueError(f"Item sparse threshold must be ≥ 1, got {sparse_item_threshold}")
        if not 0 < split_percent < 1:
            raise ValueError(f"Split percentage must be between 0 and 1, got {split_percent}")
        if normalization not in [None, 'l1', 'l2']:
            raise ValueError(f"Invalid normalization type: {normalization}. Choose 'l1', 'l2', or None.")

    def drop_sparse_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes users and items with fewer interactions than the specified threshold."""
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= self.sparse_user_threshold].index

        item_counts = df["tmdb_id"].value_counts()
        valid_items = item_counts[item_counts >= self.sparse_item_threshold].index

        filtered_df = df[df["user_id"].isin(valid_users) & df["tmdb_id"].isin(valid_items)]

        self.logger.info(f"Filtering: Users {len(valid_users)}/{len(user_counts)}, Items {len(valid_items)}/{len(item_counts)}")
        return filtered_df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing ratings using user mean, then item mean, then global mean."""
        missing_count = df["rating"].isnull().sum()
        if missing_count > 0:
            df["rating"] = df["rating"].fillna(df.groupby("user_id")["rating"].transform("mean"))
            df["rating"] = df["rating"].fillna(df.groupby("tmdb_id")["rating"].transform("mean"))
            df["rating"] = df["rating"].fillna(df["rating"].mean())

            self.logger.info(f"Imputed {missing_count} missing ratings using hierarchical mean strategy.")
        return df

    def create_mappings(self, df: pd.DataFrame) -> Tuple[dict, dict]:
        """Creates item ID mappings and returns them as dictionaries."""
        unique_items = sorted(df["tmdb_id"].unique())
        item_mapping = {int(old_id): int(new_id) for new_id, old_id in enumerate(unique_items)}
        item_reverse_mapping = {int(v): int(k) for k, v in item_mapping.items()}

        self.logger.info(f"Created mappings for {len(item_mapping)} items.")
        return item_mapping, item_reverse_mapping

    def apply_mappings(self, df: pd.DataFrame, item_mapping: dict) -> pd.DataFrame:
        """Applies item ID mappings to the dataset."""
        df = df.copy()
        df["tmdb_id"] = df["tmdb_id"].map(item_mapping)
        return df

    def create_user_item_matrix(self, df: pd.DataFrame) -> sparse.csr_matrix:
        """Creates a sparse user-item matrix using efficient chunking."""
        unique_users = sorted(df["user_id"].unique())
        user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}

        df["user_idx"] = df["user_id"].map(user_mapping)
        n_users = len(unique_users)
        n_items = df["tmdb_id"].max() + 1

        chunks = []
        for start in range(0, len(df), self.chunk_size):
            end = start + self.chunk_size
            chunk = df.iloc[start:end]

            chunk_matrix = sparse.coo_matrix(
                (chunk["rating"].values, (chunk["user_idx"].values, chunk["tmdb_id"].values)),
                shape=(n_users, n_items)
            )
            chunks.append(chunk_matrix.tocsr())

            del chunk
            gc.collect()

        user_item_matrix = sparse.vstack(chunks).tocsr()

        if self.normalization:
            user_item_matrix = normalize(user_item_matrix, norm=self.normalization, axis=1)

        density = user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1])
        self.logger.info(f"Sparse User-Item matrix created: {user_item_matrix.shape}, Density: {density:.4%}")

        return user_item_matrix

    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the dataset into train and test sets and ensures test users exist in train."""
        train, test = train_test_split(df, test_size=(1 - self.split_percent), random_state=42,
                                       stratify=df["user_id"] if len(df["user_id"].unique()) > 1 else None)

        test = test[test["user_id"].isin(train["user_id"])]  # Ensure test users exist in train
        self.logger.info(f"Train-test split: Train {len(train)}, Test {len(test)}")

        return train, test

    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict, dict, sparse.csr_matrix]:
        """
        Main processing function to clean data, create mappings, generate matrices, and split datasets.
        """
        required_columns = {"user_id", "tmdb_id", "rating", "timestamp"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Input dataframe must contain columns: {required_columns}")

        df = df.drop(columns=['timestamp'])
        df = self.drop_sparse_entities(df)
        df = self.handle_missing_values(df)

        train, test = self.split_dataset(df)

        item_mapping, item_reverse_mapping = self.create_mappings(train)  # Mapping only on train
        train = self.apply_mappings(train, item_mapping)
        test = self.apply_mappings(test, item_mapping)

        user_item_matrix = self.create_user_item_matrix(train)

        return train, test, item_mapping, item_reverse_mapping, user_item_matrix

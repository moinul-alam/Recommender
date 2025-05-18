import logging
from typing import Dict, Tuple
import pandas as pd
from scipy import sparse
import gc
from src.models.common.logger import app_logger

logger = app_logger(__name__)

class DataPreprocessing:
    def __init__(
        self, 
        sparse_user_threshold: int = 10, 
        sparse_item_threshold: int = 10,
        split_percent: float = 0.8,
        segment_size: int = 10000
    ):
        self._validate_parameters(sparse_user_threshold, sparse_item_threshold, split_percent)
        
        self.sparse_user_threshold = sparse_user_threshold
        self.sparse_item_threshold = sparse_item_threshold
        self.split_percent = split_percent
        self.segment_size = segment_size

    def _validate_parameters(self, sparse_user_threshold: int, sparse_item_threshold: int, split_percent: float) -> None:
        if sparse_user_threshold < 1:
            raise ValueError(f"User sparse threshold must be ≥ 1, got {sparse_user_threshold}")
        if sparse_item_threshold < 1:
            raise ValueError(f"Item sparse threshold must be ≥ 1, got {sparse_item_threshold}")
        if not 0 < split_percent < 1:
            raise ValueError(f"Split percentage must be between 0 and 1, got {split_percent}")

    def drop_sparse_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Dropping sparse users and items...")
        
        user_counts = df["userId"].value_counts()
        valid_users = user_counts[user_counts >= self.sparse_user_threshold].index
        
        item_counts = df["movieId"].value_counts()
        valid_items = item_counts[item_counts >= self.sparse_item_threshold].index
        
        filtered_df = df[df["userId"].isin(valid_users) & df["movieId"].isin(valid_items)]
        
        logger.info(f"Filtered dataset: {len(valid_users)} users, {len(valid_items)} items remain.")
        return filtered_df

    def create_mappings(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict, dict, dict]:
        logger.info("Creating item and user mappings...")
        
        unique_users = sorted(df["userId"].unique())
        unique_items = sorted(df["movieId"].unique())

        user_mapping = {int(old_id): int(new_id) for new_id, old_id in enumerate(unique_users)}
        user_reverse_mapping = {int(v): int(k) for k, v in user_mapping.items()}

        item_mapping = {int(old_id): int(new_id) for new_id, old_id in enumerate(unique_items)}
        item_reverse_mapping = {int(v): int(k) for k, v in item_mapping.items()}

        df["userId"] = df["userId"].map(user_mapping)
        df["movieId"] = df["movieId"].map(item_mapping)

        logger.info(f"Generated {len(user_mapping)} user mappings and {len(item_mapping)} item mappings.")
        
        return df, user_mapping, user_reverse_mapping, item_mapping, item_reverse_mapping

    def create_user_item_matrix(self, df: pd.DataFrame) -> sparse.csr_matrix:
        logger.info("Creating user-item matrix...")

        n_users = df["userId"].max() + 1
        n_items = df["movieId"].max() + 1
        
        rows, cols, data = [], [], []
        
        for start in range(0, len(df), self.segment_size):
            end = start + self.segment_size
            chunk = df.iloc[start:end]
            
            rows.extend(chunk["userId"].values)
            cols.extend(chunk["movieId"].values)
            data.extend(chunk["rating"].values)
            
            del chunk
            gc.collect()
        
        user_item_matrix = sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_items)
        ).tocsr()
        
        del rows, cols, data
        gc.collect()

        logger.info(
            f"Created sparse matrix: {user_item_matrix.shape}, "
            f"density: {user_item_matrix.nnz / (n_users * n_items):.4%}"
        )
        return user_item_matrix

    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into training and test sets based on timestamp."""
        logger.info("Splitting dataset into train and test sets...")
        
        if "timestamp" not in df.columns:
            raise ValueError("Timestamp column required for time-aware splitting")
        
        train_list = []
        test_list = []

        # Ensure sorting is done only once
        data_sorted = df.sort_values(by=['userId', 'timestamp'])

        # Split per user
        for userId, user_data in data_sorted.groupby('userId'):
            user_interactions = user_data.reset_index(drop=True)
            cutoff = int(len(user_interactions) * self.split_percent)

            # Split into train/test
            train_user = user_interactions.iloc[:cutoff]
            test_user = user_interactions.iloc[cutoff:]

            train_list.append(train_user)
            test_list.append(test_user)

        # Concatenate all users' train/test interactions
        train_data = pd.concat(train_list, ignore_index=True)
        test_data = pd.concat(test_list, ignore_index=True)

        # Optional: Filter test set to users/items seen in training
        train_users = set(train_data['userId'])
        train_items = set(train_data['movieId'])

        test_data = test_data[test_data['userId'].isin(train_users)]
        test_data = test_data[test_data['movieId'].isin(train_items)]

        logger.info(f"Split dataset: {len(train_data)} training samples, {len(test_data)} test samples")
        return train_data, test_data

    def process(self, df: pd.DataFrame) -> Dict:
        logger.info("Starting data preprocessing pipeline...")

        required_columns = ["userId", "movieId", "rating", "timestamp", "title", "genres", "tmdbId"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Input dataframe must contain columns: {required_columns}")

        # Filter sparse entities first
        df = self.drop_sparse_entities(df)

        # Split data into train and test sets
        train, test = self.split_dataset(df)

        # Create mappings based on the complete dataset to ensure consistency
        df_combined, user_mapping, user_reverse_mapping, item_mapping, item_reverse_mapping = self.create_mappings(df)
        
        # Apply mappings to train and test data
        train["userId"] = train["userId"].map(user_mapping)
        train["movieId"] = train["movieId"].map(item_mapping)
        
        test_size_before = len(test)
        test["userId"] = test["userId"].map(user_mapping)
        test["movieId"] = test["movieId"].map(item_mapping)
        test = test.dropna()
        
        logger.info(f"Dropped {test_size_before - len(test)} rows from test due to unmapped users/items.")

        # Create user-item matrix from training data
        user_item_matrix = self.create_user_item_matrix(train)

        logger.info("Data preprocessing completed successfully.")
        return {
            "train": train,
            "test": test,
            "user_mapping": user_mapping,
            "user_reverse_mapping": user_reverse_mapping,
            "item_mapping": item_mapping,
            "item_reverse_mapping": item_reverse_mapping,
            "user_item_matrix": user_item_matrix
        }
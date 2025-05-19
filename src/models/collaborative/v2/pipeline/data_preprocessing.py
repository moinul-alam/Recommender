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
        
        df = df.copy()
        
        user_counts = df["userId"].value_counts()
        valid_users = user_counts[user_counts >= self.sparse_user_threshold].index
        
        item_counts = df["movieId"].value_counts()
        valid_items = item_counts[item_counts >= self.sparse_item_threshold].index
        
        filtered_df = df[df["userId"].isin(valid_users) & df["movieId"].isin(valid_items)]
        
        logger.info(f"Filtered dataset: {len(valid_users)} users, {len(valid_items)} items remain.")
        return filtered_df

    def create_mappings(self, df: pd.DataFrame) -> Dict:
        """Create contiguous mappings for user and item IDs"""
        logger.info("Creating item and user mappings...")
        
        df = df.copy()
        
        unique_users = sorted(df["userId"].unique())
        unique_items = sorted(df["movieId"].unique())

        user_mapping = {int(old_id): int(new_id) for new_id, old_id in enumerate(unique_users)}
        user_reverse_mapping = {int(v): int(k) for k, v in user_mapping.items()}

        item_mapping = {int(old_id): int(new_id) for new_id, old_id in enumerate(unique_items)}
        item_reverse_mapping = {int(v): int(k) for k, v in item_mapping.items()}

        # Apply mappings to the dataframe
        df["userId"] = df["userId"].map(user_mapping)
        df["movieId"] = df["movieId"].map(item_mapping)

        logger.info(f"Generated {len(user_mapping)} user mappings and {len(item_mapping)} item mappings.")
        
        return {
            "user_mapping": user_mapping,
            "user_reverse_mapping": user_reverse_mapping,
            "item_mapping": item_mapping,
            "item_reverse_mapping": item_reverse_mapping,
            "mapped_df": df 
        }
        
    def create_id_mappings(self, df: pd.DataFrame, item_mapping: Dict[int, int]) -> Dict:
        """Create mappings between original movieIds, mapped movieIds, and tmdbIds"""
        logger.info("Creating comprehensive ID mappings...")
        
        df = df.copy()
        
        # Mappings to create:
        # 1. Mapped movieId to tmdbId (for model internals)
        # 2. Original movieId to tmdbId (for API request/response)
        # 3. tmdbId to original movieId (for API request/response)
        mapped_id_to_tmdb = {}  # internal mapped ID → tmdbId
        movie_to_tmdb = {}      # original movieId → tmdbId 
        tmdb_to_movie = {}      # tmdbId → original movieId
        
        # Process each row to create all necessary mappings
        for _, row in df.iterrows():
            original_movie_id = int(row["movieId"])
            tmdb_id = int(row["tmdbId"])
            
            # Convert IDs to strings for consistent key type in mappings
            str_original_id = str(original_movie_id)
            str_tmdb_id = str(tmdb_id)
            
            # Create bidirectional mappings between original movieId and tmdbId
            movie_to_tmdb[str_original_id] = str_tmdb_id
            tmdb_to_movie[str_tmdb_id] = str_original_id
            
            # Create mapping from mapped movieId to tmdbId (for internal model use)
            if original_movie_id in item_mapping:
                mapped_movie_id = item_mapping[original_movie_id]
                mapped_id_to_tmdb[mapped_movie_id] = tmdb_id
        
        logger.info(f"Generated mappings for {len(mapped_id_to_tmdb)} mapped items, " 
                    f"{len(movie_to_tmdb)} original items, and {len(tmdb_to_movie)} TMDB items.")
        
        return {
            "mapped_id_to_tmdb": mapped_id_to_tmdb,  # For model internals
            "movie_to_tmdb_mapping": movie_to_tmdb,  # For API
            "tmdb_to_movie_mapping": tmdb_to_movie   # For API
        }

    def create_user_item_matrix(self, df: pd.DataFrame) -> sparse.csr_matrix:
        """Create user-item matrix from the mapped dataframe"""
        logger.info("Creating user-item matrix...")
        
        df = df.copy()

        # Get the number of unique users and items
        n_users = df["userId"].nunique()
        n_items = df["movieId"].nunique()
        
        # Verify the maximum indices match our expectations
        max_user_idx = df["userId"].max()
        max_item_idx = df["movieId"].max()
        
        if max_user_idx >= n_users:
            logger.warning(f"Max user index {max_user_idx} >= number of unique users {n_users}")
            n_users = max_user_idx + 1
            
        if max_item_idx >= n_items:
            logger.warning(f"Max item index {max_item_idx} >= number of unique items {n_items}")
            n_items = max_item_idx + 1
        
        logger.info(f"Creating matrix with {n_users} users and {n_items} items")
        
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

        # Verify the matrix has the expected number of non-zero elements
        expected_nnz = len(df)
        if user_item_matrix.nnz != expected_nnz:
            logger.warning(f"Matrix has {user_item_matrix.nnz} non-zero elements, expected {expected_nnz}")
        
        # Log matrix density and stats
        density = user_item_matrix.nnz / (n_users * n_items)
        logger.info(
            f"Created sparse matrix: {user_item_matrix.shape}, "
            f"density: {density:.4%}, "
            f"non-zero elements: {user_item_matrix.nnz}"
        )
        
        # Count users with interactions
        users_with_interactions = sum(1 for i in range(n_users) if user_item_matrix[i].nnz > 0)
        logger.info(f"Users with at least one interaction: {users_with_interactions}/{n_users}")
        
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
            
            # Ensure each user has at least one item in training and test
            if cutoff == 0:
                cutoff = 1
            elif cutoff == len(user_interactions):
                cutoff = len(user_interactions) - 1

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

        # Drop sparse entities first
        filtered_df = self.drop_sparse_entities(df)
        
        # Create mappings on filtered data
        mapping_results = self.create_mappings(filtered_df)
        mapped_df = mapping_results.pop("mapped_df")  # Extract mapped df and remove from dict
        
        # Split the mapped dataset
        train, test = self.split_dataset(mapped_df)
        
        # Create matrices and mappings based on the mapped and filtered data
        user_item_matrix = self.create_user_item_matrix(train)
        
        # Create comprehensive ID mappings
        id_mappings = self.create_id_mappings(
            filtered_df, 
            mapping_results["item_mapping"]
        )

        logger.info("Data preprocessing completed successfully.")
        return {
            "train": train,
            "test": test,
            "user_item_mappings": mapping_results,
            "user_item_matrix": user_item_matrix,
            "mapped_id_to_tmdb": id_mappings["mapped_id_to_tmdb"],      # For model internals
            "movie_to_tmdb_mapping": id_mappings["movie_to_tmdb_mapping"],  # For API
            "tmdb_to_movie_mapping": id_mappings["tmdb_to_movie_mapping"]   # For API
        }
import logging
import os
from typing import Dict, Tuple, Optional
import pathlib
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DataPreprocessing:
    def __init__(self, sparse_threshold: int = 5, split_percent: float = 0.8):
        if not (0 < sparse_threshold < 100):
            raise ValueError("Sparse threshold must be between 0 and 100")
        if not (0 < split_percent < 1):
            raise ValueError("Split percentage must be between 0 and 1")
        
        self.sparse_threshold = sparse_threshold
        self.split_percent = split_percent

    def drop_sparse_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove users with fewer ratings than the sparse threshold.
        
        Args:
            df (pd.DataFrame): Input dataframe with user ratings
        
        Returns:
            pd.DataFrame: Filtered dataframe with sparse users removed
        """
        user_counts = df["user_id"].value_counts()
        filtered_users = user_counts[user_counts >= self.sparse_threshold].index
        filtered_df = df[df["user_id"].isin(filtered_users)]
        
        logger.info(f"Filtered out users with fewer than {self.sparse_threshold} ratings. "
                    f"Original users: {len(user_counts)}, Remaining users: {len(filtered_users)}")
        return filtered_df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the ratings column.
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Dataframe with missing values imputed
        """
        if df["rating"].isnull().any():
            mean_rating = df["rating"].mean()
            df["rating"].fillna(mean_rating, inplace=True)
            logger.info(f"Filled missing ratings using mean imputation: {mean_rating}")
        return df
    
    # def calculate_movie_popularity(self, df: pd.DataFrame):
    #     """Calculate movie popularity based on number of ratings and average rating."""
    #     movie_stats = df.groupby('tmdb_id').agg({
    #         'rating': ['count', 'mean']
    #     })
    #     movie_stats.columns = ['count', 'mean']

    #     # Normalize counts and means
    #     movie_stats['count_norm'] = (movie_stats['count'] - movie_stats['count'].min()) / \
    #                               (movie_stats['count'].max() - movie_stats['count'].min())
    #     movie_stats['mean_norm'] = (movie_stats['mean'] - movie_stats['mean'].min()) / \
    #                              (movie_stats['mean'].max() - movie_stats['mean'].min())

    #     # Combine into popularity score (weighted average of count and mean)
    #     self.movie_popularity = (movie_stats['count_norm'] * 0.7 +
    #                            movie_stats['mean_norm'] * 0.3).to_dict()
        
    #     logger.info(f"Calculated movie popularity for {len(self.movie_popularity)} movies.")


    def create_mappings(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
        """
        Create mappings for users and items to ensure contiguous integer indices.
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            Tuple containing:
            - Transformed dataframe
            - Item mapping dictionary
            - User mapping dictionary
        """
        unique_users = df["user_id"].unique()
        unique_items = df["tmdb_id"].unique()

        user_mapping = {int(old_id): int(new_id) for new_id, old_id in enumerate(unique_users)}
        item_mapping = {int(old_id): int(new_id) for new_id, old_id in enumerate(unique_items)}

        # Create reverse mappings
        user_reverse_mapping = {v: k for k, v in user_mapping.items()}
        item_reverse_mapping = {v: k for k, v in item_mapping.items()}

        df["user_id"] = df["user_id"].map(user_mapping)
        df["tmdb_id"] = df["tmdb_id"].map(item_mapping)

        logger.info(f"Created mappings. Users: {len(user_mapping)}, Items: {len(item_mapping)}")

        return df, item_mapping, user_mapping, item_reverse_mapping, user_reverse_mapping

    def create_user_item_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a user-item matrix, handling duplicate entries by aggregating ratings.
        
        Args:
            df (pd.DataFrame): Input dataframe with ratings
        
        Returns:
            pd.DataFrame: User-item interaction matrix
        """
        # Drop timestamp column if it exists
        df = df.drop(columns=['timestamp'], errors='ignore')
        
        # Aggregate duplicate entries by taking the mean rating
        df = df.groupby(['user_id', 'tmdb_id'])['rating'].mean().reset_index()
        
        user_item_matrix = df.pivot(index='user_id', columns='tmdb_id', values='rating').fillna(0)
        logger.info(f"User-Item matrix created with shape: {user_item_matrix.shape}")
        return user_item_matrix

    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train and test sets.
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            Tuple of train and test dataframes
        """
        train, test = train_test_split(df, test_size=(1 - self.split_percent), random_state=42)
        test = test[test["user_id"].isin(train["user_id"])]
        
        logger.info(f"Train-test split complete. Train size: {len(train)}, Test size: {len(test)}")
        return train, test

    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, int], Dict[int, int], pd.DataFrame]:
        """
        Complete data preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Raw input dataframe
        
        Returns:
            Tuple containing:
            - Training dataframe
            - Testing dataframe
            - Item mapping
            - User mapping
            - User-item interaction matrix
        """
        df = self.drop_sparse_users(df)
        df = self.handle_missing_values(df)
        df, item_mapping, user_mapping, item_reverse_mapping, user_reverse_mapping = self.create_mappings(df)
        
        user_item_matrix = self.create_user_item_matrix(df)
        train, test = self.split_dataset(df)

        return train, test, item_mapping, user_mapping, item_reverse_mapping, user_reverse_mapping, user_item_matrix

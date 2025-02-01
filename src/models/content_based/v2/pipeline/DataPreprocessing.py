import gc
import re
import pandas as pd
from typing import List, Optional, Tuple, Dict
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DataPreprocessing:
    LIST_COLUMNS = ['media_type', 'title', 'genres', 'keywords', 'director', 'cast']
    FULL_TEXT_COLUMNS = ['overview']
    NUMERIC_COLUMNS = ['vote_average', 'release_year']

    def __init__(self, dataset_path: Optional[str] = None, segment_size: int = 5000, 
                 keep_columns: List[str] = None, df: Optional[pd.DataFrame] = None):
        self.segment_size = segment_size
        self.keep_columns = keep_columns or [
            'tmdb_id', 'media_type', 'title', 'overview', 'vote_average',
            'release_year', 'genres', 'director', 'cast', 'keywords' 
        ]
        self.dataset_path = dataset_path
        self.df = df

        if df is None and dataset_path is None:
            raise ValueError("Either 'dataset_path' or 'df' must be provided.")
        
        if segment_size < 1:
            raise ValueError("segment_size must be at least 1")

    def load_dataset(self) -> pd.DataFrame:
        """Load dataset from CSV file."""
        if not Path(self.dataset_path).exists():
            raise FileNotFoundError(f"File not found: {self.dataset_path}")

        try:
            df = pd.read_csv(self.dataset_path)
            logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            return df.copy()
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise

    def load_dataframe(self) -> pd.DataFrame:
        """Load data from provided DataFrame."""
        if self.df is None:
            raise ValueError("DataFrame is not set. Please provide a valid DataFrame.")

        if not isinstance(self.df, pd.DataFrame):
            raise TypeError("The provided data is not a pandas DataFrame.")

        logger.info(f"Loaded DataFrame with {len(self.df)} rows and {len(self.df.columns)} columns")
        return self.df.copy()

    def handle_missing_or_duplicate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing or duplicate `tmdb_id`, `overview`, or `genres`."""
        initial_rows = len(df)
        
        # Drop rows where any of these columns are missing
        df = df.dropna(subset=['tmdb_id', 'overview', 'genres'])

        # Drop duplicates based on `tmdb_id`
        df = df.drop_duplicates(subset=['tmdb_id'])

        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} rows with missing or duplicate tmdb_id, overview, or genres")

        return df

    def handle_tmdb_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure tmdb_id is valid, numeric, and unique."""
        initial_rows = len(df)

        # Convert to numeric and drop invalid values
        df['tmdb_id'] = pd.to_numeric(df['tmdb_id'], errors='coerce')
        df = df.dropna(subset=['tmdb_id']).drop_duplicates(subset=['tmdb_id'])
        df['tmdb_id'] = df['tmdb_id'].astype(int)

        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} rows with invalid or duplicate tmdb_id")

        return df

    def handle_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle numeric data including validation and missing value imputation."""
        for col in self.NUMERIC_COLUMNS:
            if col not in df.columns:
                continue

            missing_count = df[col].isna().sum()
            if missing_count > 0:
                logger.info(f"Missing values in {col}: {missing_count}")

            df[col] = pd.to_numeric(df[col], errors='coerce')

            if col == 'release_year':
                mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)

                mask = (df[col] < 1900) | (df[col] > 2025)
                invalid_years = df.loc[mask, col].count()
                if invalid_years > 0:
                    logger.warning(f"Found {invalid_years} release years outside 1900-2025 range")

            if col == 'vote_average':
                # Round vote_average to 1 decimal place and ensure it's between 0 and 10
                df[col] = df[col].round(1)
                df[col] = df[col].clip(0, 10)

            remaining_missing = df[col].isna().sum()
            if remaining_missing > 0:
                logger.warning(f"Remaining missing values in {col}: {remaining_missing}")

        return df

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize text and list data with improved validation."""
        def preprocess_text(text: str) -> str:
            if pd.isna(text) or text == '':
                return 'Unknown'  # Explicitly assigning 'Unknown' for missing values
            text = re.sub(r'[^\w\s]', '', str(text))
            return ' '.join(text.lower().split())

        def process_list_field(field: str) -> str:
            if pd.isna(field) or field == '':
                return 'Unknown'
            field = str(field).replace('|', ' ').replace(',', ' ')
            return ' '.join(field.lower().split())

        text_transformations: Dict[str, pd.Series] = {}

        for col in self.LIST_COLUMNS:
            if col in df.columns:
                text_transformations[col] = df[col].str.lower().str.strip().apply(process_list_field)

        for col in self.FULL_TEXT_COLUMNS:
            if col in df.columns:
                text_transformations[col] = df[col].str.lower().str.strip().apply(preprocess_text)

        df = df.assign(**text_transformations)

        for col, transformed_series in text_transformations.items():
            empty_count = (transformed_series == 'Unknown').sum()
            if empty_count > 0:
                logger.info(f"Replaced {empty_count} missing values in {col} with 'Unknown'")

        return df

    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and validate required columns."""
        missing_cols = set(self.keep_columns) - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing columns in dataset: {missing_cols}")
        
        available_cols = [col for col in self.keep_columns if col in df.columns]
        return df[available_cols]

    def segment_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """Segment dataset into smaller chunks using efficient np.array_split."""
        num_segments = (len(df) // self.segment_size) + 1
        segments = np.array_split(df, num_segments)

        logger.info(f"Created {len(segments)} segments of size {self.segment_size}")
        return df, segments

    def apply_data_preprocessing(self) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """Apply all data preprocessing steps to a new dataset."""
        try:
            df = self.load_dataset()
            df = self.handle_missing_or_duplicate_data(df)  # Add the missing/duplicate handling
            df = self.handle_tmdb_id(df)
            df = self.handle_numeric_data(df)
            df = self.normalize_data(df)
            df = self.select_columns(df)
            df, segments = self.segment_dataset(df)

            gc.collect()
            return df, segments
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            raise

    def preprocess_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess new data from an existing DataFrame."""
        try:
            df = self.handle_numeric_data(df)
            df = self.normalize_data(df)
            df = self.select_columns(df)

            gc.collect()
            return df
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            raise

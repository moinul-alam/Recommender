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
    LIST_COLUMNS = ['media_type', 'title', 'spoken_languages', 'genres', 'keywords', 'director', 'cast']
    FULL_TEXT_COLUMNS = ['overview']
    NUMERIC_COLUMNS = ['vote_average', 'release_year']

    def __init__(self, dataset: Optional[pd.DataFrame] = None, segment_size: int = 5000, 
                 keep_columns: List[str] = None, df: Optional[pd.DataFrame] = None):
        self.segment_size = segment_size
        self.keep_columns = keep_columns or [
            'item_id', 'media_type', 'title', 'overview', 'spoken_languages', 'vote_average',
            'release_year', 'genres', 'director', 'cast', 'keywords' 
        ]
        self.dataset = dataset
        self.df = df
        
        if segment_size < 1:
            raise ValueError("segment_size must be at least 1")

    def load_dataframe(self) -> pd.DataFrame:
        """Load data from provided DataFrame."""
        if self.df is None:
            raise ValueError("DataFrame is not set. Please provide a valid DataFrame.")

        if not isinstance(self.df, pd.DataFrame):
            raise TypeError("The provided data is not a pandas DataFrame.")

        logger.info(f"Loaded DataFrame with {len(self.df)} rows and {len(self.df.columns)} columns")
        return self.df.copy()

    def handle_missing_or_duplicate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing or duplicate `item_id`, `overview`, or `genres`."""
        initial_rows = len(df)
        
        # Drop rows where any of these columns are missing
        df = df.dropna(subset=['item_id', 'overview', 'genres'])

        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} rows with missing or duplicate item_id, overview, or genres")

        return df

    def handle_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle numeric data including validation, missing value imputation, and empty string replacement."""
        logger.info('Handling numeric data...')

        for col in self.NUMERIC_COLUMNS:
            if col not in df.columns:
                continue

            # Replace empty strings with 'unknown'
            df[col] = df[col].replace('', 'unknown')

            missing_count = df[col].isna().sum()
            if missing_count > 0:
                logger.info(f"Missing values in {col}: {missing_count}")

            # Convert to numeric, setting errors='coerce' to handle non-numeric values
            df[col] = pd.to_numeric(df[col], errors='coerce')

            if col == 'release_year':
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 2000  # Default to 2000 if mode is empty
                df[col] = df[col].fillna(mode_value)

                mask = (df[col] < 1900) | (df[col] > 2025)
                invalid_years = df.loc[mask, col].count()
                if invalid_years > 0:
                    logger.warning(f"Found {invalid_years} release years outside 1900-2025 range")
                    df.loc[mask, col] = mode_value  # Replace invalid years with mode

            if col == 'vote_average':
                # Round vote_average to 1 decimal place and ensure it's between 0 and 10
                df[col] = df[col].round(1)
                df[col] = df[col].clip(0, 10)

            remaining_missing = df[col].isna().sum()
            if remaining_missing > 0:
                logger.warning(f"Remaining missing values in {col}: {remaining_missing}")

        return df


    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize text and list data while keeping commas intact for list fields."""
        logger.info('Normalizing.....')

        def preprocess_text(text: str) -> str:
            """Normalize general text fields by cleaning punctuation and extra spaces."""
            if pd.isna(text) or text == '':
                return 'unknown'  # Explicitly assigning 'unknown' for missing values
            text = re.sub(r'[^\w\s]', '', str(text))  # Remove punctuation except for alphanumeric and spaces
            return ' '.join(text.lower().split())  # Lowercase and remove extra spaces

        def process_list_field(field: str) -> str:
            """Ensure lowercase transformation while keeping commas intact."""
            if pd.isna(field) or field.strip() == '':
                return 'unknown'
            
            # Lowercase and clean up the string (remove extra spaces around commas)
            field = str(field).strip().lower()
            field = re.sub(r'\s*,\s*', ',', field)
            return field  # Comma-separated, lowercase genres/cast/director

        # Apply transformations for the list columns (genres, cast, director, keywords)
        text_transformations: Dict[str, pd.Series] = {}
        
        # Process 'genres', 'cast', and 'director' (comma-separated, lowercase)
        for col in ['genres', 'cast', 'director']:
            if col in df.columns:
                text_transformations[col] = df[col].apply(process_list_field)
        
        # Process 'keywords' (comma-separated, lowercase, tokenize)
        def preprocess_keywords(keywords: str) -> str:
            """Preprocess keywords by splitting on commas, lowercasing, and tokenizing."""
            if pd.isna(keywords) or keywords == '':
                return 'unknown'  # Replace missing or empty values with 'unknown'
            keywords = keywords.lower()  # Convert to lowercase
            keywords = re.sub(r'\s*,\s*', ',', keywords)  # Remove extra spaces around commas
            keywords = keywords.split(',')  # Tokenize by splitting on commas
            return ' '.join(keywords)  # Return tokenized, space-separated keywords string

        # Apply the preprocessing for keywords
        df['keywords'] = df['keywords'].apply(preprocess_keywords)

        # Process 'overview' (lowercase, punctuation removed)
        if 'overview' in df.columns:
            text_transformations['overview'] = df['overview'].apply(preprocess_text)

        # Apply the transformations to genres, cast, director, overview
        df = df.assign(**text_transformations)

        # Log any replacements of missing values with 'unknown'
        for col, transformed_series in text_transformations.items():
            empty_count = (transformed_series == 'unknown').sum()
            if empty_count > 0:
                logger.info(f"Replaced {empty_count} missing values in {col} with 'unknown'")

        return df

    
    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and validate required columns."""
        logger.info('Selecting final columns.....')
        missing_cols = set(self.keep_columns) - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing columns in dataset: {missing_cols}")
        
        available_cols = [col for col in self.keep_columns if col in df.columns]
        logger.info(f"Selected columns: {available_cols}")
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
            df = self.handle_missing_or_duplicate_data(self.dataset)
            # df = self.handle_tmdb_id(df)
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

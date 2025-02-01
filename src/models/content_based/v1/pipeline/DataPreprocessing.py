import gc
import re
import pandas as pd
from typing import List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DataPreprocessing:

    text_columns = ['overview', 'genres', 'keywords', 'cast', 'director']

    def __init__(
        self, 
        dataset_path: Optional[str] = None, 
        segment_size: int = 6000, 
        keep_columns: List[str] = None, 
        df: Optional[pd.DataFrame] = None
    ):
        self.segment_size = segment_size
        self.keep_columns = keep_columns or [
            'tmdbId', 'title', 'original_language', 'overview', 'tagline', 
            'genres', 'keywords', 'cast', 'director', 'release_year'
        ]
        self.dataset_path = dataset_path
        self.df = df

        # Ensure the class works for both use cases
        if df is None and dataset_path is None:
            raise ValueError("Either 'dataset_path' or 'df' must be provided.")

    def load_dataset(self) -> pd.DataFrame:
        if self.segment_size < 1:
            raise ValueError("segment_size must be at least 1")

        if not Path(self.dataset_path).exists():
            raise FileNotFoundError(f"File not found: {self.dataset_path}")

        try:
            df = pd.read_csv(self.dataset_path) 
            logger.info(f"Loaded dataset with {len(df)} rows")
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise

        return df.copy() 
    
    def load_dataframe(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("DataFrame is not set. Please provide a valid DataFrame.")

        if not isinstance(self.df, pd.DataFrame):
            raise TypeError("The provided data is not a pandas DataFrame.")

        logger.info(f"Loaded DataFrame with {len(self.df)} rows and {len(self.df.columns)} columns")
        return self.df.copy()

    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:

        initial_rows = len(df)
        df = df.dropna(subset=['tmdbId']).drop_duplicates(subset=['tmdbId'])
        logger.info(f"Removed {initial_rows - len(df)} rows with missing or duplicate tmdbId")
        return df

    def handle_release_date(self, df: pd.DataFrame) -> pd.DataFrame:

        def extract_year(date: str) -> Optional[int]:
            try:
                return pd.to_datetime(date).year
            except (ValueError, TypeError):  # Handle potential type errors
                return None

        df = df.assign(release_year=df['release_date'].apply(extract_year))

        missing_release_year = df['release_year'].isna().sum()
        logger.info(f"Missing Release Years: {missing_release_year}")

        mode_year = df['release_year'].mode()[0]
        df = df.fillna({
            'release_year': mode_year,
            'release_date': 'Unknown'
        })

        logger.info(f"Remaining Missing Release Years: {df['release_year'].isna().sum()}")
        return df

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:

        def preprocess_text(text: str) -> str:
            """Cleans and preprocesses text data."""
            if pd.isna(text) or text == '':
                return '<MISSING>' 

            # Remove punctuation
            text = re.sub(r'[^\w\s]', '', text) 

            # Convert to lowercase and remove extra whitespace
            return ' '.join(str(text).lower().split())

        def process_list_field(field: str) -> str:
            """Processes pipe-separated fields."""
            if pd.isna(field) or field == '':
                return '<MISSING>'

            return ' '.join(str(field).split('|'))

        text_transformations = {}
        for col in self.text_columns:
            if col in df.columns:
                temp_series = df[col].str.lower().str.strip()
                if col in ['genres', 'keywords', 'cast', 'director']:
                    text_transformations[col] = temp_series.apply(process_list_field)
                else:
                    text_transformations[col] = temp_series.apply(preprocess_text)

        # Apply all text transformations at once
        df = df.assign(**text_transformations)
        return df
    
    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:

        return df[self.keep_columns]

    def segment_dataset(self, df: pd.DataFrame) -> List[pd.DataFrame]:

        num_segments = (len(df) // self.segment_size) + 1
        segments = []
        for i in range(num_segments):
            start_idx = i * self.segment_size
            end_idx = (i + 1) * self.segment_size
            segment = df.iloc[start_idx:end_idx].copy()
            if not segment.empty:
                segments.append(segment)

        logger.info(f"Created {len(segments)} segments")
        return segments

    def apply_data_preprocessing(self) -> pd.DataFrame:
        """Applies all data preprocessing steps.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        try:
            df = self.load_dataset()
            df = self.handle_missing_data(df)
            df = self.handle_release_date(df)
            df = self.normalize_data(df)
            df = self.select_columns(df)
            df = self.segment_dataset(df)

            gc.collect()
            return df
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            raise

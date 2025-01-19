import pandas as pd
import numpy as np
from typing import List, Optional

class DataPreparation:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.df: pd.DataFrame = pd.DataFrame()  # Initialize as an empty DataFrame

    # Load the data into a DataFrame
    def load_data(self) -> None:
        try:
            self.df = pd.read_csv(self.dataset_path)
        except Exception as e:
            raise RuntimeError(f"Error Loading data: {str(e)}")
    
    # Drop rows with missing critical fields and handle missing non-critical fields
    def clean_data(self) -> None:
        self.df.dropna(subset=['tmdbId'], inplace=True)  # Drop rows where 'tmdbId' is missing
        self.df.drop_duplicates(subset=['tmdbId'], inplace=True)  # Drop duplicate rows based on 'tmdbId'

        # Fill missing values for non-critical columns
        self.df['overview'].fillna('', inplace=True)
        for col in ['genres', 'keywords', 'cast', 'director']:
            self.df[col].fillna('unknown', inplace=True)
    
    # Extract year from 'release_date' and handle errors
    def extract_year(self, date: str) -> Optional[int]:
        try:
            return pd.to_datetime(date).year
        except Exception:
            return None  # Return None if parsing fails

    def apply_year_extraction(self) -> None:
        self.df['release_year'] = self.df['release_date'].apply(self.extract_year)
    
    # Fill missing 'release_year' values with the most common year (mode)
    def fill_missing_years(self) -> None:
        missing_release_year = self.df['release_year'].isnull().sum()
        if missing_release_year > 0:
            mode_year = self.df['release_year'].mode()[0]  # Get the most common release year
            self.df['release_year'].fillna(mode_year, inplace=True)
        # Fill any remaining 'release_date' NaNs with 'Unknown'
        self.df['release_date'].fillna('Unknown', inplace=True)
    
    # Clean text columns by making all text lowercase and stripping whitespace
    def clean_text_columns(self) -> None:
        text_columns = ['overview', 'genres', 'keywords', 'cast', 'director']
        for col in text_columns:
            self.df[col] = self.df[col].str.lower().str.strip()
    
    # Segment the dataset into chunks of a given size
    def segment_dataset(self, segment_size: int) -> List[pd.DataFrame]:
        num_segments = (len(self.df) // segment_size) + 1
        segments = []
        for i in range(num_segments):
            segment = self.df.iloc[i * segment_size: (i + 1) * segment_size]
            if not segment.empty:
                segments.append(segment)
                print(f"Chunk {i + 1} created with shape: {segment.shape}")
        return segments

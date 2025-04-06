import logging
from pathlib import Path
import pandas as pd
import json
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
from src.config.config import BaseConfig
from src.recommenders.common.DataLoader import load_data
from src.recommenders.common.DataSaver import save_data, save_multiple_dataframes

from src.schemas.content_based_schema import PipelineResponse

# Set up logging globally
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Columns to retain from the dataset
KEEP_COLUMNS = [
    'tmdb_id', 'media_type', 'spoken_languages', 'title', 'overview', 'vote_average', 
    'release_date', 'genres', 'credits', 'keywords'
]

class DataPreparation:
    """Handles data preparation for content-based recommendation system."""
    
    @staticmethod
    def data_loader(content_based_dir_path: str, dataset_name: str) -> pd.DataFrame:
        """Load dataset from the specified path."""
        dataset_path = Path(content_based_dir_path) / dataset_name
        
        logger.info(f"Loading dataset from: {dataset_path}")
        if not dataset_path.exists():
            logger.error(f"File not found: {dataset_path}")
            raise FileNotFoundError(f"File not found: {dataset_path}")
        
        try:
            df = load_data(dataset_path)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    @staticmethod
    def extract_column_data(df: pd.DataFrame) -> pd.DataFrame:
        """Extract required features without preprocessing."""
        logger.info(f"Starting data extraction. Initial shape: {df.shape}")
        df = df.copy()

        # Extract 'spoken_languages' as a comma-separated string of language codes
        df['spoken_languages'] = df['spoken_languages'].apply(
            lambda x: ', '.join([lang.get('iso_639_1', '').strip() for lang in x]) 
            if isinstance(x, list) and x else ''
        )

        # Extract 'genres' as a comma-separated string
        df['genres'] = df['genres'].apply(
            lambda x: ', '.join([genre.get('name', '').strip() for genre in x]) 
            if isinstance(x, list) and x else ''
        )

        # Extract 'keywords' as a comma-separated string
        df['keywords'] = df['keywords'].apply(
            lambda x: ', '.join([keyword.get('name', '') for keyword in x]) 
            if isinstance(x, list) and x else ''
        )

        # Extract directors and cast from 'credits'
        def extract_crew_info(credits, role_types):
            if isinstance(credits, list) and credits:
                names = [person['name'] for person in credits 
                        if isinstance(person, dict) and 
                        person.get('type') in role_types and 
                        person.get('name')]
                return ', '.join(names) if names else ''
            return ''

        df['director'] = df['credits'].apply(lambda x: extract_crew_info(x, ['director', 'creator']))
        df['cast'] = df['credits'].apply(lambda x: extract_crew_info(x, ['cast']))
        df = df.drop(columns=['credits'])

        # Extract 'release_year' from 'release_date'
        def extract_release_year(date_obj):
            try:
                if isinstance(date_obj, str):
                    return datetime.strptime(date_obj.split('T')[0], '%Y-%m-%d').year
                if isinstance(date_obj, dict) and '$date' in date_obj:
                    date_value = date_obj['$date']
                    if isinstance(date_value, str):
                        return datetime.strptime(date_value.split('T')[0], '%Y-%m-%d').year
                    elif isinstance(date_value, dict) and '$numberLong' in date_value:
                        timestamp_ms = int(date_value['$numberLong'])
                        return (datetime(1970, 1, 1) + timedelta(milliseconds=timestamp_ms)).year
            except Exception as e:
                logger.warning(f"Error processing release date: {date_obj}. Error: {e}")
            return None

        df['release_year'] = df['release_date'].apply(extract_release_year)
        df = df.drop(columns=['release_date'])

        # Select and reorder columns
        columns = ['tmdb_id', 'media_type', 'title', 'overview', 'spoken_languages', 
                   'vote_average', 'release_year', 'genres', 'director', 'cast', 'keywords']
        df = df[columns]

        logger.info(f"Data extraction completed. Final shape: {df.shape}")
        return df

    @staticmethod
    def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing `overview`, or `genres`."""
        logger.info("Handling missing data")
        initial_rows = len(df)
        
        # Drop rows with NaN values
        df = df.dropna(subset=['tmdb_id', 'overview', 'genres'])
        
        # Drop rows with empty strings
        df = df[
            (df['overview'].str.strip() != '') & 
            (df['genres'].str.strip() != '') &
            (df['tmdb_id'].notna())
        ]

        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} rows with missing tmdb_id, overview, or genres")

        return df
    
    @staticmethod
    def generate_unique_id(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate sequential IDs for each unique item and create a mapping table.
        
        Returns:
            tuple: (DataFrame with new IDs, Mapping DataFrame)
        """
        logger.info("Generating sequential IDs for items")
        
        # Ensure sorting before ID assignment
        df = df.sort_values(by=['tmdb_id', 'media_type']).reset_index(drop=True)

        # Drop duplicate tmdb_id, media_type combinations
        df = df.drop_duplicates(subset=['tmdb_id', 'media_type'])

        # Create mapping DataFrame
        mapping_df = df[['tmdb_id', 'media_type', 'title']].copy()
        mapping_df['item_id'] = range(1, len(mapping_df) + 1)

        # Create a dictionary for faster mapping
        id_map = dict(zip(
            zip(mapping_df['tmdb_id'], mapping_df['media_type']), 
            mapping_df['item_id']
        ))

        # Add new sequential IDs to original DataFrame
        df['item_id'] = df.apply(
            lambda row: id_map[(row['tmdb_id'], row['media_type'])], 
            axis=1
        )

        logger.info(f"Generated {len(mapping_df)} unique sequential IDs")
        return df, mapping_df
    
    @classmethod
    def prepare_data(cls, content_based_dir_path: str, dataset_name: str) -> PipelineResponse:
        """
        Main method to prepare data for content-based recommendation.
        Handles loading, extraction, cleaning, and ID generation.
        """
        logger.info("Starting data preparation")
        try:
            # Load and process data
            df = cls.data_loader(content_based_dir_path, dataset_name)
            df = cls.extract_column_data(df)
            df = cls.handle_missing_data(df)
            df, mapping_df = cls.generate_unique_id(df)
            
            # Save processed data using the new save_data function
            directory_path = Path(content_based_dir_path)
            
            # Save both dataframes
            dataframes = {
                "processed_data": df,
                "item_mapping": mapping_df
            }
            
            saved_files = save_multiple_dataframes(
                directory_path=directory_path,
                dataframes=dataframes,
                file_type="csv",
                index=False
            )
            
            logger.info(f"Saved files: {saved_files}")
            
            return PipelineResponse(
                status="success",
                message="Data preparation completed successfully",
                output_path=str(directory_path)
            )
        except Exception as e:
            logger.error(f"Error in data preparation: {e}")
            raise

    @staticmethod
    def prepare_new_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Process new data for inference.
        Ensures data is in the correct format for model input.
        """
        try:
            logger.info("Starting new data preparation")
            df = df.copy()

            # Process JSON strings if needed
            for column in ['genres', 'keywords', 'cast', 'director']:
                if column in df.columns:
                    df[column] = df[column].apply(
                        lambda x: DataPreparation._process_list_or_string(x) if x else ''
                    )

            # Clean up whitespace and standardize separators
            for col in ['genres', 'keywords', 'cast', 'director']:
                if col in df.columns:
                    # Clean up separators and whitespace
                    df[col] = df[col].apply(
                        lambda x: ', '.join(item.strip() for item in x.split(',')) if x else ''
                    ).apply(
                        lambda x: ' '.join(x.split())
                    )

            # Convert numeric columns
            if 'tmdb_id' in df.columns:
                df['tmdb_id'] = pd.to_numeric(df['tmdb_id'], errors='coerce')

            if 'vote_average' in df.columns:
                df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').round(3)

            logger.info("New data preparation completed successfully")
            return df

        except Exception as e:
            logger.error(f"Error in new data preparation: {e}")
            raise

    @staticmethod
    def _process_list_or_string(data):
        """Helper method to process list or string data consistently."""
        if isinstance(data, str):
            try:
                data = json.loads(data.replace("'", '"'))
            except json.JSONDecodeError:
                # If not valid JSON, treat as comma-separated string
                return data
        
        if isinstance(data, list):
            return ', '.join([
                item.strip() if isinstance(item, str) 
                else item.get('name', '').strip() if isinstance(item, dict) else ''
                for item in data
            ])
        return str(data)
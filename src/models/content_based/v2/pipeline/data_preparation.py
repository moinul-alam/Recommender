import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Set up logging globally
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Columns to retain from the dataset
keep_columns = [
    'tmdb_id', 'media_type', 'spoken_languages', 'title', 'overview', 'vote_average', 
    'release_date', 'genres', 'credits', 'keywords'
]

class DataPreparation:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.item_mapping = None

    def generate_sequential_ids(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    
    def extract_column_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract required features without preprocessing."""
        logger.info(f"Starting data extraction. Initial shape: {df.shape}")

        df = df.copy()

        # Extract 'spoken_languages' as a comma-separated string of language codes
        df.loc[:, 'spoken_languages'] = df['spoken_languages'].apply(
            lambda x: ', '.join([lang.get('iso_639_1', '').strip() for lang in x]) 
            if isinstance(x, list) and x else ''
        )

        # Extract 'genres' as a comma-separated string
        df.loc[:, 'genres'] = df['genres'].apply(
            lambda x: ', '.join([genre.get('name', '').strip() for genre in x]) 
            if isinstance(x, list) and x else ''
        )

        # Extract 'keywords' as a comma-separated string
        df.loc[:, 'keywords'] = df['keywords'].apply(
            lambda x: ', '.join([keyword.get('name', '') for keyword in x]) 
            if isinstance(x, list) and x else ''
        )

        # Extract 'credits' - Extract directors and cast
        def extract_crew_info(credits, role_types):
            if isinstance(credits, list) and credits:
                names = [person['name'] for person in credits 
                        if isinstance(person, dict) and 
                        person.get('type') in role_types and 
                        person.get('name')]
                return ', '.join(names) if names else ''
            return ''

        # Extract directors and cast
        df.loc[:, 'director'] = df['credits'].apply(lambda x: extract_crew_info(x, ['director', 'creator']))
        df.loc[:, 'cast'] = df['credits'].apply(lambda x: extract_crew_info(x, ['cast']))
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

        df.loc[:, 'release_year'] = df['release_date'].apply(extract_release_year)
        df = df.drop(columns=['release_date'])

        df = df[['tmdb_id', 'media_type', 'title', 'overview', 'spoken_languages', 'vote_average', 'release_year', 'genres', 'director', 'cast', 'keywords']]

        logger.info(f"Data extraction completed. Final shape: {df.shape}")
        return df

    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def apply_data_preparation(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies full data extraction pipeline and generates sequential IDs.
        
        Returns:
            tuple: (Processed DataFrame with sequential IDs, Mapping DataFrame)
        """
        logger.info("Applying data extraction pipeline.")
        try:
            df = self.extract_column_data(self.dataset)
            df = self.handle_missing_data(df)
            df, mapping_df = self.generate_sequential_ids(df)
            self.item_mapping = mapping_df
            logger.info("Data extraction and ID generation completed successfully.")
            return df, mapping_df
        except Exception as e:
            logger.error(f"Error in data extraction: {e}")
            raise


    def prepare_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Starting new data preparation")
            df = df.copy()

            # Handle genres: convert list of dicts to comma-separated string
            if 'genres' in df.columns:
                df['genres'] = df['genres'].apply(
                    lambda x: ', '.join([genre.strip() if isinstance(genre, str) else genre['name'].strip() 
                                    for genre in (json.loads(x.replace("'", '"')) if isinstance(x, str) else x)]) 
                    if x else ''
                )

            # Handle keywords: convert list of dicts to comma-separated string
            if 'keywords' in df.columns:
                df['keywords'] = df['keywords'].apply(
                    lambda x: ', '.join([kw.strip() if isinstance(kw, str) else kw['name'].strip() 
                                    for kw in (json.loads(x.replace("'", '"')) if isinstance(x, str) else x)]) 
                    if x else ''
                )

            # Handle cast: convert list to comma-separated string
            if 'cast' in df.columns:
                df['cast'] = df['cast'].apply(
                    lambda x: ', '.join([name.strip() for name in 
                                    (json.loads(x.replace("'", '"')) if isinstance(x, str) else x)]) 
                    if x else ''
                )

            # Handle director: convert list to string
            if 'director' in df.columns:
                df['director'] = df['director'].apply(
                    lambda x: ', '.join([name.strip() for name in 
                                    (json.loads(x.replace("'", '"')) if isinstance(x, str) else x)]) 
                    if x else ''
                )

            # Clean up any extra whitespace and standardize separators
            for col in ['genres', 'keywords', 'cast', 'director']:
                if col in df.columns:
                    # Remove any extra spaces around commas
                    df[col] = df[col].apply(lambda x: ', '.join(item.strip() for item in x.split(',')) if x else '')
                    
                    # Replace multiple spaces with single space
                    df[col] = df[col].apply(lambda x: ' '.join(x.split()))

            # Ensure numeric columns are properly formatted
            if 'tmdb_id' in df.columns:
                df['tmdb_id'] = pd.to_numeric(df['tmdb_id'], errors='coerce')

            if 'vote_average' in df.columns:
                df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
                df['vote_average'] = df['vote_average'].round(3)

            logger.info("New data preparation completed successfully")
            return df

        except Exception as e:
            logger.error(f"Error in new data preparation: {e}")
            raise
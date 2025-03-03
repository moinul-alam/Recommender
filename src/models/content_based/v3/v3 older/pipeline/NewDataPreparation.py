import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NewDataPreparation:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def prepare_new_data(self) -> pd.DataFrame:
        """
        Prepare new data by transforming list/dict format into string format.
        
        Returns:
            pd.DataFrame: Prepared DataFrame with cleaned string values
        """
        try:
            logger.info("Starting new data preparation")
            df = self.df.copy()  # Use self.df instead of df

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

            if 'spoken_languages' in df.columns:
                df['spoken_languages'] = df['spoken_languages'].apply(
                    lambda x: ', '.join([lang.strip() if isinstance(lang, str) else lang['iso_639_1'].strip()
                                        for lang in (json.loads(x.replace("'", '"')) if isinstance(x, str) else x)])
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
            logger.info(f"Dataframe: {df.head()}")
            return df

        except Exception as e:
            logger.error(f"Error in new data preparation: {e}")
            raise
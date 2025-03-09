from typing import Optional, Union
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NewDataPreparation:
    def __init__(self, df: pd.DataFrame, is_custom_query: bool = False, item_mapping: Optional[pd.DataFrame] = None):
        self.df = df
        self.item_mapping = item_mapping
        self.is_custom_query = is_custom_query
        self.id_column = self._determine_id_column()

    def _determine_id_column(self) -> str:
        """
        Determine which ID column to use (tmdb_id or item_id).
        
        Returns:
            str: Name of the ID column to use
        
        Raises:
            ValueError: If no valid ID column is found
        """
        # Convert columns to numeric if they exist
        if 'tmdb_id' in self.df.columns:
            self.df['tmdb_id'] = pd.to_numeric(self.df['tmdb_id'], errors='coerce')
            return 'tmdb_id'
        elif 'item_id' in self.df.columns:
            self.df['item_id'] = pd.to_numeric(self.df['item_id'], errors='coerce')
            return 'item_id'
        else:
            raise ValueError("No valid ID column (tmdb_id or item_id) found in DataFrame")

    def prepare_data(self) -> pd.DataFrame:
        """
        Prepare new data by transforming list/dict format into string format.
        
        Returns:
            pd.DataFrame: Prepared DataFrame with cleaned string values
        """
        try:
            logger.info("Starting new data preparation")
            df = self.df.copy()

            # Validate ID column exists and convert to numeric
            if self.id_column in df.columns:
                df[self.id_column] = pd.to_numeric(df[self.id_column], errors='coerce')
                if df[self.id_column].isna().any():
                    logger.warning(f"Some {self.id_column} values could not be converted to numeric")

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
                
            if 'item_id' in df.columns:
                df['item_id'] = pd.to_numeric(df['item_id'], errors='coerce')

            if 'vote_average' in df.columns:
                df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
                df['vote_average'] = df['vote_average'].round(3)

            logger.info("New data preparation completed successfully")
            return df

        except Exception as e:
            logger.error(f"Error in new data preparation: {e}")
            raise

    def update_item_mapping(self, prepared_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Update item mapping with new items and assign sequential IDs.
        
        Args:
            prepared_df: DataFrame with prepared data
            
        Returns:
            tuple: (Updated prepared DataFrame with new IDs, Updated mapping DataFrame)
        """
        try:
            logger.info("Updating item mapping with new items")
            
            # Validate that necessary columns exist in item mapping
            required_columns = [self.id_column, 'media_type']
            if not all(col in self.item_mapping.columns for col in required_columns):
                raise ValueError(f"Item mapping missing required columns: {required_columns}")
            
            # Create a set of existing ID and media_type combinations
            existing_items = set(zip(
                self.item_mapping[self.id_column], 
                self.item_mapping['media_type']
            ))
            
            # Filter new items that don't exist in mapping
            new_items_mask = ~prepared_df.apply(
                lambda row: (row[self.id_column], row['media_type']) in existing_items, 
                axis=1
            )
            new_items_df = prepared_df[new_items_mask][[self.id_column, 'media_type', 'title']].copy()
            
            if len(new_items_df) > 0:
                # Get the next available item_id
                next_item_id = self.item_mapping['item_id'].max() + 1
                
                # Assign sequential IDs to new items
                new_items_df['item_id'] = range(next_item_id, next_item_id + len(new_items_df))
                
                # Append new items to mapping
                updated_mapping = pd.concat([self.item_mapping, new_items_df], ignore_index=True)
                
                # Create a dictionary for mapping all items
                id_map = dict(zip(
                    zip(updated_mapping[self.id_column], updated_mapping['media_type']), 
                    updated_mapping['item_id']
                ))
                
                # Add item_ids to the prepared DataFrame
                prepared_df['item_id'] = prepared_df.apply(
                    lambda row: id_map.get((row[self.id_column], row['media_type'])), 
                    axis=1
                )
                
                logger.info(f"Added {len(new_items_df)} new items to mapping")
                return prepared_df, updated_mapping
            else:
                logger.info("No new items to add to mapping")
                # Map existing IDs to prepared DataFrame
                id_map = dict(zip(
                    zip(self.item_mapping[self.id_column], self.item_mapping['media_type']), 
                    self.item_mapping['item_id']
                ))
                prepared_df['item_id'] = prepared_df.apply(
                    lambda row: id_map.get((row[self.id_column], row['media_type'])), 
                    axis=1
                )
                return prepared_df, self.item_mapping

        except Exception as e:
            logger.error(f"Error in updating item mapping: {e}")
            raise

    def prepare_new_data(self) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Complete pipeline to prepare new data and optionally update item mapping.
        
        Returns:
            Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]: 
                - For custom queries: just the prepared DataFrame
                - For normal queries: tuple of (Prepared DataFrame with IDs, Updated mapping DataFrame)
        """
        prepared_df = self.prepare_data()
        
        if self.is_custom_query:
            logger.info("Custom query detected - skipping item mapping update")
            return prepared_df
        
        if self.item_mapping is None or self.item_mapping.empty:
            logger.warning("No item mapping provided for non-custom query")
            return prepared_df
            
        return self.update_item_mapping(prepared_df)
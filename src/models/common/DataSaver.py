import os
import logging
import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_to_csv(df: pd.DataFrame, file_path: str, **kwargs) -> str:
    """Save DataFrame to a CSV file."""
    try:
        logger.info(f"Saving CSV to {file_path}")
        df.to_csv(file_path, **kwargs)
        logger.info(f"Data saved to CSV: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving CSV file {file_path}: {e}")
        raise

def save_to_json(df: pd.DataFrame, file_path: str, **kwargs) -> str:
    """Save DataFrame to a JSON file."""
    try:
        logger.info(f"Saving JSON to {file_path}")
        df.to_json(file_path, **kwargs)
        logger.info(f"Data saved to JSON: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        raise

def save_to_pickle(df: pd.DataFrame, file_path: str) -> str:
    """Save DataFrame to a pickle file."""
    try:
        logger.info(f"Saving Pickle to {file_path}")
        df.to_pickle(file_path)
        logger.info(f"Data saved to pickle: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving pickle file {file_path}: {e}")
        raise

def save_to_feather(df: pd.DataFrame, file_path: str) -> str:
    """Save DataFrame to a feather file."""
    try:
        logger.info(f"Saving Feather to {file_path}")
        df.to_feather(file_path)
        logger.info(f"Data saved to feather: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving feather file {file_path}: {e}")
        raise

def save_data(
    directory_path: Union[str, Path], 
    df: pd.DataFrame, 
    file_name: str, 
    file_type: str = "csv", 
    **kwargs
) -> str:
    """
    Save DataFrame to a file with specified directory, name, and type.
    
    Args:
        directory_path: Path to the directory where file will be saved
        df: DataFrame to save
        file_name: Name of the file (without extension)
        file_type: Type of file to save ('csv', 'json', 'pickle', 'feather')
        **kwargs: Additional arguments to pass to the respective save function
        
    Returns:
        str: Full path to the saved file
    """
    # Normalize directory path to Path object
    directory = Path(directory_path)
    
    # Create directory if it doesn't exist
    directory.mkdir(parents=True, exist_ok=True)
    
    # Ensure file_type doesn't have a leading dot
    file_type = file_type.lower().strip('.')
    
    # Create full file path
    file_path = directory / f"{file_name}.{file_type}"
    
    # Save based on file type
    if file_type == 'csv':
        return save_to_csv(df, file_path, **kwargs)
    elif file_type == 'json':
        return save_to_json(df, file_path, **kwargs)
    elif file_type in ['pkl', 'pickle']:
        return save_to_pickle(df, file_path)
    elif file_type == 'feather':
        return save_to_feather(df, file_path)
    else:
        logger.warning(f"Unsupported file format: {file_type}, defaulting to 'csv'.")
        return save_to_csv(df, file_path, **kwargs)

def save_multiple_dataframes(
    directory_path: Union[str, Path],
    dataframes: Dict[str, pd.DataFrame],
    file_type: str = "csv",
    **kwargs
) -> Dict[str, str]:
    """
    Save multiple DataFrames to files in the specified directory.
    
    Args:
        directory_path: Path to the directory where files will be saved
        dataframes: Dictionary mapping file names to DataFrames
        file_type: Type of file to save ('csv', 'json', 'pickle', 'feather')
        **kwargs: Additional arguments to pass to the respective save function
        
    Returns:
        Dict[str, str]: Dictionary mapping file names to saved file paths
    """
    saved_paths = {}
    for file_name, df in dataframes.items():
        saved_path = save_data(directory_path, df, file_name, file_type, **kwargs)
        saved_paths[file_name] = saved_path
    
    return saved_paths
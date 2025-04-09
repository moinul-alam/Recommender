import os
import logging
import joblib
import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_to_csv(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> str:
    """Save DataFrame to a CSV file."""
    try:
        str_path = str(file_path)
        logger.info(f"Saving CSV to {str_path}")
        df.to_csv(file_path, **kwargs)
        logger.info(f"Data saved to CSV: {str_path}")
        return str_path
    except Exception as e:
        logger.error(f"Error saving CSV file {file_path}: {e}")
        raise

def save_to_json(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> str:
    """Save DataFrame to a JSON file."""
    try:
        str_path = str(file_path)
        logger.info(f"Saving JSON to {str_path}")
        df.to_json(file_path, **kwargs)
        logger.info(f"Data saved to JSON: {str_path}")
        return str_path
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        raise

def save_to_pickle(df: pd.DataFrame, file_path: Union[str, Path]) -> str:
    """Save DataFrame to a pickle file."""
    try:
        str_path = str(file_path)
        logger.info(f"Saving Pickle to {str_path}")
        df.to_pickle(file_path)
        logger.info(f"Data saved to pickle: {str_path}")
        return str_path
    except Exception as e:
        logger.error(f"Error saving pickle file {file_path}: {e}")
        raise

def save_to_feather(df: pd.DataFrame, file_path: Union[str, Path]) -> str:
    """Save DataFrame to a feather file."""
    try:
        str_path = str(file_path)
        logger.info(f"Saving Feather to {str_path}")
        df.to_feather(file_path)
        logger.info(f"Data saved to feather: {str_path}")
        return str_path
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
    
    # Ensure file_name doesn't have an extension and file_type doesn't have a leading dot
    file_name = file_name.split('.')[0]
    file_type = file_type.lower().strip('.')
    
    # Create full file path
    file_path = directory / f"{file_name}.{file_type}"
    
    # Save based on file type
    save_functions = {
        'csv': save_to_csv,
        'json': save_to_json,
        'pkl': save_to_pickle,
        'pickle': save_to_pickle,
        'feather': save_to_feather
    }
    
    if file_type in save_functions:
        return save_functions[file_type](df, file_path, **kwargs)
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

def save_object(
    directory_path: Union[str, Path],
    obj: Any,
    file_name: str,
    compress: Optional[int] = None
) -> str:
    """
    Save any Python object (like ML models) to a pickle file.
    
    Args:
        directory_path: Path to the directory where file will be saved
        obj: Object to save
        file_name: Name of the file (without extension)
        compress: Compression level (0-9, None for no compression)
        
    Returns:
        str: Full path to the saved file
    """
    # Normalize directory path to Path object
    directory = Path(directory_path)
    
    # Create directory if it doesn't exist
    directory.mkdir(parents=True, exist_ok=True)
    
    # Ensure file_name doesn't have an extension
    file_name = file_name.split('.')[0]
    
    # Create full file path
    file_path = directory / f"{file_name}.pkl"
    
    try:
        str_path = str(file_path)
        logger.info(f"Saving object to {str_path}")
        joblib.dump(obj, file_path, compress=compress)
        logger.info(f"Object saved to: {str_path}")
        return str_path
    except Exception as e:
        logger.error(f"Error saving object to {file_path}: {e}")
        raise

def save_objects(
    directory_path: Union[str, Path],
    objects: Dict[str, Any],
    compress: Optional[int] = None
) -> Dict[str, str]:
    """
    Save multiple objects to pickle files in the specified directory.
    
    Args:
        directory_path: Path to the directory where files will be saved
        objects: Dictionary mapping file names to objects
        compress: Compression level (0-9, None for no compression)
        
    Returns:
        Dict[str, str]: Dictionary mapping file names to saved file paths
    """
    saved_paths = {}
    for file_name, obj in objects.items():
        saved_path = save_object(directory_path, obj, file_name, compress=compress)
        saved_paths[file_name] = saved_path
    
    return saved_paths
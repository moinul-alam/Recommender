import json
import logging
import pandas as pd
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_from_json(file_path: str) -> pd.DataFrame:
    """Load data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        raise

def load_from_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        return pd.read_csv(file_path, **kwargs)
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {e}")
        raise

def load_from_pickle(file_path: str) -> pd.DataFrame:
    """Load data from a pickle file."""
    try:
        return pd.read_pickle(file_path)
    except Exception as e:
        logger.error(f"Error loading pickle file {file_path}: {e}")
        raise

def load_from_feather(file_path: str) -> pd.DataFrame:
    """Load data from a feather file."""
    try:
        return pd.read_feather(file_path)
    except Exception as e:
        logger.error(f"Error loading feather file {file_path}: {e}")
        raise

def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from a file based on the extension in the file path.
    
    Args:
        file_path: Path to the file to load
        **kwargs: Additional arguments to pass to the loading function
        
    Returns:
        Loaded DataFrame
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is unsupported
    """
    path = Path(file_path)

    if not path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = path.suffix.lower().strip('.')

    loaders = {
        'json': load_from_json,
        'csv': load_from_csv,
        'pkl': load_from_pickle,
        'pickle': load_from_pickle,
        'feather': load_from_feather
    }

    loader = loaders.get(file_ext)

    if not loader:
        logger.error(f"Unsupported file format: .{file_ext}")
        raise ValueError(f"Unsupported file format: .{file_ext}")
    
    logger.info(f"Loading {file_ext.upper()} file: {file_path}")
    return loader(file_path, **kwargs) if file_ext == 'csv' else loader(file_path)

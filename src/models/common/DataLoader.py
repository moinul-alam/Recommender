import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

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

def load_data(file_path: str, extension: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Load data from a file based on its extension or explicitly specified format.
    
    Args:
        file_path: Path to the file to load
        extension: Optional explicit file format (json, csv, pkl, feather)
        **kwargs: Additional arguments to pass to the loading function
        
    Returns:
        Loaded DataFrame
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is unsupported
    """
    if not Path(file_path).exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Use explicit extension if provided, otherwise infer from file path
    if extension:
        # Strip dot if provided
        file_ext = extension.lower().strip('.')
    else:
        file_ext = Path(file_path).suffix.lower().strip('.')
    
    if file_ext == 'json':
        return load_from_json(file_path)
    elif file_ext == 'csv':
        return load_from_csv(file_path, **kwargs)
    elif file_ext in ['pkl', 'pickle']:
        return load_from_pickle(file_path)
    elif file_ext == 'feather':
        return load_from_feather(file_path)
    else:
        logger.error(f"Unsupported file format: {file_ext}")
        raise ValueError(f"Unsupported file format: {file_ext}")
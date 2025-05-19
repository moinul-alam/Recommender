import json
import logging
import joblib
import pandas as pd
from pathlib import Path
from typing import Any
from src.models.common.logger import app_logger

logger = app_logger(__name__)


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

def load_from_pickle(file_path: str) -> Any:
    """Load data from a pickle file using joblib."""
    try:
        logger.info(f"Loading pickle file with joblib: {file_path}")
        return joblib.load(file_path)
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

def load_data(file_path: str, **kwargs) -> Any:
    """
    Load data from a file based on the extension in the file path.
    
    Args:
        file_path: Path to the file to load
        **kwargs: Additional arguments to pass to the loading function
        
    Returns:
        Loaded DataFrame or object
    
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

def load_object(file_path: str) -> Any:
    """
    Load any Python object from a pickle file using joblib.
    
    This is an alias for load_from_pickle, providing consistent naming with save_object.
    """
    return load_from_pickle(file_path)

def load_multiple(directory_path: str, objects: dict, **kwargs) -> dict:
    """
    Load multiple objects from a given directory.

    Args:
        directory_path: Base directory path where files are located.
        objects: A dictionary mapping keys (e.g., "item_matrix") to filenames.
        **kwargs: Extra args passed to specific loaders (like for CSVs).

    Returns:
        A dictionary where keys are the same as input keys and values are the loaded objects.
    """
    results = {}

    for key, file_name in objects.items():
        file_path = Path(directory_path) / file_name
        try:
            results[key] = load_data(str(file_path), **kwargs)
        except Exception as e:
            logger.warning(f"Failed to load {key} from {file_path}: {e}")
            results[key] = None  # Optional: Skip or raise

    return results


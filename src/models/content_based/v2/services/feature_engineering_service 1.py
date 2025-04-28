import pandas as pd
import logging
import gc
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from fastapi import HTTPException
from src.models.content_based.v2.pipeline.feature_engineering import FeatureEngineering
from src.schemas.content_based_schema import PipelineResponse
from src.models.common.DataLoader import load_data
from src.models.common.DataSaver import save_data, save_objects


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FeatureEngineeringService:
    """Service for feature engineering in content-based recommendation systems."""
    
    # Default model parameters if not provided
    DEFAULT_MODEL_COMPONENTS = {
        'tfidf_overview_max_features': 5000,
        'tfidf_keywords_max_features': 1000,
        'max_cast_members': 20,
        'max_directors': 3,
        'n_components_svd_overview': 200,
        'n_components_svd_keywords': 200,
        'n_components_pca': 200,
        'random_state': 42
    }
    
    DEFAULT_FEATURE_WEIGHTS = {
        "overview": 0.40,
        "genres": 0.35,
        "keywords": 0.10,
        "cast": 0.10, 
        "director": 0.05
    }
    
    # Required columns that must be present in the dataset
    REQUIRED_COLUMNS = ['item_id', 'overview', 'genres', 'cast', 'director', 'keywords']
    
    # Required file names for the service
    REQUIRED_FILE_NAMES = [
        'preprocessed_dataset_name',
        'feature_matrix_name',
        'model_config_name',
        'tfidf_overview',
        'tfidf_keywords',
        'mlb_genres',
        'svd_overview',
        'svd_keywords',
        'pca'
    ]
    
    @staticmethod
    def validate_inputs(content_based_dir_path: Union[str, Path], file_names: Dict[str, str]) -> Path:
        """Validate input parameters and return a Path object."""
        # Convert to Path if string
        if isinstance(content_based_dir_path, str):
            content_based_dir_path = Path(content_based_dir_path)
        
        # Check if directory exists
        if not content_based_dir_path.is_dir():
            raise HTTPException(
                status_code=400, 
                detail=f"Directory not found: {content_based_dir_path}"
            )
        
        # Validate file_names dictionary
        missing_keys = [key for key in FeatureEngineeringService.REQUIRED_FILE_NAMES 
                        if key not in file_names]
        if missing_keys:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required keys in file_names: {missing_keys}"
            )
            
        return content_based_dir_path

    @staticmethod
    def engineer_features(
        content_based_dir_path: Union[str, Path], 
        file_names: Dict[str, str],
        model_components: Optional[Dict[str, Any]] = None,
        feature_weights: Optional[Dict[str, float]] = None
    ) -> PipelineResponse:
        """
        Perform feature engineering on datasets.
        
        Args:
            content_based_dir_path: Directory path containing the datasets
            file_names: Dictionary with file names for input/output files
            model_components: Optional model configuration parameters
            feature_weights: Optional feature weights
            
        Returns:
            PipelineResponse object with status and output information
        """
        try:
            # Validate inputs
            content_based_dir_path = FeatureEngineeringService.validate_inputs(
                content_based_dir_path, file_names
            )
            
            # Use default components and weights if not provided
            if model_components is None:
                model_components = FeatureEngineeringService.DEFAULT_MODEL_COMPONENTS
            
            if feature_weights is None:
                feature_weights = FeatureEngineeringService.DEFAULT_FEATURE_WEIGHTS
            
            # Ensure preprocessed dataset has correct extension
            preprocessed_dataset_name = file_names["preprocessed_dataset_name"]
            if not preprocessed_dataset_name.endswith('.csv'):
                preprocessed_dataset_name += '.csv'
            
            # Load preprocessed dataset
            preprocessed_dataset_path = content_based_dir_path / preprocessed_dataset_name
            preprocessed_dataset = load_data(preprocessed_dataset_path)
            
            if preprocessed_dataset is None or preprocessed_dataset.empty:
                raise HTTPException(status_code=400, detail="Preprocessed dataset is empty or invalid")
            
            # Validate required columns
            missing_columns = [col for col in FeatureEngineeringService.REQUIRED_COLUMNS 
                             if col not in preprocessed_dataset.columns]
            if missing_columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required columns in dataset: {missing_columns}"
                )
            
            # Initialize feature engineering
            feature_engineer = FeatureEngineering(
                model_components=model_components,
                feature_weights=feature_weights
            )
            
            # Fit the transformers
            feature_engineer.fit_transformers(preprocessed_dataset)
            
            # Save transformers
            FeatureEngineeringService._save_transformers(content_based_dir_path, feature_engineer, file_names)
            
            # Save model configuration
            model_config = {
                'weights': feature_weights,
                'components': model_components,
                'is_fitted': True
            }
            
            model_config_df = pd.DataFrame([model_config])
            model_config_name = file_names.get('model_config_name', 'model_config')
            save_data(
                directory_path=content_based_dir_path,
                df=model_config_df,
                file_name=model_config_name,
                file_type="pickle"
            )
            logger.info(f"Model config saved to {content_based_dir_path / f'{model_config_name}.pkl'}")

            # Process data (either in segments or as a whole)
            FeatureEngineeringService._process_data(
                content_based_dir_path, 
                preprocessed_dataset,
                feature_engineer,
                file_names
            )

            return PipelineResponse(
                status="Success",
                message="Feature engineering completed successfully",
                output=str(content_based_dir_path)
            )

        except HTTPException:
            # Re-raise HTTP exceptions directly
            raise
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Error in feature engineering: {str(e)}"
            )

    @staticmethod
    def _process_data(
        content_based_dir_path: Path,
        preprocessed_dataset: pd.DataFrame,
        feature_engineer: FeatureEngineering,
        file_names: Dict[str, str]
    ) -> None:
        """Process data either in segments or as a whole."""
        # Find segment files if they exist
        segment_files = []
        if "preprocessed_segment_name" in file_names:
            segment_pattern = file_names['preprocessed_segment_name']
            segment_files = sorted(
                list(content_based_dir_path.glob(f"{segment_pattern}*.csv")),
                key=lambda x: int(x.stem.split("_")[-1]) if x.stem.split("_")[-1].isdigit() else 0
            )
        
        # Process segments if found, otherwise process the entire dataset
        if not segment_files:
            logger.info("No segment files found. Processing entire preprocessed dataset.")
            engineered_df = feature_engineer.transform_features(preprocessed_dataset)
            
            # Save feature matrix
            save_data(
                directory_path=content_based_dir_path,
                df=engineered_df,
                file_name=file_names["feature_matrix_name"],
                file_type="pickle"
            )
        else:
            FeatureEngineeringService._process_segments(
                content_based_dir_path, 
                segment_files, 
                feature_engineer, 
                file_names
            )

    @staticmethod
    def _process_segments(
        content_based_dir_path: Path,
        segment_files: List[Path],
        feature_engineer: FeatureEngineering,
        file_names: Dict[str, str]
    ) -> None:
        """Process data in segments to handle large datasets."""
        logger.info(f"Found {len(segment_files)} segment files to process")
        feature_matrix_segments = []
        temp_feature_matrix_files = []
        
        for file in segment_files:
            try:
                segment_df = pd.read_csv(file)
                logger.info(f"Processing segment: {file.stem} with shape {segment_df.shape}")
                
                if segment_df.empty:
                    logger.warning(f"Skipping empty segment: {file.stem}")
                    continue
                
                # Validate required columns in segment
                missing_columns = [col for col in FeatureEngineeringService.REQUIRED_COLUMNS 
                                 if col not in segment_df.columns]
                if missing_columns:
                    logger.warning(f"Segment {file.stem} is missing columns: {missing_columns}. Skipping.")
                    continue
                
                # Transform features for this segment
                engineered_df = feature_engineer.transform_features(segment_df)
                
                # Save intermediate result
                segment_output_name = f"feature_matrix_{file.stem.split('_')[-1]}"
                saved_path = save_data(
                    directory_path=content_based_dir_path,
                    df=engineered_df,
                    file_name=segment_output_name,
                    file_type="pickle"
                )
                temp_feature_matrix_files.append(Path(saved_path))
                feature_matrix_segments.append(engineered_df)
                
                # Explicitly free memory
                del segment_df
                del engineered_df
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing segment {file.stem}: {str(e)}", exc_info=True)
                raise
        
        # Check if we have any segments to concatenate
        if not feature_matrix_segments:
            raise ValueError("No feature matrix segments were successfully processed")
        
        try:
            # Combine all segments
            feature_matrix = pd.concat(feature_matrix_segments, axis=0)
            feature_matrix.reset_index(drop=True, inplace=True)
            
            # Save combined feature matrix
            save_data(
                directory_path=content_based_dir_path,
                df=feature_matrix,
                file_name=file_names["feature_matrix_name"],
                file_type="pickle"
            )
            
            # Clean up temporary files
            FeatureEngineeringService._cleanup_temp_files(segment_files, temp_feature_matrix_files)
            
        except Exception as e:
            logger.error(f"Error combining feature matrices: {str(e)}", exc_info=True)
            raise
        finally:
            # Clean up memory regardless of success/failure
            del feature_matrix_segments
            gc.collect()

    @staticmethod
    def _cleanup_temp_files(segment_files: List[Path], temp_matrix_files: List[Path]) -> None:
        """Clean up temporary files after processing."""
        # Delete processed segment files
        logger.info("Cleaning up processed segment files...")
        for file in segment_files:
            try:
                if file.exists():
                    file.unlink()
                    logger.info(f"Deleted processed segment file: {file}")
            except Exception as e:
                logger.warning(f"Failed to delete processed segment file {file}: {str(e)}")
        
        # Delete temporary feature matrix files
        logger.info("Cleaning up feature matrix segment files...")
        for file in temp_matrix_files:
            try:
                if file.exists():
                    file.unlink()
                    logger.info(f"Deleted feature matrix segment file: {file}")
            except Exception as e:
                logger.warning(f"Failed to delete feature matrix segment file {file}: {str(e)}")

    @staticmethod
    def _save_transformers(
        content_based_dir_path: Path, 
        feature_engineer: FeatureEngineering, 
        file_names: Dict[str, str]
    ) -> None:
        """Save all transformer objects using DataSaver."""
        try:
            # Create a dictionary mapping file names to transformer objects
            transformers_to_save = {
                file_names['tfidf_overview']: feature_engineer.tfidf_overview,
                file_names['tfidf_keywords']: feature_engineer.tfidf_keywords,
                file_names['mlb_genres']: feature_engineer.mlb_genres,
                file_names['svd_overview']: feature_engineer.svd_overview,
                file_names['svd_keywords']: feature_engineer.svd_keywords,
                file_names['pca']: feature_engineer.pca
            }
            
            # Save all transformers with compression
            save_objects(
                directory_path=content_based_dir_path,
                objects=transformers_to_save,
                compress=3  # Medium compression level for good balance of speed and size
            )
            
            logger.info(f"All transformers saved to {content_based_dir_path}")
        except Exception as e:
            logger.error(f"Error saving transformers: {str(e)}", exc_info=True)
            raise
        
    @staticmethod
    def load_transformers(
        content_based_dir_path: Union[str, Path], 
        file_names: Dict[str, str]
    ) -> FeatureEngineering:
        """
        Load saved transformers for feature engineering.
        
        Args:
            content_based_dir_path: Path to the directory containing transformers
            file_names: Dictionary with file names for the transformers
            
        Returns:
            FeatureEngineering object with loaded transformers
        """
        try:
            # Validate inputs
            content_based_dir_path = FeatureEngineeringService.validate_inputs(
                content_based_dir_path, file_names
            )
            
            logger.info(f"Loading transformers from: {content_based_dir_path}")
            
            # Load configuration
            config_path = content_based_dir_path / f"{file_names['model_config_name']}.pkl"
            if not config_path.exists():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Model config file not found: {config_path}"
                )
                
            config = FeatureEngineeringService._load_object(config_path)
            if isinstance(config, pd.DataFrame):
                if len(config) == 1:
                    config = config.iloc[0].to_dict()
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Config DataFrame should have exactly one row"
                    )

            # Get model components and weights from config
            model_components = config.get('components', FeatureEngineeringService.DEFAULT_MODEL_COMPONENTS)
            feature_weights = config.get('weights', FeatureEngineeringService.DEFAULT_FEATURE_WEIGHTS)
            is_fitted = config.get('is_fitted', True)
            
            # Initialize feature engineering with configuration
            feature_engineer = FeatureEngineering(
                model_components=model_components,
                feature_weights=feature_weights
            )
            
            # Set is_fitted property after initialization
            feature_engineer.is_fitted = is_fitted

            # Map of attribute names to file names
            transformers_map = {
                'tfidf_overview': 'tfidf_overview',
                'mlb_genres': 'mlb_genres',
                'tfidf_keywords': 'tfidf_keywords',
                'svd_overview': 'svd_overview',
                'svd_keywords': 'svd_keywords',
                'pca': 'pca',
                'cast_hasher': 'cast_hasher',
                'director_hasher': 'director_hasher'
            }

            # Load all transformers
            for attr_name, file_key in transformers_map.items():
                if file_key in file_names:
                    file_path = content_based_dir_path / f"{file_names[file_key]}.pkl"
                    if file_path.exists():
                        try:
                            transformer = FeatureEngineeringService._load_object(file_path)
                            setattr(feature_engineer, attr_name, transformer)
                            logger.debug(f"Successfully loaded {attr_name} transformer")
                        except Exception as e:
                            logger.error(f"Failed to load {attr_name}: {str(e)}")
                            raise HTTPException(
                                status_code=500, 
                                detail=f"Error loading {attr_name} transformer: {str(e)}"
                            )
                    else:
                        logger.warning(f"Transformer file not found: {file_path}. Using default.")

            logger.info("All transformers loaded successfully")
            return feature_engineer
            
        except HTTPException:
            raise  # Re-raise HTTPExceptions as-is
        except Exception as e:
            logger.error(f"Error loading transformers: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Error loading transformers: {str(e)}"
            )
    
    @staticmethod
    def _load_object(file_path: Path) -> Any:
        """
        Load a pickled object, handling different storage formats.
        
        Args:
            file_path: Path to the pickle file
            
        Returns:
            The loaded object
        """
        try:
            data = pd.read_pickle(file_path)
            
            # Handle different storage formats
            if isinstance(data, pd.DataFrame):
                # If DataFrame with 'transformer' column
                if len(data) == 1 and 'transformer' in data.columns:
                    return data.iloc[0]['transformer']
                # If DataFrame without 'transformer' column (likely config)
                return data
                
            # Direct object
            return data
            
        except Exception as e:
            logger.error(f"Error loading object from {file_path}: {str(e)}")
            raise
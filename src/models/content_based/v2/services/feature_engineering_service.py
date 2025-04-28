import pandas as pd
import logging
import gc
from pathlib import Path
import os
from typing import Dict, Optional
from fastapi import HTTPException
from src.models.content_based.v2.pipeline.feature_engineering import FeatureEngineering
from src.schemas.content_based_schema import PipelineResponse
from src.models.common.DataLoader import load_data
from src.models.common.DataSaver import save_data, save_objects


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FeatureEngineeringService:
    @staticmethod
    def engineer_features(content_based_dir_path: str, file_names: dict) -> PipelineResponse:
        """
        Perform feature engineering on preprocessed dataset and save the results.
        
        Args:
            content_based_dir_path: Path to directory containing preprocessed data
            file_names: Dictionary mapping file types to their names
            
        Returns:
            PipelineResponse object with status and message
        """
        try:
            content_based_dir_path = Path(content_based_dir_path)
            
            if not content_based_dir_path.is_dir():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Directory not found: {content_based_dir_path}"
                )
            
            # Load preprocessed dataset
            preprocessed_dataset_path = content_based_dir_path / file_names["preprocessed_dataset_name"]
            if not preprocessed_dataset_path.exists():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Preprocessed dataset file not found: {preprocessed_dataset_path}"
                )
                
            preprocessed_dataset = load_data(preprocessed_dataset_path)
            
            if preprocessed_dataset is None or preprocessed_dataset.empty:
                raise HTTPException(
                    status_code=400, 
                    detail="Preprocessed dataset is empty or invalid"
                )
            
            # Model components with updated parameters for individual feature reduction
            model_components = {
                'tfidf_overview_max_features': 5000,
                'tfidf_keywords_max_features': 1000,
                'max_cast_members': 20,
                'max_directors': 3,
                'overview_tsvd_components': 200,
                'keywords_tsvd_components': 100,
                'cast_tsvd_components': 100,    # New parameter for cast dimension reduction
                'director_tsvd_components': 100, # New parameter for director dimension reduction
                'random_state': 42
            }
            
            # Feature weights
            feature_weights = {
                "overview": 0.40,
                "genres": 0.40,
                "keywords": 0.08,
                "cast": 0.07, 
                "director": 0.05
            }
            
            # Initialize feature engineering with components and weights
            feature_engineer = FeatureEngineering(model_components, feature_weights)
            
            # Check for segment files
            segment_files = sorted(
                list(content_based_dir_path.glob(f"{file_names['preprocessed_segment_name']}*.csv")),
                key=lambda x: int(x.stem.split("_")[-1])
            )
            
            # Determine processing approach based on data size and memory constraints
            sample_for_fit = preprocessed_dataset
            if len(segment_files) > 0 and len(preprocessed_dataset) > 50000:
                # If dataset is large, use a sample for fitting transformers
                logger.info("Large dataset detected - using sample for fitting transformers")
                sample_for_fit = preprocessed_dataset.sample(
                    n=min(50000, len(preprocessed_dataset)), 
                    random_state=42
                )
            
            # Fit transformers on sample or full dataset
            logger.info(f"Fitting transformers on dataset with shape {sample_for_fit.shape}")
            feature_engineer.fit_transformers(sample_for_fit)
            
            # Save transformers
            FeatureEngineeringService._save_transformers(content_based_dir_path, feature_engineer, file_names)
            
            # Save model configuration
            model_config = {
                'weights': feature_weights,
                'components': model_components,
                'is_fitted': True
            }
            
            # Save model config
            model_config_df = pd.DataFrame([model_config])
            model_config_name = file_names.get('model_config_name', 'model_config')
            save_data(
                directory_path=content_based_dir_path,
                df=model_config_df,
                file_name=model_config_name,
                file_type="pkl"
            )
            logger.info(f"Model config saved to {content_based_dir_path / f'{model_config_name}.pkl'}")
            
            # Process dataset (either whole or by segments)
            if not segment_files:
                logger.info("No segment files found. Processing entire preprocessed dataset.")
                engineered_df = feature_engineer.transform_features(preprocessed_dataset)
                
                # Save feature matrix
                save_data(
                    directory_path=content_based_dir_path,
                    df=engineered_df,
                    file_name=file_names["feature_matrix_name"],
                    file_type="pkl"
                )
                logger.info(f"Feature matrix saved with shape {engineered_df.shape}")
                
                # Clean up memory
                del engineered_df, preprocessed_dataset, sample_for_fit
                gc.collect()
            else:
                # Process segments to reduce memory usage
                logger.info(f"Found {len(segment_files)} segment files to process")
                feature_matrix_segments = []
                temp_feature_matrix_files = []
                
                # Free memory after fitting
                del sample_for_fit
                if 'preprocessed_dataset' in locals():
                    del preprocessed_dataset
                gc.collect()
                
                for file in segment_files:
                    segment_df = pd.read_csv(file)
                    logger.info(f"Processing segment: {file.stem} with shape {segment_df.shape}")
                    
                    if segment_df.empty:
                        logger.warning(f"Skipping empty segment: {file.stem}")
                        continue
                        
                    # Transform features for this segment
                    engineered_df = feature_engineer.transform_features(segment_df)
                    
                    # Save intermediate result
                    segment_output_name = f"feature_matrix_{file.stem.split('_')[-1]}"
                    saved_path = save_data(
                        directory_path=content_based_dir_path,
                        df=engineered_df,
                        file_name=segment_output_name,
                        file_type="pkl"
                    )
                    temp_feature_matrix_files.append(Path(saved_path))
                    feature_matrix_segments.append(engineered_df)
                    
                    # Clear memory after each segment
                    del segment_df, engineered_df
                    gc.collect()
                
                # Check if we have any segments to concatenate
                if not feature_matrix_segments:
                    raise ValueError("No feature matrix segments were successfully processed")
                
                # Combine all segments 
                logger.info(f"Combining {len(feature_matrix_segments)} feature matrix segments")
                feature_matrix = pd.concat(feature_matrix_segments, axis=0)
                feature_matrix.reset_index(drop=True, inplace=True)
                
                # Save combined feature matrix
                save_data(
                    directory_path=content_based_dir_path,
                    df=feature_matrix,
                    file_name=file_names["feature_matrix_name"],
                    file_type="pkl"
                )
                logger.info(f"Combined feature matrix saved with shape {feature_matrix.shape}")
                
                # Clean up memory
                del feature_matrix, feature_matrix_segments
                gc.collect()
                
                # Clean up processed segment files
                FeatureEngineeringService._cleanup_files(segment_files, "processed segment")
                
                # Clean up feature matrix segment files
                FeatureEngineeringService._cleanup_files(temp_feature_matrix_files, "feature matrix segment")

            return PipelineResponse(
                status="Success",
                message="Feature engineering completed successfully",
                output=str(content_based_dir_path)
            )

        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Error in feature engineering: {str(e)}"
            )

    @staticmethod
    def _cleanup_files(file_paths, file_description):
        """Helper method to clean up temporary files."""
        logger.info(f"Cleaning up {file_description} files...")
        for file in file_paths:
            try:
                if file.exists():
                    file.unlink()
                    logger.info(f"Deleted {file_description} file: {file}")
                else:
                    logger.warning(f"{file_description.capitalize()} file not found: {file}")
            except Exception as e:
                logger.warning(f"Failed to delete {file_description} file {file}: {str(e)}")

    @staticmethod
    def _save_transformers(content_based_dir_path: Path, feature_engineer: FeatureEngineering, file_names: dict) -> None:
        """Save all transformer objects using DataSaver."""
        try:
            # Create a dictionary mapping file names to transformer objects
            transformers_to_save = {
                file_names['tfidf_overview']: feature_engineer.tfidf_overview,
                file_names['tfidf_keywords']: feature_engineer.tfidf_keywords,
                file_names['mlb_genres']: feature_engineer.mlb_genres,
                'mlb_cast': feature_engineer.mlb_cast,                # New transformer
                'mlb_director': feature_engineer.mlb_director,        # New transformer
                'overview_tsvd': feature_engineer.overview_tsvd,
                'keywords_tsvd': feature_engineer.keywords_tsvd,
                'cast_tsvd': feature_engineer.cast_tsvd,              # New transformer
                'director_tsvd': feature_engineer.director_tsvd       # New transformer
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
    def load_transformers(content_based_dir_path: str, file_names: dict) -> FeatureEngineering:
        """Load all transformers from disk and return initialized FeatureEngineering object."""
        try:
            content_based_dir_path = Path(content_based_dir_path)
            
            if not content_based_dir_path.is_dir():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Directory not found: {content_based_dir_path}"
                )
            
            logger.info(f"Loading transformers from: {content_based_dir_path}")
            
            # Load configuration
            config_path = content_based_dir_path / f"{file_names.get('model_config', 'model_config')}.pkl"
            if not config_path.exists():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Model config file not found: {config_path}"
                )
                
            # Load config as DataFrame first
            config_df = pd.read_pickle(config_path)
            config = config_df.iloc[0].to_dict() if isinstance(config_df, pd.DataFrame) else config_df
            
            # Initialize an empty FeatureEngineering instance
            feature_engineer = FeatureEngineering(
                model_components=config.get('components', {}),
                feature_weights=config.get('weights', {})
            )
            
            # Load transformers
            tfidf_overview_path = content_based_dir_path / f"{file_names['tfidf_overview']}.pkl"
            mlb_genres_path = content_based_dir_path / f"{file_names['mlb_genres']}.pkl"
            tfidf_keywords_path = content_based_dir_path / f"{file_names['tfidf_keywords']}.pkl"
            mlb_cast_path = content_based_dir_path / "mlb_cast.pkl"
            mlb_director_path = content_based_dir_path / "mlb_director.pkl"
            overview_tsvd_path = content_based_dir_path / "overview_tsvd.pkl"
            keywords_tsvd_path = content_based_dir_path / "keywords_tsvd.pkl"
            cast_tsvd_path = content_based_dir_path / "cast_tsvd.pkl"
            director_tsvd_path = content_based_dir_path / "director_tsvd.pkl"
            
            # Check if all required files exist
            for path, name in [
                (tfidf_overview_path, "TF-IDF overview"),
                (mlb_genres_path, "MLB genres"),
                (tfidf_keywords_path, "TF-IDF keywords"),
                (mlb_cast_path, "MLB cast"),
                (mlb_director_path, "MLB director"),
                (overview_tsvd_path, "Overview TSVD"),
                (keywords_tsvd_path, "Keywords TSVD"),
                (cast_tsvd_path, "Cast TSVD"),
                (director_tsvd_path, "Director TSVD")
            ]:
                if not path.exists():
                    raise HTTPException(status_code=400, detail=f"{name} file not found: {path}")
            
            # Load transformers
            tfidf_overview_df = pd.read_pickle(tfidf_overview_path)
            mlb_genres_df = pd.read_pickle(mlb_genres_path)
            tfidf_keywords_df = pd.read_pickle(tfidf_keywords_path)
            mlb_cast_df = pd.read_pickle(mlb_cast_path)
            mlb_director_df = pd.read_pickle(mlb_director_path)
            overview_tsvd_df = pd.read_pickle(overview_tsvd_path)
            keywords_tsvd_df = pd.read_pickle(keywords_tsvd_path)
            cast_tsvd_df = pd.read_pickle(cast_tsvd_path)
            director_tsvd_df = pd.read_pickle(director_tsvd_path)
            
            # Extract transformers from DataFrames
            feature_engineer.tfidf_overview = (
                tfidf_overview_df.iloc[0]['transformer'] 
                if isinstance(tfidf_overview_df, pd.DataFrame) 
                else tfidf_overview_df
            )
            
            feature_engineer.mlb_genres = (
                mlb_genres_df.iloc[0]['transformer'] 
                if isinstance(mlb_genres_df, pd.DataFrame) 
                else mlb_genres_df
            )
            
            feature_engineer.tfidf_keywords = (
                tfidf_keywords_df.iloc[0]['transformer'] 
                if isinstance(tfidf_keywords_df, pd.DataFrame) 
                else tfidf_keywords_df
            )
            
            feature_engineer.mlb_cast = (
                mlb_cast_df.iloc[0]['transformer'] 
                if isinstance(mlb_cast_df, pd.DataFrame) 
                else mlb_cast_df
            )
            
            feature_engineer.mlb_director = (
                mlb_director_df.iloc[0]['transformer'] 
                if isinstance(mlb_director_df, pd.DataFrame) 
                else mlb_director_df
            )
            
            feature_engineer.overview_tsvd = (
                overview_tsvd_df.iloc[0]['transformer'] 
                if isinstance(overview_tsvd_df, pd.DataFrame) 
                else overview_tsvd_df
            )
            
            feature_engineer.keywords_tsvd = (
                keywords_tsvd_df.iloc[0]['transformer'] 
                if isinstance(keywords_tsvd_df, pd.DataFrame) 
                else keywords_tsvd_df
            )
            
            feature_engineer.cast_tsvd = (
                cast_tsvd_df.iloc[0]['transformer'] 
                if isinstance(cast_tsvd_df, pd.DataFrame) 
                else cast_tsvd_df
            )
            
            feature_engineer.director_tsvd = (
                director_tsvd_df.iloc[0]['transformer'] 
                if isinstance(director_tsvd_df, pd.DataFrame) 
                else director_tsvd_df
            )

            feature_engineer.max_cast_members = config['components']['max_cast_members']
            feature_engineer.max_directors = config['components']['max_directors']
            feature_engineer.is_fitted = config.get('is_fitted', True)
            
            logger.info("Transformers loaded successfully")
            return feature_engineer
            
        except Exception as e:
            logger.error(f"Error loading transformers: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Error loading transformers: {str(e)}"
            )
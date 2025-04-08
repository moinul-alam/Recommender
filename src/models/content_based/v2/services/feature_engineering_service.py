import pandas as pd
import logging
import gc
from pathlib import Path
import joblib
from fastapi import HTTPException
from src.models.content_based.v2.pipeline.feature_engineering import FeatureEngineering
from src.schemas.content_based_schema import PipelineResponse
from src.models.common.DataLoader import load_data
from src.models.common.DataSaver import save_data


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FeatureEngineeringService:
    @staticmethod
    def engineer_features(
        content_based_dir_path: str,
        file_names: dict
    ) -> PipelineResponse:
        try:
            # Initialize paths
            content_based_dir_path = Path(content_based_dir_path)
            
            if not content_based_dir_path.is_dir():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Directory not found: {content_based_dir_path}"
                )
            
            preprocessed_dataset_name = file_names["preprocessed_dataset_name"]
            if not preprocessed_dataset_name.endswith('.csv'):
                preprocessed_dataset_name += '.csv'
            
            # Load preprocessed dataset
            preprocessed_dataset_path = content_based_dir_path / preprocessed_dataset_name
            preprocessed_dataset = load_data(preprocessed_dataset_path)
            
            if preprocessed_dataset is None or preprocessed_dataset.empty:
                raise HTTPException(status_code=400, detail="Preprocessed dataset is empty or invalid")
            
            model_components = {
                'tfidf_overview_max_features': 5000,
                'tfidf_keywords_max_features': 1000,
                'max_cast_members': 20,
                'max_directors': 3,
                'n_components_svd_overview': 200,
                'n_components_svd_keywords': 200,
                'n_components_pca': 200,
                'random_state': 42
            }
            
            feature_weights = {
                "overview": 0.45,
                "genres": 0.40,
                "keywords": 0.05,
                "cast": 0.07, 
                "director": 0.03
            }
            
            # Initialize feature engineering with components and weights
            feature_engineer = FeatureEngineering(model_components, feature_weights)
            feature_engineer.fit_transformers(preprocessed_dataset)
            
            # Save transformers using joblib
            FeatureEngineeringService._save_transformers(content_based_dir_path, feature_engineer, file_names)
            
            model_config = {
                'weights': feature_weights,
                'components': model_components,
                'max_cast_members': model_components['max_cast_members'],
                'max_directors': model_components['max_directors'],
                'is_fitted': True
            }
            
            # Save model config
            model_config_name = file_names['model_config_name']
            joblib.dump(model_config, content_based_dir_path / f"{file_names.get('model_config', model_config_name)}.pkl")
            logger.info(f"Model config saved to {content_based_dir_path / 'model_config.pkl'}")

            # Process each segment
            segment_files = sorted(
                list(content_based_dir_path.glob(f"{file_names['preprocessed_segment_name']}*.csv")),
                key=lambda x: int(x.stem.split("_")[-1])
            )
            
            # Check if segment files exist
            if not segment_files:
                # If no segments found, process the entire preprocessed dataset
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
                # Process segments
                logger.info(f"Found {len(segment_files)} segment files to process")
                feature_matrix_segments = []
                temp_feature_matrix_files = []
                
                for file in segment_files:
                    segment_df = pd.read_csv(file)
                    logger.info(f"Processing segment: {file.stem} with shape {segment_df.shape}")
                    
                    if segment_df.empty:
                        logger.warning(f"Skipping empty segment: {file.stem}")
                        continue
                        
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
                    gc.collect()
                
                # Check if we have any segments to concatenate
                if not feature_matrix_segments:
                    raise ValueError("No feature matrix segments were successfully processed")
                
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
                
                # Clean up processed segment files after successful processing
                logger.info("Cleaning up processed segment files...")
                for file in segment_files:
                    try:
                        file.unlink()
                        logger.info(f"Deleted processed segment file: {file}")
                    except Exception as e:
                        logger.warning(f"Failed to delete processed segment file {file}: {str(e)}")
                
                # Clean up feature matrix segment files
                logger.info("Cleaning up feature matrix segment files...")
                for file in temp_feature_matrix_files:
                    try:
                        file.unlink()
                        logger.info(f"Deleted feature matrix segment file: {file}")
                    except Exception as e:
                        logger.warning(f"Failed to delete feature matrix segment file {file}: {str(e)}")

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
    def _save_transformers(content_based_dir_path: Path, feature_engineer: FeatureEngineering, file_names: dict) -> None:
        """Save all transformer objects using joblib."""
        try:
            # Save each transformer individually
            joblib.dump(feature_engineer.tfidf_overview, content_based_dir_path / f"{file_names['tfidf_overview']}.pkl")
            joblib.dump(feature_engineer.tfidf_keywords, content_based_dir_path / f"{file_names['tfidf_keywords']}.pkl")
            joblib.dump(feature_engineer.mlb_genres, content_based_dir_path / f"{file_names['mlb_genres']}.pkl")
            joblib.dump(feature_engineer.svd_overview, content_based_dir_path / f"{file_names['svd_overview']}.pkl")
            joblib.dump(feature_engineer.svd_keywords, content_based_dir_path / f"{file_names['svd_keywords']}.pkl")
            joblib.dump(feature_engineer.pca, content_based_dir_path / f"{file_names['pca']}.pkl")
            
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
                raise HTTPException(status_code=400, detail=f"Model config file not found: {config_path}")
                
            config = joblib.load(config_path)
            
            # Initialize an empty FeatureEngineering instance
            feature_engineer = FeatureEngineering(
                model_components=config.get('components', {}),
                feature_weights=config.get('weights', {})
            )
            
            # Load transformers
            feature_engineer.tfidf_overview = joblib.load(content_based_dir_path / f"{file_names['tfidf_overview']}.pkl")
            feature_engineer.mlb_genres = joblib.load(content_based_dir_path / f"{file_names['mlb_genres']}.pkl")
            feature_engineer.tfidf_keywords = joblib.load(content_based_dir_path / f"{file_names['tfidf_keywords']}.pkl")
            feature_engineer.svd_overview = joblib.load(content_based_dir_path / f"{file_names['svd_overview']}.pkl")
            feature_engineer.svd_keywords = joblib.load(content_based_dir_path / f"{file_names['svd_keywords']}.pkl")
            feature_engineer.pca = joblib.load(content_based_dir_path / f"{file_names['pca']}.pkl")
            
            # Set instance variables from config
            feature_engineer.max_cast_members = config.get('max_cast_members', 20)
            feature_engineer.max_directors = config.get('max_directors', 3)
            feature_engineer.is_fitted = config.get('is_fitted', True)
            
            logger.info("Transformers loaded successfully")
            return feature_engineer
            
        except Exception as e:
            logger.error(f"Error loading transformers: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Error loading transformers: {str(e)}"
            )
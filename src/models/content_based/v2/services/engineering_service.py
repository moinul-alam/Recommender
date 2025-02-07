import pandas as pd
import logging
from pathlib import Path
import gc
from fastapi import HTTPException
from src.models.content_based.v2.pipeline.FeatureEngineering import FeatureEngineering
from src.schemas.content_based_schema import PipelineResponse

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EngineeringService:
    @staticmethod
    def engineer_features(
        processed_folder_path: str, 
        features_folder_path: str, 
        transformers_folder_path: str
    ) -> PipelineResponse:
        """
        Process and engineer features from processed data files.
        
        Args:
            processed_folder_path: Path to folder containing processed data
            features_folder_path: Path to save engineered features
            transformers_folder_path: Path to save/load transformers
        """
        try:
            # Initialize paths
            processed_path = Path(processed_folder_path)
            features_path = Path(features_folder_path)
            transformers_folder_path = Path(transformers_folder_path)

            # Validate input path
            if not processed_path.is_dir():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Processed folder not found: {processed_path}"
                )

            # Create output directories
            features_path.mkdir(parents=True, exist_ok=True)
            transformers_folder_path.mkdir(parents=True, exist_ok=True)

            
            # Initialize feature engineering
            feature_engineer = FeatureEngineering(
                max_cast_members=20,
                max_directors=3,
                n_components_svd=200,
                n_components_pca=300,
                random_state=42
            )

            # Process full dataset first
            full_dataset = pd.read_csv(processed_path / "full_processed_dataset.csv")
            feature_engineer.fit_transformers(full_dataset)
            feature_engineer.save_transformers(transformers_folder_path)

            # Process individual segments
            featured_segments = []
            segment_files = sorted(
                processed_path.glob("processed_segment_*.csv"),
                key=lambda x: int(x.stem.split("_")[-1])
            )

            # Process each segment
            for file in segment_files:
                df = pd.read_csv(file)
                print(f"<--------------------Processing segment: {file.stem}-------------------->")
                engineered_df = feature_engineer.transform_features(df)
                
                # Save intermediate result
                save_path = features_path / f"feature_engineering_{file.stem}.feather"
                engineered_df.reset_index(drop=True).to_feather(save_path)
                featured_segments.append(engineered_df)
                gc.collect()

            # Combine all segments
            final_path = features_path / "engineered_features.feather"
            # pd.concat(featured_segments, axis=0).reset_index(drop=True).to_feather(final_path)

            final_features = pd.concat(featured_segments, axis=0)
            final_features = final_features.reset_index(drop=True)  # Reset index

            final_features.to_feather(final_path)

            # Cleanup intermediate files
            for file in processed_path.glob("processed_segment_*.csv"):
                file.unlink(missing_ok=True)
            for file in features_path.glob("feature_engineering_*.feather"):
                file.unlink(missing_ok=True)

            return PipelineResponse(
                status="Feature engineering completed successfully",
                output=len(segment_files),
                output_path=str(final_path)
            )

        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error in feature engineering: {str(e)}"
            )

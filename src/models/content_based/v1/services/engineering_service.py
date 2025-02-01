from pathlib import Path
import pandas as pd
import gc
import os
from fastapi import HTTPException
from src.models.content_based.v1.pipeline.FeatureEngineering import FeatureEngineering
from src.schemas.content_based_schema import PipelineResponse

class EngineeringService:
    @staticmethod
    def engineer_feature(processed_folder_path: str, features_folder_path: str) -> PipelineResponse:
        try:
            # Convert paths to Path objects
            processed_folder_path = Path(processed_folder_path)
            features_folder_path = Path(features_folder_path)

            # Validate processed folder path
            if not processed_folder_path.is_dir():
                raise HTTPException(status_code=400, detail=f"Processed folder not found: {processed_folder_path}")

            # Validate or create features folder path
            features_folder_path.mkdir(parents=True, exist_ok=True)

            # Initialize feature engineering pipeline
            feature_engineer = FeatureEngineering(processed_folder_path, features_folder_path)
            featured_segments = []

            # Sort and process files
            sorted_files = sorted(
                processed_folder_path.glob("processed_segment_*.csv"),
                key=lambda x: int(x.stem.split("_")[-1])
            )

            for segment_file in sorted_files:
                df = pd.read_csv(segment_file)
                engineered_df = feature_engineer.apply_feature_engineering(df)
                save_path = features_folder_path / f"feature_engineering_{segment_file.stem}.feather"
                engineered_df.reset_index(drop=True).to_feather(save_path)
                featured_segments.append(engineered_df)

            # Combine all engineered segments into one file
            combined_save_path = features_folder_path / "engineered_features.feather"
            combined_segments = pd.concat(featured_segments, axis=0)
            combined_segments.reset_index(drop=True).to_feather(combined_save_path)

            # Cleanup: Delete intermediate files
            for segment_file in processed_folder_path.glob("processed_segment_*.csv"):
                segment_file.unlink(missing_ok=True)
            for engineered_file in features_folder_path.glob("feature_engineering_*.feather"):
                engineered_file.unlink(missing_ok=True)

            # Clear memory
            del featured_segments, combined_segments
            gc.collect()

            # Return response
            return PipelineResponse(
                status="Feature engineering completed and results merged successfully",
                output=len(sorted_files),
                output_path=str(combined_save_path)
            )

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error in feature engineering: {str(e)}"
            )

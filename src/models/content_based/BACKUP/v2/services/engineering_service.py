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
        content_based_dir_path: str
    ) -> PipelineResponse:
        try:
            # Initialize paths
            content_based_dir_path = Path(content_based_dir_path)
            
            # Process full dataset first
            full_dataset = pd.read_csv(content_based_dir_path / "2_full_processed_dataset.csv")
             
            feature_engineer = FeatureEngineering()
            feature_engineer.fit_transformers(full_dataset)
            feature_engineer.save_transformers(content_based_dir_path)

            # Process individual segments
            featured_segments = []
            segment_files = sorted(
                content_based_dir_path.glob("2_processed_segment_*.csv"),
                key=lambda x: int(x.stem.split("_")[-1])
            )

            # Process each segment
            for file in segment_files:
                df = pd.read_csv(file)
                print(f"<--------------------Processing segment: {file.stem}-------------------->")
                engineered_df = feature_engineer.transform_features(df)
                
                # Save intermediate result
                save_path = content_based_dir_path / f"3_feature_engineering_{file.stem}.feather"
                engineered_df.reset_index(drop=True).to_feather(save_path)
                featured_segments.append(engineered_df)
                gc.collect()

            # Combine all segments
            final_path = content_based_dir_path / "3_engineered_features.feather"

            final_features = pd.concat(featured_segments, axis=0)
            final_features = final_features.reset_index(drop=True)

            final_features.to_feather(final_path)

            # Cleanup intermediate files
            for file in content_based_dir_path.glob("2_processed_segment_*.csv"):
                file.unlink(missing_ok=True)
            for file in content_based_dir_path.glob("3_feature_engineering_*.feather"):
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

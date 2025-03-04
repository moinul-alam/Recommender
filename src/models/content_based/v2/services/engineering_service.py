import pandas as pd
import numpy as np
import logging
from pathlib import Path
import gc
from fastapi import HTTPException
from scipy import sparse
import json
from src.models.content_based.v2.pipeline.FeatureEngineering import FeatureEngineering
from src.schemas.content_based_schema import PipelineResponse

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EngineeringService:
    @staticmethod
    def engineer_features(content_based_dir_path: str) -> PipelineResponse:
        try:
            # Initialize paths
            content_based_dir_path = Path(content_based_dir_path)
            
            # Process full dataset first
            full_dataset = pd.read_csv(content_based_dir_path / "2_full_processed_dataset.csv")
             
            feature_engineer = FeatureEngineering()
            feature_engineer.fit_transformers(full_dataset)
            feature_engineer.save_transformers(content_based_dir_path)

            # Process individual segments
            segment_files = sorted(
                content_based_dir_path.glob("2_processed_segment_*.csv"),
                key=lambda x: int(x.stem.split("_")[-1])
            )

            # Track segment metadata
            segment_metadata = []
            sparse_matrices = []
            all_item_ids = []
            
            # Process each segment independently
            for idx, file in enumerate(segment_files):
                df = pd.read_csv(file)
                logger.info(f"Processing segment: {file.stem}")

                # Get item IDs and sparse matrix
                item_ids, sparse_matrix = feature_engineer.transform_features_sparse(df.copy())
                
                # Generate file paths for segment outputs
                segment_id = file.stem.split("_")[-1]
                sparse_save_path = content_based_dir_path / f"3_feature_matrix_segment_{segment_id}.npz"
                item_ids_save_path = content_based_dir_path / f"3_item_ids_segment_{segment_id}.npy"
                
                # Save sparse matrix in compressed format
                sparse.save_npz(sparse_save_path, sparse_matrix)
                np.save(item_ids_save_path, np.array(item_ids))  # Save item IDs

                # Append to lists for final combination
                sparse_matrices.append(sparse_matrix)
                all_item_ids.extend(item_ids)

                # Record metadata for this segment
                segment_metadata.append({
                    "segment_id": segment_id,
                    "matrix_path": str(sparse_save_path),
                    "item_ids_path": str(item_ids_save_path),
                    "num_items": len(item_ids),
                    "matrix_shape": sparse_matrix.shape
                })
                
                del sparse_matrix, item_ids, df  # Free memory
                gc.collect()
            
            # Combine all sparse matrices
            final_sparse_matrix = sparse.vstack(sparse_matrices)
            final_item_ids = np.array(all_item_ids)

            # Save final combined matrices
            final_sparse_path = content_based_dir_path / "3_final_feature_matrix.npz"
            final_item_ids_path = content_based_dir_path / "3_final_item_ids.npy"

            sparse.save_npz(final_sparse_path, final_sparse_matrix)
            np.save(final_item_ids_path, final_item_ids)

            # Save the segment metadata for the next step
            metadata_path = content_based_dir_path / "3_engineered_features_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    "num_segments": len(segment_metadata),
                    "segments": segment_metadata,
                    "total_items": final_sparse_matrix.shape[0],
                    "final_matrix_path": str(final_sparse_path),
                    "final_item_ids_path": str(final_item_ids_path),
                    "final_matrix_shape": final_sparse_matrix.shape
                }, f, indent=2)
            
            return PipelineResponse(
                status="Feature engineering completed successfully",
                output=len(segment_files),
                output_path=str(metadata_path)
            )

        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Error in feature engineering: {str(e)}"
            )

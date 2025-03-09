import pandas as pd
import numpy as np
import logging
from pathlib import Path
import gc
from fastapi import HTTPException
from scipy import sparse
import json
import os
import joblib
from src.models.content_based.v4.pipeline.FeatureEngineering import FeatureEngineering
from src.schemas.content_based_schema import PipelineResponse

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FeatureEngineeringService:
    @staticmethod
    def engineer_features(content_based_dir_path: str) -> PipelineResponse:
        try:
            # Initialize paths
            content_based_dir_path = Path(content_based_dir_path)
            
            # Process full dataset first
            full_dataset = pd.read_csv(content_based_dir_path / "3_full_processed_dataset.csv")
             
            feature_engineer = FeatureEngineering()
            feature_engineer.fit_transformers(full_dataset)
            
            # Save transformers to disk
            FeatureEngineeringService._save_transformers(feature_engineer, content_based_dir_path)

            # Process individual segments
            segment_files = sorted(
                content_based_dir_path.glob("3_processed_segment_*.csv"),
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
                sparse_save_path = content_based_dir_path / f"4_feature_matrix_segment_{segment_id}.npz"
                item_ids_save_path = content_based_dir_path / f"4_item_ids_segment_{segment_id}.npy"
                
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
            final_sparse_path = content_based_dir_path / "4_final_feature_matrix.npz"
            final_item_ids_path = content_based_dir_path / "4_final_item_ids.npy"

            sparse.save_npz(final_sparse_path, final_sparse_matrix)
            np.save(final_item_ids_path, final_item_ids)

            # Save metadata
            metadata_path = content_based_dir_path / "4_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    "segments": segment_metadata,
                    "final_matrix_shape": final_sparse_matrix.shape,
                    "total_items": len(final_item_ids)
                }, f, indent=2)

            # Clean up temporary segment files after final matrices are created
            FeatureEngineeringService._cleanup_temp_files(content_based_dir_path)
            
            return PipelineResponse(
                status="Feature engineering completed successfully",
                output=len(segment_files),
                output_path=str(content_based_dir_path)
            )

        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Error in feature engineering: {str(e)}"
            )
    
    @staticmethod
    def _save_transformers(feature_engineer: FeatureEngineering, path: Path) -> None:
        """Save all transformers to disk."""
        if not feature_engineer.is_fitted:
            raise ValueError("Cannot save unfitted transformers")
            
        logger.info(f"Saving transformers to: {path}")
        path.mkdir(parents=True, exist_ok=True)
        
        config = {
            'weights': feature_engineer.weights,
            'max_cast_members': feature_engineer.max_cast_members,
            'max_directors': feature_engineer.max_directors,
            'is_fitted': feature_engineer.is_fitted
        }
        
        transformers = {
            '4_tfidf_overview': feature_engineer.tfidf_overview,
            '4_mlb_genres': feature_engineer.mlb_genres,
            '4_tfidf_keywords': feature_engineer.tfidf_keywords,
            '4_tfidf_cast': feature_engineer.tfidf_cast,
            '4_tfidf_director': feature_engineer.tfidf_director
        }
    
        for name, transformer in transformers.items():
            joblib.dump(transformer, path / f"{name}.pkl")
        
        joblib.dump(config, path / "4_config.pkl")
        logger.info("Transformers saved successfully")
    
    @staticmethod
    def load_transformers(path: Path) -> FeatureEngineering:
        """Load all transformers from disk and return a configured FeatureEngineering instance."""
        logger.info(f"Loading transformers from: {path}")
        
        try:
            config = joblib.load(path / "4_config.pkl")
            
            feature_engineer = FeatureEngineering(
                max_cast_members=config['max_cast_members'],
                max_directors=config['max_directors'],
                weights=config['weights']
            )
            
            feature_engineer.tfidf_overview = joblib.load(path / "4_tfidf_overview.pkl")
            feature_engineer.mlb_genres = joblib.load(path / "4_mlb_genres.pkl")
            feature_engineer.tfidf_keywords = joblib.load(path / "4_tfidf_keywords.pkl")
            feature_engineer.tfidf_cast = joblib.load(path / "4_tfidf_cast.pkl")
            feature_engineer.tfidf_director = joblib.load(path / "4_tfidf_director.pkl")
            feature_engineer.is_fitted = config['is_fitted']

            logger.info("Transformers loaded successfully")
            return feature_engineer
            
        except Exception as e:
            logger.error(f"Error loading transformers: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    def _cleanup_temp_files(content_based_dir_path: Path) -> None:
        """Clean up temporary segment files."""
        logger.info("Cleaning up temporary segment files...")
        
        # Remove processed segment CSV files
        for file in content_based_dir_path.glob("3_processed_segment_*.csv"):
            try:
                os.remove(file)
                logger.info(f"Removed file: {file}")
            except Exception as e:
                logger.warning(f"Failed to remove file {file}: {str(e)}")
        
        # Remove segment feature matrices
        for file in content_based_dir_path.glob("4_feature_matrix_segment_*.npz"):
            try:
                os.remove(file)
                logger.info(f"Removed file: {file}")
            except Exception as e:
                logger.warning(f"Failed to remove file {file}: {str(e)}")
        
        # Remove segment item IDs
        for file in content_based_dir_path.glob("4_item_ids_segment_*.npy"):
            try:
                os.remove(file)
                logger.info(f"Removed file: {file}")
            except Exception as e:
                logger.warning(f"Failed to remove file {file}: {str(e)}")
                
        logger.info("Cleanup completed")
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
import gc
from fastapi import HTTPException
from scipy import sparse
import joblib
from typing import Dict, List, Optional, Tuple, Any

from src.models.content_based.v4.pipeline.ModelTraining import ModelTraining
from src.schemas.content_based_schema import PipelineResponse

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ModelTrainingService:
    @staticmethod
    def train_model(content_based_dir_path: str) -> PipelineResponse:
        """
        Train the SVD model and save the transformed features with item IDs.
        
        Args:
            content_based_dir_path: Directory containing preprocessed feature matrices
            
        Returns:
            PipelineResponse: Response containing status and output information
        """
        try:
            # Initialize paths
            content_based_dir_path = Path(content_based_dir_path)
            
            if not content_based_dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {content_based_dir_path}")
            
            # Load data
            feature_matrix, item_ids = ModelTrainingService._load_data(content_based_dir_path)
            
            # Initialize and run model training
            model_trainer = ModelTraining(
                feature_matrix=feature_matrix,
                item_ids=item_ids,
                n_components_svd=1000,
                random_state=42            
            )
            
            # Fit SVD and transform feature matrix
            transformed_features_with_ids = model_trainer.fit_transform_and_combine()
            
            # Save model and transformed features
            model_output_path = ModelTrainingService._save_model_outputs(
                content_based_dir_path, 
                model_trainer, 
                transformed_features_with_ids
            )
            
            # Free memory
            del feature_matrix
            gc.collect()
            
            return PipelineResponse(
                status="Model training completed successfully",
                output=transformed_features_with_ids.shape[0],
                output_path=str(model_output_path)
            )

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Error in model training: {str(e)}"
            )
    
    @staticmethod
    def _load_data(base_path: Path) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Load feature matrix and item IDs from disk.
        
        Args:
            base_path: Base directory path
            
        Returns:
            Tuple containing feature matrix and item IDs
        """
        try:
            # Load sparse feature matrix
            feature_matrix = sparse.load_npz(base_path / "4_final_feature_matrix.npz")
            # Load item IDs
            item_ids = np.load(base_path / "4_final_item_ids.npy")
            
            logger.info(f"Loaded feature matrix with shape: {feature_matrix.shape}")
            logger.info(f"Loaded item_ids with shape: {item_ids.shape}")
            
            return feature_matrix, item_ids
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    def _save_model_outputs(
        base_path: Path, 
        model_trainer: ModelTraining, 
        transformed_features_df: pd.DataFrame
    ) -> Path:
        """
        Save model and transformed features to disk.
        
        Args:
            base_path: Base directory path
            model_trainer: Trained model instance
            transformed_features_df: DataFrame with item IDs and transformed features
            
        Returns:
            Path to the output directory
        """
        try:
            # Create output directory
            model_output_path = base_path / "5_model_output"
            model_output_path.mkdir(exist_ok=True, parents=True)
            
            # Save SVD model
            ModelTrainingService.save_model(
                model=model_trainer.svd,
                metadata=model_trainer.get_model_metadata(),
                output_path=model_output_path
            )
            
            # Save combined dataframe (item_ids with transformed features)
            combined_features_path = model_output_path / "transformed_features_with_ids.parquet"
            transformed_features_df.to_parquet(str(combined_features_path), index=False)
            logger.info(f"Saved combined features and IDs to: {str(combined_features_path)}")
            
            # Extract and save transformed features for FAISS
            features_array = transformed_features_df.drop(columns=['item_id']).values
            np.save(model_output_path / "transformed_features_for_faiss.npy", features_array)
            logger.info(f"Saved transformed features for FAISS to: {str(model_output_path / 'transformed_features_for_faiss.npy')}")
            
            # Save item IDs separately for FAISS index mapping
            np.save(model_output_path / "faiss_item_ids.npy", transformed_features_df['item_id'].values)
            logger.info(f"Saved item IDs for FAISS to: {str(model_output_path / 'faiss_item_ids.npy')}")
            
            return model_output_path
            
        except Exception as e:
            logger.error(f"Error saving model outputs: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    def save_model(model: Any, metadata: Dict, output_path: Path) -> None:
        """
        Save model and its metadata to disk.
        
        Args:
            model: Trained model object
            metadata: Dictionary with model metadata
            output_path: Directory to save the model
        """
        try:
            logger.info(f"Saving model to: {str(output_path)}")
            
            # Save model
            joblib.dump(model, output_path / "svd_model.pkl")
            
            # Save metadata
            with open(output_path / "svd_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info("Model and metadata saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    def load_model(model_path: str) -> Tuple[Any, Dict]:
        """
        Load a trained model and its metadata from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Tuple containing loaded model and metadata
        """
        try:
            model_path = Path(model_path)
            logger.info(f"Loading model from: {str(model_path)}")
            
            # Load model
            model = joblib.load(model_path / "svd_model.pkl")
            
            # Load metadata
            metadata = {}
            if (model_path / "svd_metadata.json").exists():
                with open(model_path / "svd_metadata.json", 'r') as f:
                    metadata = json.load(f)
            
            logger.info("Model loaded successfully")
            
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Error loading model: {str(e)}"
            )
    
    @staticmethod
    def create_model_instance_from_saved(model_path: str) -> ModelTraining:
        """
        Create a ModelTraining instance from a saved model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            ModelTraining instance with loaded model
        """
        try:
            model_path = Path(model_path)
            
            # Load model and metadata
            svd_model, metadata = ModelTrainingService.load_model(str(model_path))
            
            # Create placeholder feature matrix and item IDs
            feature_matrix_shape = metadata.get("feature_matrix_shape", (0, 0))
            placeholder_feature_matrix = sparse.csr_matrix(feature_matrix_shape)
            
            item_ids_shape = metadata.get("item_ids_shape", (0,))
            placeholder_item_ids = np.array([])
            
            # Create model instance
            model_instance = ModelTraining(
                feature_matrix=placeholder_feature_matrix,
                item_ids=placeholder_item_ids,
                n_components_svd=metadata.get("n_components", 1000),
                random_state=metadata.get("random_state", 42)
            )
            
            # Set the loaded SVD model
            model_instance.svd = svd_model
            model_instance.is_fitted = True
            
            return model_instance
            
        except Exception as e:
            logger.error(f"Error creating model instance: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Error creating model instance: {str(e)}"
            )
    
    @staticmethod
    def delete_intermediate_files(content_based_dir_path: str) -> bool:
        """
        Delete intermediate files to free up disk space after model training.
        
        Args:
            content_based_dir_path: Directory containing intermediate files
            
        Returns:
            Boolean indicating success
        """
        try:
            content_based_dir_path = Path(content_based_dir_path)
            
            # Files that can be safely deleted after model training
            intermediate_files = [
                "4_final_feature_matrix.npz",
                # Add other intermediate files that can be deleted
            ]
            
            for file_name in intermediate_files:
                file_path = content_based_dir_path / file_name
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted intermediate file: {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting intermediate files: {str(e)}", exc_info=True)
            return False
            
    @staticmethod
    def load_transformed_features_for_faiss(model_output_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load transformed features and item IDs for FAISS indexing.
        
        Args:
            model_output_path: Path to the model output directory
            
        Returns:
            Tuple containing transformed features and item IDs
        """
        try:
            model_output_path = Path(model_output_path)
            
            # Load transformed features
            features_path = model_output_path / "transformed_features_for_faiss.npy"
            if not features_path.exists():
                raise FileNotFoundError(f"Transformed features file not found: {features_path}")
                
            transformed_features = np.load(features_path)
            
            # Load item IDs
            item_ids_path = model_output_path / "faiss_item_ids.npy"
            if not item_ids_path.exists():
                raise FileNotFoundError(f"Item IDs file not found: {item_ids_path}")
                
            item_ids = np.load(item_ids_path)
            
            logger.info(f"Loaded transformed features with shape: {transformed_features.shape}")
            logger.info(f"Loaded item IDs with shape: {item_ids.shape}")
            
            return transformed_features, item_ids
            
        except Exception as e:
            logger.error(f"Error loading transformed features for FAISS: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Error loading transformed features for FAISS: {str(e)}"
            )
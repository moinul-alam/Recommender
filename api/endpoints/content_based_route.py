from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Body, HTTPException, Query
import logging
import pandas as pd
import numpy as np
import os
import gc
from app.config import settings
from app.content_based.DataPreprocessing import DataPreprocessing
from app.content_based.FeatureEngineering import FeatureEngineering
from app.content_based.SimilarityComputation import SimilarityComputation
from app.content_based.ContentBasedRecommender import ContentBasedRecommender
from app.schemas.content_based_responses import (
    Metadata,
    PreprocessingResponse,
    EngineeringResponse,
    SimilarityResponse,
    Recommendation,
    RecommendationRequest,
    RecommendationResponse
)
content_based_router = APIRouter()

# Setup logging properly
logger = logging.getLogger(__name__)

@content_based_router.post("/data-preprocessing", response_model=PreprocessingResponse)
async def preprocess_data(
    dataset_path: str = Query(
        default=settings.CONTENT_BASED_DATASET_PATH,
        description="Path to the dataset file"
    ),
    file_name: str = Query(
        default="tmdb_contents_180k.csv",
        description="Name of the dataset file"
    ),
    segment_size: int = Query(
        default=6000,
        description="Number of rows per segment (default is 6000)"
    )
):
    """
    Preprocesses the dataset and saves the results to the preprocessed folder.
    Allows customization of dataset path and segment size via query parameters.
    """
    try:
        dataset_path = os.path.join(dataset_path, file_name)

        if not os.path.isfile(dataset_path):
            raise HTTPException(status_code=400, detail=f"Dataset file not found: {dataset_path}")

        preprocessed_folder = os.path.join(os.path.dirname(dataset_path), "preprocessed")
        os.makedirs(preprocessed_folder, exist_ok=True)

        data_preprocessor = DataPreprocessing(dataset_path, segment_size) 

        processed_segments = data_preprocessor.apply_data_preprocessing()

        for i, segment in enumerate(processed_segments):
            segment_file = os.path.join(preprocessed_folder, f"processed_segment_{i + 1}.csv")
            segment.to_csv(segment_file, index=False)

        return PreprocessingResponse(
            message="Data preprocessed and saved successfully",
            processed_segments=len(processed_segments),
            saved_path=preprocessed_folder
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing data: {str(e)}")


@content_based_router.post("/feature-engineering", response_model=EngineeringResponse)
async def engineer_data(
    preprocessed_folder: str = Query(
        default=settings.CONTENT_BASED_PREPROCESSED_PATH,
        description="Path to the preprocessed dataset folder",
    ),
):
    """
    Apply feature engineering to all preprocessed datasets in the folder and merge results.
    """
    try:
        preprocessed_path = Path(preprocessed_folder)

        if not preprocessed_path.is_dir():
            raise HTTPException(
                status_code=400, detail=f"Preprocessed folder not found: {preprocessed_folder}"
            )

        engineered_folder = preprocessed_path.parent / "engineered"
        engineered_folder.mkdir(parents=True, exist_ok=True)

        # Instantiate FeatureEngineering
        feature_engineer = FeatureEngineering(
            segment_folder_path=str(preprocessed_folder),
            save_folder_path=str(engineered_folder),
        )

        # Apply feature engineering
        featured_segments = []

        # Sort files numerically based on their stem
        sorted_files = sorted(
            preprocessed_path.glob("processed_segment_*.csv"),
            key=lambda x: int(x.stem.split("_")[-1])
        )

        for segment_file in sorted_files:
            df = pd.read_csv(segment_file)
            # logger.info("Loading " + str(df))
            engineered_df = feature_engineer.apply_feature_engineering(df)
            save_path = engineered_folder / f"feature_engineering_{segment_file.stem}.feather"
            engineered_df.reset_index(drop=True).to_feather(save_path)
            featured_segments.append(engineered_df)

        combined_save_path = engineered_folder / "engineered_features.feather"
        combined_segments = pd.concat(featured_segments, axis=0)
        combined_segments.reset_index(drop=True).to_feather(combined_save_path)

        # Cleanup: Delete intermediate files
        for segment_file in preprocessed_path.glob("processed_segment_*.csv"):
            segment_file.unlink(missing_ok=True)
        for engineered_file in engineered_folder.glob("feature_engineering_*.feather"):
            engineered_file.unlink(missing_ok=True)

        # Clear memory
        for segment in featured_segments:
            del segment
        gc.collect()

        return EngineeringResponse(
            message="Feature engineering completed and results merged successfully",
            featured_segments=len(featured_segments),
            saved_path=str(combined_save_path),
            engineered_dataset=[
                segment_file.name for segment_file in engineered_folder.glob("*")
            ],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in feature engineering: {str(e)}"
        )


@content_based_router.post("/similarity-computation", response_model=SimilarityResponse)
async def compute_similarity(
    engineered_folder: str = Query(
        default=settings.CONTENT_BASED_ENGINEERED_PATH,
        description="Path to the similarity index folder",
    ),
    metric: str = Query(
        default="L2", 
        description="Similarity metric to use (e.g., L2, Inner Product)",
    ),
):
    """
    Compute similarity using FAISS.
    """
    try:
        # Validate inputs
        if metric not in ["L2", "Inner Product"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported metric: {metric}. Choose 'L2' or 'Inner Product'."
            )

        if not os.path.isdir(engineered_folder):
            raise HTTPException(status_code=400, detail=f"Engineered folder not found: {engineered_folder}")

        engineered_features_path = os.path.join(engineered_folder, "engineered_features.feather")

        if not os.path.isfile(engineered_features_path):
            raise HTTPException(status_code=400, detail=f"Combined engineered dataset not found: {engineered_features_path}")

        # Load data
        data = pd.read_feather(engineered_features_path)

        if data.empty:
            raise HTTPException(status_code=400, detail="Engineered features file is empty")

        # Ensure numeric columns
        if not all(data.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            raise HTTPException(
                status_code=400, 
                detail="Engineered features must contain only numeric columns for FAISS compatibility",
            )

        # Define similarity folder and paths
        similarity_folder = os.path.join(os.path.dirname(engineered_folder), "similarity")
        os.makedirs(similarity_folder, exist_ok=True)

        content_similarity_index_path = os.path.join(similarity_folder, "content_similarity_index.faiss")

        # Perform similarity computation
        similarity_computation = SimilarityComputation(
            save_path=content_similarity_index_path,
            metric=metric
        )
        saved_path = similarity_computation.build_index(data=data)

        return SimilarityResponse(
            message="Similarity computation completed successfully",
            saved_path=saved_path
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in similarity computation: {str(e)}")


@content_based_router.post("/similar", response_model=RecommendationResponse)
async def recommendations(
    recommendation_request: RecommendationRequest,
    n_items: int = Query(default=10, ge=1, le=100, description="Number of recommendations to return (1-100)"),
    content_similarity_index_folder: str = Query(default=settings.CONTENT_BASED_INDEX_PATH, description="Path to the FAISS index file"),
    engineered_folder: str = Query(default=settings.CONTENT_BASED_ENGINEERED_PATH, description="Path to the combined features file"),
):
    try:
        # Extract tmdbId and metadata from the request
        tmdbId = recommendation_request.tmdbId
        metadata = recommendation_request.metadata

        # Validate paths
        content_similarity_index_path = os.path.join(content_similarity_index_folder, "content_similarity_index.faiss")
        engineered_features_path = os.path.join(engineered_folder, "engineered_features.feather")

        if not os.path.isfile(content_similarity_index_path):
            raise HTTPException(status_code=400, detail=f"Index file not found at: {content_similarity_index_path}")
        if not os.path.isfile(engineered_features_path):
            raise HTTPException(status_code=400, detail=f"Features file not found at: {engineered_features_path}")

        # Instantiate and get recommendations
        recommender = ContentBasedRecommender(
            tmdbId=tmdbId,
            metadata=metadata.dict() if metadata else None,
            index_path=content_similarity_index_path,
            features_path=engineered_features_path,
            n_items=n_items
        )

        recommendations = recommender.get_recommendation()

        # Convert recommendations to the Pydantic model
        recommendation_models = [
            Recommendation(tmdbId=rec["tmdbId"], similarity=rec["similarity"])
            for rec in recommendations
        ]

        return RecommendationResponse(
            message=f"Successfully retrieved {len(recommendation_models)} recommendations",
            queriedMedia=str(tmdbId),
            similarMedia=recommendation_models
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in recommendations endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in generating recommendations: {str(e)}")
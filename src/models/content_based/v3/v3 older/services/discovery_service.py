import pandas as pd
import logging
import joblib
from pathlib import Path
from fastapi import HTTPException
from src.models.content_based.v2.pipeline.Recommender import Recommender
from src.schemas.content_based_schema import Recommendation, RecommendationRequest, RecommendationResponse
from src.models.content_based.v2.pipeline.NewDataPreparation import NewDataPreparation
from src.models.content_based.v2.pipeline.data_preprocessing import DataPreprocessing
from src.models.content_based.v2.pipeline.feature_engineering import FeatureEngineering

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DiscoveryService:
    @staticmethod
    def discover_media(
        recommendation_request: RecommendationRequest, 
        features_folder_path: str, 
        model_folder_path: str,
        processed_folder_path: str,
        transformers_folder_path: str,
        n_items: int
    ) -> RecommendationResponse:
        try:
            tmdb_id = recommendation_request.tmdb_id
            metadata = recommendation_request.metadata
            media_type = metadata.media_type
            spoken_languages = metadata.spoken_languages
            is_custom_query = True

            features_folder = Path(features_folder_path)
            model_folder = Path(model_folder_path)
            processed_folder = Path(processed_folder_path)
            transformers_folder = Path(transformers_folder_path)

            # Validate folders and files
            for folder, name in [
                (features_folder, "Features"),
                (model_folder, "Model"),
                (processed_folder, "Processed"),
                (transformers_folder, "Transformers")
            ]:
                if not folder.is_dir():
                    raise HTTPException(status_code=400, detail=f"{name} folder not found: {folder}")

            features_file = features_folder / "engineered_features.feather"
            model_file = model_folder / "content_based_model.index"
            processed_file = processed_folder / "full_processed_dataset.csv"

            for file, name in [
                (features_file, "Features"),
                (model_file, "Model"),
                (processed_file, "Processed")
            ]:
                if not file.exists():
                    raise HTTPException(status_code=400, detail=f"{name} file not found: {file}")

            feature_matrix = pd.read_feather(features_file)
            processed_df = pd.read_csv(processed_file)

            if feature_matrix.empty or processed_df.empty:
                raise HTTPException(status_code=400, detail="Dataset is empty.")

            if metadata:
                logging.info(f'New media detected.')
                metadata_df = pd.DataFrame([{
                    'tmdb_id': tmdb_id,
                    'media_type': metadata.media_type,
                    'title': metadata.title,
                    'overview': metadata.overview,
                    'spoken_languages': metadata.spoken_languages,
                    'vote_average': metadata.vote_average,
                    'release_year': metadata.release_year,
                    'genres': metadata.genres,
                    'keywords': metadata.keywords,
                    'cast': metadata.cast,
                    'director': metadata.director
                }])
                
                data_preparer = NewDataPreparation(metadata_df)
                new_prepared_data = data_preparer.prepare_new_data()

                logging.info(f'Data preparation complete.')

                preprocessor = DataPreprocessing(df=new_prepared_data)
                new_processed_data = preprocessor.preprocess_new_data(new_prepared_data)

                logging.info(f'Data preprocessing complete.')

                # Ensure new_features has the same column order as existing_feature_matrix
                new_processed_data = new_processed_data[processed_df.columns]

                # Load and apply feature engineering
                feature_engineering = FeatureEngineering()
                feature_engineering.load_transformers(transformers_folder)
                new_features = feature_engineering.transform_features(new_processed_data)
           
                recommender = Recommender(
                    tmdb_id=tmdb_id,
                    metadata=metadata.dict(),
                    features_file=features_file,
                    model_file=str(model_file),
                    n_items=n_items * 3,
                    is_custom_query=is_custom_query
                )

                recommendations = recommender.get_recommendation_for_new(new_features)

            # Filter recommendations based on media type and spoken languages
            filtered_recommendations = []
            for rec in recommendations:
                rec_row = processed_df.loc[processed_df['tmdb_id'] == rec['tmdb_id']]
                rec_media_type = rec_row['media_type'].values[0] if not rec_row.empty else None
                rec_languages = rec_row['spoken_languages'].values[0] if not rec_row.empty else ""

                # Convert rec_languages to a list
                rec_languages = rec_languages.split() if isinstance(rec_languages, str) else []

                if rec_media_type == media_type and any(lang in rec_languages for lang in spoken_languages):
                    filtered_recommendations.append(rec)

                if len(filtered_recommendations) >= n_items:
                    break

            recommendation_models = [
                Recommendation(tmdb_id=rec["tmdb_id"], similarity=rec["similarity"])
                for rec in filtered_recommendations
            ]

            return RecommendationResponse(
                status=f"Successfully retrieved {len(recommendation_models)} recommendations",
                queriedMedia=str(tmdb_id),
                similarMedia=recommendation_models,
            )

        except Exception as e:
            logger.error(f"Error during recommendation retrieval: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in recommendation service: {str(e)}")

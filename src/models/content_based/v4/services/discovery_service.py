import pandas as pd
import logging
import joblib
from pathlib import Path
from fastapi import HTTPException
from src.models.content_based.v2.pipeline.Recommender import Recommender
from src.schemas.content_based_schema import Recommendation, RecommendationRequest, RecommendationResponse
from src.models.content_based.v2.pipeline.NewDataPreparation import NewDataPreparation
from src.models.content_based.v2.pipeline.DataPreprocessing import DataPreprocessing
from src.models.content_based.v2.pipeline.FeatureEngineering import FeatureEngineering

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DiscoveryService:
    @staticmethod
    def discover_media(
        recommendation_request: RecommendationRequest, 
        content_based_dir_path: str, 
        n_items: int
    ) -> RecommendationResponse:
        try:
            tmdb_id = recommendation_request.tmdb_id
            item_id = int(tmdb_id)
            metadata = recommendation_request.metadata
            media_type = metadata.media_type
            spoken_languages = metadata.spoken_languages
            is_custom_query = True

            content_based_dir_path = Path(content_based_dir_path)

            if not content_based_dir_path.exists():
                raise HTTPException(status_code=400, detail=f"Directory not found: {content_based_dir_path}")
            
            try:
                item_mapping = pd.read_csv(content_based_dir_path / "1_item_mapping.csv")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading item_mapping.csv: {str(e)}")
            
            try:
                processed_file = content_based_dir_path / "2_full_processed_dataset.csv"
                features_file = content_based_dir_path / "3_engineered_features.feather"
                model_file = content_based_dir_path / "4_content_based_model.index"
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading pipeline files: {str(e)}")

            feature_matrix = pd.read_feather(features_file)
            processed_df = pd.read_csv(processed_file)

            if feature_matrix.empty or processed_df.empty:
                raise HTTPException(status_code=400, detail="Feature Matrix is empty.")

            if metadata:
                logging.info(f'New media detected.')
                metadata_df = pd.DataFrame([{
                    'item_id': item_id,
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
                
                data_preparer = NewDataPreparation(metadata_df, is_custom_query)
                new_prepared_data = data_preparer.prepare_new_data()

                logging.info(f'Data preparation complete.')

                preprocessor = DataPreprocessing(df=new_prepared_data)
                new_processed_data = preprocessor.preprocess_new_data(new_prepared_data)

                logging.info(f'Data preprocessing complete.')

                # Ensure new_features has the same column order as existing_feature_matrix
                new_processed_data = new_processed_data[processed_df.columns]

                # Load and apply feature engineering
                feature_engineering = FeatureEngineering()
                feature_engineering.load_transformers(content_based_dir_path)
                new_features = feature_engineering.transform_features(new_processed_data)
           
                recommender = Recommender(
                    item_id=item_id,
                    metadata=metadata.dict(),
                    features_file=features_file,
                    model_file=str(model_file),
                    n_items=n_items * 3,
                    is_custom_query=is_custom_query
                )

                recommendations = recommender.get_recommendation_for_new(new_features)

            filtered_recommendations = []
            logger.info(f"Filtering recommendations based on media type and spoken languages (if provided).")
            logger.info(f"Requested media_type: {media_type}, spoken_languages: {spoken_languages}")

            for rec in recommendations:
                rec_row = processed_df.loc[processed_df['item_id'] == rec['item_id']]
                
                if rec_row.empty:
                    logger.warning(f"Skipping item_id {rec['item_id']} - Not found in processed dataset.")
                    continue

                rec_media_type = rec_row['media_type'].values[0] if not rec_row.empty else None
                rec_languages = rec_row['spoken_languages'].values[0] if not rec_row.empty else ""

                # Convert rec_languages to a list (assuming it's stored as "en, sv, de")
                rec_languages = [lang.strip() for lang in rec_languages.split(",")] if isinstance(rec_languages, str) else []

                logger.info(f"Checking recommendation: item_id={rec['item_id']}, media_type={rec_media_type}, spoken_languages={rec_languages}")

                # Media type must match
                if rec_media_type != media_type:
                    logger.info(f"Rejected item_id={rec['item_id']} - Media type mismatch (expected {media_type}, got {rec_media_type})")
                    continue

                # If spoken_languages is provided, check if at least one matches
                if spoken_languages:
                    language_match = any(lang in rec_languages for lang in spoken_languages)
                    if not language_match:
                        logger.info(f"Rejected item_id={rec['item_id']} - No matching spoken language found.")
                        continue

                # If it passes the filters, add it
                filtered_recommendations.append(rec)
                logger.info(f"Accepted item_id={rec['item_id']} - Passed filtering.")

                if len(filtered_recommendations) >= n_items:
                    logger.info(f"Reached required {n_items} recommendations. Stopping filtering.")
                    break

            logger.info(f"Total recommendations after filtering: {len(filtered_recommendations)}")

            # Construct RecommendationResponse
            recommendation_models = []
            for rec in filtered_recommendations:
                matching_tmdb_id = item_mapping.loc[item_mapping['item_id'] == rec['item_id'], 'tmdb_id']
                
                if matching_tmdb_id.empty:
                    logger.warning(f"Skipping item_id={rec['item_id']} - No matching TMDB ID found.")
                    continue  # Skip if no matching tmdb_id is found

                recommendation_models.append(
                    Recommendation(
                        tmdb_id=matching_tmdb_id.values[0],
                        similarity=rec["similarity"]
                    )
                )

            logger.info(f"Successfully retrieved {len(recommendation_models)} recommendations")
            
            return RecommendationResponse(
                status=f"Successfully retrieved {len(recommendation_models)} recommendations",
                queriedMedia=str(tmdb_id),
                similarMedia=recommendation_models,
            )

        except Exception as e:
            logger.error(f"Error during recommendation retrieval: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in recommendation service: {str(e)}")

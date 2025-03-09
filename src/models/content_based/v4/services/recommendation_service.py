import pandas as pd
import logging
import joblib
from pathlib import Path
from fastapi import HTTPException
from src.models.content_based.v4.pipeline.Recommender import Recommender
from src.schemas.content_based_schema import Recommendation, RecommendationResponse, RecommendationRequest
from src.models.content_based.v4.pipeline.DataPreparation import DataPreparation
from src.models.content_based.v4.pipeline.NewDataPreparation import NewDataPreparation
from src.models.content_based.v4.pipeline.DataPreprocessing import DataPreprocessing
from src.models.content_based.v4.pipeline.FeatureEngineering import FeatureEngineering

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RecommendationService:
    @staticmethod
    def recommendation_service(
        recommendation_request: RecommendationRequest, 
        content_based_dir_path: str,
        n_items: int
    ) -> RecommendationResponse:
        try:
            logger.info(f"Received recommendation request")
            
            content_based_dir_path = Path(content_based_dir_path)
            if not content_based_dir_path.exists():
                raise HTTPException(status_code=400, detail=f"Directory not found: {content_based_dir_path}")
            
            # Check required files exist
            processed_file = content_based_dir_path / "2_full_processed_dataset.csv"
            features_file = content_based_dir_path / "3_engineered_features.feather"
            model_file = content_based_dir_path / "4_content_based_model.index"
            
            if not processed_file.exists() or not features_file.exists() or not model_file.exists():
                raise HTTPException(status_code=400, detail="Pipeline not executed. Please run the pipeline first.")
            
            # item_mapping = pd.read_csv(content_based_dir_path / "1_item_mapping.csv")
            tmdb_id = recommendation_request.tmdb_id
            metadata = recommendation_request.metadata
            
            try:
                item_mapping = pd.read_csv(content_based_dir_path / "1_item_mapping.csv")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading item_mapping.csv: {str(e)}")

            try:
                feature_matrix = pd.read_feather(features_file)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading features file: {str(e)}")

            try:
                processed_df = pd.read_csv(processed_file)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading processed dataset: {str(e)}")

            
            # Check if tmdb_id exists in item_mapping
            matching_items = item_mapping[item_mapping['tmdb_id'] == tmdb_id]
            is_existing = not matching_items.empty
            item_id = int(matching_items['item_id'].values[0]) if not matching_items.empty else None

            
            logger.info(f"tmdb_id: {tmdb_id}, item_id: {item_id}, is_existing: {is_existing}")
            
            media_type = metadata.media_type if metadata else None
            spoken_languages = metadata.spoken_languages if isinstance(metadata.spoken_languages, list) else []
            is_custom_query = False

            if feature_matrix.empty or processed_df.empty:
                raise HTTPException(status_code=400, detail="Dataset is empty.")

            recommender = Recommender(
                item_id=item_id,
                metadata=metadata.dict() if metadata else None,
                features_file=features_file,
                model_file=str(model_file),
                n_items=n_items * 2,
                is_custom_query=is_custom_query
            )

            if is_existing:
                logging.info(f'Existing media detected. Sending to Recommender Pipeline V4')
                recommendations = recommender.get_recommendation_for_existing()
            else:
                logging.info(f'New media detected.')
                logging.info(f'DataFrame: {pd.DataFrame([metadata.dict()])}')
                
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

                logging.info(f'New media metadata: {metadata_df}')

                data_preparer = NewDataPreparation(df=metadata_df, is_custom_query=False, item_mapping=item_mapping)
                new_prepared_data, updated_mapping = data_preparer.prepare_new_data()
                updated_mapping.to_csv(content_based_dir_path / "1_item_mapping.csv", index=False)

                logging.info(f'Data preparation complete.')

                preprocessor = DataPreprocessing(df=new_prepared_data)
                new_processed_data = preprocessor.preprocess_new_data(new_prepared_data)

                logging.info(f'Data preprocessing complete.')

                # Ensure new_features has the same column order as existing_feature_matrix
                new_processed_data = new_processed_data[processed_df.columns]

                # Append new_features to the existing feature matrix
                updated_new_processed_data = pd.concat([processed_df, new_processed_data], ignore_index=True)

                logger.info(f'new data: {updated_new_processed_data}')

                # Save back to features_file
                updated_new_processed_data.to_csv(processed_file, index=False)

                logging.info(f'Newly processed data added to the existing processed dataset.')

                # Load and apply feature engineering
                feature_engineering = FeatureEngineering()
                feature_engineering.load_transformers(content_based_dir_path)
                new_features = feature_engineering.transform_features(new_processed_data)

                recommendations = recommender.get_recommendation_for_new(new_features)

            # Filter recommendations based on media type (mandatory) and spoken languages (optional)
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

                # Media type must match
                if rec_media_type != media_type:
                    continue

                # If spoken_languages is provided, check if at least one matches
                if spoken_languages:
                    language_match = any(lang in rec_languages for lang in spoken_languages)
                    if not language_match:
                        continue

                # If it passes the filters, add it
                filtered_recommendations.append(rec)

                if len(filtered_recommendations) >= n_items:
                    logger.info(f"Reached required {n_items} recommendations. Stopping filtering.")
                    break

            logger.info(f"Total recommendations after filtering: {len(filtered_recommendations)}")

            # Construct RecommendationResponse
            recommendation_models = []
            for rec in filtered_recommendations:
                matching_tmdb_id = item_mapping.loc[item_mapping['item_id'] == rec['item_id'], 'tmdb_id']
                
                if matching_tmdb_id.empty:
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
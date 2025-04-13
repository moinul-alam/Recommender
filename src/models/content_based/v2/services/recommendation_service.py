import faiss
import pandas as pd
import logging
import joblib
from pathlib import Path
from fastapi import HTTPException
from src.models.content_based.v2.pipeline.Recommender import Recommender
from src.schemas.content_based_schema import Recommendation, RecommendationResponse, RecommendationRequest
from src.models.content_based.v2.pipeline.data_preparation import DataPreparation
from src.models.content_based.v2.pipeline.NewDataPreparation import NewDataPreparation
from src.models.content_based.v2.pipeline.data_preprocessing import DataPreprocessing
from src.models.content_based.v2.services.feature_engineering_service import FeatureEngineeringService
from src.models.content_based.v2.pipeline.feature_engineering import FeatureEngineering
from src.models.common.DataLoader import load_data


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RecommendationService:
    @staticmethod
    def recommendation_service(
        content_based_dir_path: str,
        file_names: dict,
        recommendation_request: RecommendationRequest
    ) -> RecommendationResponse:
        try:
            logger.info(f"Received recommendation request")
            content_based_dir_path = Path(content_based_dir_path)
            
            if not content_based_dir_path.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"Directory not found: {content_based_dir_path}"
                )
            
            # Defining file names and paths
            item_map_name = file_names["item_map_name"]
            item_map_path = content_based_dir_path / item_map_name
            if not item_map_path.exists():
                raise HTTPException(status_code=400, detail=f"Item mapping file not found: {item_map_path}")
            
            preprocessed_dataset_name = file_names["prepared_dataset_name"]
            preprocessed_dataset_path = content_based_dir_path / preprocessed_dataset_name
            if not preprocessed_dataset_path.exists():
                raise HTTPException(status_code=400, detail=f"Preprocessed dataset file not found: {preprocessed_dataset_path}")
            
            feature_matrix_name = file_names["feature_matrix_name"]
            feature_matrix_path = content_based_dir_path / feature_matrix_name
            if not feature_matrix_path.exists():
                raise HTTPException(status_code=400, detail=f"Feature matrix file not found: {feature_matrix_path}")
            
            index_name = file_names["index_name"]
            index_path = content_based_dir_path / index_name
            if not index_path.exists():
                raise HTTPException(status_code=400, detail=f"Index file not found: {index_path}")
            
            tmdb_id = recommendation_request.tmdb_id
            metadata = recommendation_request.metadata if recommendation_request.metadata else None
            n_recommendations = recommendation_request.num_recommendations or 10
            
            item_map = load_data(item_map_path, extension='csv')
            if item_map is None or item_map.empty:
                raise HTTPException(status_code=400, detail="Item mapping is empty or invalid")
            
            feature_matrix = load_data(feature_matrix_path, extension='pkl')
            if feature_matrix is None or feature_matrix.empty:
                raise HTTPException(status_code=400, detail="Feature matrix is empty or invalid")
            
            index = faiss.read_index(str(index_path))
            if index is None:
                raise HTTPException(status_code=500, detail="Failed to load index")

            # Check if tmdb_id exists in item_map
            existing_items = item_map[item_map['tmdb_id'] == tmdb_id]
            is_existing = not existing_items.empty
            item_id = int(existing_items['item_id'].values[0]) if not existing_items.empty else None
            
            logger.info(f"tmdb_id: {tmdb_id}, item_id: {item_id}, is_existing: {is_existing}")
            
            media_type = metadata.media_type if metadata else None
            spoken_languages = metadata.spoken_languages if isinstance(metadata.spoken_languages, list) else []
            # is_custom_query = False            
            
            content_based_recommender = Recommender(
                item_id,
                item_map,
                feature_matrix,
                index,
                n_recommendations=n_recommendations * 2
            )

            if is_existing:
                logging.info(f'Existing media detected. Sending to Recommender Pipeline V2')
                recommendations = content_based_recommender.get_recommendation_for_existing()
            else:
                logging.info(f'New item query detected.')
                preprocessed_dataset = load_data(preprocessed_dataset_path, extension='csv')
                if preprocessed_dataset is None or preprocessed_dataset.empty:
                    raise HTTPException(status_code=400, detail="Preprocessed dataset is empty or invalid")
                        
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

                # logging.info(f'New item metadata: {metadata_df}')
                data_preparer = NewDataPreparation(df=metadata_df, is_custom_query=False, item_map=item_map)
                new_prepared_data, updated_mapping = data_preparer.prepare_new_data()
                updated_mapping.to_csv(content_based_dir_path / item_map, index=False)

                logging.info(f'Data preparation complete.')

                preprocessor = DataPreprocessing(df=new_prepared_data)
                new_processed_data = preprocessor.preprocess_new_data(new_prepared_data)

                logging.info(f'Data preprocessing complete.')

                # Ensure new_features has the same column order as existing_feature_matrix
                new_processed_data = new_processed_data[preprocessed_dataset.columns]

                # Append new_features to the existing feature matrix
                updated_new_processed_data = pd.concat([preprocessed_dataset, new_processed_data], ignore_index=True)

                logger.info(f'new data: {updated_new_processed_data}')

                # Save back to features_file
                updated_new_processed_data.to_csv(preprocessed_dataset, index=False)

                logging.info(f'Newly processed data added to the existing preprocessed dataset.')

                # Load and apply feature engineering
                feature_transformers = FeatureEngineeringService.load_transformers(content_based_dir_path, file_names)
                if feature_transformers is None:
                    raise HTTPException(status_code=500, detail="Failed to load feature transformers")
                
                model_components = FeatureEngineeringService.load_transformers(content_based_dir_path, file_names)
                feature_weights = FeatureEngineeringService.load_transformers(content_based_dir_path, file_names)
                
                feature_engineer = FeatureEngineering(model_components, feature_weights)
                
                # feature_engineering.load_transformers(content_based_dir_path)
                new_features = feature_engineer.transform_features(new_processed_data)

                recommendations = content_based_recommender.get_recommendation_for_new(new_features)

            # Filter recommendations based on media type (mandatory) and spoken languages (optional)
            filtered_recommendations = []
            logger.info(f"Filtering recommendations based on media type and spoken languages (if provided).")
            logger.info(f"Requested media_type: {media_type}, spoken_languages: {spoken_languages}")

            for rec in recommendations:
                rec_row = preprocessed_dataset.loc[preprocessed_dataset['item_id'] == rec['item_id']]
                
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

                if len(filtered_recommendations) >= n_recommendations:
                    logger.info(f"Reached required {n_recommendations} recommendations. Stopping filtering.")
                    break

            logger.info(f"Total recommendations after filtering: {len(filtered_recommendations)}")

            # Construct RecommendationResponse
            recommendation_models = []
            for rec in filtered_recommendations:
                matching_tmdb_id = item_map.loc[item_map['item_id'] == rec['item_id'], 'tmdb_id']
                
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
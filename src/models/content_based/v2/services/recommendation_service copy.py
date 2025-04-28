from typing import Dict, List, Optional, Tuple
import faiss
import pandas as pd
import numpy as np
import logging
import joblib
from collections import Counter
from pathlib import Path
from fastapi import HTTPException
from src.models.content_based.v2.pipeline.Recommender import Recommender
from src.schemas.content_based_schema import Recommendation, RecommendationResponse, RecommendationRequest, RecommendationItem
from src.models.content_based.v2.pipeline.data_preparation import DataPreparation
from src.models.content_based.v2.pipeline.NewDataPreparation import NewDataPreparation
from src.models.content_based.v2.pipeline.data_preprocessing import DataPreprocessing
from models.content_based.v2.services.feature_engineering_service2 import FeatureEngineeringService
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
            logger.info("Received recommendation request")
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

            # Check if we have a single item or multiple items
            items = recommendation_request.items
            
            if len(items) == 1:
                # Single item recommendation
                logger.info("Processing single item recommendation")
                return RecommendationService._process_single_item(
                    items[0], 
                    item_map, 
                    feature_matrix, 
                    index, 
                    n_recommendations, 
                    content_based_dir_path, 
                    file_names
                )
            else:
                # Multiple items recommendation
                logger.info(f"Processing multiple items recommendation for {len(items)} items")
                return RecommendationService._process_multiple_items(
                    items, 
                    item_map, 
                    feature_matrix, 
                    index, 
                    n_recommendations, 
                    content_based_dir_path, 
                    file_names
                )

        except Exception as e:
            logger.error(f"Error during recommendation retrieval: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in recommendation service: {str(e)}")
    
    @staticmethod
    def _process_single_item(
        item: RecommendationItem,
        item_map: pd.DataFrame,
        feature_matrix: pd.DataFrame,
        index: faiss.Index,
        n_recommendations: int,
        content_based_dir_path: Path,
        file_names: dict
    ) -> RecommendationResponse:
        """Process a single item recommendation request."""
        
        tmdb_id = item.tmdb_id
        metadata = item.metadata
        
        # Check if tmdb_id exists in item_map
        existing_items = item_map[item_map['tmdb_id'] == tmdb_id]
        is_item_existing = not existing_items.empty
        item_id = int(existing_items['item_id'].values[0]) if not existing_items.empty else None
        
        logger.info(f"tmdb_id: {tmdb_id}, item_id: {item_id}, is_item_existing: {is_item_existing}")
        
        media_type = metadata.media_type if metadata else None
        spoken_languages = metadata.spoken_languages if metadata and hasattr(metadata, 'spoken_languages') else []
        
        content_based_recommender = Recommender(
            item_id,
            item_map,
            feature_matrix,
            index,
            n_recommendations=n_recommendations * 2
        )

        if is_item_existing:
            logging.info('Existing media detected. Sending to Recommender Pipeline V2')
            recommended_items = content_based_recommender.get_recommendation_for_existing()
        else:
            logging.info('New item query detected.')
            new_item_features = RecommendationService._process_new_item(tmdb_id, content_based_dir_path, file_names, metadata)
            if new_item_features is None or new_item_features.empty:
                raise HTTPException(status_code=400, detail="New item features are empty or invalid")
            recommended_items = content_based_recommender.get_recommendation_for_new(new_item_features)
            
        filtered_recommendations = RecommendationService._filter_recommendations(
            tmdb_id, recommended_items, item_map, media_type, spoken_languages, n_recommendations
        )
                        
        # Construct RecommendationResponse
        recommendation_models = RecommendationService._create_recommendation_models(filtered_recommendations, item_map)

        logger.info(f"Successfully retrieved {len(recommendation_models)} recommendations for single item")
        
        return RecommendationResponse(
            status=f"Successfully retrieved {len(recommendation_models)} recommendations",
            queriedMedia=str(tmdb_id),
            similarMedia=recommendation_models,
        )
    
    @staticmethod
    def _process_multiple_items(
        items: List[RecommendationItem],
        item_map: pd.DataFrame,
        feature_matrix: pd.DataFrame,
        index: faiss.Index,
        n_recommendations: int,
        content_based_dir_path: Path,
        file_names: dict
    ) -> RecommendationResponse:
        """Process multiple items recommendation request with weighted approach."""
        
        logger.info(f"Processing multiple items: {len(items)}")
        
        # 1. Extract common media_type and spoken_languages for average/max pooling
        media_types = [item.metadata.media_type for item in items if item.metadata and item.metadata.media_type]
        spoken_languages_lists = [item.metadata.spoken_languages for item in items 
                             if item.metadata and hasattr(item.metadata, 'spoken_languages') and item.metadata.spoken_languages]
        
        # Get most common media type
        common_media_type = Counter(media_types).most_common(1)[0][0] if media_types else None
        logger.info(f"Most common media type for pooling: {common_media_type}")
        
        # Flatten and find most common languages
        all_languages = [lang for langs in spoken_languages_lists for lang in langs]
        common_languages = [lang for lang, count in Counter(all_languages).most_common()] if all_languages else []
        logger.info(f"Common languages for pooling: {common_languages}")
        
        # 2. Get individual recommendations
        individual_recs = []
        
        for item in items:
            tmdb_id = item.tmdb_id
            metadata = item.metadata
            
            # Check if item exists
            existing_items = item_map[item_map['tmdb_id'] == tmdb_id]
            is_item_existing = not existing_items.empty
            item_id = int(existing_items['item_id'].values[0]) if not existing_items.empty else None
            
            if is_item_existing:
                item_features = feature_matrix.iloc[item_id].values.reshape(1, -1)
                individual_recs.append({
                    'tmdb_id': tmdb_id,
                    'item_id': item_id,
                    'features': item_features,
                    'media_type': metadata.media_type if metadata else None,
                    'spoken_languages': metadata.spoken_languages if metadata and hasattr(metadata, 'spoken_languages') else []
                })
            else:
                # Process new item
                new_item_features = RecommendationService._process_new_item(tmdb_id, content_based_dir_path, file_names, metadata)
                if new_item_features is not None and not new_item_features.empty:
                    individual_recs.append({
                        'tmdb_id': tmdb_id,
                        'item_id': None,  # New item doesn't have an item_id yet
                        'features': new_item_features.values,
                        'media_type': metadata.media_type if metadata else None,
                        'spoken_languages': metadata.spoken_languages if metadata and hasattr(metadata, 'spoken_languages') else []
                    })
        
        # Skip further processing if no valid items were found
        if not individual_recs:
            raise HTTPException(status_code=400, detail="No valid items found for recommendations")
        
        # 3. Get recommendations using three different approaches
        
        # a. Individual recommendations for each item (50% weight)
        individual_recommendations = []
        
        for rec in individual_recs:
            content_based_recommender = Recommender(
                rec['item_id'],
                item_map,
                feature_matrix,
                index,
                n_recommendations=n_recommendations * 3  # Get more recommendations to ensure we have enough after filtering
            )
            
            if rec['item_id'] is not None:
                recommended_items = content_based_recommender.get_recommendation_for_existing()
            else:
                recommended_items = content_based_recommender.get_recommendation_for_new(pd.DataFrame(rec['features']))
            
            # Filter recommendations for this item
            filtered_items = RecommendationService._filter_recommendations(
                rec['tmdb_id'], recommended_items, item_map, 
                rec['media_type'], rec['spoken_languages'], n_recommendations * 2
            )
            
            individual_recommendations.extend(filtered_items)
        
        # Weight and count item frequencies
        individual_weighted = {}
        for item in individual_recommendations:
            if item['item_id'] not in individual_weighted:
                individual_weighted[item['item_id']] = item['similarity'] * 0.5  # 50% weight
            else:
                individual_weighted[item['item_id']] += item['similarity'] * 0.5
        
        # b. Average pooling of features (25% weight)
        if len(individual_recs) > 0:
            features_matrix = np.vstack([rec['features'] for rec in individual_recs])
            avg_features = np.mean(features_matrix, axis=0).reshape(1, -1)
            
            avg_recommendations = RecommendationService._get_recommendations_from_features(
                avg_features, index, n_recommendations * 3
            )
            
            # Filter average pooling recommendations
            avg_filtered = RecommendationService._filter_recommendations(
                None, avg_recommendations, item_map, common_media_type, common_languages, n_recommendations * 2
            )
            
            # Add to weighted dict
            for item in avg_filtered:
                if item['item_id'] not in individual_weighted:
                    individual_weighted[item['item_id']] = item['similarity'] * 0.25  # 25% weight
                else:
                    individual_weighted[item['item_id']] += item['similarity'] * 0.25
        
        # c. Max pooling of features (25% weight)
        if len(individual_recs) > 0:
            max_features = np.max(features_matrix, axis=0).reshape(1, -1)  # Get max value for each feature
            
            max_recommendations = RecommendationService._get_recommendations_from_features(
                max_features, index, n_recommendations * 3
            )
            
            # Filter max pooling recommendations
            max_filtered = RecommendationService._filter_recommendations(
                None, max_recommendations, item_map, common_media_type, common_languages, n_recommendations * 2
            )
            
            # Add to weighted dict
            for item in max_filtered:
                if item['item_id'] not in individual_weighted:
                    individual_weighted[item['item_id']] = item['similarity'] * 0.25  # 25% weight
                else:
                    individual_weighted[item['item_id']] += item['similarity'] * 0.25
        
        # 4. Combine and sort recommendations
        combined_recommendations = []
        
        for item_id, similarity in individual_weighted.items():
            combined_recommendations.append({
                'item_id': item_id,
                'similarity': similarity
            })
        
        # Sort by weighted similarity score
        combined_recommendations.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Take top n recommendations
        top_recommendations = combined_recommendations[:n_recommendations]
        
        # Create recommendation models
        recommendation_models = RecommendationService._create_recommendation_models(top_recommendations, item_map)
        
        # Create response
        tmdb_ids = [item.tmdb_id for item in items]
        
        return RecommendationResponse(
            status=f"Successfully retrieved {len(recommendation_models)} recommendations based on {len(items)} items",
            queriedMedia=','.join([str(id) for id in tmdb_ids]),
            similarMedia=recommendation_models,
        )
    
    @staticmethod
    def _get_recommendations_from_features(features: np.ndarray, index: faiss.Index, n_recommendations: int) -> List[Dict]:
        """Get recommendations directly from feature vector using Faiss."""
        
        # Search the index
        distances, indices = index.search(features, n_recommendations)
        
        # Create recommendation list
        recommendations = []
        for i in range(len(indices[0])):
            recommendations.append({
                'item_id': int(indices[0][i]),
                'similarity': float(1.0 - distances[0][i] / 2.0)  # Convert distance to similarity
            })
        
        return recommendations
    
    @staticmethod
    def _process_new_item(
        tmdb_id: int, 
        content_based_dir_path: Path, 
        file_names: dict, 
        metadata
    ) -> pd.DataFrame:
        """Process new item metadata and return features."""
        
        try:
            item_map_path = content_based_dir_path / file_names["item_map_name"]
            preprocessed_dataset_path = content_based_dir_path / file_names["prepared_dataset_name"]
            
            item_map = load_data(item_map_path, extension='csv')
            preprocessed_dataset = load_data(preprocessed_dataset_path, extension='csv')
            if preprocessed_dataset is None or preprocessed_dataset.empty:
                raise HTTPException(status_code=400, detail="Preprocessed dataset is empty or invalid")
                    
            metadata_df = pd.DataFrame([{
                'tmdb_id': tmdb_id,
                'media_type': metadata.media_type,
                'title': metadata.title if hasattr(metadata, 'title') else None,
                'overview': metadata.overview if hasattr(metadata, 'overview') else None,
                'spoken_languages': metadata.spoken_languages if hasattr(metadata, 'spoken_languages') else [],
                'vote_average': metadata.vote_average if hasattr(metadata, 'vote_average') else None,
                'release_year': metadata.release_year if hasattr(metadata, 'release_year') else None,
                'genres': metadata.genres if hasattr(metadata, 'genres') else [],
                'keywords': metadata.keywords if hasattr(metadata, 'keywords') else [],
                'cast': metadata.cast if hasattr(metadata, 'cast') else [],
                'director': metadata.director if hasattr(metadata, 'director') else []
            }])

            data_preparer = NewDataPreparation(df=metadata_df, is_custom_query=False, item_map=item_map)
            new_prepared_data, updated_mapping = data_preparer.prepare_new_data()
            updated_mapping.to_csv(item_map_path, index=False)

            logger.info('Data preparation complete.')

            preprocessor = DataPreprocessing(df=new_prepared_data)
            new_processed_data = preprocessor.preprocess_new_data(new_prepared_data)

            logger.info('Data preprocessing complete.')

            # Ensure new_features has the same column order as existing_feature_matrix
            new_processed_data = new_processed_data[preprocessed_dataset.columns]

            # Append new_features to the existing feature matrix
            updated_new_processed_data = pd.concat([preprocessed_dataset, new_processed_data], ignore_index=True)

            # Save back to features_file
            updated_new_processed_data.to_csv(preprocessed_dataset_path, index=False)

            logger.info('Newly processed data added to the existing preprocessed dataset.')

            # Load and apply feature engineering
            feature_transformers = FeatureEngineeringService.load_transformers(content_based_dir_path, file_names)
            if feature_transformers is None:
                raise HTTPException(status_code=500, detail="Failed to load feature transformers")
            
            model_components = FeatureEngineeringService.load_transformers(content_based_dir_path, file_names)
            feature_weights = FeatureEngineeringService.load_transformers(content_based_dir_path, file_names)
            
            feature_engineer = FeatureEngineering(model_components, feature_weights)
            new_features = feature_engineer.transform_features(new_processed_data)

            return new_features
    
        except Exception as e:
            logger.error(f"Error processing new item: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing new item: {str(e)}")
    
    @staticmethod
    def _filter_recommendations(
        tmdb_id: Optional[int], 
        recommended_items: List[Dict], 
        item_map: pd.DataFrame, 
        media_type: Optional[str], 
        spoken_languages: List[str], 
        n_recommendations: int
    ) -> List[Dict]:
        """Filter recommendations based on media type and spoken languages."""
        filtered_recommendations = []
        logger.info(f"Filtering recommendations based on media type and spoken languages (if provided).")
        logger.info(f"Requested media_type: {media_type}, spoken_languages: {spoken_languages}")

        # Skip the filtering if media_type is not provided
        if not media_type:
            return recommended_items[:n_recommendations]

        for rec in recommended_items:
            rec_row = item_map.loc[item_map['item_id'] == rec['item_id']]
            
            if rec_row.empty:
                logger.warning(f"Skipping item_id {rec['item_id']} - Not found in processed dataset.")
                continue

            rec_media_type = rec_row['media_type'].values[0] if not rec_row.empty else None
            rec_languages_str = rec_row['spoken_languages'].values[0] if not rec_row.empty else ""

            # Convert rec_languages to a list (assuming it's stored as "en, sv, de")
            rec_languages = [lang.strip() for lang in rec_languages_str.split(",")] if isinstance(rec_languages_str, str) else []

            # Skip the item if it's the same as the queried item
            if tmdb_id is not None and rec_row['tmdb_id'].values[0] == tmdb_id:
                continue

            # Media type must match
            if media_type and rec_media_type != media_type:
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
        return filtered_recommendations
    
    @staticmethod
    def _create_recommendation_models(recommendations: List[Dict], item_map: pd.DataFrame) -> List[Recommendation]:
        """Convert recommendation dictionaries to Recommendation models."""
        recommendation_models = []
        
        for rec in recommendations:
            matching_tmdb_id = item_map.loc[item_map['item_id'] == rec['item_id'], 'tmdb_id']
            
            if matching_tmdb_id.empty:
                continue  # Skip if no matching tmdb_id is found

            recommendation_models.append(
                Recommendation(
                    tmdb_id=matching_tmdb_id.values[0],
                    similarity=rec["similarity"]
                )
            )
            
        return recommendation_models
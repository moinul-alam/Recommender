from typing import Dict, List, Optional, Tuple
import faiss
import pandas as pd
import numpy as np
import logging
from collections import Counter
from pathlib import Path
from fastapi import HTTPException

from src.models.content_based.v2.pipeline.Recommender import Recommender
from src.schemas.content_based_schema import Recommendation, RecommendationResponse, RecommendationRequest, RecommendationRequestedItem
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
        """Main entry point for recommendation service."""
        try:
            logger.info("Received recommendation request")
            
            # Load required resources
            resources = RecommendationService._load_resources(content_based_dir_path, file_names)
            item_map = resources['item_map']
            feature_matrix = resources['feature_matrix']
            index = resources['index']
            
            # Get number of recommendations
            n_recommendations = recommendation_request.n_recommendations or 10
            
            # Process the recommendation request
            items = recommendation_request.items
            
            if len(items) == 1:
                # Single item recommendation
                logger.info("Processing single item recommendation")
                return RecommendationService._process_single_item(
                    items[0], 
                    resources, 
                    n_recommendations, 
                    content_based_dir_path, 
                    file_names
                )
            else:
                # Multiple items recommendation
                logger.info(f"Processing multiple items recommendation for {len(items)} items")
                return RecommendationService._process_multiple_items(
                    items, 
                    resources, 
                    n_recommendations, 
                    content_based_dir_path, 
                    file_names
                )

        except Exception as e:
            logger.error(f"Error during recommendation retrieval: {str(e)}", exc_info=True)
            return RecommendationResponse(
                status=f"Error in recommendation service: {str(e)}",
                recommendations=[]
            )
    
    @staticmethod
    def _load_resources(content_based_dir_path: str, file_names: dict) -> Dict:
        """Load all required resources for recommendations."""
        content_based_dir_path = Path(content_based_dir_path)
        
        if not content_based_dir_path.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Directory not found: {content_based_dir_path}"
            )
        
        # Defining file paths
        item_map_path = content_based_dir_path / file_names["item_map_name"]
        preprocessed_dataset_path = content_based_dir_path / file_names["preprocessed_dataset_name"]
        feature_matrix_path = content_based_dir_path / file_names["feature_matrix_name"]
        index_path = content_based_dir_path / file_names["index_name"]
        
        # Check if files exist
        for path, desc in [
            (item_map_path, "Item mapping"),
            (preprocessed_dataset_path, "Preprocessed dataset"),
            (feature_matrix_path, "Feature matrix"),
            (index_path, "Index")
        ]:
            if not path.exists():
                raise HTTPException(status_code=400, detail=f"{desc} file not found: {path}")
        
        # Load resources
        item_map = load_data(item_map_path)
        if item_map is None or item_map.empty:
            raise HTTPException(status_code=400, detail="Item mapping is empty or invalid")
        
        feature_matrix = load_data(feature_matrix_path)
        if feature_matrix is None or feature_matrix.empty:
            raise HTTPException(status_code=400, detail="Feature matrix is empty or invalid")
        
        index = faiss.read_index(str(index_path))
        if index is None:
            raise HTTPException(status_code=500, detail="Failed to load index")
            
        preprocessed_dataset = load_data(preprocessed_dataset_path)
        if preprocessed_dataset is None or preprocessed_dataset.empty:
            raise HTTPException(status_code=400, detail="Preprocessed dataset is empty or invalid")
        
        return {
            'item_map': item_map,
            'feature_matrix': feature_matrix,
            'index': index,
            'preprocessed_dataset': preprocessed_dataset,
            'content_based_dir_path': content_based_dir_path
        }
    
    @staticmethod
    def _process_single_item(
        item: RecommendationRequestedItem,
        resources: Dict,
        n_recommendations: int,
        content_based_dir_path: Path,
        file_names: dict
    ) -> RecommendationResponse:
        """Process a single item recommendation request."""
        try:
            tmdb_id = item.tmdb_id
            metadata = item.metadata
            
            # Check if tmdb_id exists in item_map
            item_map = resources['item_map']
            feature_matrix = resources['feature_matrix']
            index = resources['index']
            
            existing_items = item_map[item_map['tmdb_id'] == tmdb_id]
            is_item_existing = not existing_items.empty
            item_id = int(existing_items['item_id'].values[0]) if not existing_items.empty else None
            
            logger.info(f"tmdb_id: {tmdb_id}, item_id: {item_id}, is_item_existing: {is_item_existing}")
            
            # Extract metadata if available
            media_type = metadata.media_type if metadata else None
            spoken_languages = metadata.spoken_languages if metadata and hasattr(metadata, 'spoken_languages') else []
            
            recommended_items = []
            
            try:
                # Initialize recommender
                recommender = Recommender(
                    item_id,
                    item_map,
                    feature_matrix,
                    index,
                    n_recommendations=n_recommendations * 2  # Get extra for filtering
                )

                # Get recommendations
                if is_item_existing:
                    logger.info('Existing media detected. Using existing item flow.')
                    recommended_items = recommender.get_recommendation_for_existing()
                else:
                    logger.info('New item query detected. Processing new item.')
                    new_item_features = RecommendationService._process_new_item(
                        tmdb_id, 
                        resources, 
                        file_names, 
                        metadata
                    )
                    if new_item_features is None or new_item_features.empty:
                        return RecommendationResponse(
                            status="No valid features extracted for new item",
                            recommendations=[]
                        )
                        
                    recommended_items = recommender.get_recommendation_for_new(new_item_features)
            except AssertionError as ae:
                logger.error(f"Dimension mismatch error: {str(ae)}", exc_info=True)
                return RecommendationResponse(
                    status="Error: Dimension mismatch between query features and index",
                    recommendations=[]
                )
            except Exception as e:
                logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
                return RecommendationResponse(
                    status=f"Error finding recommendation: {str(e)}",
                    recommendations=[]
                )
                
            # First filter: Exclude the input item
            filtered_recommendations = [
                rec for rec in recommended_items 
                if item_map.loc[item_map['item_id'] == rec['item_id'], 'tmdb_id'].values[0] != tmdb_id
            ]
            
            # Second filter: Apply media type and language filters if needed
            if media_type or spoken_languages:
                filtered_recommendations = recommender.filter_recommendations(
                    filtered_recommendations,
                    media_type=media_type, 
                    spoken_languages=spoken_languages
                )
            
            # Handle case where no recommendations are found
            if not filtered_recommendations:
                logger.warning("No recommendations found after filtering")
                return RecommendationResponse(
                    status="No recommendations found matching criteria",
                    recommendations=[]
                )
            
            # Limit to requested number
            filtered_recommendations = filtered_recommendations[:n_recommendations]
                            
            # Create response models
            recommendation_models = RecommendationService._create_recommendation_models(
                filtered_recommendations, 
                item_map
            )

            logger.info(f"Successfully retrieved {len(recommendation_models)} recommendations for single item")
            
            return RecommendationResponse(
                status=f"Successfully retrieved {len(recommendation_models)} recommendations",
                recommendations=recommendation_models
            )
        except Exception as e:
            logger.error(f"Error in _process_single_item: {str(e)}", exc_info=True)
            return RecommendationResponse(
                status=f"Error processing recommendation request: {str(e)}",
                recommendations=[]
            )
    
    @staticmethod
    def _process_multiple_items(
        items: List[RecommendationRequestedItem],
        resources: Dict,
        n_recommendations: int,
        content_based_dir_path: Path,
        file_names: dict
    ) -> RecommendationResponse:
        """Process a multiple items recommendation request with weighted approach."""
        try:
            logger.info(f"Processing multiple items: {len(items)}")
            item_map = resources['item_map']
            feature_matrix = resources['feature_matrix']
            index = resources['index']
            
            # Get expected feature dimension from FAISS index
            expected_dim = index.d
            logger.info(f"FAISS index expects {expected_dim} features")

            # Extract common media_type and spoken_languages for filtering
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
            
            # Extract features for each valid item and collect all input tmdb_ids
            individual_features = []
            input_tmdb_ids = set()  # Using set to avoid duplicates
            
            for item in items:
                tmdb_id = item.tmdb_id
                metadata = item.metadata
                input_tmdb_ids.add(tmdb_id)  # Add to set of IDs to exclude
                
                # Check if item exists
                existing_items = item_map[item_map['tmdb_id'] == tmdb_id]
                is_item_existing = not existing_items.empty
                item_id = int(existing_items['item_id'].values[0]) if not existing_items.empty else None
                
                if is_item_existing:
                    # Use existing item features, excluding item_id column
                    if 'item_id' in feature_matrix.columns:
                        item_features = feature_matrix.loc[item_id, feature_matrix.columns != 'item_id'].values
                    else:
                        item_features = feature_matrix.iloc[item_id].values
                    # Validate feature dimensions
                    if item_features.shape[0] != expected_dim:
                        logger.warning(f"Skipping TMDB ID {tmdb_id}: Feature vector has {item_features.shape[0]} dimensions, expected {expected_dim}")
                        continue
                    individual_features.append(item_features)
                else:
                    # Process new item
                    new_item_features = RecommendationService._process_new_item(
                        tmdb_id, 
                        resources, 
                        file_names, 
                        metadata
                    )
                    if new_item_features is not None and not new_item_features.empty:
                        new_features = new_item_features.values.flatten()
                        # Validate feature dimensions
                        if new_features.shape[0] != expected_dim:
                            logger.warning(f"Skipping TMDB ID {tmdb_id}: New feature vector has {new_features.shape[0]} dimensions, expected {expected_dim}")
                            continue
                        individual_features.append(new_features)
            
            # Skip further processing if no valid items were found
            if not individual_features:
                logger.warning("No valid items found for recommendations")
                return RecommendationResponse(
                    status="No valid items found for recommendations",
                    recommendations=[]
                )
                
            try:
                # Initialize recommender for multi-item recommendations
                recommender = Recommender(
                    None,  # No specific item_id for multiple items
                    item_map,
                    feature_matrix,
                    index,
                    n_recommendations=n_recommendations * 3  # Get extra for combining and filtering
                )
                
                # Get recommendations using three different approaches
                recommendation_sets = []
                weights = []
                
                # 1. Individual recommendations (50% weight)
                individual_recs = []
                for features in individual_features:
                    features_reshaped = features.reshape(1, -1)
                    logger.info(f"Processing individual features with shape {features_reshaped.shape}")
                    recs = recommender.get_recommendations_from_features(features_reshaped)
                    individual_recs.extend(recs)
                
                recommendation_sets.append(individual_recs)
                weights.append(0.5)
                
                # 2. Average pooling (25% weight)
                if len(individual_features) > 0:
                    features_matrix = np.vstack(individual_features)
                    logger.info(f"Features matrix shape for pooling: {features_matrix.shape}")
                    avg_features = np.mean(features_matrix, axis=0).reshape(1, -1)
                    logger.info(f"Average features shape: {avg_features.shape}")
                    avg_recs = recommender.get_recommendations_from_features(avg_features)
                    recommendation_sets.append(avg_recs)
                    weights.append(0.25)
                
                # 3. Max pooling (25% weight)
                if len(individual_features) > 0:
                    max_features = np.max(features_matrix, axis=0).reshape(1, -1)
                    logger.info(f"Max features shape: {max_features.shape}")
                    max_recs = recommender.get_recommendations_from_features(max_features)
                    recommendation_sets.append(max_recs)
                    weights.append(0.25)
                
                # Combine recommendations with weights
                combined_recommendations = recommender.combine_recommendations(recommendation_sets, weights)
            except AssertionError as ae:
                logger.error(f"Dimension mismatch error: {str(ae)}", exc_info=True)
                return RecommendationResponse(
                    status="Error: Dimension mismatch between features and index",
                    recommendations=[]
                )
            except Exception as e:
                logger.error(f"Error in recommendation processing: {str(e)}", exc_info=True)
                return RecommendationResponse(
                    status=f"Error processing recommendations: {str(e)}",
                    recommendations=[]
                )
            
            # Filter combined recommendations, excluding all input tmdb_ids
            filtered_recommendations = []
            for rec in combined_recommendations:
                # Get the tmdb_id for this recommendation
                matching_item = item_map.loc[item_map['item_id'] == rec['item_id']]
                if not matching_item.empty:
                    rec_tmdb_id = matching_item['tmdb_id'].values[0]
                    # Only include if not in our input set
                    if rec_tmdb_id not in input_tmdb_ids:
                        filtered_recommendations.append(rec)
            
            # Apply additional filters (media type, languages)
            if common_media_type or common_languages:
                filtered_recommendations = recommender.filter_recommendations(
                    filtered_recommendations,
                    excluded_tmdb_ids=None,  # Already filtered above
                    media_type=common_media_type,
                    spoken_languages=common_languages
                )
            
            # Handle case where no recommendations are found
            if not filtered_recommendations:
                logger.warning("No recommendations found after filtering")
                return RecommendationResponse(
                    status="No recommendations found matching criteria",
                    recommendations=[]
                )
            
            # Limit to requested number
            filtered_recommendations = filtered_recommendations[:n_recommendations]
            
            # Create recommendation models
            recommendation_models = RecommendationService._create_recommendation_models(
                filtered_recommendations, 
                item_map
            )
            
            return RecommendationResponse(
                status=f"Successfully retrieved {len(recommendation_models)} recommendations based on {len(items)} items",
                recommendations=recommendation_models
            )
        except Exception as e:
            logger.error(f"Error in _process_multiple_items: {str(e)}", exc_info=True)
            return RecommendationResponse(
                status=f"Error processing multiple items: {str(e)}",
                recommendations=[]
            )
    
    @staticmethod
    def _process_new_item(
        tmdb_id: int, 
        resources: Dict, 
        file_names: dict, 
        metadata
    ) -> pd.DataFrame:
        """Process new item metadata and extract features."""
        
        try:
            item_map = resources['item_map']
            preprocessed_dataset = resources['preprocessed_dataset']
            content_based_dir_path = resources['content_based_dir_path']
            
            # Create metadata dataframe
            metadata_df = pd.DataFrame([{
                'tmdb_id': tmdb_id,
                'media_type': metadata.media_type if hasattr(metadata, 'media_type') else None,
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

            # Prepare data
            data_preparer = NewDataPreparation(df=metadata_df, is_custom_query=False, item_map=item_map)
            new_prepared_data, _ = data_preparer.prepare_new_data()

            logger.info('Data preparation complete.')

            # Preprocess data
            preprocessor = DataPreprocessing(df=new_prepared_data)
            new_processed_data = preprocessor.preprocess_new_data(new_prepared_data)

            logger.info('Data preprocessing complete.')

            # Ensure new_features has the same column order as existing data
            new_processed_data = new_processed_data[preprocessed_dataset.columns]

            # Apply feature engineering
            feature_transformers = FeatureEngineeringService.load_transformers(
                content_based_dir_path, 
                file_names
            )
            if feature_transformers is None:
                logger.error("Failed to load feature transformers")
                return pd.DataFrame()
            
            feature_engineer = FeatureEngineering(feature_transformers, feature_transformers)
            new_features = feature_engineer.transform_features(new_processed_data)

            logger.info('Feature engineering complete.')
            return new_features
    
        except Exception as e:
            logger.error(f"Error processing new item: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    @staticmethod
    def _create_recommendation_models(recommendations: List[Dict], item_map: pd.DataFrame) -> List[Recommendation]:
        """Convert recommendation dictionaries to Recommendation schema models."""
        recommendation_models = []
        
        for rec in recommendations:
            item_id = rec['item_id']
            
            # Get the item details from item_map
            matching_item = item_map.loc[item_map['item_id'] == item_id]
            
            if matching_item.empty:
                logger.warning(f"Item ID {item_id} not found in item map, skipping")
                continue
                
            # Get the tmdb_id from the matching item
            tmdb_id = matching_item['tmdb_id'].values[0]
                
            # Use title column if it exists, otherwise use a default
            item_title = matching_item['title'].values[0] if 'title' in matching_item.columns else f"Item {item_id}"
                
            recommendation_models.append(
                Recommendation(
                    tmdb_id=tmdb_id,  # Use the tmdb_id from item_map
                    item_title=item_title,
                    similarity=rec["similarity"]
                )
            )
            
        return recommendation_models
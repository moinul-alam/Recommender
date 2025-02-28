import pathlib
from fastapi import HTTPException, Path
from typing import Dict, List
import pickle
import faiss
import numpy as np
import pandas as pd
from scipy import sparse
import logging
from src.models.collaborative.v2.pipeline.Recommender import BaseRecommender

logger = logging.getLogger(__name__)

class RecommendationService:

    @staticmethod
    def get_recommendations(
        items: Dict[str, float],
        processed_dir_path: str,
        model_dir_path: str,
        n_recommendations: int,
        min_similarity: float
    ) -> List[Dict]:
        try:
            # Step 1: Log received items
            logger.info(f"Received items: {items}")

            # Convert tmdb_id (string) to integer and ratings to float
            try:
                items = {int(key): float(value) for key, value in items.items()}
                logger.info(f"Converted items: {items}")
            except Exception as e:
                logger.error(f"Error in converting items: {str(e)}")
                raise HTTPException(status_code=400, detail="Invalid item format")
            
            processed_dir_path = pathlib.Path(processed_dir_path)
            model_dir_path = pathlib.Path(model_dir_path)

            # Step 2: Log directory paths
            logger.info(f"Processed directory path: {processed_dir_path}")
            logger.info(f"Model directory path: {model_dir_path}")

            # Load item mappings and model components
            item_mapping_path = processed_dir_path / "item_mapping.pkl"
            item_reverse_mapping_path = processed_dir_path / "item_reverse_mapping.pkl"
            user_item_matrix_path = processed_dir_path / "user_item_matrix.pkl"
            faiss_index_path = model_dir_path / "faiss_index.ivf"
            item_matrix_path = model_dir_path / "item_matrix.pkl"
            svd_model_path = model_dir_path / "svd_model.pkl"
            model_info_path = model_dir_path / "model_info.pkl"

            logger.info(f"Loading item mappings from {item_mapping_path} and {item_reverse_mapping_path}")

            # Load mappings
            try:
                with open(item_mapping_path, "rb") as f:
                    item_mapping = pickle.load(f)
                with open(item_reverse_mapping_path, "rb") as f:
                    item_reverse_mapping = pickle.load(f)
                logger.info("Item mappings loaded successfully")
            except Exception as e:
                logger.error(f"Error loading item mappings: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to load item mappings")

            # Step 3: Log model components loading
            logger.info(f"Loading model components from {item_matrix_path}, {svd_model_path}, {model_info_path}")

            # Load model components
            try:
                with open(item_matrix_path, "rb") as f:
                    item_matrix = pickle.load(f)
                with open(svd_model_path, "rb") as f:
                    svd_model = pickle.load(f)
                with open(model_info_path, "rb") as f:
                    model_info = pickle.load(f)
                logger.info("Model components loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model components: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to load model components")

            # Step 4: Log FAISS index loading
            logger.info(f"Loading FAISS index from {faiss_index_path}")

            # Load FAISS index
            try:
                faiss_index = faiss.read_index(str(faiss_index_path))
                logger.info("FAISS index loaded successfully")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to load FAISS index")

            # Step 5: Initialize recommender and log details
            logger.info("Initializing recommender")
            recommender = BaseRecommender(
                faiss_index=faiss_index,
                item_matrix=item_matrix,
                svd_model=svd_model,
                item_mapping=item_mapping,
                item_reverse_mapping=item_reverse_mapping,
                model_info=model_info,
                min_similarity=min_similarity
            )
            logger.info("Recommender initialized successfully")

            # Step 6: Generate recommendations and log results
            logger.info(f"Generating recommendations for items: {items} with n_recommendations={n_recommendations}")
            recommendations = recommender.generate_recommendations(
                items=items,
                n_recommendations=n_recommendations
            )

            logger.info(f"Generated {len(recommendations)} recommendations")

            if not recommendations:
                logger.warning("No recommendations generated")
                return None
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error in generating recommendations: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate recommendations")

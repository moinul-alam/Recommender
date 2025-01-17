# models/content_based/model.py
import faiss
import numpy as np
import pandas as pd
from typing import Optional
from schemas.responses import RecommendationResponse, SimilarMedia

class ContentBasedRecommender:
    def __init__(self, index_path: str, features_path: str):
        self.index_path = index_path
        self.features_path = features_path
        self.index: Optional[faiss.Index] = None
        self.features_df: Optional[pd.DataFrame] = None
        
    def load_data(self) -> None:
        try:
            self.index = faiss.read_index(self.index_path)
            self.features_df = pd.read_csv(self.features_path)
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")
    
    def get_media_features(self, tmdb_id: int) -> np.ndarray:
        media_features = self.features_df[self.features_df['tmdbId'] == tmdb_id]
        if media_features.empty:
            raise ValueError(f"Media with TMDB ID {tmdb_id} not found")
        
        return media_features.drop('tmdbId', axis=1).values.astype('float32')
    
    def find_similar_movies(self, tmdb_id: int, k: int = 10) -> RecommendationResponse:
        if self.index is None:
            raise RuntimeError("Please call load_data() first")
        
        try:
            query_vector = self.get_media_features(tmdb_id)
            distances, indices = self.index.search(query_vector, k + 1)
            
            similar_media = []
            for i, idx in enumerate(indices[0]):
                similar_tmdb_id = int(self.features_df.iloc[idx]['tmdbId'])
                if similar_tmdb_id == tmdb_id:
                    continue
                
                similarity_score = round(float(1 / (1 + distances[0][i])), 4)
                similar_media.append(SimilarMedia(
                    tmdbId=similar_tmdb_id,
                    similarity=similarity_score
                ))
                
                if len(similar_media) >= k:
                    break
            
            return RecommendationResponse(
                queriedMedia=tmdb_id,
                similarMedia=similar_media
            )
        except Exception as e:
            raise RuntimeError(f"Error finding similar media: {str(e)}")

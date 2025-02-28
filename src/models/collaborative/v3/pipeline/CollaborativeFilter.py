import numpy as np
import pandas as pd
import faiss
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from src.config.collaborative_config import ModelConfig

class CollaborativeFilter:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.faiss_index = None
        self.svd = None
        self.scaler = None
        self.user_id_map = {}
        self.reverse_user_id_map = {}
        self.movie_means = None
        self.rating_timestamps = None
        self.user_means = None
        self.user_std_dev = None
        self.preprocessed_matrix = None
        self.movie_popularity = None

    def _create_user_mapping(self, user_ids: np.ndarray) -> None:
        unique_ids = sorted(set(user_ids))
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_ids)}
        self.reverse_user_id_map = {idx: uid for uid, idx in self.user_id_map.items()}

    def _calculate_movie_popularity(self, df: pd.DataFrame) -> None:
        movie_stats = df.groupby('tmdbId').agg({'rating': ['count', 'mean']})
        movie_stats.columns = ['count', 'mean']
        movie_stats['count_norm'] = (movie_stats['count'] - movie_stats['count'].min()) / (movie_stats['count'].max() - movie_stats['count'].min())
        movie_stats['mean_norm'] = (movie_stats['mean'] - movie_stats['mean'].min()) / (movie_stats['mean'].max() - movie_stats['mean'].min())
        self.movie_popularity = (movie_stats['count_norm'] * 0.7 + movie_stats['mean_norm'] * 0.3).to_dict()

    def _preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        df['userId'] = df['userId'].astype(str)
        df['tmdbId'] = df['tmdbId'].astype(int)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        self._calculate_movie_popularity(df)
        self.rating_timestamps = df.set_index(['userId', 'tmdbId'])['timestamp'].to_dict()
        interaction_matrix = df.pivot_table(index='userId', columns='tmdbId', values='rating')
        
        self.user_means = interaction_matrix.mean(axis=1)
        self.user_std_dev = interaction_matrix.std(axis=1).replace(0, 1)
        
        self.preprocessed_matrix = (interaction_matrix.sub(self.user_means, axis=0)
                                    .div(self.user_std_dev, axis=0)
                                    .fillna(0))
        
        self.movie_means = interaction_matrix.mean()
        return self.preprocessed_matrix.values

    def _prepare_faiss_index(self, matrix: np.ndarray) -> None:
        self.scaler = StandardScaler()
        scaled_matrix = self.scaler.fit_transform(matrix)
        
        self.svd = TruncatedSVD(n_components=self.config.n_components)
        reduced_matrix = self.svd.fit_transform(scaled_matrix).astype(np.float32)
        
        quantizer = faiss.IndexFlatL2(self.config.n_components)
        self.faiss_index = faiss.IndexIVFPQ(quantizer, self.config.n_components, self.config.nlist, self.config.m, 8)
        self.faiss_index.train(reduced_matrix)
        self.faiss_index.add(reduced_matrix)

    def train(self, df: pd.DataFrame) -> None:
        self._create_user_mapping(df['userId'].unique())
        matrix = self._preprocess_data(df)
        self._prepare_faiss_index(matrix)

    def _get_similar_users(self, user_vector: np.ndarray) -> List[Tuple[str, float]]:
        query_vector = self.svd.transform(self.scaler.transform([user_vector])).astype(np.float32)
        faiss.normalize_L2(query_vector)
        
        D, I = self.faiss_index.search(query_vector, self.config.k_neighbors + 1)
        return [(self.reverse_user_id_map[idx], sim) for idx, sim in zip(I[0], D[0]) if sim > self.config.similarity_threshold]
    
    def predict_rating(self, user_id: Optional[str], movie_id: int, guest_vector: Optional[np.ndarray] = None) -> float:
        if movie_id not in self.preprocessed_matrix.columns:
            return self.movie_means.get(movie_id, self.movie_means.mean())
        
        user_vector = guest_vector if guest_vector is not None else self.preprocessed_matrix.loc[user_id].values
        similar_users = self._get_similar_users(user_vector)
        
        weighted_sum, weight_sum = 0, 0
        for similar_user_id, similarity in similar_users:
            rating = self.preprocessed_matrix.at[similar_user_id, movie_id]
            if rating == 0:
                continue
            
            timestamp = self.rating_timestamps.get((similar_user_id, movie_id))
            time_weight = np.exp(-self.config.time_weight_factor * ((datetime.now() - timestamp).total_seconds() / (365.25 * 24 * 60 * 60))) if timestamp else 1.0
            weight = similarity * time_weight
            
            weighted_sum += weight * rating
            weight_sum += abs(weight)
        
        if weight_sum == 0:
            return self.movie_means.get(movie_id, self.movie_means.mean())
        
        user_mean = self.user_means.mean() if guest_vector is not None else self.user_means[user_id]
        user_std = self.user_std_dev.mean() if guest_vector is not None else self.user_std_dev[user_id]
        
        return np.clip((weighted_sum / weight_sum * user_std) + user_mean, 1, 5)
    
    def recommend_movies(self, user_id: Optional[str] = None, n_recommendations: int = 10, guest_ratings: Optional[Dict[int, float]] = None) -> List[Tuple[int, float]]:
        if guest_ratings:
            guest_vector = np.zeros(len(self.preprocessed_matrix.columns))
            for movie_id, rating in guest_ratings.items():
                if movie_id in self.preprocessed_matrix.columns:
                    idx = self.preprocessed_matrix.columns.get_loc(movie_id)
                    guest_vector[idx] = (rating - self.user_means.mean()) / self.user_std_dev.mean()
            user_vector = guest_vector
            rated_movies = set(guest_ratings.keys())
        elif user_id:
            user_vector = self.preprocessed_matrix.loc[user_id].values
            rated_movies = set(self.preprocessed_matrix.loc[user_id][self.preprocessed_matrix.loc[user_id] != 0].index)
        else:
            return sorted(self.movie_popularity.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        predictions = [(movie, 0.8 * self.predict_rating(user_id, movie, user_vector if guest_ratings else None) + 0.2 * self.movie_popularity.get(movie, 0))
                       for movie in self.preprocessed_matrix.columns if movie not in rated_movies]
        
        return sorted(predictions, key=lambda x: x[1], reverse=True)[:n_recommendations]
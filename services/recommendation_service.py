# services/recommendation_service.py
from functools import lru_cache
from models.content_based.model import ContentBasedRecommender
from app.config import settings

@lru_cache()
def get_recommender() -> ContentBasedRecommender:
    recommender = ContentBasedRecommender(
        settings.FAISS_INDEX_PATH,
        settings.FEATURES_PATH
    )
    recommender.load_data()
    return recommender
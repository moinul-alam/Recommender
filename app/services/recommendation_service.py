# services/recommendation_service.py
from functools import lru_cache
from app.models.content_based.recommender import ContentBasedRecommender
from app.config import settings

@lru_cache()
def get_recommender() -> ContentBasedRecommender:
    recommender = ContentBasedRecommender(
        settings.CONTENT_BASED_FAISS_INDEX_PATH,
        settings.CONTENT_BASED_FEATURES_PATH
    )
    recommender.load_data()
    return recommender
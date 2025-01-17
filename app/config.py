# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    FAISS_INDEX_PATH: str = "E:/CoRE/recommender/app/index/faiss_index.index"
    FEATURES_PATH: str = "E:/CoRE/recommender/app/features/featured_combined.csv"
    
    class Config:
        env_file = ".env"

settings = Settings()

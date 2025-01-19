# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    CONTENT_BASED_FAISS_INDEX_PATH: str
    CONTENT_BASED_FEATURES_PATH: str
    
    class Config:
        env_file = ".env"

settings = Settings()

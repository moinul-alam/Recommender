# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    FAISS_INDEX_PATH: str
    FEATURES_PATH: str
    
    class Config:
        env_file = ".env"

settings = Settings()

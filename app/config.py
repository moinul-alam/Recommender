from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BASE_DIR_PATH: Path

    @property
    def CONTENT_BASED_DATASET_PATH(self) -> str:
        return str(self.BASE_DIR_PATH / "datasets")

    @property
    def CONTENT_BASED_PREPROCESSED_PATH(self) -> str:
        return str(self.BASE_DIR_PATH / "datasets" / "preprocessed")

    @property
    def CONTENT_BASED_ENGINEERED_PATH(self) -> str:
        return str(self.BASE_DIR_PATH / "datasets" / "engineered")

    @property
    def CONTENT_BASED_INDEX_PATH(self) -> str:
        return str(self.BASE_DIR_PATH / "datasets" / "similarity")

    class Config:
        env_file = ".env"

settings = Settings()

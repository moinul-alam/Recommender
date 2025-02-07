from pathlib import Path
from pydantic_settings import BaseSettings

class BaseConfig(BaseSettings):
    BASE_DATA_PATH: Path

    class Config:
        env_file = ".env"

    @property
    def CONTENT_BASED_DATA_PATH_V1(self) -> Path:
        return self.BASE_DATA_PATH / "content_based" / "v1"

    @property
    def CONTENT_BASED_DATA_PATH_V2(self) -> Path:
        return self.BASE_DATA_PATH / "content_based" / "v2"
    
    @property
    def COLLABORATIVE_PATH_V1(self) -> Path:
        return self.BASE_DATA_PATH / "collaborative" / "v1"
    
    @property
    def COLLABORATIVE_PATH_V2(self) -> Path:
        return self.BASE_DATA_PATH / "collaborative" / "v2"
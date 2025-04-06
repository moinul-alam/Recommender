from pathlib import Path
from pydantic_settings import BaseSettings

class BaseConfig(BaseSettings):
    BASE_DATA_PATH: Path

    class Config:
        env_file = ".env"

    @property
    def CONTENT_BASED_PATH(self) -> Path:
        return self.BASE_DATA_PATH / "content_based"

    @property
    def COLLABORATIVE_PATH(self) -> Path:
        return self.BASE_DATA_PATH / "collaborative"
    
    @property
    def HYBRID_PATH(self) -> Path:
        return self.BASE_DATA_PATH / "hybrid"

class ContentBasedConfig(BaseConfig):
    def get_version_path(self, version: int) -> Path:
        return self.CONTENT_BASED_PATH / f"v{version}"

class CollaborativeConfig(BaseConfig):
    def get_version_path(self, version: int) -> Path:
        return self.COLLABORATIVE_PATH / f"v{version}"
    
class HybridConfig(BaseConfig):
    def get_version_path(self, version: int) -> Path:
        return self.HYBRID_PATH / f"v{version}"

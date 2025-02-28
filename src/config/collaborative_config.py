from pathlib import Path
from src.config.base_config import BaseConfig

class CollaborativeConfigV1(BaseConfig):
    @property
    def DATASET_DIR_PATH(self) -> Path:
        return self.COLLABORATIVE_PATH_V1 / "1. dataset"

class CollaborativeConfigV1(BaseConfig):
    @property
    def DATASET_DIR_PATH(self) -> Path:
        return self.COLLABORATIVE_PATH_V1 / "1. dataset"

    @property
    def PROCESSED_DIR_PATH(self) -> Path:
        return self.COLLABORATIVE_PATH_V1 / "2. processed"
    
    @property
    def MODEL_DIR_PATH(self) -> Path:
        return self.COLLABORATIVE_PATH_V1 / "3. models"
    
class CollaborativeConfigV2(BaseConfig):
    @property
    def DIR_PATH(self) -> Path:
        return self.COLLABORATIVE_PATH_V2
    
    @property
    def DATASET_DIR_PATH(self) -> Path:
        return self.COLLABORATIVE_PATH_V2 / "1. dataset"

    @property
    def PROCESSED_DIR_PATH(self) -> Path:
        return self.COLLABORATIVE_PATH_V2 / "2. processed"
    
    @property
    def MODEL_DIR_PATH(self) -> Path:
        return self.COLLABORATIVE_PATH_V2 / "3. models"


class CollaborativeConfigV3(BaseConfig):
    @property
    def DATASET_DIR_PATH(self) -> Path:
        return self.COLLABORATIVE_PATH_V3 / "1. dataset"

    @property
    def PROCESSED_DIR_PATH(self) -> Path:
        return self.COLLABORATIVE_PATH_V3 / "2. processed"
    
    @property
    def MODEL_DIR_PATH(self) -> Path:
        return self.COLLABORATIVE_PATH_V3 / "3. models"





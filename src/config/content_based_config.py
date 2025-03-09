from pathlib import Path
from src.config.base_config import BaseConfig

raw_dataset_name = "coredb.media.json"

"""
Version 4
"""
class ContentBasedConfigV4(BaseConfig):
    @property
    def DIR_PATH(self) -> Path:
        return self.CONTENT_BASED_DATA_PATH_V4

"""
Version 3
"""
class ContentBasedConfigV3(BaseConfig):
    @property
    def DIR_PATH(self) -> Path:
        return self.CONTENT_BASED_DATA_PATH_V3

"""
Version 2
"""
class ContentBasedConfigV2(BaseConfig):
    @property
    def DIR_PATH(self) -> Path:
        return self.CONTENT_BASED_DATA_PATH_V2

    """
Version 1
"""
class ContentBasedConfigV1(BaseConfig):
    @property
    def RAW_DATA_PATH(self) -> Path:
        return self.CONTENT_BASED_DATA_PATH_V1 / "raw" / raw_dataset_name

    @property
    def PROCESSED_FOLDER_PATH(self) -> Path:
        return self.CONTENT_BASED_DATA_PATH_V1 / "processed"

    @property
    def FEATURES_FOLDER_PATH(self) -> Path:
        return self.CONTENT_BASED_DATA_PATH_V1 / "engineered"

    @property
    def MODEL_FOLDER_PATH(self) -> Path:
        return self.CONTENT_BASED_DATA_PATH_V1 / "model"
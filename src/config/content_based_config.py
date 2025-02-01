from pathlib import Path
from src.config.base_config import BaseConfig

raw_dataset_name = "coredb.media.json"

"""
Version 2
"""
class ContentBasedConfigV2(BaseConfig):
    @property
    def RAW_DATA_PATH(self) -> Path:
        return self.CONTENT_BASED_DATA_PATH_V2 / "1_raw" / raw_dataset_name
    
    @property
    def PREPARED_FOLDER_PATH(self) -> Path:
        return self.CONTENT_BASED_DATA_PATH_V2 / "2_prepared"

    @property
    def PROCESSED_FOLDER_PATH(self) -> Path:
        return self.CONTENT_BASED_DATA_PATH_V2 / "3_processed"

    @property
    def FEATURES_FOLDER_PATH(self) -> Path:
        return self.CONTENT_BASED_DATA_PATH_V2 / "4_engineered"
    
    @property
    def TRANSFORMERS_FOLDER_PATH(self) -> Path:
        return self.CONTENT_BASED_DATA_PATH_V2 / "5_transformers"

    @property
    def MODEL_FOLDER_PATH(self) -> Path:
        return self.CONTENT_BASED_DATA_PATH_V2 / "6_model"
    

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
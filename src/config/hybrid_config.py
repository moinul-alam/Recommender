from pathlib import Path
from src.config.base_config import BaseConfig

class HybridConfig(BaseConfig):
    @property
    def DATASET_PATH(self) -> Path:
        return self.BASE_DATA_PATH / "hybrid" / "dataset.csv"


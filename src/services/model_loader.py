import pandas as pd
import pickle
import faiss
import threading
from pathlib import Path
from src.config.content_based_config import ContentBasedConfigV2
from src.config.collaborative_config import CollaborativeConfigV2

class ModelLoader:
    _instance = None  # Singleton instance
    _lock = threading.Lock()  # Ensures thread safety

    def __new__(cls):
        """Ensures only one instance is created (Singleton)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelLoader, cls).__new__(cls)
                    cls._instance._load_models()  # Load models at startup
        return cls._instance

    def _load_models(self):
        """Loads all required files into memory."""
        print("🔄 Loading recommender models into memory...")

        # Paths for content-based models
        content_based_dir = ContentBasedConfigV2().DIR_PATH
        self.item_mapping = self._safe_load_csv(content_based_dir / "1_item_mapping.csv")
        self.processed_df = self._safe_load_csv(content_based_dir / "2_full_processed_dataset.csv")
        self.feature_matrix = self._safe_load_feather(content_based_dir / "3_engineered_features.feather")
        self.content_model_file = content_based_dir / "4_content_based_model.index"

        # Paths for collaborative models
        collaborative_dir = CollaborativeConfigV2().DIR_PATH
        self.user_item_matrix = self._safe_load_pickle(collaborative_dir / "2_user_item_matrix.pkl")
        self.user_mapping = self._safe_load_pickle(collaborative_dir / "2_user_mapping.pkl")
        self.user_reverse_mapping = self._safe_load_pickle(collaborative_dir / "2_user_reverse_mapping.pkl")
        self.item_mapping_collab = self._safe_load_pickle(collaborative_dir / "2_item_mapping.pkl")
        self.item_reverse_mapping = self._safe_load_pickle(collaborative_dir / "2_item_reverse_mapping.pkl")
        self.user_matrix = self._safe_load_pickle(collaborative_dir / "3_user_matrix.pkl")
        self.item_matrix = self._safe_load_pickle(collaborative_dir / "3_item_matrix.pkl")
        self.svd_user_model = self._safe_load_pickle(collaborative_dir / "3_svd_user_model.pkl")
        self.svd_item_model = self._safe_load_pickle(collaborative_dir / "3_svd_item_model.pkl")
        self.model_info = self._safe_load_pickle(collaborative_dir / "3_model_info.pkl")

        # FAISS Indexes
        self.faiss_user_index = self._safe_load_faiss(collaborative_dir / "3_faiss_user_index.flat")
        self.faiss_item_index = self._safe_load_faiss(collaborative_dir / "3_faiss_item_index.flat")

        print("✅ All models loaded successfully!")

    def _safe_load_csv(self, path):
        try:
            df = pd.read_csv(path)
            if df.empty:  # Explicitly check if DataFrame is empty
                raise ValueError(f"{path} is empty")
            return df
        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")
            return None


    def _safe_load_feather(self, path):
        """Loads Feather file safely, returns None if fails."""
        try:
            return pd.read_feather(path)
        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")
            return None

    def _safe_load_pickle(self, path):
        """Loads Pickle file safely, returns None if fails."""
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")
            return None

    def _safe_load_faiss(self, path):
        """Loads FAISS index safely, returns None if fails."""
        try:
            return faiss.read_index(str(path))
        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")
            return None

    def reload_models(self):
        """Manually reloads all models from disk."""
        print("🔄 Reloading models...")
        self._load_models()
        print("✅ Reload complete!")

# Get a Singleton instance
model_loader = ModelLoader()

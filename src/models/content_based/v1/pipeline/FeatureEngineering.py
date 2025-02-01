import pandas as pd
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from pathlib import Path
import gc
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FeatureEngineering:
    def __init__(
        self,
        segment_folder_path: str,
        save_folder_path: str,
        weights=None,
        n_components_lsa=200,
        n_components_pca=200,
    ):
        self.segment_folder_path = Path(segment_folder_path)
        self.save_folder_path = Path(save_folder_path)
        self.weights = weights or {
            "overview": 0.35,
            "genres": 0.30,
            "keywords": 0.15,
            "cast": 0.12,
            "director": 0.08,
        }
        self.tfidf = TfidfVectorizer(
            max_features=5000, stop_words="english", ngram_range=(1, 2), min_df=2
        )
        self.encoders = {
            "genres": OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            "keywords": OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            "cast": OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            "director": OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        }
        self.lsa = TruncatedSVD(n_components=n_components_lsa, random_state=42)
        self.pca = PCA(n_components=n_components_pca, random_state=42)

    def preprocess_text(self, text):
        if pd.isna(text) or text == "":
            return ""
        text = str(text)
        return " ".join(text.lower().split())

    def process_list_field(self, field):
        if pd.isna(field) or field == "":
            return ""
        return " ".join(str(field).split("|")[:10])

    def process_overview(self, df):
        logger.info("Applying TF-IDF transformation on Overview...")
        overview_features = self.tfidf.fit_transform(df["overview"])
        logger.info("Applying LSA to reduce dimensionality of Overview features...")
        overview_lsa_features = self.lsa.fit_transform(overview_features)
        overview_df = pd.DataFrame(
            overview_lsa_features * self.weights["overview"],
            columns=[f"overview_lsa_{i}" for i in range(overview_lsa_features.shape[1])],
        )
        return overview_df

    def process_categorical_fields(self, df):
        encoded_features = {}
        for field in ["genres", "keywords", "cast", "director"]:
            encoded = self.encoders[field].fit_transform(df[field].values.reshape(-1, 1))
            encoded_features[field] = pd.DataFrame(
                encoded * self.weights[field],
                columns=[f"{field}_{i}" for i in range(encoded.shape[1])],
            )
        return encoded_features

    def normalize_features(self, df_combined):
        feature_cols = [col for col in df_combined.columns if col != "tmdbId"]
        df_combined[feature_cols] = normalize(df_combined[feature_cols], norm="l2")
        return df_combined

    def apply_pca(self, df_combined):
        pca_features = self.pca.fit_transform(df_combined.drop("tmdbId", axis=1))
        pca_df = pd.DataFrame(
            pca_features, columns=[f"pca_{i}" for i in range(pca_features.shape[1])]
        )
        df_combined = pd.concat([df_combined[["tmdbId"]], pca_df], axis=1)
        return df_combined

    def apply_feature_engineering(self, df):
        try:
            df = df.copy()
            text_columns = ["overview", "genres", "keywords", "cast", "director"]
            for col in text_columns:
                df[col] = df[col].fillna("")
                if col == "overview":
                    df[col] = df[col].apply(self.preprocess_text)
                else:
                    df[col] = df[col].apply(self.process_list_field)

            overview_df = self.process_overview(df)
            encoded_features = self.process_categorical_fields(df)
            feature_dfs = [df[["tmdbId"]], overview_df] + list(encoded_features.values())
            df_combined = pd.concat(feature_dfs, axis=1)
            df_combined = self.normalize_features(df_combined)
            df_combined = self.apply_pca(df_combined)

            gc.collect()
            return df_combined
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
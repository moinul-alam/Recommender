from typing import Dict, List
import faiss
import numpy as np
import pandas as pd
import logging
from fastapi import HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer

def get_recommendation_for_new(
    tmdbId: int,
    metadata: Dict,
    n_items: int,
    content_similarity_index_path: str,
    features_combined: np.ndarray,
    tmdbId_combined: np.ndarray,
    engineered_features_path: str,
) -> List[Dict]:
    """
    Get recommendations for a new tmdbId by preprocessing and updating the index and dataset.
    """
    try:
        # Preprocess metadata to create a feature vector for the new media item
        new_feature_vector = preprocess_metadata(metadata)

        # Ensure the new vector is C-contiguous
        new_feature_vector = np.ascontiguousarray(new_feature_vector.reshape(1, -1), dtype=np.float32)

        # Load the FAISS index
        index = faiss.read_index(content_similarity_index_path)

        # Add the new feature vector to the FAISS index
        index.add(new_feature_vector)

        # Append the new media to the .feather dataset
        new_row = pd.DataFrame([[tmdbId] + new_feature_vector.flatten().tolist()], columns=["tmdbId"] + [f"feature_{i}" for i in range(new_feature_vector.shape[1])])
        existing_data = pd.read_feather(engineered_features_path)
        updated_data = pd.concat([existing_data, new_row], ignore_index=True)
        updated_data.to_feather(engineered_features_path)

        # Save the updated FAISS index
        faiss.write_index(index, content_similarity_index_path)

        # Perform the similarity search on the newly added media item
        distances, indices = index.search(new_feature_vector, k=min(5, len(features_combined) + 1))

        # Map indices back to tmdbId and similarity scores
        recommendations = [
            {"tmdbId": int(tmdbId_combined[idx]), "similarity_score": float(1 / (1 + distances[0][i]))}
            for i, idx in enumerate(indices[0]) if idx != len(features_combined)
        ]

        return recommendations
    except Exception as e:
        raise RuntimeError(f"Error in getting recommendations for new media: {e}")


def preprocess_metadata(metadata: Dict) -> np.ndarray:
    """
    Convert metadata into a feature vector for a new media item.
    """
    # Example: Combine fields into a single text and apply TF-IDF or embeddings
    combined_text = f"{metadata.get('title', '')} {metadata.get('overview', '')} {' '.join(metadata.get('genres', []))}"
    vectorizer = TfidfVectorizer(max_features=200)
    feature_vector = vectorizer.fit_transform([combined_text]).toarray()
    return feature_vector

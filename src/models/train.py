from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]

_GENRES = [
    "pop", "rock", "hip-hop", "electronic", "jazz",
    "r-n-b", "country", "classical", "other", "unknown",
]

# Synthetic user profiles: each maps audio feature names to preference weights.
# Weights don't need to sum to 1; they're used as dot-product coefficients against
# normalized [0,1] feature values.
_USER_PROFILES: dict[str, dict[str, float]] = {
    "user_energetic":  {"energy": 0.9, "danceability": 0.8, "valence": 0.8, "tempo": 0.7},
    "user_chill":      {"energy": 0.2, "acousticness": 0.8, "valence": 0.5, "tempo": 0.3},
    "user_acoustic":   {"acousticness": 0.9, "instrumentalness": 0.3, "energy": 0.3},
    "user_hiphop":     {"danceability": 0.8, "speechiness": 0.3, "energy": 0.7},
    "user_classical":  {"instrumentalness": 0.9, "acousticness": 0.8, "energy": 0.2},
    "user_melancholic":{"valence": 0.2, "energy": 0.4, "acousticness": 0.6},
    "user_party":      {"danceability": 0.9, "energy": 0.9, "valence": 0.9, "loudness": 0.8},
    "user_indie":      {"acousticness": 0.5, "energy": 0.5, "valence": 0.6},
    "user_workout":    {"energy": 0.95, "tempo": 0.9, "danceability": 0.7},
    "user_study":      {"instrumentalness": 0.7, "energy": 0.3, "acousticness": 0.6},
}


def _build_feature_matrix(
    dim_tracks: pd.DataFrame,
    fact_af: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Merge track metadata + audio features into a numeric matrix.

    Returns (matrix, track_ids, feature_cols). Tracks without audio features
    are excluded from the matrix.
    """
    merged = fact_af.merge(
        dim_tracks[["track_id", "primary_genre", "composite_popularity"]],
        on="track_id",
        how="inner",
    )

    for genre in _GENRES:
        merged[f"genre_{genre}"] = (merged["primary_genre"] == genre).astype(float)

    feature_cols = (
        _AUDIO_FEATURES
        + [f"genre_{g}" for g in _GENRES]
        + ["composite_popularity"]
    )
    available = [c for c in feature_cols if c in merged.columns]

    track_ids = merged["track_id"].tolist()
    matrix = merged[available].fillna(0.0).to_numpy(dtype=float)
    return matrix, track_ids, available


def _build_content_model(
    feature_matrix: np.ndarray,
    track_ids: list[str],
) -> dict:
    """Fit a cosine-similarity NearestNeighbors model on scaled audio features."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(scaled)

    return {
        "nn": nn,
        "scaler": scaler,
        "feature_matrix_scaled": scaled,
        "track_ids": track_ids,
    }


def _build_user_item_matrix(
    track_ids: list[str],
    feature_matrix: np.ndarray,
    feature_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Generate a synthetic (n_users × n_tracks) interaction matrix.

    Scores are derived from each user profile's dot product against audio features,
    plus small Gaussian noise, then row-normalised to [0, 1].
    """
    col_index = {f: i for i, f in enumerate(feature_cols)}
    user_ids = list(_USER_PROFILES.keys())
    n_tracks = len(track_ids)
    scores = np.zeros((len(user_ids), n_tracks))
    rng = np.random.default_rng(seed=42)

    for u_idx, user_id in enumerate(user_ids):
        for feature, weight in _USER_PROFILES[user_id].items():
            if feature in col_index:
                scores[u_idx] += weight * feature_matrix[:, col_index[feature]]
        scores[u_idx] += rng.normal(0, 0.05, n_tracks)

    row_min = scores.min(axis=1, keepdims=True)
    row_max = scores.max(axis=1, keepdims=True)
    denom = np.where(row_max - row_min == 0, 1.0, row_max - row_min)
    scores = (scores - row_min) / denom

    return scores, user_ids


def _build_collab_model(
    user_item_matrix: np.ndarray,
    user_ids: list[str],
    track_ids: list[str],
    n_components: int = 20,
) -> dict:
    """Decompose the user-item matrix via TruncatedSVD for collaborative filtering."""
    n_components = min(n_components, min(user_item_matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(user_item_matrix)
    item_factors = svd.components_.T  # (n_tracks, n_components)

    return {
        "svd": svd,
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_index": {uid: i for i, uid in enumerate(user_ids)},
        "track_ids": track_ids,
    }


def run(gold_dir: Path | None = None, models_dir: Path | None = None) -> Path:
    """Train content + collaborative models from the gold layer and save artifacts."""
    cfg = load_config()
    gold_dir = gold_dir or Path(cfg.get("storage", {}).get("gold_dir", "data/gold"))
    models_dir = models_dir or Path(cfg.get("models", {}).get("models_dir", "models"))
    n_components = cfg.get("models", {}).get("n_components", 20)

    dim_tracks_path = gold_dir / "dim_tracks.parquet"
    fact_af_path = gold_dir / "fact_audio_features.parquet"

    if not dim_tracks_path.exists():
        raise FileNotFoundError(f"dim_tracks not found: {dim_tracks_path}")
    if not fact_af_path.exists():
        raise FileNotFoundError(f"fact_audio_features not found: {fact_af_path}")

    dim_tracks = pd.read_parquet(dim_tracks_path)
    fact_af = pd.read_parquet(fact_af_path)

    logger.info("Building feature matrix from %d tracks", len(dim_tracks))
    feature_matrix, track_ids, feature_cols = _build_feature_matrix(dim_tracks, fact_af)

    if not track_ids:
        raise ValueError("No tracks with audio features — cannot train models")

    logger.info("Training content model on %d tracks × %d features", *feature_matrix.shape)
    content_model = _build_content_model(feature_matrix, track_ids)

    logger.info("Generating synthetic user-item matrix")
    user_item_matrix, user_ids = _build_user_item_matrix(track_ids, feature_matrix, feature_cols)

    logger.info(
        "Training collaborative model (%d users × %d tracks)",
        len(user_ids), len(track_ids),
    )
    collab_model = _build_collab_model(user_item_matrix, user_ids, track_ids, n_components)

    artifacts = {
        "content": content_model,
        "collab": collab_model,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / "recommendation_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifacts, f)

    logger.info("Model artifacts saved to %s", out_path)
    return out_path


if __name__ == "__main__":
    path = run()
    print(f"Trained model saved to: {path}")

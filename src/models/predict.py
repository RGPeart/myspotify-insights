from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from src.utils.config import load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_artifacts(models_dir: Path | None = None) -> dict:
    """Load trained model artifacts from disk."""
    cfg = load_config()
    models_dir = models_dir or Path(cfg.get("models", {}).get("models_dir", "models"))
    path = models_dir / "recommendation_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path} — run `python -m src.models.train` first")
    with open(path, "rb") as f:
        # NOTE: pickle.load is not secure against corrupted or malicious data.
        # For a portfolio project, we assume the model artifacts are generated
        # by trusted processes. In production, consider safer serialization
        # formats (e.g., joblib with checksums, ONNX) or integrity checks.
        return pickle.load(f)


def _content_scores(
    liked_track_ids: list[str],
    content_model: dict,
    n: int,
) -> dict[str, float]:
    """Return {track_id: similarity_score} for tracks most similar to the liked set.

    Builds an average feature vector from liked tracks, queries NearestNeighbors,
    and excludes the liked tracks themselves from results.
    """
    track_ids: list[str] = content_model["track_ids"]
    track_index = {tid: i for i, tid in enumerate(track_ids)}
    feature_matrix: np.ndarray = content_model["feature_matrix_scaled"]
    nn = content_model["nn"]

    query_indices = [track_index[tid] for tid in liked_track_ids if tid in track_index]
    if not query_indices:
        return {}

    avg_vector = feature_matrix[query_indices].mean(axis=0).reshape(1, -1)
    k = min(n + len(liked_track_ids), len(track_ids))
    distances, indices = nn.kneighbors(avg_vector, n_neighbors=k)

    liked_set = set(liked_track_ids)
    return {
        track_ids[idx]: float(max(0.0, 1.0 - dist))
        for dist, idx in zip(distances[0], indices[0])
        if track_ids[idx] not in liked_set
    }


def _collab_scores(
    user_id: str,
    collab_model: dict,
    n: int,
) -> dict[str, float]:
    """Return {track_id: predicted_score} from the collaborative filtering model.

    Scores are the dot product of the user's latent vector with each item's latent
    vector, normalised to [0, 1]. Returns empty dict for unknown users.
    """
    user_index: dict[str, int] = collab_model["user_index"]
    if user_id not in user_index:
        return {}

    user_vec: np.ndarray = collab_model["user_factors"][user_index[user_id]]
    item_factors: np.ndarray = collab_model["item_factors"]
    raw_scores = item_factors @ user_vec

    min_s, max_s = raw_scores.min(), raw_scores.max()
    norm_scores = (raw_scores - min_s) / (max_s - min_s) if max_s > min_s else np.zeros_like(raw_scores)

    track_ids: list[str] = collab_model["track_ids"]
    top_indices = np.argsort(norm_scores)[::-1][:n]
    return {track_ids[i]: float(norm_scores[i]) for i in top_indices}


def get_recommendations(
    user_id: str,
    liked_track_ids: list[str] | None,
    artifacts: dict,
    n: int = 10,
    collab_weight: float = 0.7,
) -> list[dict]:
    """Hybrid content + collaborative recommendations.

    Strategy:
    - Both signals available → hybrid score = collab_weight × cf + (1-collab_weight) × content
    - Only collab available (user known, no liked tracks) → pure collaborative
    - Only content available (user unknown, liked tracks given) → pure content-based
    - Neither → empty list (cold start with no information)
    """
    content_weight = 1.0 - collab_weight
    liked = liked_track_ids or []

    c_scores = _content_scores(liked, artifacts["content"], n * 3) if liked else {}
    cf_scores = _collab_scores(user_id, artifacts["collab"], n * 3)

    if not c_scores and not cf_scores:
        return []

    all_ids = set(c_scores) | set(cf_scores)
    results: list[tuple[str, float, str]] = []

    for tid in all_ids:
        c = c_scores.get(tid, 0.0)
        cf = cf_scores.get(tid, 0.0)

        if c_scores and cf_scores:
            score = collab_weight * cf + content_weight * c
            reason = "hybrid"
        elif c_scores:
            score = c
            reason = "content_similarity"
        else:
            score = cf
            reason = "collaborative_filtering"

        results.append((tid, score, reason))

    results.sort(key=lambda x: x[1], reverse=True)
    return [
        {"track_id": tid, "score": round(score, 4), "reason": reason}
        for tid, score, reason in results[:n]
    ]


if __name__ == "__main__":
    arts = load_artifacts()
    recs = get_recommendations("user_energetic", None, arts, n=10)
    for r in recs:
        print(r)

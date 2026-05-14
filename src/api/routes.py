from __future__ import annotations

import math

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from src.models.predict import get_recommendations as _get_recs
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _serialize(data: dict) -> dict:
    """Convert pandas/numpy scalars and NA values to JSON-safe Python types."""
    cleaned: dict = {}
    for k, v in data.items():
        if isinstance(v, np.integer):
            cleaned[k] = int(v)
        elif isinstance(v, np.floating):
            cleaned[k] = None if np.isnan(v) else float(v)
        elif isinstance(v, np.bool_):
            cleaned[k] = bool(v)
        elif isinstance(v, pd.Timestamp):
            cleaned[k] = None if pd.isna(v) else v.isoformat()
        elif isinstance(v, float) and math.isnan(v):
            cleaned[k] = None
        else:
            try:
                cleaned[k] = None if pd.isna(v) else v  # type: ignore[arg-type]
            except (TypeError, ValueError):
                cleaned[k] = v
    return cleaned


# ------------------------------------------------------------------ #
# Endpoints                                                            #
# ------------------------------------------------------------------ #

@router.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@router.get("/recommendations/{user_id}")
async def recommendations(user_id: str, request: Request, n: int = 10):
    artifacts = getattr(request.app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run `python -m src.models.train` first")

    collab_weight: float = getattr(request.app.state, "collab_weight", 0.7)
    recs = _get_recs(
        user_id=user_id,
        liked_track_ids=None,
        artifacts=artifacts,
        n=n,
        collab_weight=collab_weight,
    )

    dim_tracks = getattr(request.app.state, "dim_tracks", None)
    if dim_tracks is not None and not dim_tracks.empty:
        track_map: dict = dim_tracks.set_index("track_id").to_dict("index")
        meta_keys = ("name", "primary_artist_name", "primary_genre", "composite_popularity")
        enriched = [
            {**r, **{k: track_map.get(r["track_id"], {}).get(k) for k in meta_keys}}
            for r in recs
        ]
        return {"user_id": user_id, "count": len(enriched), "recommendations": enriched}

    return {"user_id": user_id, "count": len(recs), "recommendations": recs}


@router.get("/tracks/{track_id}")
async def track_detail(track_id: str, request: Request):
    dim_tracks = getattr(request.app.state, "dim_tracks", None)
    if dim_tracks is None:
        raise HTTPException(status_code=503, detail="Track data not loaded")

    rows = dim_tracks[dim_tracks["track_id"] == track_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found")

    track = _serialize(rows.iloc[0].to_dict())

    fact_af = getattr(request.app.state, "fact_audio_features", None)
    if fact_af is not None:
        af_rows = fact_af[fact_af["track_id"] == track_id]
        if not af_rows.empty:
            skip = {"track_id", "primary_artist_id", "album_id"}
            track["audio_features"] = {
                k: v for k, v in _serialize(af_rows.iloc[0].to_dict()).items()
                if k not in skip
            }

    return track


@router.get("/artists/{artist_id}")
async def artist_detail(artist_id: str, request: Request):
    dim_artists = getattr(request.app.state, "dim_artists", None)
    if dim_artists is None:
        raise HTTPException(status_code=503, detail="Artist data not loaded")

    rows = dim_artists[dim_artists["artist_id"] == artist_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"Artist '{artist_id}' not found")

    return _serialize(rows.iloc[0].to_dict())

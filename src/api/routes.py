from __future__ import annotations

import math

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

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
async def health(request: Request):
    return {"status": "ok", "version": request.app.version}


@router.get("/recommendations/{user_id}")
async def recommendations(
    user_id: str,
    request: Request,
    n: int = Query(default=10, ge=1, le=100),
    liked_track_ids: str | None = Query(default=None, pattern="^([a-zA-Z0-9]{22}(,[a-zA-Z0-9]{22})*)?$", description="Comma-separated list of Spotify track IDs (max 5)"),
):
    artifacts = getattr(request.app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run `python -m src.models.train` first")

    collab_weight: float = getattr(request.app.state, "collab_weight", 0.7)

    # Parse liked_track_ids string into a list
    parsed_liked_track_ids = [tid.strip() for tid in liked_track_ids.split(',')] if liked_track_ids else []
    # Limit to 5 liked tracks for performance / relevance
    parsed_liked_track_ids = parsed_liked_track_ids[:5]

    recs = _get_recs(
        user_id=user_id,
        liked_track_ids=parsed_liked_track_ids,
        artifacts=artifacts,
        n=n,
        collab_weight=collab_weight,
    )

    tracks_idx = getattr(request.app.state, "tracks_idx", None)
    if tracks_idx is not None:
        meta_keys = ("name", "primary_artist_name", "primary_genre", "composite_popularity")
        enriched = []
        for r in recs:
            try:
                track_meta = tracks_idx.loc[r["track_id"]][list(meta_keys)].to_dict()
                enriched.append({**r, **_serialize(track_meta)})
            except KeyError:
                logger.warning("Track ID %s not found in tracks_idx for enrichment", r["track_id"])
                enriched.append(r) # Fallback to non-enriched if track not found
        return {"user_id": user_id, "count": len(enriched), "recommendations": enriched}

    return {"user_id": user_id, "count": len(recs), "recommendations": recs}


@router.get("/tracks/{track_id}")
async def track_detail(track_id: str, request: Request):
    tracks_idx = getattr(request.app.state, "tracks_idx", None)
    if tracks_idx is None:
        raise HTTPException(status_code=503, detail="Track data not loaded")

    try:
        track = _serialize(tracks_idx.loc[track_id].to_dict())
        track["track_id"] = track_id
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found")

    af_idx = getattr(request.app.state, "af_idx", None)
    if af_idx is not None:
        try:
            skip = {"track_id", "primary_artist_id", "album_id"}
            track["audio_features"] = {
                k: v for k, v in _serialize(af_idx.loc[track_id].to_dict()).items()
                if k not in skip
            }
        except KeyError:
            pass

    return track


@router.get("/artists/{artist_id}")
async def artist_detail(artist_id: str, request: Request):
    artists_idx = getattr(request.app.state, "artists_idx", None)
    if artists_idx is None:
        raise HTTPException(status_code=503, detail="Artist data not loaded")

    try:
        artist = _serialize(artists_idx.loc[artist_id].to_dict())
        artist["artist_id"] = artist_id
        return artist
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Artist '{artist_id}' not found")


@router.get("/users")
async def get_users(request: Request):
    artifacts = getattr(request.app.state, "artifacts", None)
    if artifacts is None or "collab" not in artifacts:
        raise HTTPException(status_code=503, detail="Collaborative model not loaded — run `python -m src.models.train` first")

    user_ids = list(artifacts["collab"]["user_index"].keys())
    return {"user_ids": user_ids}

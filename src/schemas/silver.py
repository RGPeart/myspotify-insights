"""Pydantic contracts for the silver layer.

These models are the canonical source of truth for the shape of each silver
Parquet dataset produced by src/etl/bronze_to_silver.py. JSON Schema files under
/schemas/silver/ are generated from these models — never edit those by hand.

`extra="forbid"` makes each model a strict contract: an unexpected column is a
breaking change that must be reflected here (and recorded in schemas/CHANGELOG.md).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

# Normalized audio features share the same [0, 1] bound after bronze→silver scaling.
_UNIT = {"ge": 0.0, "le": 1.0}


class SilverTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: str
    name: str
    popularity: int = Field(ge=0, le=100)
    duration_ms: Optional[int] = Field(default=None, ge=0)
    explicit: Optional[bool] = None
    primary_artist_id: Optional[str] = None
    primary_artist_name: Optional[str] = None
    album_id: Optional[str] = None
    album_name: Optional[str] = None
    release_date: Optional[datetime] = None


class SilverAudioFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: str
    # Required: bronze→silver drops rows where any of these are null.
    danceability: float = Field(**_UNIT)
    energy: float = Field(**_UNIT)
    tempo: float = Field(**_UNIT)
    # Optional: Spotify may omit these on some tracks; still normalized when present.
    key: Optional[float] = Field(default=None, **_UNIT)
    loudness: Optional[float] = Field(default=None, **_UNIT)
    time_signature: Optional[float] = Field(default=None, **_UNIT)
    speechiness: Optional[float] = Field(default=None, **_UNIT)
    acousticness: Optional[float] = Field(default=None, **_UNIT)
    instrumentalness: Optional[float] = Field(default=None, **_UNIT)
    liveness: Optional[float] = Field(default=None, **_UNIT)
    valence: Optional[float] = Field(default=None, **_UNIT)
    mode: Optional[int] = Field(default=None, ge=0, le=1)
    duration_ms: Optional[int] = Field(default=None, ge=0)


class SilverArtist(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artist_id: str
    name: str
    popularity: int = Field(ge=0, le=100)
    followers: Optional[int] = Field(default=None, ge=0)
    genres: str = ""
    primary_genre: str

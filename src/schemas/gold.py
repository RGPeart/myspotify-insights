"""Pydantic contracts for the gold layer.

These mirror the columns emitted by the dbt models in dbt/models/gold/. Required
fields and ranges track the dbt schema tests (not_null, accepted_range) so the
Python contract and the dbt tests agree on what "valid gold data" means.

NOTE: these models are NOT wired into runtime validation. Gold is produced and
validated by dbt (dbt/models/gold/schema.yml), so re-validating it in Python would
duplicate that boundary. They exist for JSON Schema generation, documentation, and
reuse by downstream consumers. JSON Schema files under /schemas/gold/ are generated
from these models.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

_UNIT = {"ge": 0.0, "le": 1.0}


class GoldDimTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: str
    name: str
    track_popularity: Optional[int] = Field(default=None, ge=0, le=100)
    duration_ms: Optional[int] = Field(default=None, ge=0)
    explicit: Optional[bool] = None
    primary_artist_id: Optional[str] = None
    primary_artist_name: Optional[str] = None
    album_id: Optional[str] = None
    album_name: Optional[str] = None
    release_date: Optional[datetime] = None
    primary_genre: str
    composite_popularity: float = Field(**_UNIT)


class GoldDimArtist(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artist_id: str
    artist_name: str
    popularity: Optional[int] = Field(default=None, ge=0, le=100)
    followers: Optional[int] = Field(default=None, ge=0)
    genres: str = ""
    primary_genre: str


class GoldFactAudioFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: str
    primary_artist_id: str
    album_id: str
    danceability: float = Field(**_UNIT)
    energy: float = Field(**_UNIT)
    valence: Optional[float] = Field(default=None, **_UNIT)
    tempo: float = Field(**_UNIT)
    key: Optional[float] = Field(default=None, **_UNIT)
    loudness: Optional[float] = Field(default=None, **_UNIT)
    time_signature: Optional[float] = Field(default=None, **_UNIT)
    speechiness: Optional[float] = Field(default=None, **_UNIT)
    acousticness: Optional[float] = Field(default=None, **_UNIT)
    instrumentalness: Optional[float] = Field(default=None, **_UNIT)
    liveness: Optional[float] = Field(default=None, **_UNIT)
    mode: Optional[int] = Field(default=None, ge=0, le=1)
    duration_ms: Optional[int] = Field(default=None, ge=0)

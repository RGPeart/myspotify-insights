"""Schema registry: maps each pipeline dataset to its Pydantic contract.

The registry is the single place that knows about every versioned dataset schema.
It also owns JSON Schema generation (`build_json_schema`) so the generation script
and the drift test produce byte-for-byte identical output from one code path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Type

from pydantic import BaseModel

from src.schemas.gold import GoldDimArtist, GoldDimTrack, GoldFactAudioFeatures
from src.schemas.silver import SilverArtist, SilverAudioFeatures, SilverTrack

# Repo root = .../myspotify-insights (this file is src/schemas/registry.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = REPO_ROOT / "schemas"

_JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


@dataclass(frozen=True)
class SchemaSpec:
    name: str          # dataset short name, e.g. "tracks", "dim_tracks"
    layer: str         # "silver" | "gold"
    version: str       # semantic version; bump on any schema change
    model: Type[BaseModel]

    @property
    def key(self) -> str:
        return f"{self.layer}/{self.name}"

    @property
    def json_schema_path(self) -> Path:
        return SCHEMAS_DIR / self.layer / f"{self.name}.json"


SCHEMA_SPECS: list[SchemaSpec] = [
    SchemaSpec("tracks", "silver", "1.0.0", SilverTrack),
    SchemaSpec("audio_features", "silver", "1.0.0", SilverAudioFeatures),
    SchemaSpec("artists", "silver", "1.0.0", SilverArtist),
    SchemaSpec("dim_tracks", "gold", "1.0.0", GoldDimTrack),
    SchemaSpec("dim_artists", "gold", "1.0.0", GoldDimArtist),
    SchemaSpec("fact_audio_features", "gold", "1.0.0", GoldFactAudioFeatures),
]

SCHEMA_REGISTRY: dict[str, SchemaSpec] = {spec.key: spec for spec in SCHEMA_SPECS}


def build_json_schema(spec: SchemaSpec) -> dict:
    """Render a spec's Pydantic model to a versioned JSON Schema document."""
    schema = spec.model.model_json_schema()
    return {
        "$schema": _JSON_SCHEMA_DIALECT,
        "version": spec.version,
        **schema,
    }

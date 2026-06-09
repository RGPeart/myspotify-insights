"""Data-layer schema contracts (Pydantic) + JSON Schema registry.

This is distinct from src/models/, which holds the ML recommendation code.
"""
from src.schemas.gold import GoldDimArtist, GoldDimTrack, GoldFactAudioFeatures
from src.schemas.registry import SCHEMA_REGISTRY, SCHEMA_SPECS, SchemaSpec, build_json_schema
from src.schemas.silver import SilverArtist, SilverAudioFeatures, SilverTrack
from src.schemas.validate import SchemaValidationError, validate_dataframe

__all__ = [
    "SilverTrack",
    "SilverAudioFeatures",
    "SilverArtist",
    "GoldDimTrack",
    "GoldDimArtist",
    "GoldFactAudioFeatures",
    "SchemaSpec",
    "SCHEMA_SPECS",
    "SCHEMA_REGISTRY",
    "build_json_schema",
    "SchemaValidationError",
    "validate_dataframe",
]

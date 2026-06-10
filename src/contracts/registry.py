"""Contract registry: the active data contracts for each pipeline stage boundary.

Contracts are defined for the silver layer — the boundary the Python pipeline
produces (`bronze_to_silver`) and dbt consumes. Gold is produced and validated by
dbt, so its interface is enforced there rather than by a Python contract.
"""
from __future__ import annotations

from src.contracts.base import DataContract
from src.schemas.silver import SilverArtist, SilverAudioFeatures, SilverTrack

_OWNER = "Ryan Peart"
_PRODUCER = "bronze_to_silver"

silver_tracks_contract = DataContract(
    name="silver_tracks",
    version="1.0.0",
    owner=_OWNER,
    producer=_PRODUCER,
    consumer="dbt stg_silver_tracks",
    schema_model=SilverTrack,
    max_staleness_hours=25,
    quality_rules=("no_nulls_on_required_fields", "no_duplicate_track_ids"),
)

silver_audio_features_contract = DataContract(
    name="silver_audio_features",
    version="1.0.0",
    owner=_OWNER,
    producer=_PRODUCER,
    consumer="dbt stg_silver_audio_features",
    schema_model=SilverAudioFeatures,
    max_staleness_hours=25,
    quality_rules=("no_nulls_on_required_fields", "no_duplicate_track_ids"),
)

silver_artists_contract = DataContract(
    name="silver_artists",
    version="1.0.0",
    owner=_OWNER,
    producer=_PRODUCER,
    consumer="dbt stg_silver_artists",
    schema_model=SilverArtist,
    max_staleness_hours=25,
    quality_rules=("no_nulls_on_required_fields", "no_duplicate_artist_ids"),
)

CONTRACT_REGISTRY: dict[str, DataContract] = {
    c.name: c
    for c in (silver_tracks_contract, silver_audio_features_contract, silver_artists_contract)
}

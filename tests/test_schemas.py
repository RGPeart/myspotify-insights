import json

import numpy as np
import pandas as pd
import pytest

from src.schemas.registry import SCHEMA_REGISTRY, SCHEMA_SPECS, build_json_schema
from src.schemas.silver import SilverArtist, SilverAudioFeatures, SilverTrack
from src.schemas.validate import SchemaValidationError, validate_dataframe


# ------------------------------------------------------------------ #
# Valid-row fixtures matching the real ETL output                      #
# ------------------------------------------------------------------ #

def _valid_tracks_df() -> pd.DataFrame:
    return pd.DataFrame({
        "track_id": ["t0", "t1"],
        "name": ["Track 0", "Track 1"],
        "popularity": [50, 80],
        "duration_ms": [200000, 180000],
        "explicit": [False, True],
        "primary_artist_id": ["a0", "a1"],
        "primary_artist_name": ["Artist 0", "Artist 1"],
        "album_id": ["al0", "al1"],
        "album_name": ["Album 0", "Album 1"],
        "release_date": pd.to_datetime(["2023-01-01", "2024-06-15"]),
    })


def _valid_audio_features_df() -> pd.DataFrame:
    return pd.DataFrame({
        "track_id": ["t0"],
        "danceability": [0.5], "energy": [0.6], "tempo": [0.4],
        "key": [0.2], "loudness": [0.7], "time_signature": [0.8],
        "speechiness": [0.1], "acousticness": [0.3], "instrumentalness": [0.0],
        "liveness": [0.2], "valence": [0.9], "mode": [1], "duration_ms": [200000],
    })


def _valid_artists_df() -> pd.DataFrame:
    return pd.DataFrame({
        "artist_id": ["a0"],
        "name": ["Artist 0"],
        "popularity": [70],
        "followers": [100000],
        "genres": ["pop,rock"],
        "primary_genre": ["pop"],
    })


# ------------------------------------------------------------------ #
# Registry                                                             #
# ------------------------------------------------------------------ #

class TestRegistry:
    def test_has_six_specs(self):
        assert len(SCHEMA_SPECS) == 6

    def test_keys_are_layer_qualified(self):
        assert set(SCHEMA_REGISTRY) == {
            "silver/tracks", "silver/audio_features", "silver/artists",
            "gold/dim_tracks", "gold/dim_artists", "gold/fact_audio_features",
        }

    def test_every_spec_targets_a_known_layer(self):
        assert all(spec.layer in {"silver", "gold"} for spec in SCHEMA_SPECS)


# ------------------------------------------------------------------ #
# JSON Schema drift check (the CI guardrail)                           #
# ------------------------------------------------------------------ #

class TestJsonSchemaSync:
    # The checked-in JSON Schema must equal what the current Pydantic model generates.
    # If this fails, run `python scripts/generate_schemas.py` and commit the result.
    @pytest.mark.parametrize("spec", SCHEMA_SPECS, ids=lambda s: s.key)
    def test_generated_schema_matches_committed_file(self, spec):
        assert spec.json_schema_path.exists(), (
            f"Missing JSON Schema for {spec.key}; run scripts/generate_schemas.py"
        )
        committed = json.loads(spec.json_schema_path.read_text(encoding="utf-8"))
        assert committed == build_json_schema(spec)


# ------------------------------------------------------------------ #
# Model accept / reject behavior                                       #
# ------------------------------------------------------------------ #

class TestModelConstraints:
    def test_accepts_valid_track(self):
        SilverTrack(track_id="t0", name="x", popularity=50)

    def test_rejects_out_of_range_popularity(self):
        with pytest.raises(ValueError):
            SilverTrack(track_id="t0", name="x", popularity=150)

    def test_rejects_missing_required_field(self):
        with pytest.raises(ValueError):
            SilverArtist(artist_id="a0", popularity=10, primary_genre="pop")  # no name

    def test_rejects_unit_feature_above_one(self):
        with pytest.raises(ValueError):
            SilverAudioFeatures(track_id="t0", danceability=1.5, energy=0.5, tempo=0.5)

    def test_forbids_unexpected_column(self):
        with pytest.raises(ValueError):
            SilverTrack(track_id="t0", name="x", popularity=50, surprise="boom")


# ------------------------------------------------------------------ #
# validate_dataframe                                                   #
# ------------------------------------------------------------------ #

class TestValidateDataframe:
    def test_valid_tracks_pass(self):
        validate_dataframe(_valid_tracks_df(), SilverTrack, "silver/tracks")

    def test_valid_audio_features_pass(self):
        validate_dataframe(_valid_audio_features_df(), SilverAudioFeatures, "silver/audio_features")

    def test_valid_artists_pass(self):
        validate_dataframe(_valid_artists_df(), SilverArtist, "silver/artists")

    # NaN in an Optional column must be treated as None, not as a float that fails typing.
    def test_nan_optional_is_treated_as_null(self):
        df = _valid_tracks_df()
        df.loc[0, "duration_ms"] = np.nan
        validate_dataframe(df, SilverTrack, "silver/tracks")

    def test_out_of_range_row_raises(self):
        df = _valid_tracks_df()
        df.loc[0, "popularity"] = 999
        with pytest.raises(SchemaValidationError):
            validate_dataframe(df, SilverTrack, "silver/tracks")

    def test_error_reports_invalid_row_count(self):
        df = _valid_audio_features_df()
        df.loc[0, "danceability"] = 5.0
        with pytest.raises(SchemaValidationError) as exc:
            validate_dataframe(df, SilverAudioFeatures, "silver/audio_features")
        assert exc.value.invalid_rows == 1
        assert exc.value.total_rows == 1

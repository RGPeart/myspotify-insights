import pandas as pd
import pytest

from src.contracts.base import ContractViolationError, DataContract
from src.contracts.enforce import enforce_contract
from src.contracts.registry import (
    CONTRACT_REGISTRY,
    silver_artists_contract,
    silver_tracks_contract,
)
from src.contracts.rules import (
    QUALITY_RULES,
    no_duplicate_track_ids,
    no_nulls_on_required_fields,
)
from src.schemas.silver import SilverTrack


def _valid_tracks_df(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame({
        "track_id": [f"t{i}" for i in range(n)],
        "name": [f"Track {i}" for i in range(n)],
        "popularity": [50 + i for i in range(n)],
        "duration_ms": [200000] * n,
        "explicit": [False] * n,
        "primary_artist_id": [f"a{i}" for i in range(n)],
        "primary_artist_name": [f"Artist {i}" for i in range(n)],
        "album_id": [f"al{i}" for i in range(n)],
        "album_name": [f"Album {i}" for i in range(n)],
        "release_date": pd.to_datetime(["2023-01-01"] * n),
    })


# ------------------------------------------------------------------ #
# Registry                                                             #
# ------------------------------------------------------------------ #

class TestRegistry:
    def test_has_three_silver_contracts(self):
        assert set(CONTRACT_REGISTRY) == {
            "silver_tracks", "silver_audio_features", "silver_artists"
        }

    # Guard: every contract must reference only rules that exist in the registry,
    # otherwise enforce_contract raises ValueError at runtime on real data.
    def test_all_contracts_use_known_rules(self):
        for contract in CONTRACT_REGISTRY.values():
            for rule_name in contract.quality_rules:
                assert rule_name in QUALITY_RULES

    def test_contracts_carry_required_metadata(self):
        for contract in CONTRACT_REGISTRY.values():
            assert contract.version
            assert contract.owner
            assert contract.producer
            assert contract.consumer
            assert contract.max_staleness_hours > 0


# ------------------------------------------------------------------ #
# Quality rules                                                        #
# ------------------------------------------------------------------ #

class TestQualityRules:
    def test_no_nulls_passes_on_clean_data(self):
        assert no_nulls_on_required_fields(_valid_tracks_df(), silver_tracks_contract) is None

    def test_no_nulls_flags_null_required_field(self):
        df = _valid_tracks_df()
        df.loc[0, "name"] = None
        assert no_nulls_on_required_fields(df, silver_tracks_contract) is not None

    def test_no_duplicate_track_ids_passes(self):
        assert no_duplicate_track_ids(_valid_tracks_df(), silver_tracks_contract) is None

    def test_no_duplicate_track_ids_flags_dupes(self):
        df = _valid_tracks_df()
        df.loc[1, "track_id"] = "t0"
        assert no_duplicate_track_ids(df, silver_tracks_contract) is not None


# ------------------------------------------------------------------ #
# enforce_contract                                                     #
# ------------------------------------------------------------------ #

class TestEnforceContract:
    def test_valid_data_passes(self):
        enforce_contract(_valid_tracks_df(), silver_tracks_contract)

    def test_schema_violation_raises(self):
        df = _valid_tracks_df()
        df.loc[0, "popularity"] = 999  # out of [0, 100]
        with pytest.raises(ContractViolationError):
            enforce_contract(df, silver_tracks_contract)

    def test_duplicate_key_raises(self):
        df = _valid_tracks_df()
        df.loc[1, "track_id"] = "t0"
        with pytest.raises(ContractViolationError) as exc:
            enforce_contract(df, silver_tracks_contract)
        assert any("no_duplicate_track_ids" in f for f in exc.value.failures)

    def test_unknown_rule_raises_value_error(self):
        bad = DataContract(
            name="bad", version="1.0.0", owner="x", producer="p", consumer="c",
            schema_model=SilverTrack, max_staleness_hours=1,
            quality_rules=("does_not_exist",),
        )
        with pytest.raises(ValueError):
            enforce_contract(_valid_tracks_df(), bad)

    def test_violation_error_carries_contract_and_failures(self):
        df = _valid_tracks_df()
        df.loc[1, "track_id"] = "t0"
        with pytest.raises(ContractViolationError) as exc:
            enforce_contract(df, silver_tracks_contract)
        assert exc.value.contract is silver_tracks_contract
        assert exc.value.failures

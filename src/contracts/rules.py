"""Named, reusable quality rules referenced by `DataContract.quality_rules`.

These complement the per-row Pydantic schema validation: a rule operates on the
whole DataFrame, so it can assert cross-row properties (e.g. key uniqueness) that
a per-row model cannot express. Each rule returns `None` when the data passes, or
a human-readable message describing the violation.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from src.contracts.base import DataContract

QualityRule = Callable[[pd.DataFrame, "DataContract"], Optional[str]]


def _required_fields(contract: "DataContract") -> list[str]:
    return [name for name, info in contract.schema_model.model_fields.items() if info.is_required()]


def no_nulls_on_required_fields(df: pd.DataFrame, contract: "DataContract") -> Optional[str]:
    offenders = {
        col: int(df[col].isnull().sum())
        for col in _required_fields(contract)
        if col in df.columns and df[col].isnull().any()
    }
    return f"nulls in required fields {offenders}" if offenders else None


def _no_duplicate(df: pd.DataFrame, column: str) -> Optional[str]:
    if column not in df.columns:
        return f"missing key column '{column}'"
    count = int(df[column].duplicated().sum())
    return f"{count} duplicate {column}(s)" if count else None


def no_duplicate_track_ids(df: pd.DataFrame, contract: "DataContract") -> Optional[str]:
    return _no_duplicate(df, "track_id")


def no_duplicate_artist_ids(df: pd.DataFrame, contract: "DataContract") -> Optional[str]:
    return _no_duplicate(df, "artist_id")


QUALITY_RULES: dict[str, QualityRule] = {
    "no_nulls_on_required_fields": no_nulls_on_required_fields,
    "no_duplicate_track_ids": no_duplicate_track_ids,
    "no_duplicate_artist_ids": no_duplicate_artist_ids,
}

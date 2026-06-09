"""Runtime DataFrame validation against the Pydantic contracts.

Pydantic models are the canonical source of truth (see src/schemas/registry.py),
so we validate row-by-row against the model rather than against the generated JSON
Schema. A single contract violation halts the pipeline with a structured log event.
"""
from __future__ import annotations

from typing import Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Cap how many per-row errors we attach to logs/exceptions to keep them readable.
_MAX_REPORTED_ERRORS = 10


class SchemaValidationError(Exception):
    def __init__(self, dataset: str, model_name: str, invalid_rows: int, total_rows: int, detail: str) -> None:
        self.dataset = dataset
        self.model_name = model_name
        self.invalid_rows = invalid_rows
        self.total_rows = total_rows
        super().__init__(
            f"{dataset}: {invalid_rows}/{total_rows} row(s) violate {model_name} contract. {detail}"
        )


def _is_null(value) -> bool:
    """Scalar-safe null check (pd.isna raises on array-likes)."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _clean_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to plain dicts, normalizing pandas/numpy quirks.

    NaN/NaT become None (so Optional fields validate), and numpy scalars become
    native Python types (so Pydantic's int/float/bool validators accept them).
    """
    records = []
    for row in df.to_dict("records"):
        clean = {}
        for key, value in row.items():
            if _is_null(value):
                clean[key] = None
            elif isinstance(value, np.generic):
                clean[key] = value.item()
            else:
                clean[key] = value
        records.append(clean)
    return records


def validate_dataframe(df: pd.DataFrame, model: Type[BaseModel], dataset: str) -> None:
    """Validate every row of `df` against `model`; raise SchemaValidationError on any violation."""
    invalid_count = 0
    errors: list[str] = []
    for idx, record in enumerate(_clean_records(df)):
        try:
            model.model_validate(record)
        except ValidationError as exc:
            invalid_count += 1
            if len(errors) < _MAX_REPORTED_ERRORS:
                messages = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
                errors.append(f"row {idx}: {messages}")

    if invalid_count:
        detail = " | ".join(errors)
        logger.error(
            "schema_validation_failed",
            dataset=dataset,
            model=model.__name__,
            invalid_rows=invalid_count,
            total_rows=len(df),
            sample_errors=errors,
        )
        raise SchemaValidationError(dataset, model.__name__, invalid_count, len(df), detail)

    logger.info("schema_validation_passed", dataset=dataset, model=model.__name__, rows=len(df))

"""Runtime DataFrame validation against the Pydantic contracts.

Pydantic models are the canonical source of truth (see src/schemas/registry.py),
so we validate row-by-row against the model rather than against the generated JSON
Schema. A single contract violation halts the pipeline with a structured log event.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Cap how many per-row errors we attach to logs/exceptions to keep them readable.
_MAX_REPORTED_ERRORS = 10


class SchemaValidationError(Exception):
    def __init__(
        self,
        dataset: str,
        model_name: str,
        invalid_rows: int,
        total_rows: int,
        detail: str,
        *,
        truncated: bool = False,
    ) -> None:
        self.dataset = dataset
        self.model_name = model_name
        self.invalid_rows = invalid_rows
        self.total_rows = total_rows
        self.truncated = truncated
        qualifier = "at least " if truncated else ""
        super().__init__(
            f"{dataset}: {qualifier}{invalid_rows}/{total_rows} row(s) violate {model_name} contract. {detail}"
        )


def _is_null(value) -> bool:
    """Scalar-safe null check (pd.isna raises on array-likes)."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _clean_records(df: pd.DataFrame) -> Iterator[dict]:
    """Yield each row as a plain dict, normalizing pandas/numpy quirks.

    Streams via `itertuples` rather than materializing `to_dict("records")` so we
    don't double peak memory for a large silver DataFrame. NaN/NaT become None (so
    Optional fields validate) and numpy scalars become native Python types (so
    Pydantic's int/float/bool validators accept them).
    """
    columns = list(df.columns)
    for values in df.itertuples(index=False, name=None):
        clean = {}
        for key, value in zip(columns, values):
            if _is_null(value):
                clean[key] = None
            elif isinstance(value, np.generic):
                clean[key] = value.item()
            else:
                clean[key] = value
        yield clean


def validate_dataframe(df: pd.DataFrame, model: Type[BaseModel], dataset: str) -> None:
    """Validate each row of `df` against `model`; raise SchemaValidationError on any violation.

    Stops early once `_MAX_REPORTED_ERRORS` failures are collected — for a wholly
    invalid dataset there's no value in validating every remaining row just to count
    it; the pipeline halts either way. When truncated, `invalid_rows` is a floor.
    """
    errors: list[str] = []
    truncated = False
    for idx, record in enumerate(_clean_records(df)):
        try:
            model.model_validate(record)
        except ValidationError as exc:
            messages = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            errors.append(f"row {idx}: {messages}")
            if len(errors) >= _MAX_REPORTED_ERRORS:
                truncated = True
                break

    if errors:
        detail = " | ".join(errors)
        if truncated:
            detail += f" | … (stopped after first {_MAX_REPORTED_ERRORS} failures)"
        logger.error(
            "schema_validation_failed",
            dataset=dataset,
            model=model.__name__,
            invalid_rows=len(errors),
            invalid_rows_truncated=truncated,
            total_rows=len(df),
            sample_errors=errors,
        )
        raise SchemaValidationError(dataset, model.__name__, len(errors), len(df), detail, truncated=truncated)

    logger.info("schema_validation_passed", dataset=dataset, model=model.__name__, rows=len(df))

from dataclasses import dataclass, field

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DataQualityError(Exception):
    def __init__(self, report: "DataQualityReport") -> None:
        self.report = report
        super().__init__(str(report))


@dataclass
class DataQualityReport:
    table_name: str
    row_count: int
    null_counts: dict[str, int] = field(default_factory=dict)
    duplicate_count: int = 0
    schema_errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.null_counts and not self.schema_errors and self.duplicate_count == 0


def check_nulls(df: pd.DataFrame, required_cols: list[str]) -> dict[str, int]:
    """Return {col: null_count} for required columns that contain nulls."""
    result = {}
    for col in required_cols:
        if col in df.columns:
            n = int(df[col].isnull().sum())
            if n:
                result[col] = n
    return result


def check_schema(df: pd.DataFrame, expected_cols: list[str]) -> list[str]:
    """Return error strings for expected columns missing from the DataFrame."""
    return [f"Missing column: {c}" for c in expected_cols if c not in df.columns]


def check_duplicates(df: pd.DataFrame, key_cols: list[str]) -> int:
    """Return count of duplicate rows based on key columns."""
    existing = [c for c in key_cols if c in df.columns]
    if not existing:
        return 0
    return int(df.duplicated(subset=existing).sum())


def run_quality_checks(
    df: pd.DataFrame,
    table_name: str,
    required_cols: list[str],
    key_cols: list[str],
    expected_cols: list[str] | None = None,
) -> DataQualityReport:
    schema_cols = list(dict.fromkeys((expected_cols or required_cols) + key_cols))
    report = DataQualityReport(
        table_name=table_name,
        row_count=len(df),
        null_counts=check_nulls(df, required_cols),
        duplicate_count=check_duplicates(df, key_cols),
        schema_errors=check_schema(df, schema_cols),
    )
    status = "PASSED" if report.passed else "FAILED"
    logger.info(
        "[%s] %s | rows=%d nulls=%s dupes=%d schema=%s",
        status,
        table_name,
        report.row_count,
        report.null_counts or "ok",
        report.duplicate_count,
        report.schema_errors or "ok",
    )
    return report


def assert_quality(report: DataQualityReport) -> None:
    """Log all quality issues and raise DataQualityError if the report failed."""
    if report.null_counts:
        for col, n in report.null_counts.items():
            logger.warning("[%s] null check failed: %s has %d null(s)", report.table_name, col, n)
    if report.duplicate_count:
        logger.warning("[%s] duplicate check failed: %d duplicate(s)", report.table_name, report.duplicate_count)
    for err in report.schema_errors:
        logger.warning("[%s] schema check failed: %s", report.table_name, err)
    if not report.passed:
        raise DataQualityError(report)

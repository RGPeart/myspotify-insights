"""Runtime enforcement of a DataContract against a produced DataFrame.

Schema validation is delegated to the Feature 8 validator (`validate_dataframe`),
so there is one code path for per-row checks. Contract enforcement adds the
cross-row quality rules and emits versioned log events for traceability.
"""
from __future__ import annotations

import pandas as pd

from src.contracts.base import ContractViolationError, DataContract
from src.contracts.rules import QUALITY_RULES
from src.schemas.validate import SchemaValidationError, validate_dataframe
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def enforce_contract(df: pd.DataFrame, contract: DataContract) -> None:
    """Validate `df` against `contract`; raise ContractViolationError on any violation.

    Raises ValueError (not ContractViolationError) for an unknown rule name, since
    that is a contract misconfiguration rather than a data problem.
    """
    logger.info("contract_enforcement_start", contract=contract.name, version=contract.version, rows=len(df))
    failures: list[str] = []

    try:
        validate_dataframe(df, contract.schema_model, contract.name)
    except SchemaValidationError as exc:
        failures.append(f"schema: {exc}")

    for rule_name in contract.quality_rules:
        try:
            rule = QUALITY_RULES[rule_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown quality rule '{rule_name}' in contract '{contract.name}'"
            ) from exc
        message = rule(df, contract)
        if message:
            failures.append(f"{rule_name}: {message}")

    if failures:
        logger.error("contract_violation", contract=contract.name, version=contract.version, failures=failures)
        raise ContractViolationError(contract, failures)

    logger.info("contract_satisfied", contract=contract.name, version=contract.version, rows=len(df))

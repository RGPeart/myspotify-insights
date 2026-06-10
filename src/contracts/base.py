"""Core data-contract types.

A `DataContract` is the formal, versioned interface a pipeline stage promises to
its consumers: the schema (a Pydantic model from src/schemas/), the cross-row
quality rules, ownership, and a freshness budget. It turns "my pipeline writes
whatever it wants" into "my pipeline is a service with a defined API".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from pydantic import BaseModel


@dataclass(frozen=True)
class DataContract:
    name: str
    version: str
    owner: str
    producer: str               # pipeline stage that writes this dataset
    consumer: str               # pipeline stage that reads it
    schema_model: Type[BaseModel]
    max_staleness_hours: int    # freshness budget; surfaced to monitoring (Feature 11)
    quality_rules: tuple[str, ...]


class ContractViolationError(Exception):
    def __init__(self, contract: DataContract, failures: list[str]) -> None:
        self.contract = contract
        self.failures = failures
        super().__init__(
            f"Contract '{contract.name}' v{contract.version} violated: " + "; ".join(failures)
        )

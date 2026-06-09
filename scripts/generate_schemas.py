"""Generate JSON Schema files from the Pydantic contracts in src/schemas/.

Run from the repo root:

    python scripts/generate_schemas.py

Writes one file per dataset under schemas/{layer}/{name}.json. The Pydantic models
are the source of truth; these JSON Schema files are generated artifacts. CI
(tests/test_schemas.py) fails if they drift out of sync — regenerate and commit
whenever you change a model, and add an entry to schemas/CHANGELOG.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `src` importable when run as a plain script (not just `python -m`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schemas.registry import SCHEMA_SPECS, build_json_schema  # noqa: E402


def write_schema_files() -> list[Path]:
    written: list[Path] = []
    for spec in SCHEMA_SPECS:
        path = spec.json_schema_path
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(build_json_schema(spec), indent=2) + "\n"
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    for path in write_schema_files():
        print(f"wrote {path.relative_to(Path(__file__).resolve().parents[1])}")

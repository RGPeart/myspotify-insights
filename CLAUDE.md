# CLAUDE.md

## Project Overview

**MySpotify Insights** is a Python-based data engineering portfolio project implementing an end-to-end music recommendation system: ETL pipelines, Azure cloud infrastructure, ML-based recommendations, a REST API, and an analytics dashboard.

## Development Commands

```bash
# Environment setup
python -m venv venv
.\venv\Scripts\activate           # Windows
pip install -r requirements.txt

# Copy and populate credentials (see .env.example for required variables)
cp .env.example .env

# Start Airflow stack (orchestrates the full pipeline)
docker compose up -d
# Trigger the 'spotify_etl_pipeline' DAG in the Airflow UI at http://localhost:8080

# Train recommendation model (run after pipeline produces gold data)
python -m src.models.train

# Start API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8001

# Start dashboard
streamlit run src/dashboard/app.py

# Run tests
pytest
pytest --cov=src tests/
pytest tests/path/to/test_file.py::test_function
```

## Architecture

The system follows a **medallion data lakehouse pattern** (Bronze → Silver → Gold) with data flowing one-way through the pipeline before being served via API and dashboard.

```text
Spotify API
    │
    ▼
src/ingestion/spotify_client.py    → data/bronze/   (raw JSON)
    │
    ▼
src/etl/bronze_to_silver.py        → data/silver/   (cleaned, validated)
    │
    ▼
src/etl/silver_to_gold.py          → data/gold/     (aggregated, feature-engineered)
    │
    ├──► src/models/train.py        (trains collaborative + content-based recommendation model)
    │         │
    │         ▼
    │    src/models/predict.py      (inference, called by API routes)
    │
    └──► src/api/routes.py          (FastAPI endpoints, reads gold layer + model predictions)
              │
              ▼
         src/dashboard/app.py       (Streamlit, reads gold layer + API)
```

**Key relationships:**
- `src/utils/data_quality.py` — `DataQualityReport`, `run_quality_checks`, `assert_quality` used across all ETL stages
- `src/utils/logging_config.py` — shared structured logger used across all modules
- `src/utils/config.py` — `load_config()` reads `config/config.yaml` with UTF-8 encoding; called lazily inside `run()` functions (not at import time)
- `src/utils/parquet_io.py` — `write_parquet()` used by both ETL modules
- `config/config.yaml` holds non-secret configuration; secrets go in `.env` (never `config/secrets.yaml`)
- Azure Blob Storage mirrors the `data/` directory structure in the cloud

## ETL Pipeline Details

### Bronze → Silver (`src/etl/bronze_to_silver.py`)
- Globs `data/bronze/{data_type}/**/*.json`; skips malformed JSON and non-list files with a warning
- Normalizes audio features: range features (key, loudness, tempo, time_signature) scaled to [0, 1]; unit features clipped to [0, 1]
- Logs a warning when feature values fall outside the defined normalization bounds before clipping
- Categorizes genres via ordered substring matching in `_GENRE_PATTERNS` (most-specific first; "pop" is intentionally the catch-all tail)
- Quality gate: `assert_quality(report)` raises `DataQualityError` before writing Parquet if checks fail

### Silver → Gold (`src/etl/silver_to_gold.py`)
- Builds `dim_tracks` with composite popularity: `0.6 × track_pop/100 + 0.4 × artist_pop/100`
- Unmatched artist join rows get `primary_genre = "unknown"`; artist popularity fill uses column median (falls back to 0 when all are absent)
- Fails fast if silver tracks table is missing or empty
- Quality gate: `assert_quality(report)` raises `DataQualityError` before writing Parquet if checks fail

### Airflow DAG (`dags/spotify_etl_dag.py`)
- DAG ID: `spotify_etl_pipeline`; no schedule (manual trigger only); `catchup=False`
- Three tasks in sequence: `_ingest_data` → `_bronze_to_silver` → `_silver_to_gold`; each has `retries=2, retry_delay=1m, execution_timeout=1h`
- Each task calls the corresponding Python module directly (`SpotifyIngestionClient().ingest()`, `bronze_to_silver.run()`, `silver_to_gold.run()`)
- `dags/.airflowignore` excludes `test_dag.py` from the scheduler

## Branching Strategy

```
feature/* → release → main
```

## CI/CD

GitHub Actions workflows:
- `.github/workflows/ci.yml` — lint and test on push/PR to `main`; steps are currently commented out
- `.github/workflows/claude-code-review.yml` — Claude code review on PRs; excludes workflow-file-only changes via `paths-ignore`

## Decision Records

Major decisions — infrastructure choices, design trade-offs, accepted limitations, established conventions — are recorded as Architecture Decision Records (ADRs) in `docs/decisions/`, one file per date named `YYYY-MM-DD.md`.

**When to add a decision record:**
- Choosing between competing technical approaches with non-trivial trade-offs
- Resolving infrastructure or configuration issues a future contributor might re-hit
- Establishing a convention that is not obvious from the code (e.g., "always start the stack with `--profile lineage`")
- Accepting a known limitation or explicitly deferring a fix to a later iteration

**Format per decision:** `Status`, `Context`, `Decision`, `Consequences` sections.

**Same-day rule:** If multiple decisions are made in a single day or session, append them to that date's file as numbered sections (`## 1. …`, `## 2. …`). Do not create multiple files for the same date — grouping related context together keeps the record readable.

**Immutability:** Past decision files are historical records and should not be edited. To supersede a prior decision, write a new entry in the current day's file that references and overrides the older one by date and number (e.g., "Supersedes 2026-05-18 §2").

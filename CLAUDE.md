# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MySpotify Insights** is a Python-based data engineering portfolio project implementing an end-to-end music recommendation system. It demonstrates production-grade ETL pipelines, Azure cloud infrastructure, ML-based recommendations, a REST API, and an analytics dashboard.

## Development Commands

```bash
# Environment setup
python -m venv venv
.\venv\Scripts\activate           # Windows
pip install -r requirements.txt

# Copy and populate credentials
cp .env.example .env

# Run ingestion and ETL stages in order
python -m src.ingestion.spotify_client
python -m src.etl.bronze_to_silver
python -m src.etl.silver_to_gold

# Train recommendation model
python -m src.models.train

# Start API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8001

# Start dashboard
streamlit run src/dashboard/app.py

# Run tests
pytest
pytest --cov=src tests/
pytest tests/path/to/test_file.py::test_function   # single test
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
- `src/utils/config.py` — shared `load_config()` reads `config/config.yaml` with UTF-8 encoding; called lazily inside `run()` functions (not at import time)
- `src/utils/parquet_io.py` — shared `write_parquet()` used by both ETL modules
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
- Three tasks in sequence: `_ingest_data` → `_bronze_to_silver` → `_silver_to_gold`
- Each task calls the corresponding Python module directly (`SpotifyIngestionClient().ingest()`, `bronze_to_silver.run()`, `silver_to_gold.run()`)
- Runs via Docker Compose (`docker compose up -d`); Airflow UI at `http://localhost:8080`

## Environment Variables

Required in `.env` (see `.env.example`):

| Variable | Purpose |
|---|---|
| `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` | Spotify API OAuth credentials |
| `SPOTIFY_REDIRECT_URI` | OAuth callback (e.g. `http://localhost:8888/callback`) |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Blob Storage access |
| `AZURE_STORAGE_CONTAINER_NAME` | Target container for data layers |
| `DB_CONNECTION_STRING` | Optional database backend |
| `API_HOST` / `API_PORT` | FastAPI bind address (defaults: `0.0.0.0` / `8000`) |

## Branching Strategy

```
feature/* → release → main
```

All feature branches target `release`. Once a feature is reviewed and merged to `release`, a consolidated PR moves it into `main`.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) triggers on push/PR to `main` and `develop`. The test and lint steps are currently commented out — uncomment them to re-enable pytest and flake8 runs on CI.

The Claude Code Review workflow (`.github/workflows/claude-code-review.yml`) runs on PRs but excludes workflow-file-only changes via `paths-ignore` to avoid OIDC validation errors.

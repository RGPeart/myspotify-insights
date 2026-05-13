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
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start dashboard
streamlit run src/dashboard/app.py

# Run tests
pytest
pytest --cov=src tests/
pytest tests/path/to/test_file.py::test_function   # single test
```

## Architecture

The system follows a **medallion data lakehouse pattern** (Bronze → Silver → Gold) with data flowing one-way through the pipeline before being served via API and dashboard.

```
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
- `src/utils/data_quality.py` is used across ETL stages for validation
- `src/utils/logging_config.py` is shared across all modules
- `config/config.yaml` holds non-secret configuration; secrets go in `.env` (never `config/secrets.yaml`)
- Azure Blob Storage mirrors the `data/` directory structure in the cloud

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

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) triggers on push/PR to `main` and `develop`. The test and lint steps are currently commented out — uncomment them to re-enable pytest and flake8 runs on CI
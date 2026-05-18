# MySpotify Insights

A data engineering portfolio project featuring an end-to-end ETL pipeline and ML-powered music recommendation engine using Spotify data and Azure cloud services.

## Project Overview

MySpotify Insights demonstrates production-grade data engineering practices by building a complete music recommendation system that:
- Ingests personal Spotify listening history via API
- Processes data through a multi-stage ETL pipeline (Bronze → Silver → Gold)
- Trains a hybrid recommendation model (collaborative + content-based filtering)
- Serves recommendations via a REST API
- Visualizes insights through an interactive dashboard
- Tracks full data lineage with OpenLineage and Marquez
- Emits structured JSON logs via structlog for queryable observability

**Tech Stack:** Python | Apache Airflow | Azure Blob Storage | FastAPI | Scikit-learn | Streamlit | structlog | OpenLineage | Marquez

## Architecture

The system follows a **medallion data lakehouse pattern** with data flowing one-way through the pipeline before being served by the API and dashboard. All pipeline stages emit structured logs and data lineage events automatically.

```text
Spotify API
    │
    ▼
src/ingestion/spotify_client.py    →  data/bronze/   (raw JSON)
    │                                      │ lineage event emitted
    ▼                                      ▼
src/etl/bronze_to_silver.py        →  data/silver/   (cleaned, validated Parquet)
    │                                      │ lineage event emitted
    ▼                                      ▼
src/etl/silver_to_gold.py          →  data/gold/     (dimensional model, composite scores)
    │                                      │ lineage event emitted
    ├──► src/models/train.py        (collaborative + content-based recommendation model)
    │         │
    │         ▼
    │    src/models/predict.py      (inference, called by API routes)
    │
    └──► src/api/routes.py          (FastAPI endpoints)
              │
              ▼
         src/dashboard/app.py       (Streamlit analytics dashboard + lineage graph)

Observability layer (runs alongside):
    ├──► structlog          → JSON-structured logs on stdout / forwarded to Azure Monitor
    └──► OpenLineage/Marquez → dataset + job lineage graph at http://localhost:3000
```

## Setup

### Prerequisites
- Python 3.11+
- Docker Desktop (for Airflow and optional Marquez lineage store)
- Spotify Developer account (for API credentials)
- Azure account (optional — for cloud storage and Azure Monitor)

### Installation

```bash
# Clone and enter the repo
git clone https://github.com/RGPeart/myspotify-insights.git
cd myspotify-insights

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # macOS/Linux

# Install all dependencies (includes structlog and openlineage providers)
pip install -r requirements.txt

# Install the project in editable mode so src.* imports resolve without PYTHONPATH
pip install -e .

# Configure credentials
cp .env.example .env
# Edit .env — at minimum set SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, FERNET_KEY,
# and AIRFLOW__API_AUTH__JWT_SECRET (see generation commands inside .env.example)
```

### Running the pipeline (via Airflow)

```bash
# 1. Start the Airflow stack
docker compose up airflow-init   # first time only — creates DB and admin user
docker compose up -d             # start all Airflow services

# 2. Open Airflow UI at http://localhost:8080
#    Default credentials: airflow / airflow
#    Unpause and manually trigger the 'spotify_etl_pipeline' DAG

# 3. Train recommendation model (after the DAG has produced gold data)
python -m src.models.train

# 4. Start the API server (in a dedicated terminal)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8001

# 5. Start the dashboard (in another dedicated terminal)
streamlit run src/dashboard/app.py
#    Open http://localhost:8501
```

## Observability & Data Lineage

### Structured Logging

All pipeline modules emit **JSON-structured logs** via [structlog](https://www.structlog.org/). Each log line is a machine-readable JSON object:

```json
{"event": "etl_stage_complete", "table": "tracks", "status": "PASS", "row_count": 342, "logger": "dags.spotify_etl_dag", "level": "info", "timestamp": "2026-05-17T06:12:34.123456Z"}
{"event": "ingestion_complete",  "tracks": 342, "audio_features": 342, "artists": 87, "level": "info", "timestamp": "2026-05-17T06:11:58.001234Z"}
```

Logs can be piped to any log aggregator (Datadog, CloudWatch, Azure Monitor) without additional parsing.

#### Optional: Forward to Azure Monitor

Set `AZURE_MONITOR_CONNECTION_STRING` in your `.env` to automatically forward logs and traces to Azure Application Insights:

```bash
# .env
AZURE_MONITOR_CONNECTION_STRING=InstrumentationKey=xxxxxxxx-...
```

The pipeline calls `src/utils/azure_monitor.py` at startup, which configures the OpenTelemetry Azure Monitor exporter. If the env var is absent the pipeline runs normally without any Azure dependency.

### Data Lineage with OpenLineage + Marquez

The Airflow DAG is instrumented with the `apache-airflow-providers-openlineage` package, which **automatically emits lineage events** each time a task runs — no code changes needed beyond setting two environment variables.

#### Start the Marquez lineage store

Marquez runs as an opt-in Docker Compose profile so it doesn't slow down the base dev stack:

```bash
# Start Marquez alongside Airflow
docker compose --profile lineage up -d

# Or start only Marquez (if Airflow is already running)
docker compose --profile lineage up -d marquez-db marquez marquez-web
```

| Service | URL | Description |
|---|---|---|
| Marquez UI | http://localhost:3000 | Browse the lineage graph interactively |
| Marquez API | http://localhost:5000/api/v1 | REST API consumed by the dashboard |
| Airflow UI | http://localhost:8080 | Trigger DAG runs |
| Dashboard | http://localhost:8501 | Live lineage panel + recommendations |

#### What you'll see

After triggering the `spotify_etl_pipeline` DAG at least once, Marquez will show:

- **Jobs:** `_ingest_data`, `_bronze_to_silver`, `_silver_to_gold` (one node per Airflow task)
- **Datasets:** `bronze/tracks.json`, `silver/tracks.parquet`, `gold/dim_tracks.parquet`, etc.
- **Edges:** which task produced each dataset and which task consumed it

The **Streamlit dashboard** includes a static pipeline topology diagram and a live panel that queries the Marquez API to show tracked job/dataset counts.

#### Lineage environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENLINEAGE_URL` | `http://marquez:5000` | Marquez API URL (inside Docker network) |
| `OPENLINEAGE_NAMESPACE` | `myspotify-insights` | Namespace grouping all jobs and datasets |
| `MARQUEZ_URL` | `http://localhost:5000` | Marquez API URL for the dashboard (outside Docker) |

These are pre-configured in `.env.example` and `docker-compose.yaml`. No additional setup is needed beyond starting the lineage profile.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run a single test file
pytest tests/test_etl.py::TestBronzeToSilverRun::test_run_end_to_end
```

## Project Status

| Feature | Status | Branch / PR |
|---|---|---|
| Project setup & PRD | Done | `main` |
| Feature 1: Spotify API ingestion | Done | `main` |
| Feature 2: ETL pipeline (Bronze → Silver → Gold) | Done | `main` |
| Feature 3: Recommendation model | Done | `main` |
| Feature 4: FastAPI service | Done | `main` |
| Feature 5: Streamlit dashboard | Done | `main` |
| Feature 6: Observability & Data Lineage | In Progress | `feature/data-lineage` |
| Feature 7: SQL Transformation Layer (dbt + DuckDB) | Planned | — |
| Feature 8: Schema Registry & Schema Evolution | Planned | — |
| Feature 9: Data Contracts | Planned | — |
| Feature 10: Idempotent DAGs & Backfill | Planned | — |
| Feature 11: SLA Monitoring & Alerting | Planned | — |
| Feature 12: Cloud Cost Monitoring | Planned | — |

## Branching Strategy

```
feature/* → release → main
```

Feature branches are opened against `release`. Once reviewed and merged to `release`, a consolidated PR moves the changes into `main`.

## Author

**Ryan Peart**
- Portfolio: [rgpeart.github.io/portfolio](https://rgpeart.github.io/portfolio)
- LinkedIn: [Ryan Peart](https://www.linkedin.com/in/ryan-peart/)
- GitHub: [@RGPeart](https://github.com/RGPeart)

## License

MIT License — see [LICENSE](LICENSE) for details.

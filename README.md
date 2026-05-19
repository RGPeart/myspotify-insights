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

**Tech Stack:** Python | Apache Airflow | dbt + DuckDB | Azure Blob Storage | FastAPI | Scikit-learn | Streamlit | structlog | OpenLineage | Marquez

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
dbt run (dbt/models/gold/*.sql)    →  data/gold/     (dimensional model in SQL,
    │                                      │              external Parquet via DuckDB)
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

## SQL Transformation Layer (dbt + DuckDB)

The Silver → Gold step is implemented as a dbt project at [`dbt/`](dbt/) using the [`dbt-duckdb`](https://github.com/duckdb/dbt-duckdb) adapter. SQL models replace the legacy `src/etl/silver_to_gold.py` (kept in the repo for reference but no longer invoked by the DAG).

**Layout:**

```
dbt/
├── dbt_project.yml
├── profiles.yml                 # DuckDB profile (dev + prod targets)
├── packages.yml                 # dbt_utils dependency
├── package-lock.yml             # pinned dbt_utils version
└── models/
    ├── staging/                 # views over silver/*.parquet via read_parquet()
    │   ├── stg_silver_tracks.sql
    │   ├── stg_silver_artists.sql
    │   ├── stg_silver_audio_features.sql
    │   └── schema.yml
    └── gold/                    # materialized='external' → writes data/gold/*.parquet
        ├── dim_tracks.sql
        ├── dim_artists.sql
        ├── fact_audio_features.sql
        └── schema.yml           # unique / not_null / accepted_range tests
```

Gold models use `materialized='external'` so the outputs land in `data/gold/*.parquet` exactly where the API and Streamlit dashboard already read from — no downstream changes were needed. The `data/spotify.duckdb` database is created alongside as a queryable view layer over those Parquet files.

### Run dbt locally

From the **repo root** with the virtualenv active:

```bash
# One-time: install dbt_utils package
cd dbt && dbt deps --profiles-dir . && cd ..

# Build the gold models (writes data/gold/*.parquet and data/spotify.duckdb)
cd dbt
dbt run  --profiles-dir . --vars "{silver_dir: '$PWD/../data/silver', gold_dir: '$PWD/../data/gold'}"
dbt test --profiles-dir . --vars "{silver_dir: '$PWD/../data/silver', gold_dir: '$PWD/../data/gold'}"
```

**Windows (PowerShell):**

```powershell
cd dbt
dbt deps --profiles-dir .
$repo = (Resolve-Path ..).Path
$env:DBT_DUCKDB_PATH = "$repo\data\spotify.duckdb"
$vars = "{silver_dir: '$($repo -replace '\\','/')/data/silver', gold_dir: '$($repo -replace '\\','/')/data/gold'}"
dbt run  --profiles-dir . --vars $vars
dbt test --profiles-dir . --vars $vars
```

Expected output: `PASS=6 WARN=0 ERROR=0` (3 staging views + 3 gold external models) and `PASS=35` data tests.

### Query the DuckDB database

```bash
python -c "import duckdb; con = duckdb.connect('data/spotify.duckdb', read_only=True); print(con.execute('SELECT count(*) FROM dim_tracks').fetchone())"
```

Or with the DuckDB CLI:

```bash
duckdb data/spotify.duckdb -c "SELECT primary_genre, count(*) FROM dim_tracks GROUP BY 1 ORDER BY 2 DESC;"
```

### Browse the lineage graph

```bash
cd dbt
dbt docs generate --profiles-dir .
dbt docs serve    --profiles-dir .   # opens http://localhost:8080
```

The dbt docs site complements the OpenLineage/Marquez graph from Feature 6: Marquez shows job-level Airflow lineage; dbt shows model-level SQL lineage within the gold layer.

### Run dbt via Airflow

The `spotify_etl_pipeline` DAG executes `dbt deps → dbt run → dbt test` after `_bronze_to_silver`. `docker-compose.yaml` mounts `./dbt → /opt/airflow/dbt`, and `dbt-core` + `dbt-duckdb` are installed inside the Airflow image via `requirements.txt`. No manual setup is needed inside Airflow — trigger the DAG and dbt runs automatically.

### dbt vars

| Var | Default | Override via |
|---|---|---|
| `silver_dir` | `data/silver` | `--vars '{"silver_dir": "/abs/path"}'` |
| `gold_dir`   | `data/gold`   | `--vars '{"gold_dir": "/abs/path"}'` |

The DuckDB file path is set with the `DBT_DUCKDB_PATH` env var (defaults to `data/spotify.duckdb` in `profiles.yml`).

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
| Marquez API | http://localhost:5002/api/v1 | REST API consumed by the dashboard |
| Airflow UI | http://localhost:8080 | Trigger DAG runs |
| Dashboard | http://localhost:8501 | Live lineage panel + recommendations |

#### What you'll see

After triggering the `spotify_etl_pipeline` DAG at least once, Marquez will show:

- **Jobs:** `_ingest_data`, `_bronze_to_silver`, `dbt_deps`, `dbt_run_gold`, `dbt_test_gold` (one node per Airflow task)
- **Datasets:** `bronze/tracks.json`, `silver/tracks.parquet`, `gold/dim_tracks.parquet`, etc.
- **Edges:** which task produced each dataset and which task consumed it

The **Streamlit dashboard** includes a static pipeline topology diagram and a live panel that queries the Marquez API to show tracked job/dataset counts.

#### Lineage environment variables

| Variable | Default | Description |
|---|---|---|
| `MARQUEZ_URL` | `http://marquez:5002` | Marquez API URL (inside Docker network) |
| `OPENLINEAGE_NAMESPACE` | `myspotify-insights` | Namespace grouping all jobs and datasets |
| `MARQUEZ_URL` | `http://localhost:5002` | Marquez API URL for the dashboard (outside Docker) |

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
| Feature 6: Observability & Data Lineage | In Progress | `main` |
| Feature 7: SQL Transformation Layer (dbt + DuckDB) | In Progress | `feature/dbt-duckdb` |
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

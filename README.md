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
# Edit .env — at minimum set SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET,
# SPOTIFY_REFRESH_TOKEN (see "One-time Spotify authorization" below), FERNET_KEY,
# and AIRFLOW__API_AUTH__JWT_SECRET (see generation commands inside .env.example)
```

### One-time Spotify authorization

Ingestion uses the Spotify **Authorization Code** flow to read *your* listening
data (`/me`, `/me/top/{tracks,artists}`, `/me/following`), so it needs a user
refresh token in addition to the client ID/secret. Obtain one once:

```bash
python -m src.ingestion.spotify_auth_login
```

This opens a browser, asks you to grant the `user-top-read`, `user-follow-read`,
and `user-read-private` scopes, then prints a line like
`SPOTIFY_REFRESH_TOKEN=...`. Paste that value into your `.env`. The token is
long-lived; every ingest run (including headless Airflow runs) exchanges it for a
short-lived access token automatically — no further browser interaction needed.

> The redirect URI registered in your Spotify app **must exactly match**
> `SPOTIFY_REDIRECT_URI` in `.env` (default `http://127.0.0.1:8888/callback`).
> Spotify requires `127.0.0.1` for loopback redirects, not `localhost`.

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

### Running the pipeline without Docker (direct modules)

If Docker is unavailable, you can run every stage the DAG orchestrates as a
plain module from the **repo root** (venv active). This is the exact sequence the
DAG performs, minus Airflow:

```bash
# 1. Ingestion → data/bronze/  (requires SPOTIFY_REFRESH_TOKEN; see above)
python -m src.ingestion.spotify_client

# 2. Bronze → Silver → data/silver/*.parquet
python -m src.etl.bronze_to_silver

# 3. Silver → Gold (dbt + DuckDB) → data/gold/*.parquet
#    Run from the repo root so DuckDB's relative paths resolve against ./data
dbt deps --project-dir dbt --profiles-dir dbt
dbt run  --project-dir dbt --profiles-dir dbt
dbt test --project-dir dbt --profiles-dir dbt

# 4. Train the recommendation model → models/recommendation_model.pkl
python -m src.models.train

# 5. Serve the API (separate terminal)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8001

# 6. Serve the dashboard (separate terminal)
streamlit run src/dashboard/app.py
```

Running dbt with `--project-dir dbt` from the repo root keeps the working
directory at the project root, so the default `silver_dir`/`gold_dir` vars
(`data/silver`, `data/gold`) and the DuckDB path resolve correctly without
passing absolute paths. A healthy run reports `PASS=6` for `dbt run` and
`PASS=37` for `dbt test`.

### What ingestion produces

`python -m src.ingestion.spotify_client` reads your refresh token, then:

- fetches your profile, your top tracks and top artists (`short_term` +
  `medium_term`), and your followed artists;
- derives the most common genres across your top + followed artists and uses
  those to drive a catalog-expansion search (falling back to
  `spotify.fallback_genres` from `config/config.yaml` only if no personal genres
  are available);
- merges your top tracks/artists into the standard `tracks`/`artists` layers so
  they reach silver/gold;
- writes raw JSON to `data/bronze/`, including the personal-data partitions:

  ```text
  data/bronze/
  ├── tracks/ , audio_features/ , artists/          (standard catalog)
  ├── user_profile/<date>/
  ├── user_top_tracks/{short_term,medium_term}/<date>/
  ├── user_top_artists/{short_term,medium_term}/<date>/
  └── followed_artists/<date>/
  ```

  The `user_*` and `followed_artists` partitions are landed for future
  personalised-ML work; the standard `tracks`/`audio_features`/`artists`
  partitions feed the existing silver → gold pipeline unchanged.

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

Expected output: `PASS=6 WARN=0 ERROR=0` (3 staging views + 3 gold external models) and `PASS=37` data tests.

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
dbt docs serve    --profiles-dir . --port 18080   # opens http://localhost:18080 (8080 is Airflow)
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

### Switching the warehouse to Postgres

`dbt/profiles.yml` defines a `pg` target alongside DuckDB. To route the gold layer through Postgres:

1. Register an Airflow Connection named `dbt_postgres` (CLI: `airflow connections add dbt_postgres --conn-type postgres ...`, or via the Airflow UI).
2. Set `DBT_TARGET=pg` in `.env` and recreate the Airflow containers.

The DAG's `BashOperator`s pull credentials from the Connection at task-execution time via Jinja (`{{ conn.dbt_postgres.host }}` etc.) and export them as env vars that `profiles.yml` reads. Gold model SQL automatically switches from `external` Parquet materialisation to regular tables when `target.type == 'postgres'`. See [`docs/setup.md`](docs/setup.md#switching-the-warehouse-to-postgres-future) for the full procedure and a note on the silver-loading limitation.

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

## Schema Registry & Schema Evolution

Each pipeline dataset has an explicit, versioned schema contract. **Pydantic models
in [`src/schemas/`](src/schemas/) are the canonical source of truth**; the JSON
Schema files in [`schemas/`](schemas/) are generated artifacts kept in sync via CI.

**What's covered:** the silver datasets (`tracks`, `audio_features`, `artists`) and
the gold datasets (`dim_tracks`, `dim_artists`, `fact_audio_features`). Bronze is
excluded — it stores raw nested Spotify API JSON whose shape Spotify controls.

```
src/schemas/
├── silver.py        # SilverTrack, SilverAudioFeatures, SilverArtist
├── gold.py          # GoldDimTrack, GoldDimArtist, GoldFactAudioFeatures
├── registry.py      # SCHEMA_REGISTRY + build_json_schema()
└── validate.py      # validate_dataframe() + SchemaValidationError

schemas/             # generated JSON Schema (do not edit by hand)
├── silver/*.json
├── gold/*.json
└── CHANGELOG.md     # human-readable log of every schema change
```

**Runtime enforcement:** `src/etl/bronze_to_silver.py` calls `validate_dataframe()`
on each silver dataset after the quality gate and before writing Parquet. A contract
violation raises `SchemaValidationError` and halts the pipeline with a structured
`schema_validation_failed` log event naming the offending rows.

**Evolution workflow** — whenever you change a model:

1. Edit the Pydantic model in `src/schemas/` and bump its `version` in `registry.py`.
2. Regenerate the JSON Schema files: `python scripts/generate_schemas.py`.
3. Add an entry to [`schemas/CHANGELOG.md`](schemas/CHANGELOG.md).
4. Commit all of the above together.

CI guards this: `tests/test_schemas.py` regenerates each schema from its model and
fails if it differs from the committed file, so the JSON Schema can never silently
drift from the Pydantic source of truth.

## Data Contracts

Each silver stage boundary has an explicit, versioned **data contract** — the formal
interface the producer promises its consumers. Contracts live in
[`src/contracts/`](src/contracts/) and build directly on the schema registry above.

A `DataContract` bundles:
- **`schema_model`** — the Pydantic contract from `src/schemas/` (per-row validation).
- **`quality_rules`** — named cross-row checks (e.g. key uniqueness) the per-row schema can't express.
- **`owner` / `producer` / `consumer`** — who owns it and which stages write/read it.
- **`version`** and **`max_staleness_hours`** — for traceability and freshness budgeting.

```
src/contracts/
├── base.py       # DataContract dataclass + ContractViolationError
├── rules.py      # QUALITY_RULES registry (name → predicate)
├── enforce.py    # enforce_contract(df, contract)
└── registry.py   # silver_tracks / silver_audio_features / silver_artists
```

**Runtime enforcement:** `bronze_to_silver.run()` calls `enforce_contract(df, contract)`
after the operational quality gate and before writing Parquet. It delegates per-row
schema checks to the Feature 8 validator, then runs the contract's quality rules. A
violation raises `ContractViolationError` and halts the pipeline, logging a versioned
`contract_violation` event. An unknown rule name raises `ValueError` (a misconfiguration).

The active contracts and their metadata are surfaced in the Streamlit dashboard's
**Data Contracts** panel. Gold's interface is enforced by the dbt schema tests rather
than a duplicate Python contract.

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
| Feature 6: Observability & Data Lineage | Done | `main` |
| Feature 7: SQL Transformation Layer (dbt + DuckDB) | Done | `main` |
| Feature 8: Schema Registry & Schema Evolution | Done | `feature/schema-registry` |
| Feature 9: Data Contracts | Done | `feature/data-contracts` |
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

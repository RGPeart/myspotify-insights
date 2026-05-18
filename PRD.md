# Product Requirements Document: Music Recommendation Engine

**Project Name:** MySpotify Insights  
**Version:** 2.0  
**Author:** Ryan Peart  
**Last Updated:** May 17, 2026  
**Status:** In Progress

---

## 1. Overview

MySpotify Insights is a data engineering portfolio project that demonstrates end-to-end pipeline development, cloud infrastructure management, and machine learning integration. The project builds a music recommendation system powered by Spotify data, featuring a robust ETL pipeline, intelligent recommendation engine, and interactive web analytics dashboard.

**Problem Statement:**  
Music streaming platforms generate massive amounts of user behavior and audio feature data, but understanding how to process, model, and serve recommendations at scale requires sophisticated data engineering. This project simulates real-world challenges faced by companies like Spotify, demonstrating skills in data ingestion, transformation, storage optimization, and ML model deployment.

**Solution:**  
A full-stack data product that:
- Ingests music data from Spotify API (tracks, artists, audio features)
- Processes and transforms raw data through an ETL pipeline
- Tracks data lineage and schema evolution across all pipeline stages
- Enforces data contracts between pipeline producers and consumers
- Generates personalized music recommendations using collaborative filtering
- Exposes recommendations via a RESTful API
- Visualizes pipeline performance, data quality, and recommendation quality through a dashboard
- Monitors SLA compliance and cloud infrastructure costs

---

## 2. Goals and Non-Goals

### Goals
1. **Demonstrate Data Engineering Skills**
   - Build production-quality ETL pipeline with orchestration
   - Implement data quality checks, error handling, and idempotent DAGs
   - Design scalable data models and storage architecture
   - Enforce data contracts between pipeline stages using Pydantic models

2. **Showcase Cloud Infrastructure Knowledge**
   - Utilize Azure services cost-effectively (Blob Storage, Functions, App Service)
   - Implement CI/CD practices with GitHub Actions
   - Deploy live, publicly accessible demo
   - Track and report actual cloud infrastructure costs

3. **Integrate Machine Learning**
   - Build recommendation engine using audio features and user behavior patterns
   - Deploy ML model as a REST API
   - Track model performance metrics

4. **Build Observable, Trustworthy Pipelines**
   - Implement full data lineage tracking with OpenLineage/Marquez
   - Structured logging and Azure Monitor integration
   - SLA monitoring with alerting for downstream dependency awareness
   - Schema registry with schema evolution changelog

5. **Create Portfolio-Ready Deliverable**
   - Comprehensive documentation with architecture diagrams
   - Clean, well-tested codebase
   - Live demo with compelling use case
   - Blog post explaining technical decisions

### Non-Goals
1. **Not building a production music streaming service** - Focus is on data engineering infrastructure, not full-featured product
2. **Not implementing real-time streaming** - Batch processing is sufficient for portfolio demonstration (can be Phase 2)
3. **Not optimizing for scale** - Designed for thousands of tracks, not millions (cost constraint)
4. **Not building mobile apps** - Web-only interface sufficient for demo

---

## 3. Target Users

### Primary Audience: Recruiters & Hiring Managers
**Profile:**
- Data Engineering, Analytics Engineering, or ML Engineering roles
- Companies: Spotify, WHOOP, Microsoft, or similar tech companies
- Looking for candidates with end-to-end pipeline development experience

**What they want to see:**
- Clean, documented code with best practices
- Understanding of cloud architecture and cost optimization
- Ability to design scalable data models
- Integration of multiple technologies (ETL, ML, APIs, cloud)
- Problem-solving and technical decision-making
- Evidence of production-readiness thinking: lineage, contracts, observability, SLAs

### Secondary Audience: Technical Community
**Profile:**
- Engineers learning data pipelines
- Portfolio project inspiration seekers
- Blog readers interested in Spotify API + Azure

**What they want:**
- Step-by-step technical breakdown
- Open-source code to learn from
- Architectural insights and tradeoffs

---

## 4. Core Features

### Feature 1: Data Ingestion Pipeline
**Description:** Automated system to extract music data from Spotify API and load into Azure Blob Storage.

**Technical Details:**
- Azure Functions (timer-triggered) for scheduled API calls
- Rate limiting and error handling for API reliability
- Incremental data loading (avoid reprocessing)
- Store raw JSON data in "bronze" layer

**Data Sources:**
- Spotify Web API endpoints: tracks, artists, audio features, genres
- Personal Spotify listening history

---

### Feature 2: ETL Pipeline with Orchestration
**Description:** Transform raw Spotify data into analytics-ready models with data quality checks.

**Technical Details:**
- **Bronze → Silver:** Clean and normalize JSON data
- **Silver → Gold:** Create dimensional models (fact_listening_history, dim_tracks, dim_artists)
- Orchestration via Airflow
- Data quality tests: null checks, schema validation, duplicate detection
- Store processed data in Parquet files with DuckDB integration

**Transformations:**
- Audio feature normalization (scale 0-1)
- Genre categorization and tagging
- Popularity scoring algorithms
- Time-based listening pattern aggregations

---

### Feature 3: Recommendation Engine
**Description:** Machine learning model that generates personalized track recommendations.

**Technical Details:**
- **Algorithm:** Collaborative filtering (user-item matrix) + content-based filtering (audio features)
- **Training:** Scikit-learn or lightweight library
- **Features:** danceability, energy, valence, tempo, genre similarity, artist popularity
- **Output:** Top 10 recommended tracks per user with confidence scores
- Model retraining schedule (weekly batch job)
- **Data Volume:** Target 1,000+ tracks of historical data for recommendations

**Recommendation Logic:**
- Hybrid approach: 70% collaborative filtering, 30% content similarity
- Cold-start handling: Use audio features for new users/tracks
- Diversity bonus: Penalize genre over-concentration

---

### Feature 4: REST API for Recommendations
**Description:** FastAPI service that serves recommendations and track metadata.

**Endpoints:**
- `GET /recommendations/{user_id}` - Get top N recommendations
- `GET /tracks/{track_id}` - Get track details and audio features
- `GET /artists/{artist_id}` - Get artist information
- `GET /health` - Service health check

**Deployment:**
- Azure App Service (free tier) or Azure Container Instances
- OpenAPI documentation (Swagger UI)
- Response caching for performance

---

### Feature 5: Analytics Dashboard
**Description:** Web-based dashboard showing pipeline metrics and recommendation insights.

**Metrics Displayed:**
- Pipeline run status and data freshness
- Recommendation model performance (precision, recall, diversity)
- Data quality KPIs (completeness, validity)
- Most recommended tracks/artists
- Audio feature distributions
- Data lineage graph visualization
- SLA compliance status
- Cloud cost tracking panel

**Technology:**
- Streamlit or React + Chart.js
- User feedback loop for recommendation ratings
- Real user authentication via Spotify
- Deployed on Streamlit Community Cloud or Azure Static Web Apps (free)

---

### Feature 6: Observability & Data Lineage Tracking
**Description:** End-to-end visibility into data movement, transformation history, and pipeline health using OpenLineage and structured logging.

**Why This Matters:**
One of the hardest problems in production data engineering is answering "where did this bad data come from?" This feature directly addresses that by tracking data lineage at every stage and emitting structured logs that can be queried and alerted on.

**Technical Details:**

**OpenLineage / Marquez Integration:**
- Deploy Marquez locally (Docker) or use the Marquez Cloud free tier as a lineage metadata store
- Instrument each Airflow DAG task with the `openlineage-airflow` provider package to automatically emit lineage events
- Track dataset-level lineage: which task consumed `bronze/tracks.json` and produced `silver/tracks.parquet`
- Store lineage events in Marquez and expose the lineage graph in the analytics dashboard

```python
# Example: Airflow DAG with OpenLineage instrumentation
# openlineage-airflow emits events automatically when OPENLINEAGE_URL is set
# Set in Airflow environment:
# OPENLINEAGE_URL=http://localhost:5002
# OPENLINEAGE_NAMESPACE=myspotify-insights

from airflow import DAG
from airflow.operators.python import PythonOperator

def transform_bronze_to_silver():
    # OpenLineage events emitted automatically via the Airflow provider
    pass

dag = DAG("etl_pipeline", ...)
```

**Structured Logging with `structlog`:**
- Replace all `print()` and basic `logging` calls with `structlog` for machine-readable, queryable logs
- Emit structured log events at key pipeline checkpoints: ingestion start/end, record counts, transformation errors, model retraining

```python
import structlog

log = structlog.get_logger()

log.info(
    "ingestion_complete",
    source="spotify_api",
    endpoint="/audio-features",
    records_fetched=342,
    duration_seconds=4.7,
    layer="bronze",
    run_id="2026-05-17T06:00:00Z"
)
```

**Azure Monitor Integration:**
- Forward structured logs to Azure Monitor using the `azure-monitor-opentelemetry` SDK
- Create metric alerts: trigger when record count drops >20% vs. prior run (potential API issue)
- Create availability alerts: trigger when a DAG task hasn't run in >26 hours (missed schedule)
- Dashboard panel showing last 7 days of alert history

**Dashboard Lineage Graph:**
- Render the Marquez lineage graph in the Streamlit/React dashboard using the Marquez REST API
- Show: `Spotify API → bronze/tracks.json → silver/tracks.parquet → gold/dim_tracks → recommendations API`
- Clicking a node shows the task that produced it, row counts, and last updated timestamp

**Implementation Steps:**
1. `pip install openlineage-airflow structlog azure-monitor-opentelemetry`
2. Deploy Marquez via Docker: `docker run -p 5002:5003 marquezproject/marquez`
3. Set `OPENLINEAGE_URL` in Airflow environment variables
4. Replace logging calls with `structlog` throughout codebase
5. Configure Azure Monitor workspace and set connection string
6. Add lineage graph panel to dashboard using Marquez `/api/v1/lineage` endpoint

---

### Feature 7: SQL Transformation Layer with dbt + DuckDB
**Description:** Replaces the Silver → Gold pandas transformation with dbt (data build tool) SQL models backed by DuckDB, bringing software-engineering practices — version-controlled models, automated testing, and auto-generated documentation with a visual lineage DAG — to the gold layer.

**Why This Matters:**
dbt is mentioned in the majority of senior data engineering job descriptions and has become the de-facto standard for transformation in the modern data stack. But beyond the resume signal, it is the architecturally correct tool here: SQL is the right language for dimensional modeling (joins, window functions, aggregations), while Python/pandas remains the right tool for the complex algorithmic cleaning in Bronze → Silver. Choosing each tool where it fits demonstrates engineering judgement, not just familiarity with buzzwords.

**Architecture Change:**
- **Bronze → Silver:** Stays as Python/pandas — audio feature normalization, genre classification, and malformed-record handling are algorithms, not queries.
- **Silver → Gold:** Moves to dbt + DuckDB — dimensional modeling, popularity scoring, and artist joins are relational operations that are cleaner and more testable in SQL.

**Technical Details:**

**dbt Project Structure:**
```
dbt/
  models/
    staging/
      stg_silver_tracks.sql      # Thin wrapper — selects from silver Parquet
      stg_silver_artists.sql
    gold/
      dim_tracks.sql             # Core dimension with composite popularity
      dim_artists.sql
      fact_listening_history.sql
  tests/
    assert_composite_popularity_range.sql   # Custom singular test
  schema.yml                     # Column-level docs + built-in tests (unique, not_null)
  dbt_project.yml
  profiles.yml                   # DuckDB connection config (database path; Parquet paths are resolved in model SQL via read_parquet())
```

**Staging model (`stg_silver_tracks.sql`):**
```sql
{{ config(materialized='view') }}

select
    track_id,
    name,
    artist_id,
    danceability,
    energy,
    valence,
    tempo,
    track_popularity,
    primary_genre
from read_parquet('{{ var("silver_dir") }}/tracks/*.parquet')
```

Set `silver_dir` in `dbt_project.yml` so it has a default and can be overridden at the CLI with `--vars`:
```yaml
# dbt_project.yml (vars block)
vars:
  silver_dir: "data/silver"
```
Override for CI or Airflow: `dbt run --vars '{"silver_dir": "/opt/airflow/data/silver"}'`

**Core gold model (`dim_tracks.sql`):**
```sql
{{ config(materialized='table') }}

with tracks as (
    select * from {{ ref('stg_silver_tracks') }}
),
artists as (
    select * from {{ ref('stg_silver_artists') }}
),
median_pop as (
    select percentile_cont(0.5) within group (order by artist_popularity) as value
    from artists
)

select
    t.track_id,
    t.name,
    t.artist_id,
    a.artist_name,
    coalesce(a.primary_genre, 'unknown')                                  as primary_genre,
    t.danceability,
    t.energy,
    t.valence,
    t.tempo,
    round(
        0.6 * t.track_popularity / 100.0
        + 0.4 * coalesce(a.artist_popularity, m.value, 0) / 100.0,
    4)                                                                     as composite_popularity
from tracks t
left join artists a on t.artist_id = a.artist_id
cross join median_pop m
```

**`dim_artists.sql`:**
```sql
{{ config(materialized='table') }}

with artists as (
    select * from {{ ref('stg_silver_artists') }}
),
median_pop as (
    select percentile_cont(0.5) within group (order by artist_popularity) as value
    from artists
    where artist_popularity is not null
)

select
    a.artist_id,
    a.artist_name,
    coalesce(a.artist_popularity, m.value, 0) as artist_popularity,
    coalesce(a.primary_genre, 'unknown')       as primary_genre,
    a.follower_count
from artists a
cross join median_pop m
```

**`fact_listening_history.sql`:**
```sql
{{ config(materialized='table') }}

with history as (
    select * from read_parquet('{{ var("silver_dir") }}/listening_history/*.parquet')
),
tracks as (
    select track_id from {{ ref('dim_tracks') }}
)

select
    h.played_at,
    h.track_id,
    h.context_type,
    h.ms_played
from history h
inner join tracks t on h.track_id = t.track_id
```

**schema.yml (tests + column docs):**
```yaml
version: 2

models:
  - name: dim_tracks
    description: "Gold layer track dimension — composite popularity scores, genre, audio features."
    columns:
      - name: track_id
        description: "Spotify track ID — primary key."
        tests:
          - unique
          - not_null
      - name: composite_popularity
        description: "Weighted score: 60% track popularity + 40% artist popularity, scaled 0–1."
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
              max_value: 1
      - name: primary_genre
        tests:
          - not_null

  - name: dim_artists
    description: "Gold layer artist dimension — popularity, genre, follower count."
    columns:
      - name: artist_id
        description: "Spotify artist ID — primary key."
        tests:
          - unique
          - not_null
      - name: artist_popularity
        description: "Artist popularity 0–100; median-imputed when null."
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
              max_value: 100
      - name: primary_genre
        tests:
          - not_null

  - name: fact_listening_history
    description: "Fact table of Spotify play events joined to dim_tracks."
    columns:
      - name: played_at
        description: "ISO 8601 timestamp of the play event."
        tests:
          - not_null
      - name: track_id
        description: "FK → dim_tracks.track_id."
        tests:
          - not_null
      - name: ms_played
        description: "Milliseconds the track was played."
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
```

**Airflow DAG integration:**
```python
from airflow.operators.bash import BashOperator

dbt_run = BashOperator(
    task_id="dbt_run_gold",
    bash_command="cd /opt/airflow/dbt && dbt run --profiles-dir . --target prod",
)

dbt_test = BashOperator(
    task_id="dbt_test_gold",
    bash_command="cd /opt/airflow/dbt && dbt test --profiles-dir . --target prod",
)

# Reference the existing task instances (do NOT call the @task functions again)
# Inside the with DAG(...) block:
ingest_data >> transform_bronze_to_silver >> dbt_run >> dbt_test
```

**dbt Docs Site:**
- `dbt docs generate && dbt docs serve` produces a browsable documentation site with an interactive lineage DAG (source → staging → gold models)
- This DAG complements the Marquez lineage from Feature 6: Marquez shows task-level Airflow lineage; dbt shows model-level SQL lineage within the gold layer
- Screenshot both for the portfolio README — together they demonstrate end-to-end lineage awareness

**Implementation Steps:**
1. `pip install dbt-core dbt-duckdb` (`dbt-utils` is a **dbt package**, not a pip package — see step 3)
2. `cd dbt && dbt init spotify_gold` — initialise the project structure
3. Create `packages.yml` alongside `dbt_project.yml` and run `dbt deps` to install `dbt-utils`:
   ```yaml
   # dbt/packages.yml
   packages:
     - package: dbt-labs/dbt_utils
       version: [">=1.0.0", "<2.0.0"]
   ```
4. Configure `profiles.yml` with a DuckDB profile (set the `path` to your DuckDB database file, e.g. `data/spotify.duckdb`)
5. Write staging models as thin views over the silver Parquet files (use `var("silver_dir")` for Parquet paths)
6. Translate `silver_to_gold.py` logic into `dim_tracks.sql`, `dim_artists.sql`, `fact_listening_history.sql`
7. Add `schema.yml` with `unique`, `not_null`, and `accepted_range` tests for all key columns across all three gold models
8. Replace the `_silver_to_gold` Airflow task with `dbt_run` + `dbt_test` BashOperators
9. Run `dbt docs generate` and add a screenshot of the lineage graph to the README under "Architecture"
10. Add `dbt-core` and `dbt-duckdb` to `requirements.txt`

---

### Feature 8: Schema Registry & Schema Evolution Management
**Description:** A lightweight schema registry that tracks the shape of data at each pipeline layer and maintains a changelog of schema changes — enabling safe API evolution and rapid diagnosis of breaking upstream changes.

**Why This Matters:**
Spotify has changed their API response shapes before, and every data engineer has a story about a silent schema change that corrupted downstream models for days before anyone noticed. This feature shows you've thought beyond "write the pipeline" to "maintain the pipeline."

**Technical Details:**

**Schema Definition with JSON Schema:**
- Define a JSON Schema file for each layer (bronze, silver, gold) and the Spotify API response
- Check schemas into version control under `/schemas/` with a `CHANGELOG.md` tracking every change

```
/schemas/
  bronze/
    tracks.json          # JSON Schema for raw Spotify API response
    audio_features.json
  silver/
    tracks.json          # JSON Schema for cleaned/normalized tracks
  gold/
    dim_tracks.json
    fact_listening_history.json
  CHANGELOG.md           # Human-readable log of every schema change
```

**Example schema (`/schemas/silver/tracks.json`):**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "SilverTrack",
  "version": "1.2.0",
  "type": "object",
  "required": ["track_id", "name", "artist_id", "danceability", "energy"],
  "properties": {
    "track_id": { "type": "string" },
    "name": { "type": "string" },
    "artist_id": { "type": "string" },
    "danceability": { "type": "number", "minimum": 0, "maximum": 1 },
    "energy": { "type": "number", "minimum": 0, "maximum": 1 }
  }
}
```

**Runtime Validation in the ETL Pipeline:**
- Validate every dataset against its schema at the start of each transformation step
- If validation fails, raise a `SchemaValidationError`, halt the pipeline, and log the diff between expected and actual schema

```python
import jsonschema
import json
import structlog

log = structlog.get_logger()

def validate_schema(data: dict, schema_path: str, layer: str) -> None:
    with open(schema_path) as f:
        schema = json.load(f)
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        log.error(
            "schema_validation_failed",
            layer=layer,
            schema_path=schema_path,
            field=e.path,
            message=e.message
        )
        raise
```

**Pydantic Models as the Canonical Source of Truth:**
- Define Pydantic models for each layer in `/src/models/` — these are the authoritative contracts between pipeline stages
- Auto-generate JSON Schema files from Pydantic models to keep them in sync

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class SilverTrack(BaseModel):
    track_id: str
    name: str
    artist_id: str
    danceability: float = Field(ge=0.0, le=1.0)
    energy: float = Field(ge=0.0, le=1.0)
    valence: float = Field(ge=0.0, le=1.0)
    tempo: float = Field(ge=0.0)
    genre: Optional[str] = None

    @field_validator("tempo")
    @classmethod
    def tempo_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Tempo must be positive")
        return v

# Auto-generate JSON Schema from Pydantic model
import json
schema = SilverTrack.model_json_schema()
with open("schemas/silver/tracks.json", "w") as f:
    json.dump(schema, f, indent=2)
```

**Schema Evolution Changelog (`/schemas/CHANGELOG.md`):**
```markdown
# Schema Changelog

## [1.2.0] - 2026-05-10
### Added
- `genre` field to SilverTrack (optional, nullable) — Spotify added genre tagging to track endpoint

## [1.1.0] - 2026-04-28
### Changed
- `tempo` field: changed type from int to float — Spotify now returns fractional BPM values
### Migration
- Existing silver data backfilled via `scripts/migrations/001_tempo_float.py`

## [1.0.0] - 2026-04-14
### Initial schema definitions for bronze, silver, and gold layers
```

**Implementation Steps:**
1. `pip install pydantic jsonschema`
2. Create `/src/models/` directory and define Pydantic models for all layers
3. Add schema validation step at the start of each Airflow task
4. Add a pre-commit hook or CI check that fails if Pydantic models and JSON Schema files are out of sync
5. Document the schema evolution process in README

---

### Feature 9: Data Contracts Between Pipeline Stages
**Description:** Explicit, versioned contracts that define what each pipeline stage promises to produce and what downstream stages are allowed to expect — enforced at runtime with Pydantic.

**Why This Matters:**
Data contracts are the interface design of data engineering. They shift the culture from "my pipeline writes whatever it wants" to "my pipeline is a service with a defined API." This is a concept currently transforming how data teams operate at companies like Airbnb and Netflix.

**Technical Details:**

**Contract Structure:**
Each data contract covers four concerns: the schema (via Pydantic model), the freshness SLA, the quality rules, and the owner.

```python
# /src/contracts/silver_tracks_contract.py
from dataclasses import dataclass
from typing import Type
from pydantic import BaseModel

@dataclass
class DataContract:
    name: str
    version: str
    owner: str
    producer: str          # Which pipeline stage writes this data
    consumer: str          # Which pipeline stage reads this data
    schema_model: Type[BaseModel]
    max_staleness_hours: int
    quality_rules: list[str]

silver_tracks_contract = DataContract(
    name="silver_tracks",
    version="1.2.0",
    owner="Ryan Peart",
    producer="bronze_to_silver_transform",
    consumer="silver_to_gold_transform",
    schema_model=SilverTrack,
    max_staleness_hours=25,
    quality_rules=[
        "no_nulls_on_required_fields",
        "danceability_between_0_and_1",
        "no_duplicate_track_ids",
        "tempo_positive"
    ]
)
```

**Contract Enforcement at Runtime:**
- At the start of each Airflow task, load and validate the relevant contract
- If the data violates any contract rule, raise a `ContractViolationError` and halt the task
- Log violations with full context for debugging

```python
from src.contracts.silver_tracks_contract import silver_tracks_contract
from src.models.silver import SilverTrack
import pandas as pd

def enforce_contract(df: pd.DataFrame, contract: DataContract) -> None:
    # Schema validation via Pydantic
    for record in df.to_dict("records"):
        contract.schema_model(**record)  # raises ValidationError on bad data

    # Quality rules
    assert df["track_id"].notna().all(), "Contract violation: null track_ids found"
    assert df["danceability"].between(0, 1).all(), "Contract violation: danceability out of range"
    assert not df["track_id"].duplicated().any(), "Contract violation: duplicate track_ids"
```

**Contract Registry:**
- Store all contracts in `/src/contracts/` with a `registry.py` that maps dataset names to contract objects
- The dashboard displays all active contracts, their versions, and last validation status

**Implementation Steps:**
1. Define `DataContract` dataclass in `/src/contracts/base.py`
2. Create one contract file per dataset in `/src/contracts/`
3. Add `enforce_contract()` call at the start and end of each Airflow task
4. Add contract version to every log event for traceability
5. Add contracts summary table to the dashboard

---

### Feature 10: Idempotent DAGs with Backfill Capability
**Description:** All Airflow DAGs are designed to be safely re-run for any historical date without creating duplicate data or corrupting state — and support a `backfill_date` parameter for historical reprocessing.

**Why This Matters:**
Idempotency is a foundational property of reliable pipelines. It means you can re-run any task at any time and get the same result — no duplicates, no gaps, no side effects. Every DE interview will eventually ask about this.

**Technical Details:**

**Idempotency Design Pattern:**
- Every write operation uses upsert logic (insert-or-replace) keyed on a natural key (e.g., `track_id + date`)
- Parquet files are written to date-partitioned paths and overwritten on re-run
- Azure Blob Storage writes use deterministic blob names derived from the logical date, not wall-clock time

```python
from airflow.decorators import dag, task
from datetime import datetime
import pendulum

@dag(
    schedule="@daily",
    start_date=pendulum.datetime(2026, 4, 1),
    catchup=True,  # Enable catchup so missed runs are automatically backfilled
    params={"backfill_date": None}  # Optional manual override
)
def etl_pipeline():

    @task
    def ingest_spotify_data(ds=None, params=None):
        # Use logical date (ds), not datetime.now()
        logical_date = params.get("backfill_date") or ds
        output_path = f"bronze/tracks/date={logical_date}/tracks.json"

        # Idempotent write: overwrite if exists
        write_to_blob(output_path, data, overwrite=True)

    @task
    def transform_bronze_to_silver(ds=None, params=None):
        logical_date = params.get("backfill_date") or ds
        input_path = f"bronze/tracks/date={logical_date}/tracks.json"
        output_path = f"silver/tracks/date={logical_date}/tracks.parquet"

        # Upsert pattern: read existing, merge, write back
        df = load_bronze(input_path)
        df_validated = validate_and_clean(df)
        write_parquet(output_path, df_validated, overwrite=True)

etl_pipeline()
```

**Backfill CLI Command:**
- Document a simple `airflow dags backfill` command in the README for re-running historical dates

```bash
# Re-run the ETL pipeline for the past 7 days
airflow dags backfill etl_pipeline \
  --start-date 2026-05-10 \
  --end-date 2026-05-17 \
  --reset-dagruns
```

**Duplicate Detection:**
- Before writing to the gold layer, run a deduplication check and log the number of duplicates found and removed
- Track deduplication metrics over time in the dashboard

**Implementation Steps:**
1. Audit all existing DAG tasks and replace `datetime.now()` with `ds` (logical date)
2. Change all write operations to use `overwrite=True` or upsert logic
3. Enable `catchup=True` on all DAGs
4. Add a `backfill_date` param to all DAGs
5. Add a deduplication step before every gold layer write
6. Document the backfill procedure in README with example commands

---

### Feature 11: SLA Monitoring & Alerting
**Description:** Formal SLA definitions for each pipeline stage with automated alerting when deadlines are missed — demonstrating awareness of downstream data dependencies.

**Why This Matters:**
A pipeline that runs eventually is not the same as a pipeline that can be depended on. SLA monitoring shows you understand that data products have consumers with their own deadlines — a dashboard that needs to be ready before business hours, a model that needs to be retrained before Monday's recommendations go out.

**Technical Details:**

**SLA Definitions:**
Define expected completion times for each pipeline stage. If a stage misses its SLA, log a warning and (optionally) send an alert.

```python
# /src/sla/definitions.py
from dataclasses import dataclass
from datetime import time

@dataclass
class PipelineSLA:
    task_name: str
    must_complete_by: time   # UTC
    downstream_dependency: str
    alert_channel: str       # "log" | "email" | "slack"

SLAS = [
    PipelineSLA(
        task_name="ingest_spotify_data",
        must_complete_by=time(4, 0),   # 4:00 AM UTC
        downstream_dependency="bronze_to_silver_transform",
        alert_channel="log"
    ),
    PipelineSLA(
        task_name="bronze_to_silver_transform",
        must_complete_by=time(5, 0),   # 5:00 AM UTC
        downstream_dependency="silver_to_gold_transform",
        alert_channel="log"
    ),
    PipelineSLA(
        task_name="gold_layer_ready",
        must_complete_by=time(6, 0),   # 6:00 AM UTC — dashboard must be fresh by this time
        downstream_dependency="analytics_dashboard",
        alert_channel="log"
    ),
    PipelineSLA(
        task_name="recommendation_model_retrain",
        must_complete_by=time(3, 0),   # 3:00 AM UTC every Monday
        downstream_dependency="recommendations_api",
        alert_channel="log"
    ),
]
```

**Airflow SLA Miss Callbacks:**
- Use Airflow's built-in `sla_miss_callback` to trigger alerts when tasks exceed their SLA

```python
from airflow import DAG
from datetime import timedelta
import structlog

log = structlog.get_logger()

def sla_miss_callback(dag, task_list, blocking_task_list, slas, blocking_tis):
    for sla in slas:
        log.warning(
            "sla_miss",
            task_id=sla.task_id,
            dag_id=dag.dag_id,
            execution_date=str(sla.execution_date),
            alert="Gold layer not ready by 06:00 UTC — downstream dashboard may show stale data"
        )

dag = DAG(
    "etl_pipeline",
    sla_miss_callback=sla_miss_callback,
    default_args={"sla": timedelta(hours=6)}
)
```

**SLA Dashboard Panel:**
- Display a rolling 7-day SLA compliance table in the dashboard
- Color-code: green (on time), yellow (within 30 min of SLA), red (SLA missed)
- Show the downstream impact note for any missed SLA (e.g., "Dashboard may show data from prior day")

**Implementation Steps:**
1. Define all SLA targets in `/src/sla/definitions.py`
2. Add `sla_miss_callback` to all production DAGs
3. Log SLA miss events with `structlog` using the `sla_miss` event key
4. Add SLA compliance panel to the dashboard
5. Document SLA definitions and escalation path in README

---

### Feature 12: Cloud Cost Monitoring & Reporting
**Description:** A lightweight cost tracking layer that measures actual Azure spend against estimates, documents resource right-sizing decisions, and surfaces cost data in the dashboard.

**Why This Matters:**
Data engineers are frequently the source of unexpected cloud bills. Showing that you proactively track costs and make deliberate right-sizing decisions signals production maturity. It also makes for a compelling blog post section: "here's what I expected to spend and here's what I actually spent."

**Technical Details:**

**Azure Cost Tracking:**
- Enable Azure Cost Management on the resource group
- Use the Azure Cost Management REST API to pull daily spend by service
- Store cost data as a CSV in Blob Storage (updated daily via Azure Function)

```python
# /src/monitoring/cost_tracker.py
import requests
from datetime import datetime, timedelta

def fetch_daily_costs(subscription_id: str, resource_group: str, token: str) -> dict:
    """Fetch daily costs from Azure Cost Management API."""
    url = f"https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.CostManagement/query?api-version=2023-03-01"

    body = {
        "type": "Usage",
        "timeframe": "LastMonth",
        "dataset": {
            "granularity": "Daily",
            "aggregation": {"totalCost": {"name": "Cost", "function": "Sum"}},
            "grouping": [{"type": "Dimension", "name": "ServiceName"}]
        }
    }

    response = requests.post(url, json=body, headers={"Authorization": f"Bearer {token}"})
    return response.json()
```

**Cost Comparison Notebook:**
- Maintain a Jupyter notebook (`/notebooks/cost_analysis.ipynb`) that compares:
  - Estimated monthly cost (from PRD) vs. actual Azure spend
  - Cost per pipeline run
  - Cost per 1,000 recommendations served
  - Breakdown by service: Blob Storage, App Service, Azure Functions

**Right-Sizing Documentation:**
- Document at least 3 resource right-sizing decisions in the README with before/after cost impact:
  - Example: "Chose DuckDB over Azure SQL Database for the silver layer — saves ~$5/month with equivalent query performance at this data volume"
  - Example: "Used Azure Functions Consumption plan instead of App Service for ingestion — $0 cost at low invocation frequency"
  - Example: "Parquet + DuckDB vs. Azure SQL: Parquet files in Blob Storage cost ~$0.02/GB vs. $4.99/month minimum for Azure SQL"

**Cost Dashboard Panel:**
- Show current month spend by service (bar chart)
- Show estimated vs. actual cumulative spend (line chart)
- Display cost per pipeline run as a KPI metric
- Include a "Total project cost to date" counter

**Implementation Steps:**
1. Enable Azure Cost Management on the resource group (free feature)
2. Create `/src/monitoring/cost_tracker.py` to query Cost Management API
3. Schedule a daily Azure Function to fetch and store cost data to Blob Storage
4. Create `/notebooks/cost_analysis.ipynb` with estimate vs. actual comparison
5. Add cost panel to the dashboard
6. Document right-sizing decisions in README under "Infrastructure Decisions"

---

## 5. User Stories

### For Recruiters/Hiring Managers

**US-1:** As a hiring manager, I want to see a live demo URL so I can quickly evaluate the candidate's work without setup.

**US-2:** As a technical recruiter, I want to read a clear README with architecture diagrams so I can understand the system design at a glance.

**US-3:** As a data engineering manager, I want to review the codebase and see data quality checks so I can assess production-readiness thinking.

**US-4:** As an interviewer, I want to see metrics and monitoring so I can ask questions about operational considerations.

**US-5:** As a data engineering hiring manager, I want to see a data lineage graph so I can assess whether the candidate understands data provenance and pipeline debugging.

**US-6:** As a senior DE interviewer, I want to see data contracts between pipeline stages so I can evaluate whether the candidate thinks about interfaces and downstream consumers, not just pipeline execution.

**US-7:** As a hiring manager evaluating cost awareness, I want to see a cloud cost breakdown so I can assess whether the candidate makes deliberate infrastructure decisions.

**US-8:** As an interviewer, I want to see SLA definitions and compliance tracking so I can ask about operational considerations and on-call responsibilities.

---

### For Data Pipeline (Technical)

**US-9:** As the ETL pipeline, I want to incrementally load only new Spotify data so I minimize API calls and costs.

**US-10:** As the data quality module, I want to validate schemas and detect anomalies so bad data doesn't corrupt downstream models.

**US-11:** As the orchestrator, I want to retry failed tasks with exponential backoff so transient errors don't break the pipeline.

**US-12:** As the pipeline operator, I want to re-run any DAG task for a past date without creating duplicate records so I can safely recover from failures.

**US-13:** As the ETL pipeline, I want to validate incoming data against a Pydantic contract at each stage boundary so schema violations are caught immediately rather than silently corrupting downstream models.

**US-14:** As the observability layer, I want to emit structured log events at every key checkpoint so I can query pipeline behavior and build alerts without parsing unstructured text.

**US-15:** As the SLA monitor, I want to trigger a logged warning when a pipeline stage misses its completion deadline so downstream consumers are aware of potential data staleness.

---

### For End Users (Demo Context)

**US-16:** As a music listener, I want to receive 10 personalized track recommendations so I can discover new music matching my taste.

**US-17:** As a user, I want to see why a track was recommended (e.g., "Similar to artists you like") so recommendations feel transparent.

**US-18:** As a dashboard viewer, I want to explore audio feature distributions so I can understand what makes recommendations work.

**US-19:** As a dashboard viewer, I want to see a data lineage graph so I understand how my recommendations were produced and what data they depend on.

**US-20:** As a dashboard viewer, I want to see whether the pipeline met its SLA today so I know if the data I'm looking at is current.

**US-21:** As a dashboard viewer, I want to see this month's cloud infrastructure cost so I understand the economics of running this system.

---

## 6. Success Metrics

### Technical Metrics (Primary Focus)

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Pipeline Reliability** | 95%+ successful runs | Airflow task success rate |
| **Data Freshness** | Updated within 24 hours | Timestamp of latest ingestion |
| **API Response Time** | < 500ms (p95) | FastAPI logging/monitoring |
| **Data Quality Score** | 90%+ passing checks | Great Expectations or custom tests |
| **Code Coverage** | > 70% | pytest coverage reports |

### Model Performance Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Recommendation Precision@10** | > 0.20 | Offline evaluation on test set |
| **Genre Diversity** | > 3 genres in top 10 | Diversity calculation |
| **Cold Start Coverage** | > 80% of tracks | % tracks that can be recommended |

### Observability & Reliability Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **SLA Compliance Rate** | > 90% of runs meet deadline | SLA miss callback logs |
| **Lineage Coverage** | 100% of DAG tasks emit lineage events | Marquez task count vs. Airflow task count |
| **Structured Log Coverage** | 100% of pipeline modules use structlog | Code review / linting rule |
| **Schema Validation Pass Rate** | > 98% of runs pass all contract checks | ContractViolationError log count |
| **Idempotency Verified** | All DAGs pass a backfill re-run test | CI test: re-run last 3 days, assert no duplicates |

### Data Contract & Schema Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Schema Coverage** | JSON Schema defined for all 5 datasets | File count in `/schemas/` |
| **Contract Coverage** | Data contract defined for all stage boundaries | Contract registry completeness |
| **Schema Changelog Entries** | At least 1 documented evolution | `/schemas/CHANGELOG.md` |
| **Pydantic / JSON Schema Sync** | 0 drift between Pydantic models and JSON Schema files | CI check: auto-generate and diff |

### Cost Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Actual vs. Estimated Cost** | Within 20% of PRD estimate | Azure Cost Management vs. PRD table |
| **Cost per Pipeline Run** | < $0.05 | Total monthly cost / run count |
| **Right-Sizing Decisions Documented** | At least 3 | README "Infrastructure Decisions" section |

### Portfolio Impact Metrics (Qualitative)

- GitHub stars/forks received
- Blog post engagement (views, shares)
- Recruiter inquiries mentioning the project
- Interview conversations driven by project

---

## 7. Open Questions

### Technical Decisions

- **Marquez hosting:** Self-host via Docker (simpler) vs. Marquez Cloud free tier (demonstrates cloud deployment)?
- **SLA alerting channel:** Log-only sufficient for portfolio, or worth setting up a free Slack webhook to demonstrate real alerting?
- **Pydantic v1 vs. v2:** Scikit-learn compatibility constraints may require Pydantic v1; evaluate before committing to v2 patterns.

### Scope Questions

- **Schema registry tooling:** JSON Schema files in git (simple, sufficient) vs. a dedicated registry like Confluent Schema Registry (complex, overcomplicated for this scale)?
- **Cost tracking:** Is the Azure Cost Management API accessible on the free tier, or does it require a paid subscription?

### Data Questions

---

## 8. Milestones

### Milestone 1: Foundation & Data Ingestion (Week 1)
**Deliverables:**
- [ ] GitHub repository set up with README
- [ ] Azure account configured with resource group
- [ ] Spotify API integration working
- [ ] Azure Function that fetches track data to Blob Storage
- [ ] Initial architecture diagram
- [ ] `structlog` integrated as the logging standard across all modules
- [ ] Azure Cost Management enabled on resource group

**Success Criteria:** Can extract 100+ tracks with audio features and store as JSON in Azure; all log output is structured JSON via structlog.

---

### Milestone 2: ETL Pipeline & Data Models (Week 2)
**Deliverables:**
- [ ] Bronze → Silver transformation (Python/pandas)
- [ ] Silver → Gold transformation migrated to dbt SQL models (dim_tracks, dim_artists, fact_listening_history)
- [ ] dbt project initialised under `/dbt/` with DuckDB profile
- [ ] dbt staging models reading from silver Parquet files
- [ ] dbt schema tests: `unique`, `not_null`, `accepted_range` on all key columns
- [ ] `dbt run && dbt test` wired into the Airflow DAG after `_bronze_to_silver`
- [ ] `dbt docs generate` screenshot added to README lineage section
- [ ] Pydantic models defined for all 5 datasets in `/src/models/`
- [ ] JSON Schema files generated from Pydantic models in `/schemas/`
- [ ] `DataContract` dataclass defined and contracts created for all stage boundaries
- [ ] `enforce_contract()` called at the start and end of each Airflow task
- [ ] All DAG tasks updated to use logical date (`ds`) instead of `datetime.now()`
- [ ] All write operations updated to be idempotent (overwrite or upsert)
- [ ] `catchup=True` and `backfill_date` param added to all DAGs

**Success Criteria:** Automated pipeline processes raw Spotify data into analytics-ready gold tables via dbt; `dbt test` passes with 0 failures; re-running any task for a past date produces identical output with no duplicates; contract violations raise and halt the pipeline within 1 second of a bad record.

---

### Milestone 3: Recommendation Engine, API & Observability (Week 3)
**Deliverables:**
- [ ] Trained collaborative filtering model
- [ ] Model evaluation metrics calculated
- [ ] FastAPI service with 4 core endpoints
- [ ] API deployed to Azure App Service
- [ ] OpenAPI documentation live
- [ ] Marquez deployed (Docker or cloud) and `OPENLINEAGE_URL` configured
- [ ] All Airflow DAGs instrumented with `openlineage-airflow` provider
- [ ] Lineage graph visible in Marquez UI showing full bronze → gold flow
- [ ] Azure Monitor workspace configured; structured logs forwarded
- [ ] SLA definitions created in `/src/sla/definitions.py`
- [ ] `sla_miss_callback` added to all production DAGs
- [ ] Daily cost fetch Azure Function deployed and storing cost data to Blob Storage

**Success Criteria:** API returns 10 recommendations per user with <500ms latency; lineage graph shows all 5 datasets and producing tasks; an intentionally failed DAG run triggers a logged SLA miss warning.

---

### Milestone 4: Dashboard, Schema Registry & Portfolio Polish (Week 4)
**Deliverables:**
- [ ] Interactive dashboard with 5+ core visualizations
- [ ] Lineage graph panel in dashboard (via Marquez REST API)
- [ ] SLA compliance table (rolling 7-day) in dashboard
- [ ] Cloud cost panel in dashboard (actual vs. estimated)
- [ ] Active contracts summary table in dashboard
- [ ] `/schemas/CHANGELOG.md` with at least one documented schema evolution
- [ ] CI check added: Pydantic models and JSON Schema files must be in sync
- [ ] `/notebooks/cost_analysis.ipynb` with estimate vs. actual comparison
- [ ] At least 3 right-sizing decisions documented in README under "Infrastructure Decisions"
- [ ] Backfill procedure documented in README with example `airflow dags backfill` command
- [ ] Dashboard deployed publicly
- [ ] Architecture diagram finalized
- [ ] README with setup instructions and demo links
- [ ] Code cleaned, commented, tested
- [ ] Demo video or GIF created
- [ ] Blog post drafted

**Success Criteria:** Complete portfolio-ready project with public demo URL, lineage graph, SLA panel, cost panel, and comprehensive docs; blog post draft covers at least one of: idempotency design, schema evolution story, or cost right-sizing decisions.

---

### Milestone 5: Optional Enhancements (Post-MVP)
**Future Ideas:**
- Real-time streaming with Azure Event Hubs
- A/B testing framework for recommendation algorithms
- Integration with additional music APIs (Last.fm, MusicBrainz)
- User feedback loop and retraining pipeline
- Cost optimization analysis and report
- Slack webhook integration for real SLA miss alerting
- Automated schema compatibility checks (breaking vs. non-breaking change detection)

---

## Appendix

### Tech Stack Summary
- **Cloud:** Azure (Blob Storage, Functions, App Service, SQL Database, Monitor, Cost Management)
- **Orchestration:** Airflow
- **Transformation (Gold layer):** dbt Core + DuckDB
- **Processing:** Python (pandas)
- **ML:** Scikit-learn, Surprise (collaborative filtering)
- **API:** FastAPI
- **Dashboard:** Streamlit or React
- **CI/CD:** GitHub Actions
- **IaC:** Azure CLI or Terraform (optional)
- **Observability:** OpenLineage, Marquez, structlog, Azure Monitor
- **Data Contracts & Schemas:** Pydantic, JSON Schema, jsonschema
- **Cost Monitoring:** Azure Cost Management API

### Estimated Costs
- Azure Free Tier: $0 (first 12 months)
- After free tier: ~$5-10/month (Blob Storage + App Service)
- Azure Monitor (basic logs): $0 (5GB/month free)
- Streamlit Community Cloud: $0
- Marquez (self-hosted Docker): $0

### Timeline
- **Total Duration:** 4 weeks (part-time)
- **Effort:** 10-15 hours/week
- **Target Completion:** June 9, 2026
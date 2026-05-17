# MySpotify Insights

A data engineering portfolio project featuring an end-to-end ETL pipeline and ML-powered music recommendation engine using Spotify data and Azure cloud services.

## Project Overview

MySpotify Insights demonstrates production-grade data engineering practices by building a complete music recommendation system that:
- Ingests personal Spotify listening history via API
- Processes data through a multi-stage ETL pipeline (Bronze → Silver → Gold)
- Trains a hybrid recommendation model (collaborative + content-based filtering)
- Serves recommendations via a REST API
- Visualizes insights through an interactive dashboard

**Tech Stack:** Python | Azure Blob Storage | FastAPI | Scikit-learn | Streamlit | Prefect (optional)

## Architecture

The system follows a **medallion data lakehouse pattern** with data flowing one-way through each layer before being served by the API and dashboard.

```text
Spotify API
    │
    ▼
src/ingestion/spotify_client.py    →  data/bronze/   (raw JSON, manifest-tracked)
    │
    ▼
src/etl/bronze_to_silver.py        →  data/silver/   (cleaned, normalized, validated Parquet)
    │
    ▼
src/etl/silver_to_gold.py          →  data/gold/     (dimensional model, composite scores)
    │
    ├──► src/models/train.py        (collaborative + content-based recommendation model)
    │         │
    │         ▼
    │    src/models/predict.py      (inference, called by API routes)
    │
    └──► src/api/routes.py          (FastAPI endpoints)
              │
              ▼
         src/dashboard/app.py       (Streamlit analytics dashboard)
```

## Setup

### Prerequisites
- Python 3.11+
- Spotify Developer account (for API credentials)
- Azure account (optional — for cloud storage)

### Installation

```bash
# Clone and enter the repo
git clone https://github.com/RGPeart/myspotify-insights.git
cd myspotify-insights

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Edit .env with your Spotify client ID/secret and (optionally) Azure connection string
```

### Running the pipeline

```bash
# 1. Set up and run Airflow (using Docker Compose)
# Ensure Docker is running.
# Create a dags directory if it doesn't exist and copy the DAG file to your Airflow DAGs folder (e.g., ./dags/spotify_etl_dag.py)
# From the project root, navigate into the 'airflow' directory:
# cd airflow
# Initialize the Airflow database and create a user (first time only):
# docker compose up airflow-init
# Start Airflow services:
# docker compose up -d
# Access Airflow UI at http://localhost:8080 (default credentials: airflow/airflow)
# Unpause the 'spotify_etl_pipeline' DAG and trigger it manually or await schedule.

# 2. Ingest raw data from Spotify API → data/bronze/
#    (Orchestrated by Airflow: Trigger the 'spotify_etl_pipeline' DAG in Airflow UI)

# 3. Bronze → Silver (clean, normalize, validate)
#    (Orchestrated by Airflow: Trigger the 'spotify_etl_pipeline' DAG in Airflow UI)

# 4. Silver → Gold (dimensional model, composite popularity scores)
#    (Orchestrated by Airflow: Trigger the 'spotify_etl_pipeline' DAG in Airflow UI)

# 5. Train recommendation model
python -m src.models.train

# 6. Start API server (in a dedicated terminal)
bash -c "source .venv/bin/activate && PYTHONPATH=. uvicorn src/api/main:app --reload --host 0.0.0.0 --port 8001"

# 7. Start dashboard (in another dedicated terminal)
streamlit run src/dashboard/app.py
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run a single test
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
| Feature 6: CI/CD & cloud deployment | Planned | — |

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

# Setup Guide

This guide walks you through setting up and running every part of the MySpotify Insights project from scratch. No prior familiarity with the codebase is assumed — just follow each section in order.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the Repository](#2-clone-the-repository)
3. [Create a Python Virtual Environment](#3-create-a-python-virtual-environment)
4. [Install Dependencies](#4-install-dependencies)
5. [Get Spotify API Credentials](#5-get-spotify-api-credentials)
6. [Configure Environment Variables](#6-configure-environment-variables)
7. [Feature 1 — Data Ingestion](#7-feature-1--data-ingestion)
8. [Feature 2 — ETL Pipeline](#8-feature-2--etl-pipeline)
9. [Feature 3 — Recommendation Model](#9-feature-3--recommendation-model) *(coming soon)*
10. [Feature 4 — REST API](#10-feature-4--rest-api) *(coming soon)*
11. [Feature 5 — Analytics Dashboard](#11-feature-5--analytics-dashboard) *(coming soon)*
12. [Running Tests](#12-running-tests)
13. [Optional: Azure Cloud Storage](#13-optional-azure-cloud-storage)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Prerequisites

Before starting, make sure you have the following installed on your machine:

| Tool | Minimum version | How to check |
|---|---|---|
| Python | 3.11+ | `python --version` |
| Git | any recent | `git --version` |

You will also need:

- A **Spotify account** (free or paid) to access the Spotify Developer Portal
- A **GitHub account** if you want to push changes

You do **not** need an Azure account to run the project locally. Azure Blob Storage is optional and used only for cloud backups of the raw data.

---

## 2. Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/RGPeart/myspotify-insights.git
cd myspotify-insights
```

All commands in the rest of this guide should be run from inside this `myspotify-insights` folder.

---

## 3. Create a Python Virtual Environment

A virtual environment keeps project dependencies isolated from the rest of your system.

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt, which means the environment is active. Every time you open a new terminal, you need to re-run the `activate` command above before working on this project.

---

## 4. Install Dependencies

With the virtual environment active, install all required packages:

```bash
pip install -r requirements.txt
```

This installs libraries including `pandas`, `pyarrow`, `spotipy`, `fastapi`, `streamlit`, and others. It may take a minute or two.

To verify the install worked:

```bash
python -c "import pandas, spotipy, pyarrow; print('All good!')"
```

---

## 5. Get Spotify API Credentials

The ingestion pipeline connects to the Spotify Web API using OAuth. You need to register a free developer application to get credentials.

1. Go to [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) and log in with your Spotify account.

2. Click **Create app**.

3. Fill in any name and description (e.g. "MySpotify Insights"). Set the **Redirect URI** to:
   ```
   http://localhost:5000/callback
   ```
   Accept the terms and click **Save**.

4. On your new app's page, click **Settings**. You will see your **Client ID** and **Client Secret** — keep this page open for the next step.

---

## 6. Configure Environment Variables

The project reads secrets (API keys, connection strings) from a `.env` file that lives at the root of the repository and is never committed to Git.

Copy the provided template:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` in any text editor and fill in your values:

```
# Spotify API Credentials (from step 5)
SPOTIFY_CLIENT_ID=paste_your_client_id_here
SPOTIFY_CLIENT_SECRET=paste_your_client_secret_here
SPOTIFY_REDIRECT_URI=http://localhost:5000/callback

# Azure Storage — leave as-is if running locally only
AZURE_STORAGE_CONNECTION_STRING=your_connection_string_here
AZURE_STORAGE_CONTAINER_NAME=spotify-data

# API Configuration — defaults are fine for local development
API_HOST=0.0.0.0
API_PORT=5000
```

Save the file. You do **not** need to fill in the Azure variables to run Features 1 and 2 locally.

---

## 7. Feature 1 — Data Ingestion

**What it does:** Connects to the Spotify Web API, fetches tracks, artists, and audio features, and saves them as raw JSON files under `data/bronze/`.

### Run it

```bash
python -m src.ingestion.spotify_client
```

### What happens

The first time you run this, your browser will open and ask you to log in to Spotify and authorise the app. After you confirm, the page will redirect to `localhost:5000/callback` — the page will appear to fail (that is expected), but the token has been captured.

The script will then:
- Fetch tracks across several genre categories
- Fetch audio features for each track
- Fetch artist metadata
- Write raw JSON files to:
  ```
  data/
  └── bronze/
      ├── tracks/YYYY-MM-DD/tracks_YYYYMMDDTHHMMSSZ.json
      ├── audio_features/YYYY-MM-DD/audio_features_...json
      └── artists/YYYY-MM-DD/artists_...json
  ```
- Write a manifest file to `data/bronze/manifest.json` that records which data has already been ingested (so re-running skips already-fetched batches)

### Verify it worked

```bash
# Windows
dir data\bronze\tracks

# macOS / Linux
ls data/bronze/tracks/
```

You should see a dated folder containing a JSON file. Open it to confirm it contains track objects.

---

## 8. Feature 2 — ETL Pipeline

**What it does:** Reads the raw JSON from the bronze layer, cleans and normalises the data, runs data quality checks, and saves structured Parquet files to `data/silver/` and `data/gold/`.

You must complete Feature 1 first so there is data in `data/bronze/` to process.

### Option A — Run all stages at once (recommended)

```bash
python -m src.etl.pipeline
```

This runs Bronze → Silver → Gold sequentially and prints a summary at the end.

### Option B — Run each stage separately

```bash
# Stage 1: Bronze → Silver
python -m src.etl.bronze_to_silver

# Stage 2: Silver → Gold
python -m src.etl.silver_to_gold
```

### What Bronze → Silver does

Reads from `data/bronze/` and writes to `data/silver/`:

| Input (raw JSON) | Output (Parquet) | Key transformations |
|---|---|---|
| `bronze/tracks/` | `silver/tracks.parquet` | Extracts fields, parses release dates, deduplicates on `track_id` |
| `bronze/audio_features/` | `silver/audio_features.parquet` | Normalises all features to [0, 1] range |
| `bronze/artists/` | `silver/artists.parquet` | Categorises genres, joins genre strings |

Data quality checks run after each transformation. If a check fails (e.g. required columns are missing or duplicates are found), the pipeline raises an error and stops — no bad data is written to disk.

### What Silver → Gold does

Reads from `data/silver/` and writes to `data/gold/`:

| Output (Parquet) | Description |
|---|---|
| `gold/dim_tracks.parquet` | Track dimension table with composite popularity score |
| `gold/dim_artists.parquet` | Artist dimension table |
| `gold/fact_audio_features.parquet` | Fact table with normalised audio features linked to tracks and artists |

The composite popularity score is calculated as:
```
composite_popularity = 0.6 × (track_popularity / 100) + 0.4 × (artist_popularity / 100)
```

### Verify it worked

```bash
python -c "
import pandas as pd
tracks = pd.read_parquet('data/gold/dim_tracks.parquet')
print(f'Gold dim_tracks: {len(tracks)} rows')
print(tracks[['track_id', 'name', 'primary_genre', 'composite_popularity']].head())
"
```

### Optional: Prefect orchestration

Prefect is an orchestration tool that adds scheduling, retries, and a monitoring UI. It is optional — the pipeline runs fine without it.

To use Prefect:

```bash
pip install "prefect>=3.0"
prefect server start          # opens the UI at http://localhost:4200
python -m src.etl.pipeline    # now tracked by Prefect
```

---

## 9. Feature 3 — Recommendation Model

> **Status: Coming soon.**

This feature will train a hybrid recommendation model (collaborative filtering + content-based) using the gold-layer data.

Once implemented, you will run it with:

```bash
python -m src.models.train
```

---

## 10. Feature 4 — REST API

> **Status: Coming soon.**

This feature will expose a FastAPI service that serves track recommendations and metadata.

Once implemented, you will start it with:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

The interactive API docs will be available at `http://localhost:8000/docs`.

---

## 11. Feature 5 — Analytics Dashboard

> **Status: Coming soon.**

This feature will provide a Streamlit dashboard showing pipeline metrics, recommendation performance, and audio feature visualisations.

Once implemented, you will start it with:

```bash
streamlit run src/dashboard/app.py
```

---

## 12. Running Tests

The test suite covers the data quality framework, all ETL transforms, and the pipeline orchestration.

```bash
# Run all tests
pytest

# Run with a coverage report
pytest --cov=src tests/

# Run a specific test file
pytest tests/test_etl.py

# Run a single test by name
pytest tests/test_etl.py::TestBronzeToSilverRun::test_run_end_to_end -v
```

All tests are in-memory and do not require real Spotify credentials or actual data files.

---

## 13. Optional: Azure Cloud Storage

By default, all data is stored locally under the `data/` folder. If you want to mirror data to Azure Blob Storage:

1. Create a free Azure account at [azure.microsoft.com](https://azure.microsoft.com)
2. Create a **Storage Account** in the Azure Portal
3. Create a **container** inside it (e.g. `spotify-data`)
4. In the storage account's **Access keys** page, copy the connection string
5. Paste it into your `.env` file:
   ```
   AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
   AZURE_STORAGE_CONTAINER_NAME=spotify-data
   ```

When valid Azure credentials are present, the ingestion script will automatically upload each JSON batch to Blob Storage in addition to saving it locally.

---

## 14. Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

You are probably running `python src/etl/bronze_to_silver.py` directly instead of using the module flag. Always use:

```bash
python -m src.etl.bronze_to_silver   # correct
python src/etl/bronze_to_silver.py   # incorrect
```

### Spotify authentication browser does not open

If your environment cannot open a browser automatically, Spotipy will print a URL to the terminal. Copy and paste it into a browser manually, log in, and paste the full redirect URL back into the terminal when prompted.

### `DataQualityError` raised during ETL

The pipeline enforces data quality checks before writing any output. If a check fails, the full error message will tell you which table failed and why (e.g. `silver/tracks: FAILED (nulls={'track_id': 3})`).

Common causes:
- Incomplete bronze data — re-run ingestion, then retry
- Corrupted JSON file — delete the file from `data/bronze/` and re-run ingestion

### `FileNotFoundError` when reading silver/gold data

Bronze → Silver must run before Silver → Gold, and ingestion must run before Bronze → Silver. Run the stages in order:

```bash
python -m src.ingestion.spotify_client
python -m src.etl.bronze_to_silver
python -m src.etl.silver_to_gold
```

### Tests fail with import errors

Make sure your virtual environment is active (`(venv)` in prompt) and dependencies are installed:

```bash
pip install -r requirements.txt
```

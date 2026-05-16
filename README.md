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
# 1. Ingest raw data from Spotify API → data/bronze/
python -m src.ingestion.spotify_client

# 2. Bronze → Silver (clean, normalize, validate)
python -m src.etl.bronze_to_silver

# 3. Silver → Gold (dimensional model, composite popularity scores)
python -m src.etl.silver_to_gold

# Or run the full pipeline in one step (with optional Prefect orchestration)
python -m src.etl.pipeline

# 4. Train recommendation model
python -m src.models.train

# 5. Start API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 6. Start dashboard
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
| Feature 2: ETL pipeline (Bronze → Silver → Gold) | In review | PR #9 |
| Feature 3: Recommendation model | Planned | — |
| Feature 4: FastAPI service | Planned | — |
| Feature 5: Streamlit dashboard | Planned | — |
| Feature 6: CI/CD & cloud deployment | Planned | — |

## Branching Strategy

```
feature/* → release → main
```

Feature branches are opened against `release`. Once reviewed and merged to `release`, a consolidated PR moves the changes into `main`.

## Claude Code Setup

To use Claude Code with Google Gemini via LiteLLM:

1.  **Install LiteLLM:**
    ```bash
    pip install litellm
    ```

2.  **Configure LiteLLM for Gemini:**
    Create a `litellm_config.yml` file in the root of your project:
    ```yaml
    # litellm_config.yml
    model_list:
      - model_name: gemini-2.5-flash
        litellm_params:
          model: gemini/gemini-2.5-flash
          api_key: os.environ/GEMINI_API_KEY
    ```

3.  **Get a Gemini API Key:**
    Obtain a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

4.  **Add API Key to .env:**
    Add your Gemini API key to your `.env` file:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```

5. **Start the LiteLLM Proxy in your terminal:**
    Run the LiteLLM Proxy within it's own terminal to run alongside your Claude Code session:
    ```bash
    litellm --config litellm_config.yaml --port 4000
    ```

6.  **Launch Claude Code in another terminal window:**
    Run the following command in a separate, new terminal window to start using Claude Code with Google Geminia as the model:
    ```bash
    set -a && source .env && set +a 
    claude
    ```
    _You might get a message to use the detected ANTHROPIC API KEY key in the environment (the DUMMY key). Enter "No" to use the correct Google AI Studio API Key_

Now, Claude Code will use Google Gemini (via LiteLLM) for its responses, allowing you to leverage Gemini's capabilities within your development workflow.

## Author

**Ryan Peart**
- Portfolio: [rgpeart.github.io/portfolio](https://rgpeart.github.io/portfolio)
- LinkedIn: [Ryan Peart](https://www.linkedin.com/in/ryan-peart/)
- GitHub: [@RGPeart](https://github.com/RGPeart)

## License

MIT License — see [LICENSE](LICENSE) for details.

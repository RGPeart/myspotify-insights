# See README.md for instructions on how to run this Streamlit application.

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")

# --- Constants ---
_DUMMY_AUDIO_FEATURES_DF = pd.DataFrame({
    "danceability": [0.7, 0.8, 0.6, 0.9],
    "energy": [0.6, 0.7, 0.8, 0.5],
    "valence": [0.5, 0.6, 0.7, 0.8],
    "tempo": [120, 130, 110, 140],
    "acousticness": [0.1, 0.2, 0.3, 0.05],
    "instrumentalness": [0.01, 0.0, 0.02, 0.03],
    "liveness": [0.15, 0.25, 0.1, 0.3],
    "speechiness": [0.05, 0.03, 0.07, 0.04],
})

st.set_page_config(layout="wide", page_title="MySpotify Insights Dashboard")

st.title("🎶 MySpotify Insights Dashboard")

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_data(endpoint: str, params: dict = None):
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from API endpoint '{endpoint}': {e}")
        return None

@st.cache_data(ttl=3600) # Cache user IDs for an hour
def fetch_user_ids():
    users_data = fetch_data("users")
    if users_data:
        if "user_ids" in users_data:
            return sorted(users_data["user_ids"])
        else:
            st.error("API response for /users is missing 'user_ids'.")
    return []

@st.cache_data(ttl=3600) # Cache track IDs for an hour
def fetch_track_ids():
    tracks_data = fetch_data("tracks/ids")
    if tracks_data:
        if "track_ids" in tracks_data:
            return tracks_data["track_ids"]
        else:
            st.error("API response for /tracks/ids is missing 'track_ids'.")
    return []

# -----------------------------------------------------------------------------
# Dashboard Sections
# -----------------------------------------------------------------------------

st.header("📈 Overall Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Tracks in Gold", "N/A", help="Run the ETL pipeline to populate this value")
with col2:
    st.metric("Total Artists in Gold", "N/A", help="Run the ETL pipeline to populate this value")
with col3:
    api_health_data = fetch_data("health")
    st.metric("API Health", "OK" if api_health_data else "DOWN", help="Status of the FastAPI service")
    if not api_health_data:
        st.error(f"Error fetching API health data. API may be down or unreachable.")

st.markdown("---")

st.header("📊 Recommendation Engine Performance (Offline Metrics)")
st.write("*(These metrics would typically come from an MLOps platform or dedicated evaluation pipeline)*")

rec_metrics_data = {
    "Metric": ["Precision@10", "Genre Diversity", "Cold Start Coverage"],
    "Value": [0.25, 3.5, 0.85], # Placeholder values
    "Target": ["> 0.20", "> 3 genres", "> 80%"]
}
st.dataframe(pd.DataFrame(rec_metrics_data), use_container_width=True)

st.markdown("---")

st.header("🎶 User Recommendations (Live Demo)")

user_ids = fetch_user_ids()

user_id_example = None
if user_ids:
    user_id_example = st.selectbox("Select User ID", options=user_ids, index=0)
else:
    st.warning("No user IDs available from the API. Please run the ETL and model training.")

num_recs = st.slider("Number of Recommendations", min_value=1, max_value=20, value=10)

if st.button("Get Recommendations"):
    if user_id_example:
        recommendations = fetch_data(f"recommendations/{user_id_example}", params={"n": num_recs})
        if recommendations:
            if recommendations["recommendations"]:
                st.subheader(f"Recommendations for {user_id_example}")
                recs_df = pd.DataFrame(recommendations["recommendations"])
                st.dataframe(recs_df, use_container_width=True)

                # Display track details for one recommendation
                if not recs_df.empty:
                    first_track_id = recs_df.iloc[0]["track_id"]
                    st.subheader(f"Details for first recommendation ({first_track_id})")
                    track_detail = fetch_data(f"tracks/{first_track_id}")
                    if track_detail:
                        st.json(track_detail)
                    else:
                        st.error(f"Error fetching details for track ID: {first_track_id}")
            else:
                st.info(f"No recommendations found for user ID: {user_id_example}")
        else:
            st.error(f"Error fetching recommendations for user ID: {user_id_example}")
    else:
        st.warning("Please select a User ID.")

st.markdown("---")

st.header("🔊 Audio Features for a Sample Track")
st.write("*(Audio features for a single track from your ingested data)*")

all_track_ids = fetch_track_ids()

audio_features_df = pd.DataFrame() # Initialize an empty DataFrame
if all_track_ids:
    track_id_for_features = all_track_ids[0] # Using the first track ID for consistency in demo
    tracks_data = fetch_data(f"tracks/{track_id_for_features}")
    if tracks_data:
        if "audio_features" in tracks_data:
            audio_features_df = pd.DataFrame([tracks_data["audio_features"]])
        else:
            st.error(f"API response for /tracks/{{track_id}} is missing 'audio_features' for track ID: {track_id_for_features}. Showing dummy data.")
            audio_features_df = _DUMMY_AUDIO_FEATURES_DF.copy()
    else:
        st.error(f"Error fetching audio features for track ID: {track_id_for_features}. Showing dummy data.")
        audio_features_df = _DUMMY_AUDIO_FEATURES_DF.copy()
else:
    st.warning("No track IDs available from the API for audio feature visualization. Showing dummy data.")
    audio_features_df = _DUMMY_AUDIO_FEATURES_DF.copy()


if not audio_features_df.empty:
    feature_to_plot = st.selectbox(
        "Select Audio Feature to Visualize",
        options=audio_features_df.columns.tolist()
    )
    fig = px.histogram(audio_features_df, x=feature_to_plot, title=f"Distribution of {feature_to_plot}")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.header("🧪 Data Quality & Freshness")

# Placeholder for data quality report
dq_data = {
    "Check": ["Nulls in Track Name", "Valid Genres", "Duplicate Tracks"],
    "Status": ["PASS", "PASS", "FAIL"], # Placeholder
    "Details": ["0 nulls", "100% valid", "5 duplicates"]
}
st.dataframe(pd.DataFrame(dq_data), use_container_width=True)

st.write(f"Last Pipeline Run: N/A (Placeholder)")
st.write(f"Data Freshness: 1 day (Placeholder)")

st.markdown("---")

# -----------------------------------------------------------------------------
# Data Lineage Graph (Marquez)
# -----------------------------------------------------------------------------

MARQUEZ_URL = os.getenv("MARQUEZ_URL", "http://localhost:5002")
MARQUEZ_WEB_URL = os.getenv("MARQUEZ_WEB_URL", "http://localhost:3000")
MARQUEZ_NAMESPACE = os.getenv("OPENLINEAGE_NAMESPACE", "myspotify-insights")

st.header("🔗 Data Lineage")
st.write(
    "Pipeline lineage tracked via [OpenLineage](https://openlineage.io) and "
    "[Marquez](https://marquezproject.ai). "
    f"Marquez UI: `{MARQUEZ_WEB_URL}`"
)

# Static pipeline topology (always shown)
st.subheader("Pipeline Topology")
st.graphviz_chart("""
digraph pipeline {
    rankdir=LR;
    node [fontname="Helvetica", fontsize=11]
    edge [fontsize=10]

    "Spotify API" [shape=cylinder, style=filled, fillcolor="#a8d8ea"]

    "bronze/tracks"         [shape=note, style=filled, fillcolor="#ffe082", label="bronze/\\ntracks.json"]
    "bronze/artists"        [shape=note, style=filled, fillcolor="#ffe082", label="bronze/\\nartists.json"]
    "bronze/audio_features" [shape=note, style=filled, fillcolor="#ffe082", label="bronze/\\naudio_features.json"]

    "silver/tracks"   [shape=note, style=filled, fillcolor="#c8e6c9", label="silver/\\ntracks.parquet"]
    "silver/artists"  [shape=note, style=filled, fillcolor="#c8e6c9", label="silver/\\nartists.parquet"]

    "gold/dim_tracks"   [shape=note, style=filled, fillcolor="#b2dfdb", label="gold/\\ndim_tracks.parquet"]
    "gold/dim_artists"  [shape=note, style=filled, fillcolor="#b2dfdb", label="gold/\\ndim_artists.parquet"]
    "gold/fact_history" [shape=note, style=filled, fillcolor="#b2dfdb", label="gold/\\nfact_listening_history.parquet"]

    "_ingest_data"       [shape=box, style=filled, fillcolor="#f8bbd0", label="_ingest_data\\n(Airflow task)"]
    "_bronze_to_silver"  [shape=box, style=filled, fillcolor="#f8bbd0", label="_bronze_to_silver\\n(Airflow task)"]
    "_silver_to_gold"    [shape=box, style=filled, fillcolor="#f8bbd0", label="_silver_to_gold\\n(Airflow task)"]
    "recommendations_api" [shape=component, style=filled, fillcolor="#ce93d8", label="Recommendations\\nAPI"]

    "Spotify API" -> "_ingest_data"
    "_ingest_data" -> "bronze/tracks"
    "_ingest_data" -> "bronze/artists"
    "_ingest_data" -> "bronze/audio_features"

    "bronze/tracks"         -> "_bronze_to_silver"
    "bronze/artists"        -> "_bronze_to_silver"
    "bronze/audio_features" -> "_bronze_to_silver"
    "_bronze_to_silver" -> "silver/tracks"
    "_bronze_to_silver" -> "silver/artists"

    "silver/tracks"  -> "_silver_to_gold"
    "silver/artists" -> "_silver_to_gold"
    "_silver_to_gold" -> "gold/dim_tracks"
    "_silver_to_gold" -> "gold/dim_artists"
    "_silver_to_gold" -> "gold/fact_history"

    "gold/dim_tracks"   -> "recommendations_api"
    "gold/dim_artists"  -> "recommendations_api"
    "gold/fact_history" -> "recommendations_api"
}
""")

# Live Marquez dataset/job counts (shown only when Marquez is running)
st.subheader("Live Lineage (Marquez)")
with st.spinner("Querying Marquez..."):
    try:
        marquez_resp = requests.get(
            f"{MARQUEZ_URL}/api/v1/namespaces/{MARQUEZ_NAMESPACE}/jobs",
            timeout=3,
        )
        marquez_resp.raise_for_status()
        jobs = marquez_resp.json().get("jobs", [])

        dataset_resp = requests.get(
            f"{MARQUEZ_URL}/api/v1/namespaces/{MARQUEZ_NAMESPACE}/datasets",
            timeout=3,
        )
        dataset_resp.raise_for_status()
        datasets = dataset_resp.json().get("datasets", [])

        col_j, col_d = st.columns(2)
        with col_j:
            st.metric("Tracked Jobs", len(jobs))
        with col_d:
            st.metric("Tracked Datasets", len(datasets))

        if jobs:
            jobs_df = pd.DataFrame([
                {
                    "job": j.get("name"),
                    "namespace": j.get("namespace", ""),
                    "latest_run": j.get("latestRun", {}).get("state", "N/A") if j.get("latestRun") else "N/A",
                    "updated_at": j.get("updatedAt", "N/A"),
                }
                for j in jobs
            ])
            st.dataframe(jobs_df, use_container_width=True)
        else:
            st.info("No jobs tracked yet. Trigger the Airflow DAG to populate lineage.")

    except requests.exceptions.ConnectionError:
        st.info(
            "Marquez is not running. Start it with: "
            "`docker compose --profile lineage up -d marquez marquez-web`"
        )
    except Exception as e:
        st.warning(f"Could not reach Marquez: {e}")

st.markdown("---")
st.caption("Built with Streamlit and FastAPI. Data from Spotify & ReccoBeats.")

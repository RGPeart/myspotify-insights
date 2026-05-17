# See README.md for instructions on how to run this Streamlit application.

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os
from datetime import datetime

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

@st.cache_data(ttl=600)
def fetch_data(endpoint: str, params: dict = None):
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from API endpoint /{endpoint}: {e}")
        return None

@st.cache_data(ttl=3600) # Cache user IDs for an hour
def fetch_user_ids():
    users_data = fetch_data("users")
    if users_data and "user_ids" in users_data:
        return sorted(users_data["user_ids"])
    return []

@st.cache_data(ttl=3600) # Cache track IDs for an hour
def fetch_track_ids():
    tracks_data = fetch_data("tracks/ids")
    if tracks_data and "track_ids" in tracks_data:
        return tracks_data["track_ids"]
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
    st.metric("API Health", "OK" if fetch_data("health") else "DOWN", help="Status of the FastAPI service")

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
        if recommendations and recommendations["recommendations"]:
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
            st.info(f"No recommendations found for user ID: {user_id_example}")
    else:
        st.warning("Please enter a User ID.")

st.markdown("---")

st.header("🔊 Audio Feature Distributions")
st.write("*(Distribution of audio features across your ingested tracks)*")

all_track_ids = fetch_track_ids()

audio_features_df = pd.DataFrame() # Initialize an empty DataFrame
if all_track_ids:
    track_id_for_features = all_track_ids[0] # Using the first track ID for consistency in demo
    tracks_data = fetch_data(f"tracks/{track_id_for_features}")
    if tracks_data and "audio_features" in tracks_data:
        audio_features_df = pd.DataFrame([tracks_data["audio_features"]])
    else:
        st.warning(f"Could not fetch audio features for track ID: {track_id_for_features}. Showing dummy data.")
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
st.caption("Built with Streamlit and FastAPI. Data from Spotify & ReccoBeats.")

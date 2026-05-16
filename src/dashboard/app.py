# To run this Streamlit application, follow these steps:
# 1. Open a new terminal window and navigate to the root of your project.
# 2. Start the FastAPI server by running the following command:
#    bash -c "source .venv/bin/activate && PYTHONPATH=. uvicorn src/api/main:app --reload --host 0.0.0.0 --port 8001"
#    Keep this terminal open and running in the background.
# 3. Open another new terminal window and navigate to the root of your project.
# 4. Run the Streamlit dashboard with the command:
#    streamlit run src/dashboard/app.py
#    This will open the dashboard in your web browser.

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta

# Configuration
API_BASE_URL = "http://localhost:8001" # Assuming local API for now

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

# -----------------------------------------------------------------------------
# Dashboard Sections
# -----------------------------------------------------------------------------

st.header("📈 Overall Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Tracks in Gold", "N/A", "Update ETL") # Placeholder
with col2:
    st.metric("Total Artists in Gold", "N/A", "Update ETL") # Placeholder
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

user_id_example = st.text_input("Enter User ID (e.g., user_energetic)", "user_energetic")
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

# Placeholder for audio feature data - would fetch from API if available
# For now, generate some dummy data or load from a sample
try:
    tracks_data = fetch_data("tracks/TRvQ4qB0D3X0j3J3G0J0J0") # Example track ID, replace with actual
    if tracks_data and "audio_features" in tracks_data:
        dummy_audio_features = pd.DataFrame([tracks_data["audio_features"]])
    else:
        dummy_audio_features = pd.DataFrame({
            "danceability": [0.7, 0.8, 0.6, 0.9],
            "energy": [0.6, 0.7, 0.8, 0.5],
            "valence": [0.5, 0.6, 0.7, 0.8],
            "tempo": [120, 130, 110, 140],
            "acousticness": [0.1, 0.2, 0.3, 0.05],
            "instrumentalness": [0.01, 0.0, 0.02, 0.03],
            "liveness": [0.15, 0.25, 0.1, 0.3],
            "speechiness": [0.05, 0.03, 0.07, 0.04],
        })
except Exception: # Fallback for when API is not running or track ID is invalid
    dummy_audio_features = pd.DataFrame({
        "danceability": [0.7, 0.8, 0.6, 0.9],
        "energy": [0.6, 0.7, 0.8, 0.5],
        "valence": [0.5, 0.6, 0.7, 0.8],
        "tempo": [120, 130, 110, 140],
        "acousticness": [0.1, 0.2, 0.3, 0.05],
        "instrumentalness": [0.01, 0.0, 0.02, 0.03],
        "liveness": [0.15, 0.25, 0.1, 0.3],
        "speechiness": [0.05, 0.03, 0.07, 0.04],
    })

if not dummy_audio_features.empty:
    feature_to_plot = st.selectbox(
        "Select Audio Feature to Visualize",
        options=dummy_audio_features.columns.tolist()
    )
    fig = px.histogram(dummy_audio_features, x=feature_to_plot, title=f"Distribution of {feature_to_plot}")
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

st.write(f"Last Pipeline Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Placeholder)")
st.write(f"Data Freshness: {timedelta(days=1)} (Placeholder)")

st.markdown("---")
st.caption("Built with Streamlit and FastAPI. Data from Spotify & ReccoBeats.")

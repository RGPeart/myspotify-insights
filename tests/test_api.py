from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import router
from src.models.train import (
    _build_collab_model,
    _build_content_model,
    _build_feature_matrix,
    _build_user_item_matrix,
    _AUDIO_FEATURES,
    _GENRES,
)


# ------------------------------------------------------------------ #
# Shared test data                                                     #
# ------------------------------------------------------------------ #

def _make_dim_tracks(n: int = 4) -> pd.DataFrame:
    genres = (_GENRES * n)[:n]
    return pd.DataFrame({
        "track_id": [f"t{i}" for i in range(n)],
        "name": [f"Track {i}" for i in range(n)],
        "primary_artist_id": [f"a{i}" for i in range(n)],
        "primary_artist_name": [f"Artist {i}" for i in range(n)],
        "primary_genre": genres,
        "composite_popularity": [0.5 + 0.05 * i for i in range(n)],
        "track_popularity": [50 + 5 * i for i in range(n)],
        "album_id": [f"al{i}" for i in range(n)],
        "album_name": [f"Album {i}" for i in range(n)],
        "release_date": pd.to_datetime(["2023-01-01"] * n),
        "duration_ms": [200000] * n,
        "explicit": [False] * n,
    })


def _make_dim_artists(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "artist_id": [f"a{i}" for i in range(n)],
        "artist_name": [f"Artist {i}" for i in range(n)],
        "popularity": [70 + i for i in range(n)],
        "followers": [100000 + i * 1000 for i in range(n)],
        "genres": ["pop,rock"] * n,
        "primary_genre": (_GENRES * n)[:n],
    })


def _make_fact_af(n: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        row = {"track_id": f"t{i}", "primary_artist_id": f"a{i}", "album_id": f"al{i}"}
        for feat in _AUDIO_FEATURES:
            row[feat] = float(rng.uniform(0, 1))
        rows.append(row)
    return pd.DataFrame(rows)


def _make_artifacts(dim_tracks: pd.DataFrame, fact_af: pd.DataFrame) -> dict:
    matrix, track_ids, feature_cols = _build_feature_matrix(dim_tracks, fact_af)
    content = _build_content_model(matrix, track_ids)
    ui, user_ids = _build_user_item_matrix(track_ids, matrix, feature_cols)
    collab = _build_collab_model(ui, user_ids, track_ids, n_components=3)
    return {"content": content, "collab": collab, "trained_at": "2026-05-14T00:00:00+00:00"}


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def shared_data():
    dim_tracks = _make_dim_tracks()
    dim_artists = _make_dim_artists()
    fact_af = _make_fact_af()
    artifacts = _make_artifacts(dim_tracks, fact_af)
    return dim_tracks, dim_artists, fact_af, artifacts


def _make_test_app(state_overrides: dict) -> FastAPI:
    """Create a minimal FastAPI app with router and pre-populated state."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for key, value in state_overrides.items():
            setattr(app.state, key, value)
        yield

    test_app = FastAPI(lifespan=lifespan)
    test_app.include_router(router)
    return test_app


@pytest.fixture(scope="module")
def full_client(shared_data):
    """Client with all state loaded."""
    dim_tracks, dim_artists, fact_af, artifacts = shared_data
    app = _make_test_app({
        "artifacts": artifacts,
        "dim_tracks": dim_tracks,
        "dim_artists": dim_artists,
        "fact_audio_features": fact_af,
        "collab_weight": 0.7,
        "n_recommendations": 10,
    })
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def empty_client():
    """Client with no data loaded (simulates cold start)."""
    app = _make_test_app({
        "artifacts": None,
        "dim_tracks": None,
        "dim_artists": None,
        "fact_audio_features": None,
        "collab_weight": 0.7,
    })
    with TestClient(app) as client:
        yield client


# ------------------------------------------------------------------ #
# GET /health                                                          #
# ------------------------------------------------------------------ #

class TestHealth:
    def test_returns_200(self, full_client):
        response = full_client.get("/health")
        assert response.status_code == 200

    def test_returns_ok_status(self, full_client):
        data = full_client.get("/health").json()
        assert data["status"] == "ok"

    def test_returns_version(self, full_client):
        data = full_client.get("/health").json()
        assert "version" in data

    def test_works_without_data(self, empty_client):
        response = empty_client.get("/health")
        assert response.status_code == 200


# ------------------------------------------------------------------ #
# GET /recommendations/{user_id}                                       #
# ------------------------------------------------------------------ #

class TestRecommendations:
    def test_known_user_returns_200(self, full_client):
        response = full_client.get("/recommendations/user_energetic")
        assert response.status_code == 200

    def test_response_contains_recommendations_list(self, full_client):
        data = full_client.get("/recommendations/user_energetic").json()
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)

    def test_response_contains_user_id(self, full_client):
        data = full_client.get("/recommendations/user_energetic").json()
        assert data["user_id"] == "user_energetic"

    def test_count_field_matches_list_length(self, full_client):
        data = full_client.get("/recommendations/user_energetic").json()
        assert data["count"] == len(data["recommendations"])

    def test_n_query_param_limits_results(self, full_client):
        data = full_client.get("/recommendations/user_chill?n=2").json()
        assert len(data["recommendations"]) <= 2

    def test_each_recommendation_has_required_fields(self, full_client):
        data = full_client.get("/recommendations/user_energetic").json()
        for rec in data["recommendations"]:
            assert "track_id" in rec
            assert "score" in rec
            assert "reason" in rec

    def test_enriched_with_track_metadata(self, full_client):
        data = full_client.get("/recommendations/user_energetic").json()
        if data["recommendations"]:
            rec = data["recommendations"][0]
            assert "name" in rec

    def test_unknown_user_with_no_liked_returns_empty(self, full_client):
        data = full_client.get("/recommendations/completely_unknown_user").json()
        assert data["recommendations"] == []

    def test_no_model_returns_503(self, empty_client):
        response = empty_client.get("/recommendations/user_energetic")
        assert response.status_code == 503


# ------------------------------------------------------------------ #
# GET /tracks/{track_id}                                               #
# ------------------------------------------------------------------ #

class TestTrackDetail:
    def test_existing_track_returns_200(self, full_client):
        response = full_client.get("/tracks/t0")
        assert response.status_code == 200

    def test_returns_track_id(self, full_client):
        data = full_client.get("/tracks/t0").json()
        assert data["track_id"] == "t0"

    def test_returns_track_name(self, full_client):
        data = full_client.get("/tracks/t0").json()
        assert "name" in data

    def test_includes_audio_features(self, full_client):
        data = full_client.get("/tracks/t0").json()
        assert "audio_features" in data
        assert "danceability" in data["audio_features"]

    def test_audio_features_exclude_key_columns(self, full_client):
        data = full_client.get("/tracks/t0").json()
        af = data.get("audio_features", {})
        assert "track_id" not in af
        assert "primary_artist_id" not in af

    def test_missing_track_returns_404(self, full_client):
        response = full_client.get("/tracks/nonexistent_track_id")
        assert response.status_code == 404

    def test_no_data_returns_503(self, empty_client):
        response = empty_client.get("/tracks/t0")
        assert response.status_code == 503


# ------------------------------------------------------------------ #
# GET /artists/{artist_id}                                             #
# ------------------------------------------------------------------ #

class TestArtistDetail:
    def test_existing_artist_returns_200(self, full_client):
        response = full_client.get("/artists/a0")
        assert response.status_code == 200

    def test_returns_artist_id(self, full_client):
        data = full_client.get("/artists/a0").json()
        assert data["artist_id"] == "a0"

    def test_returns_artist_name(self, full_client):
        data = full_client.get("/artists/a0").json()
        assert "artist_name" in data

    def test_missing_artist_returns_404(self, full_client):
        response = full_client.get("/artists/nonexistent_artist")
        assert response.status_code == 404

    def test_no_data_returns_503(self, empty_client):
        response = empty_client.get("/artists/a0")
        assert response.status_code == 503

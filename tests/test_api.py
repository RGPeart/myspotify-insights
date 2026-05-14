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
        "track_map": dim_tracks.set_index("track_id").to_dict("index"),
        "tracks_idx": dim_tracks.set_index("track_id"),
        "artists_idx": dim_artists.set_index("artist_id"),
        "af_idx": fact_af.set_index("track_id"),
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
        "track_map": {},
        "tracks_idx": None,
        "artists_idx": None,
        "af_idx": None,
        "collab_weight": 0.7,
    })
    with TestClient(app) as client:
        yield client


# ------------------------------------------------------------------ #
# GET /health                                                          #
# ------------------------------------------------------------------ #

class TestHealth:
    # The health endpoint must always return HTTP 200 so load balancers and uptime
    # monitors can confirm the service process is running.
    def test_returns_200(self, full_client):
        response = full_client.get("/health")
        assert response.status_code == 200

    # The status field must equal "ok" so automated monitors can parse the response
    # body and trigger alerts when the value changes.
    def test_returns_ok_status(self, full_client):
        data = full_client.get("/health").json()
        assert data["status"] == "ok"

    # A version field must be present to help operators correlate health check
    # responses with specific deployments during incident investigations.
    def test_returns_version(self, full_client):
        data = full_client.get("/health").json()
        assert "version" in data

    # The health check must succeed even when no model or gold data is loaded,
    # because it tests process availability rather than data readiness.
    def test_works_without_data(self, empty_client):
        response = empty_client.get("/health")
        assert response.status_code == 200


# ------------------------------------------------------------------ #
# GET /recommendations/{user_id}                                       #
# ------------------------------------------------------------------ #

class TestRecommendations:
    # A synthetic user present in the training set must receive a successful 200 response,
    # confirming the full recommendation path executes without error.
    def test_known_user_returns_200(self, full_client):
        response = full_client.get("/recommendations/user_energetic")
        assert response.status_code == 200

    # The response body must contain a recommendations list so client code can iterate
    # over results without needing to handle a non-list payload.
    def test_response_contains_recommendations_list(self, full_client):
        data = full_client.get("/recommendations/user_energetic").json()
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)

    # The user_id must be echoed back in the response so clients batching multiple
    # requests can match each response to its originating user.
    def test_response_contains_user_id(self, full_client):
        data = full_client.get("/recommendations/user_energetic").json()
        assert data["user_id"] == "user_energetic"

    # The count field must equal the actual list length; a mismatch would confuse
    # clients that use count to allocate or validate pagination.
    def test_count_field_matches_list_length(self, full_client):
        data = full_client.get("/recommendations/user_energetic").json()
        assert data["count"] == len(data["recommendations"])

    # The n query parameter must cap the number of returned recommendations; ignoring
    # it would always return the default count regardless of what the client requested.
    def test_n_query_param_limits_results(self, full_client):
        data = full_client.get("/recommendations/user_chill?n=2").json()
        assert len(data["recommendations"]) <= 2

    # Each recommendation item must include track_id, score, and reason so clients
    # can display and act on the data without a second lookup.
    def test_each_recommendation_has_required_fields(self, full_client):
        data = full_client.get("/recommendations/user_energetic").json()
        for rec in data["recommendations"]:
            assert "track_id" in rec
            assert "score" in rec
            assert "reason" in rec

    # When gold data is loaded, each recommendation must include a human-readable
    # track name so dashboard consumers can display results without an additional API call.
    def test_enriched_with_track_metadata(self, full_client):
        data = full_client.get("/recommendations/user_energetic").json()
        if data["recommendations"]:
            rec = data["recommendations"][0]
            assert "name" in rec

    # A completely unknown user with no liked tracks has no usable signal and must
    # receive an empty recommendations list rather than a 404 or 500 error.
    def test_unknown_user_with_no_liked_returns_empty(self, full_client):
        data = full_client.get("/recommendations/completely_unknown_user").json()
        assert data["recommendations"] == []

    # Without a trained model, the endpoint must return 503 to signal that the
    # service is not ready rather than pretending to serve recommendations.
    def test_no_model_returns_503(self, empty_client):
        response = empty_client.get("/recommendations/user_energetic")
        assert response.status_code == 503


# ------------------------------------------------------------------ #
# GET /tracks/{track_id}                                               #
# ------------------------------------------------------------------ #

class TestTrackDetail:
    # A track ID present in dim_tracks must return HTTP 200, confirming the lookup
    # and serialisation path work correctly end-to-end.
    def test_existing_track_returns_200(self, full_client):
        response = full_client.get("/tracks/t0")
        assert response.status_code == 200

    # The response must echo back the track_id so clients can confirm they received
    # the correct record when looking up by ID.
    def test_returns_track_id(self, full_client):
        data = full_client.get("/tracks/t0").json()
        assert data["track_id"] == "t0"

    # The track name must be present so clients can display the result without an
    # additional lookup; a missing name field would break the dashboard UI.
    def test_returns_track_name(self, full_client):
        data = full_client.get("/tracks/t0").json()
        assert "name" in data

    # Audio features must be nested in an audio_features sub-object so clients can
    # access them in a single request rather than needing a separate feature endpoint.
    def test_includes_audio_features(self, full_client):
        data = full_client.get("/tracks/t0").json()
        assert "audio_features" in data
        assert "danceability" in data["audio_features"]

    # Join key columns (track_id, primary_artist_id, album_id) must not appear inside
    # audio_features to avoid confusing duplication in the response structure.
    def test_audio_features_exclude_key_columns(self, full_client):
        data = full_client.get("/tracks/t0").json()
        af = data.get("audio_features", {})
        assert "track_id" not in af
        assert "primary_artist_id" not in af

    # An unknown track ID must return 404 rather than 500 so clients can distinguish
    # a not-found result from an internal server error.
    def test_missing_track_returns_404(self, full_client):
        response = full_client.get("/tracks/nonexistent_track_id")
        assert response.status_code == 404

    # When dim_tracks hasn't been loaded, the endpoint must return 503 so clients
    # know the service is unavailable rather than getting an unexpected 500 crash.
    def test_no_data_returns_503(self, empty_client):
        response = empty_client.get("/tracks/t0")
        assert response.status_code == 503


# ------------------------------------------------------------------ #
# GET /artists/{artist_id}                                             #
# ------------------------------------------------------------------ #

class TestArtistDetail:
    # A known artist ID must return HTTP 200, confirming the artist lookup and
    # serialisation work correctly end-to-end.
    def test_existing_artist_returns_200(self, full_client):
        response = full_client.get("/artists/a0")
        assert response.status_code == 200

    # The response must include the artist_id field confirming the correct record
    # was retrieved rather than a neighbour row.
    def test_returns_artist_id(self, full_client):
        data = full_client.get("/artists/a0").json()
        assert data["artist_id"] == "a0"

    # The artist name must be present for display in the dashboard and by API
    # consumers that list artist metadata alongside recommendations.
    def test_returns_artist_name(self, full_client):
        data = full_client.get("/artists/a0").json()
        assert "artist_name" in data

    # An unknown artist ID must return 404 so clients can distinguish a not-found
    # result from an internal server error and handle it appropriately.
    def test_missing_artist_returns_404(self, full_client):
        response = full_client.get("/artists/nonexistent_artist")
        assert response.status_code == 404

    # When dim_artists hasn't been loaded, the endpoint must return 503 so clients
    # know the service is unavailable rather than receiving an unexpected 500 crash.
    def test_no_data_returns_503(self, empty_client):
        response = empty_client.get("/artists/a0")
        assert response.status_code == 503

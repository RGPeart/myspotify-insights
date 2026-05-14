import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.train import (
    _AUDIO_FEATURES,
    _GENRES,
    _USER_PROFILES,
    _build_collab_model,
    _build_content_model,
    _build_feature_matrix,
    _build_user_item_matrix,
    run as train_run,
)
from src.models.predict import (
    _collab_scores,
    _content_scores,
    get_recommendations,
)


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

def _make_dim_tracks(n: int = 5) -> pd.DataFrame:
    genres = (_GENRES * n)[:n]
    return pd.DataFrame({
        "track_id": [f"t{i}" for i in range(n)],
        "name": [f"Track {i}" for i in range(n)],
        "primary_genre": genres,
        "composite_popularity": [0.5 + 0.05 * i for i in range(n)],
    })


def _make_fact_af(n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        row = {"track_id": f"t{i}"}
        for feat in _AUDIO_FEATURES:
            row[feat] = float(rng.uniform(0, 1))
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def dim_tracks():
    return _make_dim_tracks()


@pytest.fixture
def fact_af():
    return _make_fact_af()


@pytest.fixture
def feature_data(dim_tracks, fact_af):
    matrix, track_ids, feature_cols = _build_feature_matrix(dim_tracks, fact_af)
    return matrix, track_ids, feature_cols


@pytest.fixture
def content_model(feature_data):
    matrix, track_ids, _ = feature_data
    return _build_content_model(matrix, track_ids)


@pytest.fixture
def collab_model(feature_data):
    matrix, track_ids, feature_cols = feature_data
    ui_matrix, user_ids = _build_user_item_matrix(track_ids, matrix, feature_cols)
    return _build_collab_model(ui_matrix, user_ids, track_ids, n_components=3)


@pytest.fixture
def artifacts(content_model, collab_model):
    return {"content": content_model, "collab": collab_model}


# ------------------------------------------------------------------ #
# _build_feature_matrix                                               #
# ------------------------------------------------------------------ #

class TestBuildFeatureMatrix:
    def test_returns_correct_track_count(self, dim_tracks, fact_af):
        matrix, track_ids, _ = _build_feature_matrix(dim_tracks, fact_af)
        assert len(track_ids) == 5
        assert matrix.shape[0] == 5

    def test_feature_cols_include_audio_features(self, dim_tracks, fact_af):
        _, _, feature_cols = _build_feature_matrix(dim_tracks, fact_af)
        for feat in _AUDIO_FEATURES:
            assert feat in feature_cols

    def test_feature_cols_include_genre_encoding(self, dim_tracks, fact_af):
        _, _, feature_cols = _build_feature_matrix(dim_tracks, fact_af)
        assert "genre_pop" in feature_cols
        assert "genre_rock" in feature_cols

    def test_feature_cols_include_composite_popularity(self, dim_tracks, fact_af):
        _, _, feature_cols = _build_feature_matrix(dim_tracks, fact_af)
        assert "composite_popularity" in feature_cols

    def test_excludes_tracks_without_audio_features(self):
        dim = _make_dim_tracks(3)
        af = _make_fact_af(2)  # only 2 of 3 tracks have audio features
        matrix, track_ids, _ = _build_feature_matrix(dim, af)
        assert len(track_ids) == 2

    def test_no_nans_in_matrix(self, dim_tracks, fact_af):
        matrix, _, _ = _build_feature_matrix(dim_tracks, fact_af)
        assert not np.isnan(matrix).any()

    def test_empty_audio_features_returns_empty(self, dim_tracks):
        af = pd.DataFrame(columns=["track_id"] + _AUDIO_FEATURES)
        matrix, track_ids, _ = _build_feature_matrix(dim_tracks, af)
        assert len(track_ids) == 0
        assert matrix.shape[0] == 0


# ------------------------------------------------------------------ #
# _build_content_model                                                 #
# ------------------------------------------------------------------ #

class TestBuildContentModel:
    def test_returns_required_keys(self, feature_data):
        matrix, track_ids, _ = feature_data
        model = _build_content_model(matrix, track_ids)
        assert "nn" in model
        assert "scaler" in model
        assert "feature_matrix_scaled" in model
        assert "track_ids" in model

    def test_scaled_matrix_shape_matches_input(self, feature_data):
        matrix, track_ids, _ = feature_data
        model = _build_content_model(matrix, track_ids)
        assert model["feature_matrix_scaled"].shape == matrix.shape

    def test_track_ids_preserved(self, feature_data):
        matrix, track_ids, _ = feature_data
        model = _build_content_model(matrix, track_ids)
        assert model["track_ids"] == track_ids

    def test_nn_can_query(self, feature_data):
        matrix, track_ids, _ = feature_data
        model = _build_content_model(matrix, track_ids)
        query = model["feature_matrix_scaled"][[0]]
        distances, indices = model["nn"].kneighbors(query, n_neighbors=2)
        assert distances.shape == (1, 2)
        assert indices.shape == (1, 2)


# ------------------------------------------------------------------ #
# _build_user_item_matrix                                              #
# ------------------------------------------------------------------ #

class TestBuildUserItemMatrix:
    def test_shape(self, feature_data):
        matrix, track_ids, feature_cols = feature_data
        ui, user_ids = _build_user_item_matrix(track_ids, matrix, feature_cols)
        assert ui.shape == (len(_USER_PROFILES), len(track_ids))
        assert len(user_ids) == len(_USER_PROFILES)

    def test_values_in_unit_interval(self, feature_data):
        matrix, track_ids, feature_cols = feature_data
        ui, _ = _build_user_item_matrix(track_ids, matrix, feature_cols)
        assert ui.min() >= 0.0
        assert ui.max() <= 1.0

    def test_user_ids_match_profiles(self, feature_data):
        matrix, track_ids, feature_cols = feature_data
        _, user_ids = _build_user_item_matrix(track_ids, matrix, feature_cols)
        assert set(user_ids) == set(_USER_PROFILES.keys())

    def test_deterministic_with_seed(self, feature_data):
        matrix, track_ids, feature_cols = feature_data
        ui1, _ = _build_user_item_matrix(track_ids, matrix, feature_cols)
        ui2, _ = _build_user_item_matrix(track_ids, matrix, feature_cols)
        np.testing.assert_array_equal(ui1, ui2)


# ------------------------------------------------------------------ #
# _build_collab_model                                                  #
# ------------------------------------------------------------------ #

class TestBuildCollabModel:
    def test_returns_required_keys(self, feature_data):
        matrix, track_ids, feature_cols = feature_data
        ui, user_ids = _build_user_item_matrix(track_ids, matrix, feature_cols)
        model = _build_collab_model(ui, user_ids, track_ids, n_components=3)
        for key in ("svd", "user_factors", "item_factors", "user_index", "track_ids"):
            assert key in model

    def test_user_index_maps_all_users(self, feature_data):
        matrix, track_ids, feature_cols = feature_data
        ui, user_ids = _build_user_item_matrix(track_ids, matrix, feature_cols)
        model = _build_collab_model(ui, user_ids, track_ids, n_components=3)
        assert set(model["user_index"].keys()) == set(user_ids)

    def test_item_factors_shape(self, feature_data):
        matrix, track_ids, feature_cols = feature_data
        ui, user_ids = _build_user_item_matrix(track_ids, matrix, feature_cols)
        n_components = 3
        model = _build_collab_model(ui, user_ids, track_ids, n_components=n_components)
        assert model["item_factors"].shape == (len(track_ids), n_components)

    def test_n_components_clamped_below_matrix_rank(self, feature_data):
        matrix, track_ids, feature_cols = feature_data
        ui, user_ids = _build_user_item_matrix(track_ids, matrix, feature_cols)
        # Request more components than matrix rank allows
        model = _build_collab_model(ui, user_ids, track_ids, n_components=999)
        actual_k = model["item_factors"].shape[1]
        assert actual_k < min(ui.shape)


# ------------------------------------------------------------------ #
# _content_scores                                                      #
# ------------------------------------------------------------------ #

class TestContentScores:
    def test_returns_scores_for_similar_tracks(self, content_model):
        scores = _content_scores(["t0"], content_model, n=3)
        assert len(scores) > 0
        assert "t0" not in scores  # liked track excluded

    def test_all_scores_in_unit_interval(self, content_model):
        scores = _content_scores(["t0"], content_model, n=3)
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_unknown_track_returns_empty(self, content_model):
        scores = _content_scores(["unknown_track"], content_model, n=3)
        assert scores == {}

    def test_empty_liked_list_returns_empty(self, content_model):
        scores = _content_scores([], content_model, n=3)
        assert scores == {}

    def test_multiple_liked_tracks_excluded(self, content_model):
        liked = ["t0", "t1"]
        scores = _content_scores(liked, content_model, n=5)
        for tid in liked:
            assert tid not in scores


# ------------------------------------------------------------------ #
# _collab_scores                                                        #
# ------------------------------------------------------------------ #

class TestCollabScores:
    def test_returns_scores_for_known_user(self, collab_model):
        scores = _collab_scores("user_energetic", collab_model, n=3)
        assert len(scores) > 0

    def test_returns_empty_for_unknown_user(self, collab_model):
        scores = _collab_scores("user_who_does_not_exist", collab_model, n=3)
        assert scores == {}

    def test_scores_in_unit_interval(self, collab_model):
        scores = _collab_scores("user_chill", collab_model, n=5)
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_respects_n_limit(self, collab_model):
        scores = _collab_scores("user_workout", collab_model, n=2)
        assert len(scores) <= 2


# ------------------------------------------------------------------ #
# get_recommendations                                                   #
# ------------------------------------------------------------------ #

class TestGetRecommendations:
    def test_collab_only_for_known_user(self, artifacts):
        recs = get_recommendations("user_energetic", None, artifacts, n=3)
        assert len(recs) > 0
        for r in recs:
            assert r["reason"] == "collaborative_filtering"

    def test_content_only_for_unknown_user_with_liked(self, artifacts):
        recs = get_recommendations("unknown_user", ["t0"], artifacts, n=3)
        assert len(recs) > 0
        for r in recs:
            assert r["reason"] == "content_similarity"

    def test_hybrid_for_known_user_with_liked(self, artifacts):
        recs = get_recommendations("user_chill", ["t0"], artifacts, n=3)
        assert len(recs) > 0
        for r in recs:
            assert r["reason"] == "hybrid"

    def test_empty_when_no_user_and_no_liked(self, artifacts):
        recs = get_recommendations("unknown_user", None, artifacts, n=5)
        assert recs == []

    def test_respects_n_parameter(self, artifacts):
        recs = get_recommendations("user_party", None, artifacts, n=2)
        assert len(recs) <= 2

    def test_result_has_required_keys(self, artifacts):
        recs = get_recommendations("user_indie", None, artifacts, n=3)
        for r in recs:
            assert "track_id" in r
            assert "score" in r
            assert "reason" in r

    def test_scores_are_sorted_descending(self, artifacts):
        recs = get_recommendations("user_workout", None, artifacts, n=5)
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_scores_rounded_to_four_decimal_places(self, artifacts):
        recs = get_recommendations("user_energetic", None, artifacts, n=3)
        for r in recs:
            assert round(r["score"], 4) == r["score"]


# ------------------------------------------------------------------ #
# train.run() end-to-end                                               #
# ------------------------------------------------------------------ #

class TestTrainRun:
    def test_saves_pkl_file(self, tmp_path):
        gold = tmp_path / "gold"
        gold.mkdir()
        models = tmp_path / "models"
        _make_dim_tracks(10).to_parquet(gold / "dim_tracks.parquet", index=False)
        _make_fact_af(10).to_parquet(gold / "fact_audio_features.parquet", index=False)

        out = train_run(gold_dir=gold, models_dir=models)

        assert out == models / "recommendation_model.pkl"
        assert out.exists()

    def test_artifacts_contain_expected_keys(self, tmp_path):
        gold = tmp_path / "gold"
        gold.mkdir()
        models = tmp_path / "models"
        _make_dim_tracks(10).to_parquet(gold / "dim_tracks.parquet", index=False)
        _make_fact_af(10).to_parquet(gold / "fact_audio_features.parquet", index=False)

        out = train_run(gold_dir=gold, models_dir=models)
        with open(out, "rb") as f:
            arts = pickle.load(f)

        assert "content" in arts
        assert "collab" in arts
        assert "trained_at" in arts

    def test_raises_when_dim_tracks_missing(self, tmp_path):
        gold = tmp_path / "gold"
        gold.mkdir()
        _make_fact_af(5).to_parquet(gold / "fact_audio_features.parquet", index=False)
        with pytest.raises(FileNotFoundError):
            train_run(gold_dir=gold, models_dir=tmp_path / "models")

    def test_raises_when_audio_features_missing(self, tmp_path):
        gold = tmp_path / "gold"
        gold.mkdir()
        _make_dim_tracks(5).to_parquet(gold / "dim_tracks.parquet", index=False)
        with pytest.raises(FileNotFoundError):
            train_run(gold_dir=gold, models_dir=tmp_path / "models")

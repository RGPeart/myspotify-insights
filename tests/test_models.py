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
    # The matrix row count must match the number of tracks that have audio features,
    # since each row is one model input and missing rows would silently reduce coverage.
    def test_returns_correct_track_count(self, dim_tracks, fact_af):
        matrix, track_ids, _ = _build_feature_matrix(dim_tracks, fact_af)
        assert len(track_ids) == 5
        assert matrix.shape[0] == 5

    # All 11 audio feature columns must be present in the feature list; a missing feature
    # would silently degrade recommendation quality without any error.
    def test_feature_cols_include_audio_features(self, dim_tracks, fact_af):
        _, _, feature_cols = _build_feature_matrix(dim_tracks, fact_af)
        for feat in _AUDIO_FEATURES:
            assert feat in feature_cols

    # Genre one-hot columns must be present in the feature list because they carry
    # important categorical signal for content-based similarity scoring.
    def test_feature_cols_include_genre_encoding(self, dim_tracks, fact_af):
        _, _, feature_cols = _build_feature_matrix(dim_tracks, fact_af)
        assert "genre_pop" in feature_cols
        assert "genre_rock" in feature_cols

    # composite_popularity must be included in the feature vector so the model can
    # factor popularity into track similarity during content-based scoring.
    def test_feature_cols_include_composite_popularity(self, dim_tracks, fact_af):
        _, _, feature_cols = _build_feature_matrix(dim_tracks, fact_af)
        assert "composite_popularity" in feature_cols

    # Tracks with no matching audio features must be excluded; including them with
    # zero-filled vectors would produce meaningless similarity scores.
    def test_excludes_tracks_without_audio_features(self):
        dim = _make_dim_tracks(3)
        af = _make_fact_af(2)  # only 2 of 3 tracks have audio features
        matrix, track_ids, _ = _build_feature_matrix(dim, af)
        assert len(track_ids) == 2

    # NaNs in the feature matrix would corrupt cosine distance calculations; all
    # missing values must be filled to zero before the matrix is returned.
    def test_no_nans_in_matrix(self, dim_tracks, fact_af):
        matrix, _, _ = _build_feature_matrix(dim_tracks, fact_af)
        assert not np.isnan(matrix).any()

    # When no audio features exist at all, the function must return an empty matrix
    # without raising so train.run() can surface a more descriptive error.
    def test_empty_audio_features_returns_empty(self, dim_tracks):
        af = pd.DataFrame(columns=["track_id"] + _AUDIO_FEATURES)
        matrix, track_ids, _ = _build_feature_matrix(dim_tracks, af)
        assert len(track_ids) == 0
        assert matrix.shape[0] == 0


# ------------------------------------------------------------------ #
# _build_content_model                                                 #
# ------------------------------------------------------------------ #

class TestBuildContentModel:
    # All four artifact keys must be present so predict.py can load the model without
    # KeyError; a missing key would only surface at inference time, not training time.
    def test_returns_required_keys(self, feature_data):
        matrix, track_ids, _ = feature_data
        model = _build_content_model(matrix, track_ids)
        assert "nn" in model
        assert "scaler" in model
        assert "feature_matrix_scaled" in model
        assert "track_ids" in model

    # The scaled matrix must have the same shape as the raw input; a different shape
    # would misalign the track_ids index and cause wrong tracks to be recommended.
    def test_scaled_matrix_shape_matches_input(self, feature_data):
        matrix, track_ids, _ = feature_data
        model = _build_content_model(matrix, track_ids)
        assert model["feature_matrix_scaled"].shape == matrix.shape

    # track_ids must be stored unchanged in the model because they are used to map
    # neighbor indices back to Spotify track IDs at inference time.
    def test_track_ids_preserved(self, feature_data):
        matrix, track_ids, _ = feature_data
        model = _build_content_model(matrix, track_ids)
        assert model["track_ids"] == track_ids

    # The fitted NearestNeighbors model must accept a query vector and return neighbors;
    # this confirms the model is usable immediately after training.
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
    # The matrix must be exactly (n_users × n_tracks) so TruncatedSVD receives
    # the correct input dimensions and the resulting factors align with track_ids.
    def test_shape(self, feature_data):
        matrix, track_ids, feature_cols = feature_data
        ui, user_ids = _build_user_item_matrix(track_ids, matrix, feature_cols)
        assert ui.shape == (len(_USER_PROFILES), len(track_ids))
        assert len(user_ids) == len(_USER_PROFILES)

    # Row-normalised scores must be in [0, 1] so they can be combined with content
    # scores on equal footing; values outside this range would skew the hybrid blend.
    def test_values_in_unit_interval(self, feature_data):
        matrix, track_ids, feature_cols = feature_data
        ui, _ = _build_user_item_matrix(track_ids, matrix, feature_cols)
        assert ui.min() >= 0.0
        assert ui.max() <= 1.0

    # Every user profile must appear in user_ids; a missing user would cause a silent
    # cold-start fallback at inference rather than a clear training-time error.
    def test_user_ids_match_profiles(self, feature_data):
        matrix, track_ids, feature_cols = feature_data
        _, user_ids = _build_user_item_matrix(track_ids, matrix, feature_cols)
        assert set(user_ids) == set(_USER_PROFILES.keys())

    # The same seed must produce identical matrices across invocations so model
    # training is reproducible and differences between runs can be attributed to data.
    def test_deterministic_with_seed(self, feature_data):
        matrix, track_ids, feature_cols = feature_data
        ui1, _ = _build_user_item_matrix(track_ids, matrix, feature_cols)
        ui2, _ = _build_user_item_matrix(track_ids, matrix, feature_cols)
        np.testing.assert_array_equal(ui1, ui2)


# ------------------------------------------------------------------ #
# _build_collab_model                                                  #
# ------------------------------------------------------------------ #

class TestBuildCollabModel:
    # All five keys must be present in the collab artifact dict so predict.py can
    # load every component without a KeyError at inference time.
    def test_returns_required_keys(self, feature_data):
        matrix, track_ids, feature_cols = feature_data
        ui, user_ids = _build_user_item_matrix(track_ids, matrix, feature_cols)
        model = _build_collab_model(ui, user_ids, track_ids, n_components=3)
        for key in ("svd", "user_factors", "item_factors", "user_index", "track_ids"):
            assert key in model

    # Every synthetic user must be addressable by user_id so the prediction path can
    # look up their latent vector without falling through to cold-start.
    def test_user_index_maps_all_users(self, feature_data):
        matrix, track_ids, feature_cols = feature_data
        ui, user_ids = _build_user_item_matrix(track_ids, matrix, feature_cols)
        model = _build_collab_model(ui, user_ids, track_ids, n_components=3)
        assert set(model["user_index"].keys()) == set(user_ids)

    # item_factors must be (n_tracks × n_components) for the dot-product scoring in
    # predict.py to produce one score per track correctly.
    def test_item_factors_shape(self, feature_data):
        matrix, track_ids, feature_cols = feature_data
        ui, user_ids = _build_user_item_matrix(track_ids, matrix, feature_cols)
        n_components = 3
        model = _build_collab_model(ui, user_ids, track_ids, n_components=n_components)
        assert model["item_factors"].shape == (len(track_ids), n_components)

    # TruncatedSVD raises if n_components ≥ min(matrix dimensions); this clamp must
    # prevent that crash when a large value is requested on a small dataset.
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
    # A known track must produce at least one similar neighbour, confirming the content
    # model is functioning and can generate recommendations from audio features.
    def test_returns_scores_for_similar_tracks(self, content_model):
        scores = _content_scores(["t0"], content_model, n=3)
        assert len(scores) > 0
        assert "t0" not in scores  # liked track excluded

    # Scores must be clipped to [0, 1] because cosine distance can exceed 1.0 for
    # anti-correlated tracks, producing negative similarities without clipping.
    def test_all_scores_in_unit_interval(self, content_model):
        scores = _content_scores(["t0"], content_model, n=3)
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    # A track ID not present in the model index must return an empty dict rather than
    # raising a KeyError so callers can handle unknown tracks gracefully.
    def test_unknown_track_returns_empty(self, content_model):
        scores = _content_scores(["unknown_track"], content_model, n=3)
        assert scores == {}

    # With no liked tracks there is no query vector to build, so the function must
    # short-circuit and return empty rather than producing a nonsense average vector.
    def test_empty_liked_list_returns_empty(self, content_model):
        scores = _content_scores([], content_model, n=3)
        assert scores == {}

    # Every track in the liked list must be excluded from the results so the engine
    # doesn't recommend tracks the user has already indicated they like.
    def test_multiple_liked_tracks_excluded(self, content_model):
        liked = ["t0", "t1"]
        scores = _content_scores(liked, content_model, n=5)
        for tid in liked:
            assert tid not in scores


# ------------------------------------------------------------------ #
# _collab_scores                                                        #
# ------------------------------------------------------------------ #

class TestCollabScores:
    # A user present in the training index must produce a non-empty score dict,
    # confirming the dot-product inference path is working correctly.
    def test_returns_scores_for_known_user(self, collab_model):
        scores = _collab_scores("user_energetic", collab_model, n=3)
        assert len(scores) > 0

    # An unknown user triggers the cold-start fallback path; the function must return
    # empty rather than raising a KeyError or returning garbage scores.
    def test_returns_empty_for_unknown_user(self, collab_model):
        scores = _collab_scores("user_who_does_not_exist", collab_model, n=3)
        assert scores == {}

    # All collaborative scores must be in [0, 1] after normalisation so they can be
    # linearly combined with content scores in the hybrid blend without distortion.
    def test_scores_in_unit_interval(self, collab_model):
        scores = _collab_scores("user_chill", collab_model, n=5)
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    # The function must not return more candidates than n; returning too many would
    # cause the hybrid scorer to over-weight collaborative results.
    def test_respects_n_limit(self, collab_model):
        scores = _collab_scores("user_workout", collab_model, n=2)
        assert len(scores) <= 2


# ------------------------------------------------------------------ #
# get_recommendations                                                   #
# ------------------------------------------------------------------ #

class TestGetRecommendations:
    # When a user is known but no liked tracks are provided, every recommendation must
    # come from collaborative filtering since there is no content query vector.
    def test_collab_only_for_known_user(self, artifacts):
        recs = get_recommendations("user_energetic", None, artifacts, n=3)
        assert len(recs) > 0
        for r in recs:
            assert r["reason"] == "collaborative_filtering"

    # An unknown user with liked tracks must fall back to pure content-based filtering
    # since there is no latent vector available for the user.
    def test_content_only_for_unknown_user_with_liked(self, artifacts):
        recs = get_recommendations("unknown_user", ["t0"], artifacts, n=3)
        assert len(recs) > 0
        for r in recs:
            assert r["reason"] == "content_similarity"

    # When both a known user and liked tracks are present, the hybrid path must be taken
    # and every recommendation must reflect the combined signal.
    def test_hybrid_for_known_user_with_liked(self, artifacts):
        recs = get_recommendations("user_chill", ["t0"], artifacts, n=3)
        assert len(recs) > 0
        for r in recs:
            assert r["reason"] == "hybrid"

    # With neither a known user nor liked tracks the engine has no signal at all and
    # must return an empty list rather than generating meaningless recommendations.
    def test_empty_when_no_user_and_no_liked(self, artifacts):
        recs = get_recommendations("unknown_user", None, artifacts, n=5)
        assert recs == []

    # The returned list must not exceed n items; violating this would break the API's
    # count field and potentially overwhelm client-side rendering.
    def test_respects_n_parameter(self, artifacts):
        recs = get_recommendations("user_party", None, artifacts, n=2)
        assert len(recs) <= 2

    # Each recommendation dict must contain track_id, score, and reason so the API
    # route can serialize and enrich results without a KeyError.
    def test_result_has_required_keys(self, artifacts):
        recs = get_recommendations("user_indie", None, artifacts, n=3)
        for r in recs:
            assert "track_id" in r
            assert "score" in r
            assert "reason" in r

    # Results must be ordered by descending score so the highest-confidence
    # recommendations appear first, as callers expect.
    def test_scores_are_sorted_descending(self, artifacts):
        recs = get_recommendations("user_workout", None, artifacts, n=5)
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    # Scores must be rounded to four decimal places to keep API responses readable
    # without losing meaningful precision in the ranking signal.
    def test_scores_rounded_to_four_decimal_places(self, artifacts):
        recs = get_recommendations("user_energetic", None, artifacts, n=3)
        for r in recs:
            assert round(r["score"], 4) == r["score"]


# ------------------------------------------------------------------ #
# train.run() end-to-end                                               #
# ------------------------------------------------------------------ #

class TestTrainRun:
    # The run() function must write a .pkl file at the expected path so predict.py
    # can load it; a missing file would cause a FileNotFoundError at API startup.
    def test_saves_pkl_file(self, tmp_path):
        gold = tmp_path / "gold"
        gold.mkdir()
        models = tmp_path / "models"
        _make_dim_tracks(10).to_parquet(gold / "dim_tracks.parquet", index=False)
        _make_fact_af(10).to_parquet(gold / "fact_audio_features.parquet", index=False)

        out = train_run(gold_dir=gold, models_dir=models)

        assert out == models / "recommendation_model.pkl"
        assert out.exists()

    # The saved artifacts dict must contain content, collab, and trained_at so the
    # inference layer and monitoring tools can load all required components.
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

    # Missing dim_tracks must raise FileNotFoundError immediately rather than proceeding
    # to train a model with no track data, which would be silently wrong.
    def test_raises_when_dim_tracks_missing(self, tmp_path):
        gold = tmp_path / "gold"
        gold.mkdir()
        _make_fact_af(5).to_parquet(gold / "fact_audio_features.parquet", index=False)
        with pytest.raises(FileNotFoundError):
            train_run(gold_dir=gold, models_dir=tmp_path / "models")

    # Missing audio features must also raise immediately; training without feature
    # data would produce a model that cannot generate meaningful recommendations.
    def test_raises_when_audio_features_missing(self, tmp_path):
        gold = tmp_path / "gold"
        gold.mkdir()
        _make_dim_tracks(5).to_parquet(gold / "dim_tracks.parquet", index=False)
        with pytest.raises(FileNotFoundError):
            train_run(gold_dir=gold, models_dir=tmp_path / "models")

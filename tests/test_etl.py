import json
import pytest
import pandas as pd

from src.utils.data_quality import (
    DataQualityError,
    DataQualityReport,
    assert_quality,
    check_duplicates,
    check_nulls,
    check_schema,
    run_quality_checks,
)
from src.etl.bronze_to_silver import (
    _categorize_genres,
    transform_audio_features,
    transform_artists,
    transform_tracks,
    run as b2s_run,
)
from src.etl.silver_to_gold import (
    build_dim_artists,
    build_dim_tracks,
    build_fact_audio_features,
    run as s2g_run,
)


# ------------------------------------------------------------------ #
# Sample data factories                                               #
# ------------------------------------------------------------------ #

def make_track(track_id="t1", name="Track One", popularity=70,
               duration_ms=200000, explicit=False,
               artist_id="a1", artist_name="Artist One",
               album_id="al1", album_name="Album One", release_date="2023-01-15"):
    return {
        "id": track_id, "name": name, "popularity": popularity,
        "duration_ms": duration_ms, "explicit": explicit,
        "artists": [{"id": artist_id, "name": artist_name}],
        "album": {"id": album_id, "name": album_name, "release_date": release_date},
    }


def make_audio_feature(track_id="t1", danceability=0.8, energy=0.7,
                       key=5, loudness=-8.0, mode=1, speechiness=0.05,
                       acousticness=0.1, instrumentalness=0.0,
                       liveness=0.1, valence=0.6, tempo=120.0,
                       duration_ms=200000, time_signature=4):
    return {
        "id": track_id, "danceability": danceability, "energy": energy,
        "key": key, "loudness": loudness, "mode": mode,
        "speechiness": speechiness, "acousticness": acousticness,
        "instrumentalness": instrumentalness, "liveness": liveness,
        "valence": valence, "tempo": tempo,
        "duration_ms": duration_ms, "time_signature": time_signature,
    }


def make_artist(artist_id="a1", name="Artist One", popularity=80,
                followers=500000, genres=None):
    return {
        "id": artist_id, "name": name, "popularity": popularity,
        "followers": {"total": followers},
        "genres": genres if genres is not None else ["pop", "dance pop"],
    }


# ------------------------------------------------------------------ #
# Data quality                                                        #
# ------------------------------------------------------------------ #

class TestDataQuality:
    def test_check_nulls_clean(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        assert check_nulls(df, ["a", "b"]) == {}

    def test_check_nulls_detects_missing(self):
        df = pd.DataFrame({"a": [1, None], "b": ["x", "y"]})
        result = check_nulls(df, ["a", "b"])
        assert result == {"a": 1}

    def test_check_nulls_ignores_non_required_cols(self):
        df = pd.DataFrame({"a": [1, None], "b": [None, None]})
        assert check_nulls(df, ["a"]) == {"a": 1}

    def test_check_schema_missing_col(self):
        df = pd.DataFrame({"a": [1]})
        errors = check_schema(df, ["a", "b"])
        assert any("b" in e for e in errors)

    def test_check_schema_passes(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert check_schema(df, ["a", "b"]) == []

    def test_check_duplicates_finds_dups(self):
        df = pd.DataFrame({"id": ["x", "x", "y"]})
        assert check_duplicates(df, ["id"]) == 1

    def test_check_duplicates_clean(self):
        df = pd.DataFrame({"id": ["x", "y"]})
        assert check_duplicates(df, ["id"]) == 0

    def test_check_duplicates_missing_key_col(self):
        df = pd.DataFrame({"a": [1, 2]})
        assert check_duplicates(df, ["nonexistent"]) == 0

    def test_run_quality_checks_passed(self):
        df = pd.DataFrame({"id": ["a", "b"], "val": [1, 2]})
        report = run_quality_checks(df, "test", required_cols=["id"], key_cols=["id"])
        assert report.passed is True
        assert report.row_count == 2

    def test_run_quality_checks_failed_nulls(self):
        df = pd.DataFrame({"id": ["a", None]})
        report = run_quality_checks(df, "test", required_cols=["id"], key_cols=["id"])
        assert report.passed is False
        assert "id" in report.null_counts

    def test_assert_quality_raises_on_failure(self):
        report = DataQualityReport(
            table_name="test", row_count=1,
            null_counts={"col": 1}, duplicate_count=0, schema_errors=[],
        )
        with pytest.raises(DataQualityError):
            assert_quality(report)

    def test_assert_quality_passes_silently(self):
        report = DataQualityReport(table_name="test", row_count=5)
        assert_quality(report)  # should not raise


# ------------------------------------------------------------------ #
# Bronze → Silver                                                     #
# ------------------------------------------------------------------ #

class TestTransformTracks:
    def test_basic(self):
        df = transform_tracks([make_track()])
        assert len(df) == 1
        assert "track_id" in df.columns
        assert df.iloc[0]["track_id"] == "t1"

    def test_renames_id_to_track_id(self):
        df = transform_tracks([make_track()])
        assert "id" not in df.columns
        assert "track_id" in df.columns

    def test_deduplication(self):
        df = transform_tracks([make_track("t1"), make_track("t1", name="Dup")])
        assert len(df) == 1

    def test_drops_null_name(self):
        bad = make_track()
        bad["name"] = None
        df = transform_tracks([bad, make_track("t2", name="Good")])
        assert len(df) == 1
        assert df.iloc[0]["track_id"] == "t2"

    def test_drops_null_id(self):
        bad = make_track()
        bad["id"] = None
        df = transform_tracks([bad])
        assert df.empty

    def test_extracts_primary_artist(self):
        df = transform_tracks([make_track(artist_id="a99", artist_name="Star")])
        assert df.iloc[0]["primary_artist_id"] == "a99"
        assert df.iloc[0]["primary_artist_name"] == "Star"

    def test_release_date_year_only(self):
        t = make_track(release_date="1985")
        df = transform_tracks([t])
        # year-only dates parse to Jan 1 of that year, not NaT
        assert len(df) == 1

    def test_release_date_full_date(self):
        df = transform_tracks([make_track(release_date="2023-06-15")])
        assert pd.notna(df.iloc[0]["release_date"])

    def test_empty_input(self):
        assert transform_tracks([]).empty

    def test_null_item_skipped(self):
        df = transform_tracks([None, make_track("t1")])
        assert len(df) == 1


class TestTransformAudioFeatures:
    def test_basic(self):
        df = transform_audio_features([make_audio_feature()])
        assert len(df) == 1
        assert "track_id" in df.columns

    def test_deduplication(self):
        df = transform_audio_features([make_audio_feature("t1"), make_audio_feature("t1")])
        assert len(df) == 1

    def test_drops_null_id(self):
        bad = make_audio_feature()
        bad["id"] = None
        df = transform_audio_features([bad])
        assert df.empty

    def test_key_normalization_max(self):
        df = transform_audio_features([make_audio_feature(key=11)])
        assert abs(df.iloc[0]["key"] - 1.0) < 1e-6

    def test_key_normalization_min(self):
        df = transform_audio_features([make_audio_feature(key=0)])
        assert abs(df.iloc[0]["key"] - 0.0) < 1e-6

    def test_loudness_normalization(self):
        df_max = transform_audio_features([make_audio_feature(loudness=0)])
        assert abs(df_max.iloc[0]["loudness"] - 1.0) < 1e-6
        df_min = transform_audio_features([make_audio_feature(loudness=-60)])
        assert abs(df_min.iloc[0]["loudness"] - 0.0) < 1e-6

    def test_tempo_normalization(self):
        df_max = transform_audio_features([make_audio_feature(tempo=250)])
        assert abs(df_max.iloc[0]["tempo"] - 1.0) < 1e-6
        df_min = transform_audio_features([make_audio_feature(tempo=50)])
        assert abs(df_min.iloc[0]["tempo"] - 0.0) < 1e-6

    def test_unit_features_clipped(self):
        # Spotify guarantees 0-1 but we clip just in case
        df = transform_audio_features([make_audio_feature(danceability=1.5, energy=-0.1)])
        assert df.iloc[0]["danceability"] == 1.0
        assert df.iloc[0]["energy"] == 0.0

    def test_empty_input(self):
        assert transform_audio_features([]).empty


class TestTransformArtists:
    def test_basic(self):
        df = transform_artists([make_artist()])
        assert len(df) == 1
        assert "artist_id" in df.columns

    def test_deduplication(self):
        df = transform_artists([make_artist("a1"), make_artist("a1")])
        assert len(df) == 1

    def test_drops_null_id(self):
        bad = make_artist()
        bad["id"] = None
        df = transform_artists([bad])
        assert df.empty

    def test_genre_categorization_pop(self):
        df = transform_artists([make_artist(genres=["pop", "dance pop"])])
        assert df.iloc[0]["primary_genre"] == "pop"

    def test_genre_categorization_electronic(self):
        df = transform_artists([make_artist(genres=["deep house", "progressive house"])])
        assert df.iloc[0]["primary_genre"] == "electronic"

    def test_genre_categorization_hiphop(self):
        df = transform_artists([make_artist(genres=["trap", "rap"])])
        assert df.iloc[0]["primary_genre"] == "hip-hop"

    def test_empty_genres_returns_other(self):
        df = transform_artists([make_artist(genres=[])])
        assert df.iloc[0]["primary_genre"] == "other"

    def test_unknown_genre_returns_other(self):
        df = transform_artists([make_artist(genres=["zzz obscure genre"])])
        assert df.iloc[0]["primary_genre"] == "other"

    def test_genres_joined_as_string(self):
        df = transform_artists([make_artist(genres=["pop", "dance pop"])])
        assert df.iloc[0]["genres"] == "pop,dance pop"

    def test_empty_input(self):
        assert transform_artists([]).empty


class TestCategorizeGenres:
    def test_rock(self):
        assert _categorize_genres(["indie rock", "alternative rock"]) == "rock"

    def test_jazz(self):
        assert _categorize_genres(["bebop", "jazz fusion"]) == "jazz"

    def test_rnb(self):
        assert _categorize_genres(["neo soul", "r&b"]) == "r-n-b"

    def test_classical(self):
        assert _categorize_genres(["orchestra", "chamber music"]) == "classical"

    def test_empty(self):
        assert _categorize_genres([]) == "other"


class TestBronzeToSilverRun:
    def test_run_end_to_end(self, tmp_path):
        bronze = tmp_path / "bronze"
        silver = tmp_path / "silver"

        # Write fake bronze files
        (bronze / "tracks" / "2026-01-01").mkdir(parents=True)
        (bronze / "audio_features" / "2026-01-01").mkdir(parents=True)
        (bronze / "artists" / "2026-01-01").mkdir(parents=True)

        (bronze / "tracks" / "2026-01-01" / "tracks_20260101T000000Z.json").write_text(
            json.dumps([make_track("t1"), make_track("t2", artist_id="a2")])
        )
        (bronze / "audio_features" / "2026-01-01" / "audio_features_20260101T000000Z.json").write_text(
            json.dumps([make_audio_feature("t1"), make_audio_feature("t2")])
        )
        (bronze / "artists" / "2026-01-01" / "artists_20260101T000000Z.json").write_text(
            json.dumps([make_artist("a1"), make_artist("a2", name="Artist Two")])
        )

        reports = b2s_run(bronze_dir=bronze, silver_dir=silver)

        assert "tracks" in reports
        assert "audio_features" in reports
        assert "artists" in reports
        assert reports["tracks"].row_count == 2
        assert (silver / "tracks.parquet").exists()
        assert (silver / "audio_features.parquet").exists()
        assert (silver / "artists.parquet").exists()

    def test_run_skips_missing_bronze_type(self, tmp_path):
        bronze = tmp_path / "bronze"
        silver = tmp_path / "silver"
        (bronze / "tracks" / "2026-01-01").mkdir(parents=True)
        (bronze / "tracks" / "2026-01-01" / "tracks.json").write_text(
            json.dumps([make_track()])
        )
        # No audio_features or artists bronze files
        reports = b2s_run(bronze_dir=bronze, silver_dir=silver)
        assert "tracks" in reports
        assert "audio_features" not in reports
        assert "artists" not in reports

    def test_skips_malformed_json(self, tmp_path):
        from src.etl.bronze_to_silver import _load_bronze_files
        bronze = tmp_path / "bronze"
        (bronze / "tracks").mkdir(parents=True)
        (bronze / "tracks" / "bad.json").write_text("{not valid json}")
        (bronze / "tracks" / "good.json").write_text(json.dumps([make_track()]))
        records = _load_bronze_files("tracks", bronze)
        assert len(records) == 1

    def test_skips_non_list_json(self, tmp_path):
        from src.etl.bronze_to_silver import _load_bronze_files
        bronze = tmp_path / "bronze"
        (bronze / "tracks").mkdir(parents=True)
        (bronze / "tracks" / "file.json").write_text(json.dumps({"id": "t1"}))
        records = _load_bronze_files("tracks", bronze)
        assert records == []

    def test_quality_failure_prevents_write(self, tmp_path, monkeypatch):
        from src.etl import bronze_to_silver

        bronze = tmp_path / "bronze"
        silver = tmp_path / "silver"
        (bronze / "tracks" / "2026-01-01").mkdir(parents=True)
        (bronze / "tracks" / "2026-01-01" / "tracks.json").write_text(
            json.dumps([make_track()])
        )

        failing_report = DataQualityReport(
            table_name="silver/tracks", row_count=1,
            null_counts={"track_id": 1}, duplicate_count=0, schema_errors=[],
        )
        monkeypatch.setattr(bronze_to_silver, "run_quality_checks", lambda *a, **kw: failing_report)

        with pytest.raises(DataQualityError):
            b2s_run(bronze_dir=bronze, silver_dir=silver)

        assert not (silver / "tracks.parquet").exists()


# ------------------------------------------------------------------ #
# Silver → Gold                                                       #
# ------------------------------------------------------------------ #

def _make_silver_tracks(rows=None):
    data = rows or [
        {"track_id": "t1", "name": "Track One", "popularity": 80, "duration_ms": 210000,
         "explicit": False, "primary_artist_id": "a1", "primary_artist_name": "Artist One",
         "album_id": "al1", "album_name": "Album One", "release_date": pd.Timestamp("2023-01-15")},
        {"track_id": "t2", "name": "Track Two", "popularity": 60, "duration_ms": 180000,
         "explicit": True, "primary_artist_id": "a2", "primary_artist_name": "Artist Two",
         "album_id": "al2", "album_name": "Album Two", "release_date": pd.Timestamp("2022-06-01")},
    ]
    return pd.DataFrame(data)


def _make_silver_artists(rows=None):
    data = rows or [
        {"artist_id": "a1", "name": "Artist One", "popularity": 90, "followers": 500000,
         "genres": "pop,dance pop", "primary_genre": "pop"},
        {"artist_id": "a2", "name": "Artist Two", "popularity": 70, "followers": 200000,
         "genres": "rock,indie rock", "primary_genre": "rock"},
    ]
    return pd.DataFrame(data)


def _make_silver_features(rows=None):
    data = rows or [
        {"track_id": "t1", "danceability": 0.8, "energy": 0.7, "key": 0.45,
         "loudness": 0.87, "mode": 1, "speechiness": 0.05, "acousticness": 0.1,
         "instrumentalness": 0.0, "liveness": 0.1, "valence": 0.6,
         "tempo": 0.35, "duration_ms": 210000, "time_signature": 0.25},
        {"track_id": "t2", "danceability": 0.5, "energy": 0.6, "key": 0.27,
         "loudness": 0.7, "mode": 0, "speechiness": 0.1, "acousticness": 0.3,
         "instrumentalness": 0.0, "liveness": 0.2, "valence": 0.4,
         "tempo": 0.5, "duration_ms": 180000, "time_signature": 0.25},
    ]
    return pd.DataFrame(data)


class TestBuildDimTracks:
    def test_basic(self):
        dim = build_dim_tracks(_make_silver_tracks(), _make_silver_artists())
        assert len(dim) == 2
        assert "composite_popularity" in dim.columns
        assert "primary_genre" in dim.columns

    def test_composite_popularity_formula(self):
        tracks = pd.DataFrame([{
            "track_id": "t1", "name": "T", "popularity": 80, "duration_ms": 200000,
            "explicit": False, "primary_artist_id": "a1", "primary_artist_name": "A",
            "album_id": "al1", "album_name": "Al", "release_date": None,
        }])
        artists = pd.DataFrame([{
            "artist_id": "a1", "name": "A", "popularity": 60,
            "followers": 100, "genres": "pop", "primary_genre": "pop",
        }])
        dim = build_dim_tracks(tracks, artists)
        expected = 0.6 * (80 / 100) + 0.4 * (60 / 100)
        assert abs(dim.iloc[0]["composite_popularity"] - expected) < 1e-6

    def test_no_artists_fallback(self):
        dim = build_dim_tracks(_make_silver_tracks(), None)
        assert len(dim) == 2
        assert (dim["primary_genre"] == "unknown").all()
        # all artist_popularity are NaN → median is NaN → fill is 0; composite ≤ 0.6
        assert all(dim["composite_popularity"] <= 0.6)

    def test_unmatched_artist_still_present(self):
        tracks = _make_silver_tracks()
        # artist_id "a1" is not in artists — unmatched track should get "unknown" genre, not NaN
        artists = _make_silver_artists().query("artist_id == 'a2'").reset_index(drop=True)
        dim = build_dim_tracks(tracks, artists)
        assert len(dim) == 2  # both tracks present
        t1 = dim[dim["track_id"] == "t1"].iloc[0]
        assert t1["primary_genre"] == "unknown"  # NaN would be a data leak


class TestBuildDimArtists:
    def test_basic(self):
        dim = build_dim_artists(_make_silver_artists())
        assert "artist_name" in dim.columns
        assert "name" not in dim.columns

    def test_row_count(self):
        dim = build_dim_artists(_make_silver_artists())
        assert len(dim) == 2

    def test_columns_present(self):
        dim = build_dim_artists(_make_silver_artists())
        for col in ["artist_id", "artist_name", "popularity", "followers", "primary_genre"]:
            assert col in dim.columns


class TestBuildFactAudioFeatures:
    def test_basic(self):
        fact = build_fact_audio_features(_make_silver_features(), _make_silver_tracks())
        assert "track_id" in fact.columns
        assert "primary_artist_id" in fact.columns
        assert len(fact) == 2

    def test_joins_primary_artist_id(self):
        fact = build_fact_audio_features(_make_silver_features(), _make_silver_tracks())
        assert fact[fact["track_id"] == "t1"].iloc[0]["primary_artist_id"] == "a1"

    def test_no_tracks_fallback(self):
        fact = build_fact_audio_features(_make_silver_features(), None)
        assert len(fact) == 2
        assert "track_id" in fact.columns


class TestSilverToGoldRun:
    def test_run_end_to_end(self, tmp_path):
        silver = tmp_path / "silver"
        gold = tmp_path / "gold"
        silver.mkdir()

        _make_silver_tracks().to_parquet(silver / "tracks.parquet", index=False)
        _make_silver_features().to_parquet(silver / "audio_features.parquet", index=False)
        _make_silver_artists().to_parquet(silver / "artists.parquet", index=False)

        reports = s2g_run(silver_dir=silver, gold_dir=gold)

        assert "dim_tracks" in reports
        assert "dim_artists" in reports
        assert "fact_audio_features" in reports
        assert (gold / "dim_tracks.parquet").exists()
        assert (gold / "dim_artists.parquet").exists()
        assert (gold / "fact_audio_features.parquet").exists()

    def test_run_returns_empty_when_tracks_missing(self, tmp_path):
        silver = tmp_path / "silver"
        gold = tmp_path / "gold"
        silver.mkdir()
        # No tracks.parquet
        reports = s2g_run(silver_dir=silver, gold_dir=gold)
        assert reports == {}

    def test_run_without_artists(self, tmp_path):
        silver = tmp_path / "silver"
        gold = tmp_path / "gold"
        silver.mkdir()
        _make_silver_tracks().to_parquet(silver / "tracks.parquet", index=False)
        _make_silver_features().to_parquet(silver / "audio_features.parquet", index=False)
        # No artists.parquet

        reports = s2g_run(silver_dir=silver, gold_dir=gold)
        assert "dim_tracks" in reports
        assert "dim_artists" not in reports

    def test_quality_failure_prevents_write(self, tmp_path, monkeypatch):
        from src.etl import silver_to_gold

        silver = tmp_path / "silver"
        gold = tmp_path / "gold"
        silver.mkdir()
        _make_silver_tracks().to_parquet(silver / "tracks.parquet", index=False)

        failing_report = DataQualityReport(
            table_name="gold/dim_tracks", row_count=2,
            null_counts={"track_id": 1}, duplicate_count=0, schema_errors=[],
        )
        monkeypatch.setattr(silver_to_gold, "run_quality_checks", lambda *a, **kw: failing_report)

        with pytest.raises(DataQualityError):
            s2g_run(silver_dir=silver, gold_dir=gold)

        assert not (gold / "dim_tracks.parquet").exists()


# ------------------------------------------------------------------ #
# Pipeline                                                            #
# ------------------------------------------------------------------ #

class TestPipeline:
    def test_pipeline_calls_both_stages(self, monkeypatch):
        from src.etl import pipeline, bronze_to_silver, silver_to_gold
        from src.utils.data_quality import DataQualityReport

        b2s_calls = []
        s2g_calls = []

        def fake_b2s(**_):
            b2s_calls.append(1)
            return {"tracks": DataQualityReport("tracks", 10)}

        def fake_s2g(**_):
            s2g_calls.append(1)
            return {"dim_tracks": DataQualityReport("dim_tracks", 10)}

        monkeypatch.setattr(bronze_to_silver, "run", fake_b2s)
        monkeypatch.setattr(silver_to_gold, "run", fake_s2g)

        result = pipeline.spotify_etl_pipeline()
        assert len(b2s_calls) == 1
        assert len(s2g_calls) == 1
        assert "silver" in result
        assert "gold" in result

    def test_pipeline_result_structure(self, monkeypatch):
        from src.etl import pipeline, bronze_to_silver, silver_to_gold
        from src.utils.data_quality import DataQualityReport

        monkeypatch.setattr(bronze_to_silver, "run", lambda **_: {
            "tracks": DataQualityReport("tracks", 50),
        })
        monkeypatch.setattr(silver_to_gold, "run", lambda **_: {
            "dim_tracks": DataQualityReport("dim_tracks", 50),
        })

        result = pipeline.spotify_etl_pipeline()
        assert result["silver"] == {"tracks": 50}
        assert result["gold"] == {"dim_tracks": 50}

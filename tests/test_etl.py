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
    # A fully populated DataFrame with no missing values must return an empty dict;
    # this is the expected baseline behaviour for clean data.
    def test_check_nulls_clean(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        assert check_nulls(df, ["a", "b"]) == {}

    # Missing values in required columns must be detected and reported so bad data
    # doesn't silently flow into the silver or gold layers.
    def test_check_nulls_detects_missing(self):
        df = pd.DataFrame({"a": [1, None], "b": ["x", "y"]})
        result = check_nulls(df, ["a", "b"])
        assert result == {"a": 1}

    # Columns not in the required list must not appear in the result; widening the
    # check to all columns would generate false positives on optional fields.
    def test_check_nulls_ignores_non_required_cols(self):
        df = pd.DataFrame({"a": [1, None], "b": [None, None]})
        assert check_nulls(df, ["a"]) == {"a": 1}

    # A missing required column must appear in schema errors so callers immediately
    # know the table is structurally invalid before attempting to use it.
    def test_check_schema_missing_col(self):
        df = pd.DataFrame({"a": [1]})
        errors = check_schema(df, ["a", "b"])
        assert any("b" in e for e in errors)

    # All required columns present must produce no errors; confirms the check does not
    # generate false positives on valid tables.
    def test_check_schema_passes(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert check_schema(df, ["a", "b"]) == []

    # Duplicate key rows must be counted accurately; undetected duplicates would inflate
    # aggregation metrics and cause fan-out in downstream joins.
    def test_check_duplicates_finds_dups(self):
        df = pd.DataFrame({"id": ["x", "x", "y"]})
        assert check_duplicates(df, ["id"]) == 1

    # A deduplicated table must return zero so the check doesn't produce false positives
    # on clean data.
    def test_check_duplicates_clean(self):
        df = pd.DataFrame({"id": ["x", "y"]})
        assert check_duplicates(df, ["id"]) == 0

    # If the specified key column doesn't exist in the DataFrame, the function must
    # return zero rather than raising a KeyError.
    def test_check_duplicates_missing_key_col(self):
        df = pd.DataFrame({"a": [1, 2]})
        assert check_duplicates(df, ["nonexistent"]) == 0

    # A clean DataFrame must produce a passing report with the correct row count;
    # this validates the full quality check pipeline end-to-end.
    def test_run_quality_checks_passed(self):
        df = pd.DataFrame({"id": ["a", "b"], "val": [1, 2]})
        report = run_quality_checks(df, "test", required_cols=["id"], key_cols=["id"])
        assert report.passed is True
        assert report.row_count == 2

    # Nulls in a required column must flip report.passed to False and populate null_counts
    # so the pipeline can block the write and surface a clear error.
    def test_run_quality_checks_failed_nulls(self):
        df = pd.DataFrame({"id": ["a", None]})
        report = run_quality_checks(df, "test", required_cols=["id"], key_cols=["id"])
        assert report.passed is False
        assert "id" in report.null_counts

    # A failing report must raise DataQualityError so callers can halt processing
    # and prevent bad data from advancing through the pipeline.
    def test_assert_quality_raises_on_failure(self):
        report = DataQualityReport(
            table_name="test", row_count=1,
            null_counts={"col": 1}, duplicate_count=0, schema_errors=[],
        )
        with pytest.raises(DataQualityError):
            assert_quality(report)

    # A passing report must not raise; the guard should fire only on actual failures
    # so clean data flows through without interruption.
    def test_assert_quality_passes_silently(self):
        report = DataQualityReport(table_name="test", row_count=5)
        assert_quality(report)  # should not raise

    # An empty table signals a broken upstream stage; it must fail quality checks
    # so the pipeline does not write zero-row Parquet files to the data lake.
    def test_empty_dataframe_fails_quality_check(self):
        df = pd.DataFrame({"id": pd.Series([], dtype=str)})
        report = run_quality_checks(df, "test", required_cols=["id"], key_cols=["id"])
        assert report.passed is False
        assert report.row_count == 0

    # The raised error message must specifically identify the empty-table cause so
    # engineers can diagnose the failure without inspecting log files.
    def test_assert_quality_raises_on_empty_dataframe(self):
        report = DataQualityReport(table_name="test", row_count=0)
        with pytest.raises(DataQualityError) as exc_info:
            assert_quality(report)
        assert "empty (0 rows)" in str(exc_info.value)


# ------------------------------------------------------------------ #
# Bronze → Silver                                                     #
# ------------------------------------------------------------------ #

class TestTransformTracks:
    # A single well-formed track record must produce exactly one output row with the
    # expected track_id column, validating the core transform path.
    def test_basic(self):
        df = transform_tracks([make_track()])
        assert len(df) == 1
        assert "track_id" in df.columns
        assert df.iloc[0]["track_id"] == "t1"

    # The raw Spotify "id" field must be renamed to "track_id" since all downstream
    # joins and gold-layer schemas depend on that column name.
    def test_renames_id_to_track_id(self):
        df = transform_tracks([make_track()])
        assert "id" not in df.columns
        assert "track_id" in df.columns

    # Duplicate track IDs must be collapsed to a single row to prevent fan-out
    # when joining to audio features or artist tables.
    def test_deduplication(self):
        df = transform_tracks([make_track("t1"), make_track("t1", name="Dup")])
        assert len(df) == 1

    # Tracks without a name are not useful for recommendations or display and must
    # be silently dropped rather than propagating null names downstream.
    def test_drops_null_name(self):
        bad = make_track()
        bad["name"] = None
        df = transform_tracks([bad, make_track("t2", name="Good")])
        assert len(df) == 1
        assert df.iloc[0]["track_id"] == "t2"

    # A track without an ID cannot be joined or referenced; it must be excluded to
    # prevent orphaned rows in the silver and gold layers.
    def test_drops_null_id(self):
        bad = make_track()
        bad["id"] = None
        df = transform_tracks([bad])
        assert df.empty

    # The first artist's ID and name must be promoted to primary_artist_id and
    # primary_artist_name because the gold layer join uses these top-level columns.
    def test_extracts_primary_artist(self):
        df = transform_tracks([make_track(artist_id="a99", artist_name="Star")])
        assert df.iloc[0]["primary_artist_id"] == "a99"
        assert df.iloc[0]["primary_artist_name"] == "Star"

    # Spotify returns some release dates as a bare year string; these must parse to
    # a valid date (Jan 1 of that year) rather than NaT to avoid null date columns.
    def test_release_date_year_only(self):
        t = make_track(release_date="1985")
        df = transform_tracks([t])
        # year-only dates parse to Jan 1 of that year, not NaT
        assert len(df) == 1

    # A full YYYY-MM-DD release date must parse to a non-null Timestamp confirming
    # the standard date format is handled correctly.
    def test_release_date_full_date(self):
        df = transform_tracks([make_track(release_date="2023-06-15")])
        assert pd.notna(df.iloc[0]["release_date"])

    # An empty input list must return an empty DataFrame without raising so callers
    # don't need to guard against the empty case before calling transform_tracks.
    def test_empty_input(self):
        assert transform_tracks([]).empty

    # None entries in the input list (from partial API responses) must be silently
    # skipped rather than causing a KeyError or AttributeError.
    def test_null_item_skipped(self):
        df = transform_tracks([None, make_track("t1")])
        assert len(df) == 1


class TestTransformAudioFeatures:
    # A single well-formed audio feature record must produce one output row with the
    # expected track_id column confirming the base transform works.
    def test_basic(self):
        df = transform_audio_features([make_audio_feature()])
        assert len(df) == 1
        assert "track_id" in df.columns

    # Duplicate audio feature records for the same track must be collapsed to one row
    # to avoid incorrect row counts in the fact_audio_features table.
    def test_deduplication(self):
        df = transform_audio_features([make_audio_feature("t1"), make_audio_feature("t1")])
        assert len(df) == 1

    # Audio features with no track ID cannot be joined to any track and must be
    # excluded to prevent orphaned rows in the fact table.
    def test_drops_null_id(self):
        bad = make_audio_feature()
        bad["id"] = None
        df = transform_audio_features([bad])
        assert df.empty

    # Musical key 11 (B) is the maximum value and must normalize to exactly 1.0;
    # an off-by-one would shift every key feature in the model.
    def test_key_normalization_max(self):
        df = transform_audio_features([make_audio_feature(key=11)])
        assert abs(df.iloc[0]["key"] - 1.0) < 1e-6

    # Musical key 0 (C) is the minimum and must normalize to exactly 0.0, confirming
    # the lower bound of the scale is correct.
    def test_key_normalization_min(self):
        df = transform_audio_features([make_audio_feature(key=0)])
        assert abs(df.iloc[0]["key"] - 0.0) < 1e-6

    # Loudness 0 dB (the ceiling) must map to 1.0; this confirms the direction of the
    # loudness normalization (louder = higher) is correct.
    def test_loudness_normalization(self):
        df_max = transform_audio_features([make_audio_feature(loudness=0)])
        assert abs(df_max.iloc[0]["loudness"] - 1.0) < 1e-6
        df_min = transform_audio_features([make_audio_feature(loudness=-60)])
        assert abs(df_min.iloc[0]["loudness"] - 0.0) < 1e-6

    # Tempo at both extremes of its [50, 250] BPM range must map to [0, 1]; incorrect
    # bounds would silently compress or expand the tempo distribution used by the model.
    def test_tempo_normalization(self):
        df_max = transform_audio_features([make_audio_feature(tempo=250)])
        assert abs(df_max.iloc[0]["tempo"] - 1.0) < 1e-6
        df_min = transform_audio_features([make_audio_feature(tempo=50)])
        assert abs(df_min.iloc[0]["tempo"] - 0.0) < 1e-6

    # Values outside [0, 1] from upstream data anomalies must be clipped so the
    # unit-scale invariant assumed by the recommendation model is always preserved.
    def test_unit_features_clipped(self):
        # Spotify guarantees 0-1 but we clip just in case
        df = transform_audio_features([make_audio_feature(danceability=1.5, energy=-0.1)])
        assert df.iloc[0]["danceability"] == 1.0
        assert df.iloc[0]["energy"] == 0.0

    # An empty list must return an empty DataFrame without raising so callers don't
    # need to handle the empty case separately.
    def test_empty_input(self):
        assert transform_audio_features([]).empty


class TestTransformArtists:
    # A single well-formed artist record must produce one output row with the expected
    # artist_id column confirming the base transform works.
    def test_basic(self):
        df = transform_artists([make_artist()])
        assert len(df) == 1
        assert "artist_id" in df.columns

    # Duplicate artist IDs must be collapsed to one row; duplicate rows would inflate
    # join results when linking tracks to their artists.
    def test_deduplication(self):
        df = transform_artists([make_artist("a1"), make_artist("a1")])
        assert len(df) == 1

    # Artists without an ID cannot be referenced by any track and must be excluded
    # to prevent null artist_id values in the gold layer.
    def test_drops_null_id(self):
        bad = make_artist()
        bad["id"] = None
        df = transform_artists([bad])
        assert df.empty

    # Pop subgenres must resolve to the "pop" category, validating that the catch-all
    # pattern at the end of the genre matching chain works correctly.
    def test_genre_categorization_pop(self):
        df = transform_artists([make_artist(genres=["pop", "dance pop"])])
        assert df.iloc[0]["primary_genre"] == "pop"

    # Subgenres like "deep house" and "progressive house" must map to "electronic"
    # via substring matching rather than falling through to "other".
    def test_genre_categorization_electronic(self):
        df = transform_artists([make_artist(genres=["deep house", "progressive house"])])
        assert df.iloc[0]["primary_genre"] == "electronic"

    # Trap and rap subgenres must map to "hip-hop", validating that the hip-hop
    # pattern correctly captures common subgenre strings.
    def test_genre_categorization_hiphop(self):
        df = transform_artists([make_artist(genres=["trap", "rap"])])
        assert df.iloc[0]["primary_genre"] == "hip-hop"

    # An artist with no genres listed must get "other" as their primary genre rather
    # than a null value, which would break the gold-layer genre join.
    def test_empty_genres_returns_other(self):
        df = transform_artists([make_artist(genres=[])])
        assert df.iloc[0]["primary_genre"] == "other"

    # Unrecognized genre strings must fall back to "other" instead of leaving
    # primary_genre null, which would propagate unknown values into recommendations.
    def test_unknown_genre_returns_other(self):
        df = transform_artists([make_artist(genres=["zzz obscure genre"])])
        assert df.iloc[0]["primary_genre"] == "other"

    # The genres list must be stored as a comma-separated string for Parquet and API
    # compatibility; leaving it as a Python list would cause serialisation errors.
    def test_genres_joined_as_string(self):
        df = transform_artists([make_artist(genres=["pop", "dance pop"])])
        assert df.iloc[0]["genres"] == "pop,dance pop"

    # An empty list must return an empty DataFrame without raising so callers don't
    # need to guard against the empty case.
    def test_empty_input(self):
        assert transform_artists([]).empty


class TestCategorizeGenres:
    # Rock subgenres must resolve to "rock", validating that the ordered substring
    # matching correctly handles genre specificity before reaching the pop catch-all.
    def test_rock(self):
        assert _categorize_genres(["indie rock", "alternative rock"]) == "rock"

    # Bebop and jazz fusion must resolve to "jazz", confirming that jazz subgenre
    # strings are matched before the more generic patterns.
    def test_jazz(self):
        assert _categorize_genres(["bebop", "jazz fusion"]) == "jazz"

    # Neo soul and r&b strings must map to "r-n-b" via substring matching, confirming
    # the pattern covers the common spelling variations.
    def test_rnb(self):
        assert _categorize_genres(["neo soul", "r&b"]) == "r-n-b"

    # Classical subgenre strings must resolve to "classical" so orchestral and chamber
    # music are grouped correctly for content-based filtering.
    def test_classical(self):
        assert _categorize_genres(["orchestra", "chamber music"]) == "classical"

    # An empty genres list must return "other" to prevent null primary_genre values
    # from entering the artist table.
    def test_empty(self):
        assert _categorize_genres([]) == "other"


class TestBronzeToSilverRun:
    # Verifies the full bronze→silver pipeline reads all three data types, transforms
    # them correctly, and writes valid Parquet files to the silver directory.
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

    # If only tracks bronze files are present, the run must still succeed for that data
    # type without raising errors for the absent audio_features and artists types.
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

    # Corrupt JSON files must be skipped with a warning so a single bad file does not
    # abort processing of all other valid files in the same batch.
    def test_skips_malformed_json(self, tmp_path):
        from src.etl.bronze_to_silver import _load_bronze_files
        bronze = tmp_path / "bronze"
        (bronze / "tracks").mkdir(parents=True)
        (bronze / "tracks" / "bad.json").write_text("{not valid json}")
        (bronze / "tracks" / "good.json").write_text(json.dumps([make_track()]))
        records = _load_bronze_files("tracks", bronze)
        assert len(records) == 1

    # Files containing a JSON object instead of a list must be skipped; only list
    # payloads are valid bronze records and objects would break the transform logic.
    def test_skips_non_list_json(self, tmp_path):
        from src.etl.bronze_to_silver import _load_bronze_files
        bronze = tmp_path / "bronze"
        (bronze / "tracks").mkdir(parents=True)
        (bronze / "tracks" / "file.json").write_text(json.dumps({"id": "t1"}))
        records = _load_bronze_files("tracks", bronze)
        assert records == []

    # A failing quality report must raise DataQualityError before any Parquet is written
    # so corrupted data cannot advance to the silver layer.
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
    # Verifies dim_tracks is built with the correct row count and contains both
    # composite_popularity and primary_genre columns required by the gold schema.
    def test_basic(self):
        dim = build_dim_tracks(_make_silver_tracks(), _make_silver_artists())
        assert len(dim) == 2
        assert "composite_popularity" in dim.columns
        assert "primary_genre" in dim.columns

    # Validates the exact 0.6 × track_pop + 0.4 × artist_pop weighting; an incorrect
    # formula would silently skew composite scores for all tracks in the catalog.
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

    # When the artists table is absent, dim_tracks must still be built with "unknown"
    # genre and a valid composite score so the gold layer is never blocked on missing artists.
    def test_no_artists_fallback(self):
        dim = build_dim_tracks(_make_silver_tracks(), None)
        assert len(dim) == 2
        assert (dim["primary_genre"] == "unknown").all()
        # all artist_popularity are NaN → median is NaN → fill is 0; composite ≤ 0.6
        assert all(dim["composite_popularity"] <= 0.6)

    # Tracks whose artist didn't match the join must remain in the output with "unknown"
    # genre rather than being dropped or carrying a null value that would break downstream queries.
    def test_unmatched_artist_still_present(self):
        tracks = _make_silver_tracks()
        # artist_id "a1" is not in artists — unmatched track should get "unknown" genre, not NaN
        artists = _make_silver_artists().query("artist_id == 'a2'").reset_index(drop=True)
        dim = build_dim_tracks(tracks, artists)
        assert len(dim) == 2  # both tracks present
        t1 = dim[dim["track_id"] == "t1"].iloc[0]
        assert t1["primary_genre"] == "unknown"  # NaN would be a data leak


class TestBuildDimArtists:
    # Confirms the "name" column is renamed to "artist_name" so dim_artists column
    # names are consistent with the gold schema used by the API artist endpoint.
    def test_basic(self):
        dim = build_dim_artists(_make_silver_artists())
        assert "artist_name" in dim.columns
        assert "name" not in dim.columns

    # All input artist rows must be preserved in the output without any unexpected
    # filtering or deduplication at this stage.
    def test_row_count(self):
        dim = build_dim_artists(_make_silver_artists())
        assert len(dim) == 2

    # All five required columns must be present in dim_artists; a missing column
    # would cause a KeyError in the API artist detail endpoint.
    def test_columns_present(self):
        dim = build_dim_artists(_make_silver_artists())
        for col in ["artist_id", "artist_name", "popularity", "followers", "primary_genre"]:
            assert col in dim.columns


class TestBuildFactAudioFeatures:
    # Verifies the fact table contains track_id and primary_artist_id with the correct
    # row count, confirming the join from tracks was applied correctly.
    def test_basic(self):
        fact = build_fact_audio_features(_make_silver_features(), _make_silver_tracks())
        assert "track_id" in fact.columns
        assert "primary_artist_id" in fact.columns
        assert len(fact) == 2

    # The join must correctly populate primary_artist_id from dim_tracks so audio features
    # can be filtered or aggregated by artist in ML training and the API.
    def test_joins_primary_artist_id(self):
        fact = build_fact_audio_features(_make_silver_features(), _make_silver_tracks())
        assert fact[fact["track_id"] == "t1"].iloc[0]["primary_artist_id"] == "a1"

    # When the tracks table is absent, the fact table must still be built from audio
    # features alone so audio data is not lost even when track metadata is missing.
    def test_no_tracks_fallback(self):
        fact = build_fact_audio_features(_make_silver_features(), None)
        assert len(fact) == 2
        assert "track_id" in fact.columns


class TestSilverToGoldRun:
    # Verifies the full silver→gold pipeline writes all three gold tables as Parquet
    # when all silver inputs are present, confirming the end-to-end flow works.
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

    # When the mandatory tracks table is absent, the run must fail fast and return an
    # empty dict rather than attempting to build a gold layer with no track data.
    def test_run_returns_empty_when_tracks_missing(self, tmp_path):
        silver = tmp_path / "silver"
        gold = tmp_path / "gold"
        silver.mkdir()
        # No tracks.parquet
        reports = s2g_run(silver_dir=silver, gold_dir=gold)
        assert reports == {}

    # Without an artists table, dim_tracks must still be built while dim_artists is
    # correctly omitted from the reports rather than raising.
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

    # A failing quality check must raise DataQualityError before any Parquet is written
    # so corrupted gold data cannot reach the API or recommendation model.
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

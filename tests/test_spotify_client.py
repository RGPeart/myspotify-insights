import json
import pytest
import spotipy
from unittest.mock import MagicMock, patch

import src.ingestion.spotify_client as spotify_module
from src.ingestion.spotify_client import SpotifyIngestionClient, _batched, _default_genres


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "test_id")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "test_secret")
    monkeypatch.setattr(spotify_module, "BRONZE_DIR", tmp_path / "bronze")
    monkeypatch.setattr(spotify_module, "MANIFEST_PATH", tmp_path / "bronze" / "manifest.json")

    with patch("src.ingestion.spotify_client.SpotifyClientCredentials"), \
         patch("src.ingestion.spotify_client.spotipy.Spotify") as mock_sp_cls:
        mock_sp = MagicMock()
        mock_sp_cls.return_value = mock_sp
        c = SpotifyIngestionClient()
        yield c, mock_sp, tmp_path


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def make_track(track_id: str, artist_id: str = "artist1") -> dict:
    return {
        "id": track_id,
        "name": f"Track {track_id}",
        "artists": [{"id": artist_id, "name": "Artist"}],
        "album": {"id": "album1", "name": "Album"},
        "popularity": 70,
    }


def make_audio_feature(track_id: str) -> dict:
    return {
        "id": track_id,
        "danceability": 0.8,
        "energy": 0.7,
        "valence": 0.6,
        "tempo": 120.0,
    }


def make_rb_track_lookup(spotify_id: str, rb_id: str, title: str = "Track") -> dict:
    """Fake ReccoBeats /v1/track response item."""
    return {
        "id": rb_id,
        "trackTitle": title,
        "href": f"https://open.spotify.com/track/{spotify_id}",
    }


def make_rb_audio_feature(rb_id: str, spotify_id: str) -> dict:
    """Fake ReccoBeats /v1/track/{id}/audio-features response."""
    return {
        "id": rb_id,
        "href": f"https://open.spotify.com/track/{spotify_id}",
        "acousticness": 0.5,
        "danceability": 0.7,
        "energy": 0.6,
        "instrumentalness": 0.01,
        "key": 5,
        "liveness": 0.1,
        "loudness": -8.0,
        "mode": 1,
        "speechiness": 0.04,
        "tempo": 120.0,
        "valence": 0.6,
    }


# ------------------------------------------------------------------ #
# Incremental loading                                                  #
# ------------------------------------------------------------------ #

class TestIncrementalLoading:
    # On the very first run the manifest is empty, so every discovered ID must be
    # treated as new to ensure all available tracks are fetched.
    def test_filter_new_ids_all_new(self, client):
        c, _, _ = client
        assert c._filter_new_ids("tracks", ["a", "b", "c"]) == {"a", "b", "c"}

    # Already-ingested IDs must be excluded so subsequent runs don't waste API quota
    # re-fetching data that's already in bronze storage.
    def test_filter_new_ids_skips_known(self, client):
        c, _, _ = client
        c._manifest["tracks"] = {"ids": ["a", "b"], "last_updated": None}
        assert c._filter_new_ids("tracks", ["a", "b", "c"]) == {"c"}

    # If all discovered IDs are already known, the result must be empty so the ingest
    # loop short-circuits and makes no further API calls.
    def test_filter_new_ids_all_known(self, client):
        c, _, _ = client
        c._manifest["tracks"] = {"ids": ["a", "b", "c"], "last_updated": None}
        assert c._filter_new_ids("tracks", ["a", "b", "c"]) == set()

    # New IDs must be recorded with a timestamp so the next invocation knows what
    # has already been ingested and can correctly filter them out.
    def test_update_manifest_adds_ids(self, client):
        c, _, _ = client
        c._update_manifest("tracks", {"x", "y"})
        assert set(c._manifest["tracks"]["ids"]) == {"x", "y"}
        assert c._manifest["tracks"]["last_updated"] is not None

    # The manifest must accumulate IDs across runs rather than replacing them;
    # overwriting would cause previously-ingested tracks to be re-fetched.
    def test_update_manifest_merges_existing(self, client):
        c, _, _ = client
        c._manifest["tracks"] = {"ids": ["a"], "last_updated": None}
        c._update_manifest("tracks", {"b", "c"})
        assert set(c._manifest["tracks"]["ids"]) == {"a", "b", "c"}

    # The manifest must be written to disk so it persists across process restarts;
    # an in-memory-only manifest would lose all incremental state.
    def test_manifest_persisted_to_disk(self, client, tmp_path):
        c, _, _ = client
        c._update_manifest("tracks", {"t1", "t2"})
        c._save_manifest()
        manifest_path = tmp_path / "bronze" / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert set(data["tracks"]["ids"]) == {"t1", "t2"}


# ------------------------------------------------------------------ #
# Bronze storage                                                       #
# ------------------------------------------------------------------ #

class TestBronzeStorage:
    # Records must be written as valid JSON so the ETL pipeline can read them back;
    # any serialisation error here would silently break the entire downstream pipeline.
    def test_save_creates_file_with_correct_content(self, client):
        c, _, _ = client
        records = [{"id": "1", "name": "Track One"}]
        path = c._save_to_bronze(records, "tracks")
        assert path.exists()
        assert json.loads(path.read_text()) == records

    # An empty record list means there is nothing to store; returning None signals callers
    # not to attempt an upload or update the manifest for this run.
    def test_save_returns_none_for_empty_records(self, client):
        c, _, _ = client
        assert c._save_to_bronze([], "tracks") is None

    # The bronze directory structure must match the glob pattern used by bronze_to_silver
    # (data_type/YYYY-MM-DD/); any deviation would cause the ETL to silently skip files.
    def test_save_partitioned_by_data_type_and_date(self, client):
        c, _, _ = client
        path = c._save_to_bronze([{"id": "1"}], "audio_features")
        assert path.parent.parent.name == "audio_features"
        assert path.parent.name.count("-") == 2  # YYYY-MM-DD

    # Filenames must be prefixed with the data type so they can be identified without
    # opening each file to inspect its contents.
    def test_save_filename_includes_data_type(self, client):
        c, _, _ = client
        path = c._save_to_bronze([{"id": "1"}], "tracks")
        assert path.name.startswith("tracks_")
        assert path.suffix == ".json"


# ------------------------------------------------------------------ #
# Retry logic (Spotify API)                                           #
# ------------------------------------------------------------------ #

class TestRetryLogic:
    # A single 429 response must not abort the ingest; the client must wait and retry,
    # returning successfully on the next attempt.
    def test_retries_once_on_rate_limit_then_succeeds(self, client):
        c, _, _ = client
        exc = spotipy.SpotifyException(429, -1, "rate limited", headers={"Retry-After": "0"})
        mock_fn = MagicMock(side_effect=[exc, {"ok": True}])
        with patch("src.ingestion.spotify_client.time.sleep"):
            result = c._call_api(mock_fn)
        assert result == {"ok": True}
        assert mock_fn.call_count == 2

    # Persistent rate-limiting must eventually surface as a RuntimeError so callers know
    # the request failed, rather than retrying indefinitely.
    def test_raises_runtime_error_after_max_retries(self, client):
        c, _, _ = client
        exc = spotipy.SpotifyException(429, -1, "rate limited", headers={"Retry-After": "0"})
        mock_fn = MagicMock(side_effect=exc)
        with patch("src.ingestion.spotify_client.time.sleep"), \
             pytest.raises(RuntimeError, match="failed after"):
            c._call_api(mock_fn)
        assert mock_fn.call_count == c.MAX_RETRIES

    # 4xx errors other than 429 are non-transient and must be re-raised immediately;
    # retrying a 404 would waste API quota with no chance of success.
    def test_reraises_non_retryable_client_errors(self, client):
        c, _, _ = client
        exc = spotipy.SpotifyException(404, -1, "not found")
        mock_fn = MagicMock(side_effect=exc)
        with pytest.raises(spotipy.SpotifyException):
            c._call_api(mock_fn)
        assert mock_fn.call_count == 1

    # 5xx server errors are transient and must be retried with the same logic as
    # rate limits so temporary Spotify outages don't abort the ingestion run.
    def test_retries_on_server_error(self, client):
        c, _, _ = client
        exc = spotipy.SpotifyException(503, -1, "service unavailable")
        mock_fn = MagicMock(side_effect=[exc, {"ok": True}])
        with patch("src.ingestion.spotify_client.time.sleep"):
            result = c._call_api(mock_fn)
        assert result == {"ok": True}


# ------------------------------------------------------------------ #
# Batching                                                             #
# ------------------------------------------------------------------ #

class TestBatching:
    # Lists must be split into correctly-sized chunks including the final partial batch,
    # since Spotify's batch endpoints have a fixed maximum items per request.
    def test_splits_into_batches(self):
        batches = list(_batched(list(range(7)), 3))
        assert batches == [[0, 1, 2], [3, 4, 5], [6]]

    # When the list length divides evenly into the batch size, no empty trailing batch
    # should be produced.
    def test_exact_multiple(self):
        batches = list(_batched(list(range(6)), 3))
        assert batches == [[0, 1, 2], [3, 4, 5]]

    # An empty input must yield no batches without raising so callers don't need to
    # guard against empty lists before calling _batched.
    def test_empty_list(self):
        assert list(_batched([], 10)) == []

    # When the batch size exceeds the list length, the entire list must be returned
    # as a single batch rather than an error.
    def test_batch_size_larger_than_list(self):
        assert list(_batched([1, 2], 10)) == [[1, 2]]


# ------------------------------------------------------------------ #
# ReccoBeats audio features fetch                                      #
# ------------------------------------------------------------------ #

class TestFetchAudioFeaturesReccoBeats:
    def _mock_rb(self, c, spotify_ids, rb_ids):
        """Patch _call_rb to return realistic ReccoBeats responses."""
        lookup_resp = {"content": [
            make_rb_track_lookup(sid, rid) for sid, rid in zip(spotify_ids, rb_ids)
        ]}
        features_resps = {
            rid: make_rb_audio_feature(rid, sid)
            for sid, rid in zip(spotify_ids, rb_ids)
        }

        def fake_call_rb(url):
            if "/audio-features" in url:
                rb_id = url.removesuffix("/audio-features").rsplit("/", 1)[-1]
                return features_resps[rb_id]
            return lookup_resp

        c._call_rb = fake_call_rb

    # A successful two-step lookup must return one feature record per Spotify ID,
    # confirming both the track resolution and the audio features fetch work end-to-end.
    def test_returns_one_record_per_track(self, client):
        c, _, _ = client
        self._mock_rb(c, ["s1", "s2"], ["rb1", "rb2"])
        with patch("src.ingestion.spotify_client.time.sleep"):
            results = c._fetch_audio_features(["s1", "s2"])
        assert len(results) == 2

    # Each returned record must use the Spotify track ID as "id" so the bronze →
    # silver transform can join on track_id without any schema changes.
    def test_records_use_spotify_id_not_reccobeats_id(self, client):
        c, _, _ = client
        self._mock_rb(c, ["s1"], ["rb1"])
        with patch("src.ingestion.spotify_client.time.sleep"):
            results = c._fetch_audio_features(["s1"])
        assert results[0]["id"] == "s1"
        assert results[0]["id"] != "rb1"

    # Expected audio feature fields must be present so the silver transform can
    # normalize and validate them correctly.
    def test_records_contain_audio_feature_fields(self, client):
        c, _, _ = client
        self._mock_rb(c, ["s1"], ["rb1"])
        with patch("src.ingestion.spotify_client.time.sleep"):
            results = c._fetch_audio_features(["s1"])
        for field in ("danceability", "energy", "tempo", "key", "loudness", "valence"):
            assert field in results[0]

    # When a track is not in ReccoBeats' catalog, the lookup returns no content for it;
    # the method must skip it gracefully and return only the tracks that resolved.
    def test_skips_tracks_not_in_reccobeats(self, client):
        c, _, _ = client
        # Only s1 resolves; s2 is absent from the lookup response
        c._call_rb = lambda url: (
            {"content": [make_rb_track_lookup("s1", "rb1")]}
            if "/audio-features" not in url
            else make_rb_audio_feature("rb1", "s1")
        )
        with patch("src.ingestion.spotify_client.time.sleep"):
            results = c._fetch_audio_features(["s1", "s2"])
        assert len(results) == 1
        assert results[0]["id"] == "s1"

    # If the audio features call raises (e.g. network error), the track must be skipped
    # rather than aborting the entire fetch so other tracks still get processed.
    def test_skips_on_audio_features_error(self, client):
        c, _, _ = client
        lookup_resp = {"content": [
            make_rb_track_lookup("s1", "rb1"),
            make_rb_track_lookup("s2", "rb2"),
        ]}

        def fake_call_rb(url):
            if "/audio-features" not in url:
                return lookup_resp
            if "rb1" in url:
                raise Exception("network error")
            return make_rb_audio_feature("rb2", "s2")

        c._call_rb = fake_call_rb
        with patch("src.ingestion.spotify_client.time.sleep"):
            results = c._fetch_audio_features(["s1", "s2"])
        assert len(results) == 1
        assert results[0]["id"] == "s2"

    # An empty input must return an empty list immediately without making any HTTP calls.
    def test_empty_input_returns_empty(self, client):
        c, _, _ = client
        rb_called = []
        c._call_rb = lambda url: rb_called.append(url) or {}
        results = c._fetch_audio_features([])
        assert results == []
        assert rb_called == []

    # If the batch track lookup itself fails, the method must log and return an empty
    # list rather than crashing the ingest run.
    def test_lookup_failure_returns_empty(self, client):
        c, _, _ = client
        c._call_rb = MagicMock(side_effect=Exception("connection refused"))
        with patch("src.ingestion.spotify_client.time.sleep"):
            results = c._fetch_audio_features(["s1", "s2"])
        assert results == []


# ------------------------------------------------------------------ #
# Backfill audio features                                              #
# ------------------------------------------------------------------ #

class TestBackfillAudioFeatures:
    # When the manifest has no tracks, there is nothing to backfill and the method
    # must return 0 without making any HTTP calls.
    def test_returns_zero_when_manifest_empty(self, client):
        c, _, _ = client
        result = c.backfill_audio_features()
        assert result == 0

    # If all manifest track IDs already have a bronze audio_features record, the method
    # must skip them all and return 0 to avoid duplicate files.
    def test_skips_ids_already_in_bronze(self, client, tmp_path):
        c, _, _ = client
        c._manifest["tracks"] = {"ids": ["t1", "t2"], "last_updated": None}

        # Write existing audio_features bronze file
        af_dir = tmp_path / "bronze" / "audio_features" / "2026-01-01"
        af_dir.mkdir(parents=True)
        (af_dir / "af.json").write_text(json.dumps([{"id": "t1"}, {"id": "t2"}]))

        result = c.backfill_audio_features()
        assert result == 0

    # IDs missing from bronze must be fetched and written; confirms the backfill only
    # targets the gap rather than re-fetching everything.
    def test_fetches_only_missing_ids(self, client, tmp_path):
        c, _, _ = client
        c._manifest["tracks"] = {"ids": ["t1", "t2"], "last_updated": None}

        # t1 is already in bronze; only t2 is missing
        af_dir = tmp_path / "bronze" / "audio_features" / "2026-01-01"
        af_dir.mkdir(parents=True)
        (af_dir / "af.json").write_text(json.dumps([{"id": "t1"}]))

        fetched_ids = []

        def fake_fetch(ids):
            fetched_ids.extend(ids)
            return [make_audio_feature(tid) for tid in ids]

        c._fetch_audio_features = fake_fetch
        c.backfill_audio_features()
        assert fetched_ids == ["t2"]

    # A successful backfill must write a bronze audio_features file and return
    # the count of records written.
    def test_writes_bronze_file_and_returns_count(self, client, tmp_path):
        c, _, _ = client
        c._manifest["tracks"] = {"ids": ["t1", "t2"], "last_updated": None}
        c._fetch_audio_features = lambda ids: [make_audio_feature(tid) for tid in ids]

        result = c.backfill_audio_features()

        assert result == 2
        assert any((tmp_path / "bronze").glob("audio_features/**/*.json"))

    # If _fetch_audio_features returns empty (e.g. ReccoBeats unreachable), the method
    # must return 0 and write no files rather than writing an empty bronze record.
    def test_returns_zero_when_fetch_returns_empty(self, client):
        c, _, _ = client
        c._manifest["tracks"] = {"ids": ["t1"], "last_updated": None}
        c._fetch_audio_features = lambda ids: []

        result = c.backfill_audio_features()
        assert result == 0


# ------------------------------------------------------------------ #
# Full ingest (mocked Spotify + ReccoBeats)                           #
# ------------------------------------------------------------------ #

class TestIngest:
    def _setup_mocks(self, c, mock_sp, track_ids, artist_id="artist1"):
        mock_sp.search.return_value = {
            "tracks": {"items": [make_track(tid, artist_id) for tid in track_ids]}
        }
        mock_sp.artists.return_value = {
            "artists": [{"id": artist_id, "name": "Artist", "genres": ["pop"]}]
        }
        # Stub ReccoBeats at the method level so ingest tests focus on orchestration
        c._fetch_audio_features = lambda ids: [make_audio_feature(tid) for tid in ids]

    # The summary dict is used by monitoring and CI; all three counts must reflect
    # the actual number of records fetched and stored.
    def test_ingest_returns_correct_counts(self, client):
        c, mock_sp, _ = client
        self._setup_mocks(c, mock_sp, ["t1", "t2"])
        summary = c.ingest(genres=["pop"], tracks_per_genre=2)
        assert summary["tracks"] == 2
        assert summary["audio_features"] == 2
        assert summary["artists"] == 1

    # If all discovered tracks are already in the manifest, the run must return zero
    # counts and make no downstream calls, fully honouring the incremental contract.
    def test_ingest_skips_all_known_tracks(self, client):
        c, mock_sp, _ = client
        self._setup_mocks(c, mock_sp, ["t1", "t2"])
        c._manifest["tracks"] = {"ids": ["t1", "t2"], "last_updated": "2026-01-01T00:00:00+00:00"}
        summary = c.ingest(genres=["pop"], tracks_per_genre=2)
        assert summary == {"tracks": 0, "audio_features": 0, "artists": 0}

    # Only previously-unseen tracks must be fetched and stored; this validates the
    # incremental filter correctly partitions new from known IDs mid-run.
    def test_ingest_processes_only_new_tracks(self, client):
        c, mock_sp, _ = client
        self._setup_mocks(c, mock_sp, ["t1", "t2", "t3"])
        c._manifest["tracks"] = {"ids": ["t1"], "last_updated": "2026-01-01T00:00:00+00:00"}
        summary = c.ingest(genres=["pop"], tracks_per_genre=3)
        assert summary["tracks"] == 2

    # The manifest must be updated and written to disk after each run so that the next
    # invocation correctly identifies what is already known.
    def test_ingest_writes_manifest_after_run(self, client, tmp_path):
        c, mock_sp, _ = client
        self._setup_mocks(c, mock_sp, ["t1", "t2"])
        c.ingest(genres=["pop"], tracks_per_genre=2)
        manifest = json.loads((tmp_path / "bronze" / "manifest.json").read_text())
        assert "t1" in manifest["tracks"]["ids"]
        assert "t2" in manifest["tracks"]["ids"]

    # All three data types must produce bronze JSON files so the ETL pipeline has
    # all the inputs it needs to build the silver layer.
    def test_ingest_writes_bronze_files(self, client, tmp_path):
        c, mock_sp, _ = client
        self._setup_mocks(c, mock_sp, ["t1", "t2"])
        c.ingest(genres=["pop"], tracks_per_genre=2)
        bronze = tmp_path / "bronze"
        assert any(bronze.glob("tracks/**/*.json"))
        assert any(bronze.glob("audio_features/**/*.json"))
        assert any(bronze.glob("artists/**/*.json"))


# ------------------------------------------------------------------ #
# Azure upload integration                                             #
# ------------------------------------------------------------------ #

class TestAzureUpload:
    # One upload call per data type confirms that tracks, audio features, and artists
    # all land in Azure Blob Storage after a successful ingest run.
    def test_upload_called_for_each_data_type(self, client):
        c, mock_sp, _ = client
        mock_azure = MagicMock()
        c._azure = mock_azure
        c._fetch_audio_features = lambda ids: [make_audio_feature(tid) for tid in ids]

        mock_sp.search.return_value = {
            "tracks": {"items": [make_track("t1", "a1"), make_track("t2", "a1")]}
        }
        mock_sp.artists.return_value = {
            "artists": [{"id": "a1", "name": "Artist", "genres": ["pop"]}]
        }

        c.ingest(genres=["pop"], tracks_per_genre=2)

        # tracks + audio_features + artists files should each be uploaded
        assert mock_azure.upload_file.call_count == 3
        blob_args = [call.args[0].name for call in mock_azure.upload_file.call_args_list]
        assert any("tracks_" in name for name in blob_args)
        assert any("artists_" in name for name in blob_args)

    # When Azure credentials are not configured, the ingest must still complete
    # successfully and write bronze files locally without attempting any uploads.
    def test_no_upload_when_azure_not_configured(self, client):
        c, mock_sp, _ = client
        c._azure = None  # explicitly no Azure
        c._fetch_audio_features = lambda ids: [make_audio_feature(tid) for tid in ids]

        mock_sp.search.return_value = {"tracks": {"items": [make_track("t1")]}}
        mock_sp.artists.return_value = {"artists": [{"id": "a1", "name": "Artist", "genres": []}]}

        # Should complete without errors - no upload attempted
        summary = c.ingest(genres=["pop"], tracks_per_genre=1)
        assert summary["tracks"] == 1

    # The relative_to base must be the data/ root so that blob names preserve the full
    # local path hierarchy (e.g. bronze/tracks/...) in Azure Blob Storage.
    def test_upload_uses_bronze_dir_parent_as_base(self, client):
        c, mock_sp, _ = client
        mock_azure = MagicMock()
        c._azure = mock_azure
        c._fetch_audio_features = lambda ids: []

        mock_sp.search.return_value = {"tracks": {"items": [make_track("t1")]}}
        mock_sp.artists.return_value = {"artists": [{"id": "a1", "name": "Artist", "genres": []}]}

        c.ingest(genres=["pop"], tracks_per_genre=1)

        import src.ingestion.spotify_client as m
        expected_base = m.BRONZE_DIR.parent
        for call in mock_azure.upload_file.call_args_list:
            assert call.kwargs["relative_to"] == expected_base

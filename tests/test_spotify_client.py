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


# ------------------------------------------------------------------ #
# Incremental loading                                                  #
# ------------------------------------------------------------------ #

class TestIncrementalLoading:
    def test_filter_new_ids_all_new(self, client):
        c, _, _ = client
        assert c._filter_new_ids("tracks", ["a", "b", "c"]) == {"a", "b", "c"}

    def test_filter_new_ids_skips_known(self, client):
        c, _, _ = client
        c._manifest["tracks"] = {"ids": ["a", "b"], "last_updated": None}
        assert c._filter_new_ids("tracks", ["a", "b", "c"]) == {"c"}

    def test_filter_new_ids_all_known(self, client):
        c, _, _ = client
        c._manifest["tracks"] = {"ids": ["a", "b", "c"], "last_updated": None}
        assert c._filter_new_ids("tracks", ["a", "b", "c"]) == set()

    def test_update_manifest_adds_ids(self, client):
        c, _, _ = client
        c._update_manifest("tracks", {"x", "y"})
        assert set(c._manifest["tracks"]["ids"]) == {"x", "y"}
        assert c._manifest["tracks"]["last_updated"] is not None

    def test_update_manifest_merges_existing(self, client):
        c, _, _ = client
        c._manifest["tracks"] = {"ids": ["a"], "last_updated": None}
        c._update_manifest("tracks", {"b", "c"})
        assert set(c._manifest["tracks"]["ids"]) == {"a", "b", "c"}

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
    def test_save_creates_file_with_correct_content(self, client):
        c, _, _ = client
        records = [{"id": "1", "name": "Track One"}]
        path = c._save_to_bronze(records, "tracks")
        assert path.exists()
        assert json.loads(path.read_text()) == records

    def test_save_returns_none_for_empty_records(self, client):
        c, _, _ = client
        assert c._save_to_bronze([], "tracks") is None

    def test_save_partitioned_by_data_type_and_date(self, client):
        c, _, _ = client
        path = c._save_to_bronze([{"id": "1"}], "audio_features")
        assert path.parent.parent.name == "audio_features"
        assert path.parent.name.count("-") == 2  # YYYY-MM-DD

    def test_save_filename_includes_data_type(self, client):
        c, _, _ = client
        path = c._save_to_bronze([{"id": "1"}], "tracks")
        assert path.name.startswith("tracks_")
        assert path.suffix == ".json"


# ------------------------------------------------------------------ #
# Retry logic                                                          #
# ------------------------------------------------------------------ #

class TestRetryLogic:
    def test_retries_once_on_rate_limit_then_succeeds(self, client):
        c, _, _ = client
        exc = spotipy.SpotifyException(429, -1, "rate limited", headers={"Retry-After": "0"})
        mock_fn = MagicMock(side_effect=[exc, {"ok": True}])
        with patch("src.ingestion.spotify_client.time.sleep"):
            result = c._call_api(mock_fn)
        assert result == {"ok": True}
        assert mock_fn.call_count == 2

    def test_raises_runtime_error_after_max_retries(self, client):
        c, _, _ = client
        exc = spotipy.SpotifyException(429, -1, "rate limited", headers={"Retry-After": "0"})
        mock_fn = MagicMock(side_effect=exc)
        with patch("src.ingestion.spotify_client.time.sleep"), \
             pytest.raises(RuntimeError, match="failed after"):
            c._call_api(mock_fn)
        assert mock_fn.call_count == c.MAX_RETRIES

    def test_reraises_non_retryable_client_errors(self, client):
        c, _, _ = client
        exc = spotipy.SpotifyException(404, -1, "not found")
        mock_fn = MagicMock(side_effect=exc)
        with pytest.raises(spotipy.SpotifyException):
            c._call_api(mock_fn)
        assert mock_fn.call_count == 1

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
    def test_splits_into_batches(self):
        batches = list(_batched(list(range(7)), 3))
        assert batches == [[0, 1, 2], [3, 4, 5], [6]]

    def test_exact_multiple(self):
        batches = list(_batched(list(range(6)), 3))
        assert batches == [[0, 1, 2], [3, 4, 5]]

    def test_empty_list(self):
        assert list(_batched([], 10)) == []

    def test_batch_size_larger_than_list(self):
        assert list(_batched([1, 2], 10)) == [[1, 2]]


# ------------------------------------------------------------------ #
# Full ingest (mocked Spotify API)                                    #
# ------------------------------------------------------------------ #

class TestIngest:
    def _setup_mocks(self, mock_sp, track_ids, artist_id="artist1"):
        mock_sp.search.return_value = {
            "tracks": {"items": [make_track(tid, artist_id) for tid in track_ids]}
        }
        mock_sp.audio_features.return_value = [make_audio_feature(tid) for tid in track_ids]
        mock_sp.artists.return_value = {
            "artists": [{"id": artist_id, "name": "Artist", "genres": ["pop"]}]
        }

    def test_ingest_returns_correct_counts(self, client):
        c, mock_sp, _ = client
        self._setup_mocks(mock_sp, ["t1", "t2"])
        summary = c.ingest(genres=["pop"], tracks_per_genre=2)
        assert summary["tracks"] == 2
        assert summary["audio_features"] == 2
        assert summary["artists"] == 1

    def test_ingest_skips_all_known_tracks(self, client):
        c, mock_sp, _ = client
        self._setup_mocks(mock_sp, ["t1", "t2"])
        c._manifest["tracks"] = {"ids": ["t1", "t2"], "last_updated": "2026-01-01T00:00:00+00:00"}
        summary = c.ingest(genres=["pop"], tracks_per_genre=2)
        assert summary == {"tracks": 0, "audio_features": 0, "artists": 0}

    def test_ingest_processes_only_new_tracks(self, client):
        c, mock_sp, _ = client
        self._setup_mocks(mock_sp, ["t1", "t2", "t3"])
        c._manifest["tracks"] = {"ids": ["t1"], "last_updated": "2026-01-01T00:00:00+00:00"}
        summary = c.ingest(genres=["pop"], tracks_per_genre=3)
        assert summary["tracks"] == 2

    def test_ingest_writes_manifest_after_run(self, client, tmp_path):
        c, mock_sp, _ = client
        self._setup_mocks(mock_sp, ["t1", "t2"])
        c.ingest(genres=["pop"], tracks_per_genre=2)
        manifest = json.loads((tmp_path / "bronze" / "manifest.json").read_text())
        assert "t1" in manifest["tracks"]["ids"]
        assert "t2" in manifest["tracks"]["ids"]

    def test_ingest_writes_bronze_files(self, client, tmp_path):
        c, mock_sp, _ = client
        self._setup_mocks(mock_sp, ["t1", "t2"])
        c.ingest(genres=["pop"], tracks_per_genre=2)
        bronze = tmp_path / "bronze"
        assert any(bronze.glob("tracks/**/*.json"))
        assert any(bronze.glob("audio_features/**/*.json"))
        assert any(bronze.glob("artists/**/*.json"))


# ------------------------------------------------------------------ #
# Azure upload integration                                             #
# ------------------------------------------------------------------ #

class TestAzureUpload:
    def test_upload_called_for_each_data_type(self, client):
        c, mock_sp, tmp_path = client
        mock_azure = MagicMock()
        c._azure = mock_azure

        mock_sp.search.return_value = {
            "tracks": {"items": [make_track("t1", "a1"), make_track("t2", "a1")]}
        }
        mock_sp.audio_features.return_value = [make_audio_feature("t1"), make_audio_feature("t2")]
        mock_sp.artists.return_value = {
            "artists": [{"id": "a1", "name": "Artist", "genres": ["pop"]}]
        }

        c.ingest(genres=["pop"], tracks_per_genre=2)

        # tracks + artists files should be uploaded (audio_features returns [] due to mock)
        assert mock_azure.upload_file.call_count >= 2
        blob_args = [call.args[0].name for call in mock_azure.upload_file.call_args_list]
        assert any("tracks_" in name for name in blob_args)
        assert any("artists_" in name for name in blob_args)

    def test_no_upload_when_azure_not_configured(self, client):
        c, mock_sp, _ = client
        c._azure = None  # explicitly no Azure

        mock_sp.search.return_value = {"tracks": {"items": [make_track("t1")]}}
        mock_sp.audio_features.return_value = [make_audio_feature("t1")]
        mock_sp.artists.return_value = {"artists": [{"id": "a1", "name": "Artist", "genres": []}]}

        # Should complete without errors - no upload attempted
        summary = c.ingest(genres=["pop"], tracks_per_genre=1)
        assert summary["tracks"] == 1

    def test_upload_uses_bronze_dir_parent_as_base(self, client):
        c, mock_sp, tmp_path = client
        mock_azure = MagicMock()
        c._azure = mock_azure

        mock_sp.search.return_value = {"tracks": {"items": [make_track("t1")]}}
        mock_sp.audio_features.return_value = []
        mock_sp.artists.return_value = {"artists": [{"id": "a1", "name": "Artist", "genres": []}]}

        c.ingest(genres=["pop"], tracks_per_genre=1)

        import src.ingestion.spotify_client as m
        expected_base = m.BRONZE_DIR.parent
        for call in mock_azure.upload_file.call_args_list:
            assert call.kwargs["relative_to"] == expected_base

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.ingestion.azure_uploader import AzureBlobUploader


@pytest.fixture
def uploader():
    mock_service = MagicMock()
    with patch("src.ingestion.azure_uploader.BlobServiceClient") as mock_cls:
        mock_cls.from_connection_string.return_value = mock_service
        u = AzureBlobUploader("DefaultEndpointsProtocol=https;AccountName=x;AccountKey=y;", "my-container")
    u._service = mock_service
    return u


# ------------------------------------------------------------------ #
# from_env factory                                                     #
# ------------------------------------------------------------------ #

class TestFromEnv:
    # When neither credential env var is set, the factory must return None to avoid crashing
    # on environments where Azure isn't configured rather than raising an exception.
    def test_returns_none_when_no_env_vars(self, monkeypatch):
        monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
        monkeypatch.delenv("AZURE_STORAGE_CONTAINER_NAME", raising=False)
        assert AzureBlobUploader.from_env() is None

    # A connection string alone is insufficient; the factory must reject partial configuration
    # and return None when the container name is missing.
    def test_returns_none_when_container_missing(self, monkeypatch):
        monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "conn_str")
        monkeypatch.delenv("AZURE_STORAGE_CONTAINER_NAME", raising=False)
        assert AzureBlobUploader.from_env() is None

    # A container name alone cannot establish a connection; confirms the factory
    # returns None rather than attempting to build an unusable uploader.
    def test_returns_none_when_connection_string_missing(self, monkeypatch):
        monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
        monkeypatch.setenv("AZURE_STORAGE_CONTAINER_NAME", "spotify-data")
        assert AzureBlobUploader.from_env() is None

    # With both vars present, the factory must return a live uploader instance;
    # this is the happy path that production ingestion relies on.
    def test_returns_uploader_when_both_vars_set(self, monkeypatch):
        monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "conn_str")
        monkeypatch.setenv("AZURE_STORAGE_CONTAINER_NAME", "spotify-data")
        with patch("src.ingestion.azure_uploader.BlobServiceClient"):
            result = AzureBlobUploader.from_env()
        assert isinstance(result, AzureBlobUploader)


# ------------------------------------------------------------------ #
# upload_file                                                          #
# ------------------------------------------------------------------ #

class TestUploadFile:
    # The blob name must mirror the local directory hierarchy relative to the base path
    # so Azure storage structure stays consistent with the bronze/ layout.
    def test_blob_name_is_path_relative_to_base(self, uploader, tmp_path):
        file = tmp_path / "bronze" / "tracks" / "2026-05-13" / "tracks_20260513T120000Z.json"
        file.parent.mkdir(parents=True)
        file.write_text("[]")
        mock_blob_client = MagicMock()
        uploader._service.get_blob_client.return_value = mock_blob_client

        blob_name = uploader.upload_file(file, relative_to=tmp_path)

        assert blob_name == "bronze/tracks/2026-05-13/tracks_20260513T120000Z.json"

    # The SDK must be called with the correct container and blob name; passing wrong values
    # would silently upload data to the wrong location without any error.
    def test_correct_container_and_blob_passed_to_client(self, uploader, tmp_path):
        file = tmp_path / "bronze" / "artists" / "2026-05-13" / "artists_20260513T120000Z.json"
        file.parent.mkdir(parents=True)
        file.write_text("[]")
        mock_blob_client = MagicMock()
        uploader._service.get_blob_client.return_value = mock_blob_client

        blob_name = uploader.upload_file(file, relative_to=tmp_path)

        uploader._service.get_blob_client.assert_called_once_with(
            container="my-container",
            blob=blob_name,
        )

    # overwrite=True must be passed to the SDK so re-running ingestion replaces stale blobs
    # rather than failing on files that already exist.
    def test_upload_blob_called_with_overwrite_true(self, uploader, tmp_path):
        file = tmp_path / "test.json"
        file.write_text("[]")
        mock_blob_client = MagicMock()
        uploader._service.get_blob_client.return_value = mock_blob_client

        uploader.upload_file(file, relative_to=tmp_path)

        _, kwargs = mock_blob_client.upload_blob.call_args
        assert kwargs.get("overwrite") is True

    # The return value is used by callers to record which blobs were uploaded;
    # it must equal the relative path that was constructed.
    def test_returns_blob_name(self, uploader, tmp_path):
        file = tmp_path / "subdir" / "file.json"
        file.parent.mkdir()
        file.write_text("[]")
        uploader._service.get_blob_client.return_value = MagicMock()

        result = uploader.upload_file(file, relative_to=tmp_path)
        assert result == "subdir/file.json"

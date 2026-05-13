import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class AzureBlobUploader:
    """Uploads files to Azure Blob Storage, preserving directory structure as blob names."""

    def __init__(self, connection_string: str, container_name: str) -> None:
        self._service = BlobServiceClient.from_connection_string(connection_string)
        self._container = container_name

    @classmethod
    def from_env(cls) -> "AzureBlobUploader | None":
        """Returns an uploader from env vars, or None if Azure is not configured."""
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
        if not conn_str or not container:
            return None
        return cls(conn_str, container)

    def upload_file(self, local_path: Path, relative_to: Path) -> str:
        """
        Uploads local_path to the container.
        The blob name is derived from local_path relative to relative_to (POSIX-formatted).

        Example:
            upload_file(Path("data/bronze/tracks/2026-05-13/f.json"), Path("data"))
            -> blob name: "bronze/tracks/2026-05-13/f.json"

        Returns the blob name.
        """
        blob_name = local_path.relative_to(relative_to).as_posix()
        blob_client = self._service.get_blob_client(container=self._container, blob=blob_name)
        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        logger.info("Uploaded %s -> azure://%s/%s", local_path.name, self._container, blob_name)
        return blob_name

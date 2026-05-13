import json
import logging
import os
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials

from src.ingestion.azure_uploader import AzureBlobUploader
from src.utils.logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


_CONFIG = _load_config()

BRONZE_DIR = Path(_CONFIG.get("storage", {}).get("bronze_dir", "data/bronze"))
MANIFEST_PATH = BRONZE_DIR / "manifest.json"


class SpotifyIngestionClient:
    """Extracts tracks, audio features, and artists from Spotify API into the bronze layer."""

    _spotify_cfg = _CONFIG.get("spotify", {})
    MAX_RETRIES: int = _spotify_cfg.get("max_retries", 3)
    BACKOFF_BASE: float = _spotify_cfg.get("backoff_base_seconds", 1.0)
    MARKET: str = _spotify_cfg.get("market", "US")

    def __init__(self) -> None:
        auth = SpotifyClientCredentials(
            client_id=os.environ["SPOTIFY_CLIENT_ID"],
            client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
        )
        self.sp = spotipy.Spotify(auth_manager=auth)
        self._manifest = self._load_manifest()
        self._azure = AzureBlobUploader.from_env()
        if self._azure:
            logger.info("Azure Blob Storage upload enabled (container: %s)", os.getenv("AZURE_STORAGE_CONTAINER_NAME"))
        else:
            logger.info("Azure Blob Storage not configured - writing to local bronze layer only")

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def ingest(self, genres: list[str] | None = None, tracks_per_genre: int | None = None) -> dict:
        """
        Run a full ingestion cycle: search tracks by genre → fetch audio features
        and artist metadata → save new records to the bronze layer.

        Returns a summary dict: {"tracks": N, "audio_features": N, "artists": N}.
        """
        genres = genres or _default_genres()
        if tracks_per_genre is None:
            tracks_per_genre = int(_CONFIG.get("ingestion", {}).get("tracks_per_genre", 50))
        logger.info("Starting ingestion | genres=%s, tracks_per_genre=%d", genres, tracks_per_genre)

        all_tracks = self._fetch_tracks_by_genre(genres, tracks_per_genre)

        new_track_ids = self._filter_new_ids("tracks", [t["id"] for t in all_tracks])
        new_tracks = [t for t in all_tracks if t["id"] in new_track_ids]

        if not new_tracks:
            logger.info("No new tracks to ingest")
            return {"tracks": 0, "audio_features": 0, "artists": 0}

        track_ids = [t["id"] for t in new_tracks]
        artist_ids = list({a["id"] for t in new_tracks for a in t.get("artists", [])})
        new_artist_ids = self._filter_new_ids("artists", artist_ids)

        audio_features = self._fetch_audio_features(track_ids)
        artists = self._fetch_artists(list(new_artist_ids)) if new_artist_ids else []

        self._save_to_bronze(new_tracks, "tracks")
        self._save_to_bronze(audio_features, "audio_features")
        if artists:
            self._save_to_bronze(artists, "artists")

        self._update_manifest("tracks", new_track_ids)
        self._update_manifest("artists", new_artist_ids)
        self._save_manifest()

        summary = {
            "tracks": len(new_tracks),
            "audio_features": len(audio_features),
            "artists": len(artists),
        }
        logger.info("Ingestion complete | %s", summary)
        return summary

    # ------------------------------------------------------------------ #
    # Private fetch methods                                                #
    # ------------------------------------------------------------------ #

    def _fetch_tracks_by_genre(self, genres: list[str], tracks_per_genre: int) -> list[dict]:
        seen: dict[str, dict] = {}
        for genre in genres:
            fetched = 0
            offset = 0
            while fetched < tracks_per_genre:
                limit = min(50, tracks_per_genre - fetched)
                result = self._call_api(
                    self.sp.search,
                    q=f"genre:{genre}",
                    type="track",
                    limit=limit,
                    offset=offset,
                    market=self.MARKET,
                )
                items = [i for i in result["tracks"]["items"] if i]
                if not items:
                    break
                for item in items:
                    seen[item["id"]] = item
                fetched += len(items)
                offset += len(items)
                if len(items) < limit:
                    break
        logger.info("Fetched %d unique tracks across %d genres", len(seen), len(genres))
        return list(seen.values())

    def _fetch_audio_features(self, track_ids: list[str]) -> list[dict]:
        # Spotify deprecated this endpoint for new apps in Nov 2024 (requires Extended Quota Mode).
        # A 403 means the app lacks access; we skip gracefully rather than aborting ingestion.
        spotipy_logger = logging.getLogger("spotipy.client")
        features: list[dict] = []
        for batch in _batched(track_ids, 100):
            try:
                # Temporarily silence spotipy's HTTP error log for this known-deprecated call.
                prev_level = spotipy_logger.level
                spotipy_logger.setLevel(logging.CRITICAL)
                try:
                    result = self._call_api(self.sp.audio_features, batch)
                finally:
                    spotipy_logger.setLevel(prev_level)
                features.extend(r for r in result if r is not None)
            except spotipy.SpotifyException as exc:
                if exc.http_status == 403:
                    logger.warning(
                        "audio-features endpoint returned 403 (deprecated for new Spotify apps, "
                        "Extended Quota Mode required). Audio features will not be stored."
                    )
                    return []
                raise
        return features

    def _fetch_artists(self, artist_ids: list[str]) -> list[dict]:
        artists: list[dict] = []
        for batch in _batched(artist_ids, 50):
            result = self._call_api(self.sp.artists, batch)
            artists.extend(a for a in result["artists"] if a is not None)
        return artists

    # ------------------------------------------------------------------ #
    # Bronze layer storage                                                 #
    # ------------------------------------------------------------------ #

    def _save_to_bronze(self, records: list[dict], data_type: str) -> Path | None:
        if not records:
            return None
        now_utc = datetime.now(timezone.utc)
        timestamp = now_utc.strftime("%Y%m%dT%H%M%SZ")
        date_str = now_utc.strftime("%Y-%m-%d")
        dest_dir = BRONZE_DIR / data_type / date_str
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{data_type}_{timestamp}.json"
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d %s records -> %s", len(records), data_type, dest_path)
        if self._azure:
            try:
                self._azure.upload_file(dest_path, relative_to=BRONZE_DIR.parent)
            except Exception as e:
                logger.error("Azure upload failed for %s: %s", dest_path, e)
        return dest_path

    # ------------------------------------------------------------------ #
    # Manifest (incremental loading)                                      #
    # ------------------------------------------------------------------ #

    def _load_manifest(self) -> dict:
        if MANIFEST_PATH.exists():
            try:
                with open(MANIFEST_PATH, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError, ValueError) as e:
                logger.warning("Manifest at %s is invalid (%s); starting fresh.", MANIFEST_PATH, e)
        return {}

    def _save_manifest(self) -> None:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MANIFEST_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2)
        tmp.replace(MANIFEST_PATH)

    def _filter_new_ids(self, data_type: str, ids: list[str]) -> set[str]:
        known = set(self._manifest.get(data_type, {}).get("ids", []))
        return set(ids) - known

    def _update_manifest(self, data_type: str, new_ids: set[str]) -> None:
        entry = self._manifest.setdefault(data_type, {"ids": [], "last_updated": None})
        entry["ids"] = list(set(entry["ids"]) | new_ids)
        entry["last_updated"] = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------ #
    # API retry wrapper                                                    #
    # ------------------------------------------------------------------ #

    def _call_api(self, fn, *args, **kwargs):
        for attempt in range(self.MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except spotipy.SpotifyException as exc:
                if exc.http_status == 429:
                    headers = getattr(exc, "headers", None) or {}
                    wait = float(headers.get("Retry-After", self.BACKOFF_BASE * (2 ** attempt)))
                    logger.warning(
                        "Rate limited. Retry in %.1fs (attempt %d/%d)", wait, attempt + 1, self.MAX_RETRIES
                    )
                    time.sleep(wait)
                elif exc.http_status and 500 <= exc.http_status < 600:
                    wait = self.BACKOFF_BASE * (2 ** attempt)
                    logger.warning("Server error %d. Retry in %.1fs", exc.http_status, wait)
                else:
                    raise
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(wait)
        raise RuntimeError(f"Spotify API call failed after {self.MAX_RETRIES} retries")


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _default_genres() -> list[str]:
    return _CONFIG.get("spotify", {}).get(
        "default_genres",
        ["pop", "rock", "hip-hop", "electronic", "jazz", "r-n-b", "country", "classical"],
    )


if __name__ == "__main__":
    client = SpotifyIngestionClient()
    result = client.ingest()
    print("Ingestion summary:", result)

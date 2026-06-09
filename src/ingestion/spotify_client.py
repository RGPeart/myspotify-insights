import json
import os
import re
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path

import requests
import spotipy
from azure.core.exceptions import AzureError
from dotenv import load_dotenv

from src.ingestion.azure_uploader import AzureBlobUploader
from src.ingestion.spotify_auth import get_authenticated_client
from src.utils.logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)

_RECCOBEATS_BASE = "https://api.reccobeats.com/v1"
_RECCOBEATS_HEADERS = {"User-Agent": "MySpotifyInsights/1.0", "Accept": "application/json"}


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
    """Extracts the authenticated user's listening data and a derived genre-search catalog into the bronze layer."""

    _spotify_cfg = _CONFIG.get("spotify", {})
    MAX_RETRIES: int = _spotify_cfg.get("max_retries", 3)
    BACKOFF_BASE: float = _spotify_cfg.get("backoff_base_seconds", 1.0)
    MARKET: str = _spotify_cfg.get("market", "US")
    TOP_TIME_RANGES: tuple[str, ...] = _spotify_cfg.get("top_time_ranges", ("short_term", "medium_term"))
    TOP_LIMIT: int = _spotify_cfg.get("top_limit", 50)
    FOLLOWED_LIMIT: int = _spotify_cfg.get("followed_artists_limit", 50)
    DERIVED_GENRES_MAX: int = _spotify_cfg.get("derived_genres_max", 10)

    def __init__(self, bronze_dir: Path = BRONZE_DIR) -> None:
        self.sp = get_authenticated_client()
        self.bronze_dir = bronze_dir
        self._manifest_path = self.bronze_dir / "manifest.json"
        self._manifest = self._load_manifest()
        self._azure = AzureBlobUploader.from_env()
        self._rb_session = requests.Session()
        self._rb_session.headers.update(_RECCOBEATS_HEADERS)
        if self._azure:
            logger.info("Azure Blob Storage upload enabled (container: %s)", os.getenv("AZURE_STORAGE_CONTAINER_NAME"))
        else:
            logger.info("Azure Blob Storage not configured - writing to local bronze layer only")

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def ingest(self, genres: list[str] | None = None, tracks_per_genre: int | None = None) -> dict:
        """Run a full ingestion cycle.

        Order of operations:
          1. User profile + per-range top tracks/artists + followed artists → bronze.
          2. Derive genre seeds from the user's top + followed artists.
          3. Genre-search expansion using those seeds (or ``genres`` override / fallback).
          4. Merge top tracks/artists into the standard tracks/artists data types so
             they flow through silver/gold even if not surfaced by search.
          5. Audio features (ReccoBeats) + missing artist details + manifest update.

        Returns a summary dict with the counts of bronze records written for
        the standard tracks / audio_features / artists data types.
        """
        if tracks_per_genre is None:
            tracks_per_genre = _CONFIG.get("ingestion", {}).get("tracks_per_genre", 50)

        # 1. Personal data
        profile = self._fetch_user_profile()
        if profile:
            self._save_to_bronze([profile], "user_profile")

        top_tracks_by_id: dict[str, dict] = {}
        top_artists_by_id: dict[str, dict] = {}
        for time_range in self.TOP_TIME_RANGES:
            tracks = self._fetch_user_top_items("tracks", time_range, self.TOP_LIMIT)
            if tracks:
                self._save_to_bronze(tracks, "user_top_tracks", partition=time_range)
                for t in tracks:
                    top_tracks_by_id.setdefault(t["id"], t)
            artists = self._fetch_user_top_items("artists", time_range, self.TOP_LIMIT)
            if artists:
                self._save_to_bronze(artists, "user_top_artists", partition=time_range)
                for a in artists:
                    _merge_artist(top_artists_by_id, a)

        followed = self._fetch_followed_artists(self.FOLLOWED_LIMIT)
        if followed:
            self._save_to_bronze(followed, "followed_artists")
            for a in followed:
                _merge_artist(top_artists_by_id, a)

        # 2. Decide which genres drive catalog expansion
        if genres is not None:
            search_genres = genres
            logger.info("Using %d caller-supplied genres for search: %s", len(search_genres), search_genres)
        else:
            search_genres = self._derive_search_genres(list(top_artists_by_id.values()))
            if search_genres:
                logger.info("Using %d derived genres for search: %s", len(search_genres), search_genres)
            else:
                search_genres = _fallback_genres()
                logger.info("No genres derivable from user data; using fallback: %s", search_genres)

        # 3. Genre-search catalog expansion
        search_tracks = self._fetch_tracks_by_genre(search_genres, tracks_per_genre)

        # 4. Merge top tracks into the standard tracks set
        all_tracks_by_id: dict[str, dict] = {t["id"]: t for t in search_tracks}
        for tid, t in top_tracks_by_id.items():
            all_tracks_by_id.setdefault(tid, t)
        all_tracks = list(all_tracks_by_id.values())

        new_track_ids = self._filter_new_ids("tracks", [t["id"] for t in all_tracks])
        new_tracks = [t for t in all_tracks if t["id"] in new_track_ids]

        if not new_tracks:
            logger.info("No new tracks to ingest")
            self._save_manifest()
            return {"tracks": 0, "audio_features": 0, "artists": 0}

        # 5. Audio features for the new track IDs
        track_ids = [t["id"] for t in new_tracks]
        audio_features = self._fetch_audio_features(track_ids)

        # 6. Artists: union of artists referenced by new tracks + top/followed (which we already have full payloads for)
        artist_ids_from_tracks = {a["id"] for t in new_tracks for a in t.get("artists", [])}
        candidate_artist_ids = artist_ids_from_tracks | set(top_artists_by_id.keys())
        new_artist_ids = self._filter_new_ids("artists", list(candidate_artist_ids))

        already_have = {aid: top_artists_by_id[aid] for aid in new_artist_ids if aid in top_artists_by_id}
        need_to_fetch = [aid for aid in new_artist_ids if aid not in already_have]
        fetched_artists = self._fetch_artists(need_to_fetch) if need_to_fetch else []
        artists = list(already_have.values()) + fetched_artists

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

    def backfill_audio_features(self) -> int:
        """Fetch and store audio features for all tracks already in the manifest.

        Needed for tracks ingested before the ReccoBeats integration was added.
        Skips any Spotify IDs that already have a bronze audio_features record.
        Returns the number of feature records written.
        """
        track_ids = self._manifest.get("tracks", {}).get("ids", [])
        if not track_ids:
            logger.info("No tracks in manifest — nothing to backfill")
            return 0

        existing_af: set[str] = set()
        for f in self.bronze_dir.glob("audio_features/**/*.json"):
            try:
                records = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(records, list):
                    existing_af.update(r["id"] for r in records if r and "id" in r)
            except (json.JSONDecodeError, OSError):
                pass

        missing = [tid for tid in track_ids if tid not in existing_af]
        if not missing:
            logger.info("All %d tracks already have audio features in bronze", len(track_ids))
            return 0

        logger.info(
            "Backfilling audio features for %d tracks (%d already present, skipped)",
            len(missing), len(track_ids) - len(missing),
        )
        features = self._fetch_audio_features(missing)
        if not features:
            logger.warning("Backfill returned no features — check ReccoBeats connectivity")
            return 0

        self._save_to_bronze(features, "audio_features")
        logger.info("Backfill complete: %d audio feature records written", len(features))
        return len(features)

    # ------------------------------------------------------------------ #
    # Personal data fetches                                                #
    # ------------------------------------------------------------------ #

    def _fetch_user_profile(self) -> dict | None:
        try:
            return self._call_api(self.sp.current_user)
        except spotipy.SpotifyException as exc:
            logger.warning("Failed to fetch user profile: %s", exc)
            return None

    def _fetch_user_top_items(self, item_type: str, time_range: str, limit: int) -> list[dict]:
        """Fetch /me/top/{tracks|artists} for a single time_range."""
        if item_type == "tracks":
            fn = self.sp.current_user_top_tracks
        elif item_type == "artists":
            fn = self.sp.current_user_top_artists
        else:
            raise ValueError(f"item_type must be 'tracks' or 'artists', got {item_type!r}")
        try:
            result = self._call_api(fn, limit=limit, time_range=time_range)
        except spotipy.SpotifyException as exc:
            logger.warning("Failed to fetch top %s (%s): %s", item_type, time_range, exc)
            return []
        items = [i for i in (result or {}).get("items", []) if i]
        logger.info("Fetched %d top %s for time_range=%s", len(items), item_type, time_range)
        return items

    def _fetch_followed_artists(self, limit: int) -> list[dict]:
        try:
            result = self._call_api(self.sp.current_user_followed_artists, limit=limit)
        except spotipy.SpotifyException as exc:
            logger.warning("Failed to fetch followed artists: %s", exc)
            return []
        items = [i for i in (result or {}).get("artists", {}).get("items", []) if i]
        logger.info("Fetched %d followed artists", len(items))
        return items

    def _derive_search_genres(self, artists: list[dict]) -> list[str]:
        """Return the top N most-common genres across the supplied artists."""
        counts: dict[str, int] = {}
        for a in artists:
            for g in a.get("genres") or []:
                counts[g] = counts.get(g, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [g for g, _ in ranked[: self.DERIVED_GENRES_MAX]]

    # ------------------------------------------------------------------ #
    # Catalog expansion (genre search) & enrichment                        #
    # ------------------------------------------------------------------ #

    def _fetch_tracks_by_genre(self, genres: list[str], tracks_per_genre: int) -> list[dict]:
        seen: dict[str, dict] = {}
        for genre in genres:
            # Quote multi-word genres so Spotify's parser treats them as a single term.
            query_genre = f'"{genre}"' if " " in genre else genre
            fetched = 0
            offset = 0
            while fetched < tracks_per_genre:
                limit = min(50, tracks_per_genre - fetched)
                result = self._call_api(
                    self.sp.search,
                    q=f"genre:{query_genre}",
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
        """Fetch audio features from ReccoBeats in two steps:
        1. Batch-resolve Spotify IDs to ReccoBeats UUIDs.
        2. Fetch audio features per ReccoBeats UUID.
        The returned records use the original Spotify track ID as "id" so the
        existing bronze → silver transform requires no changes.
        """
        # Short-circuit if no track IDs provided, to avoid unnecessary ReccoBeats calls and log noise.
        if len(track_ids) == 0:
            return []

        # Step 1: resolve Spotify IDs → ReccoBeats IDs (max 40 per request)
        rb_id_map: dict[str, str] = {}  # spotify_id → reccobeats_id
        for batch in _batched(track_ids, 40):
            url = f"{_RECCOBEATS_BASE}/track?ids={','.join(batch)}"
            try:
                data = self._call_rb(url)
                for item in data.get("content", []):
                    # href is "https://open.spotify.com/track/{spotify_id}"
                    spotify_id = item.get("href", "").rsplit("/", 1)[-1]
                    if spotify_id and item.get("id"):
                        rb_id_map[spotify_id] = item["id"]
            except requests.exceptions.RequestException as exc:
                logger.warning("ReccoBeats track lookup failed for batch: %s", exc)

        if not rb_id_map:
            logger.warning("ReccoBeats resolved 0 of %d track IDs", len(track_ids))
            return []

        logger.info("Resolved %d / %d tracks to ReccoBeats IDs", len(rb_id_map), len(track_ids))

        # Step 2: fetch audio features for each resolved ReccoBeats ID
        features: list[dict] = []
        for spotify_id, rb_id in rb_id_map.items():
            url = f"{_RECCOBEATS_BASE}/track/{rb_id}/audio-features"
            try:
                data = self._call_rb(url)
                # Replace ReccoBeats UUID with Spotify ID so transform_audio_features
                # can join on track_id without any schema changes.
                data["id"] = spotify_id
                features.append(data)
            except requests.exceptions.RequestException as exc:
                logger.warning("ReccoBeats audio features failed for %s (%s): %s", spotify_id, rb_id, exc)
            time.sleep(0.05)  # stay polite to the public API

        logger.info("Fetched %d audio feature records from ReccoBeats", len(features))
        return features

    def _fetch_artists(self, artist_ids: list[str]) -> list[dict]:
        artists: list[dict] = []
        for batch in _batched(artist_ids, 50):
            result = self._call_api(self.sp.artists, batch)
            artists.extend(a for a in result["artists"] if a is not None)
        return artists

    def _call_rb(self, url: str) -> dict:
        """GET a ReccoBeats URL and return the parsed JSON, with retry logic on transient errors."""
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self._rb_session.get(url, timeout=10)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError as exc:
                if exc.response.status_code == 429:
                    headers = exc.response.headers
                    wait = float(headers.get("Retry-After", self.BACKOFF_BASE * (2 ** attempt)))
                    logger.warning(
                        "ReccoBeats rate limited. Retry in %.1fs (attempt %d/%d)", wait, attempt + 1, self.MAX_RETRIES
                    )
                elif 500 <= exc.response.status_code < 600:
                    wait = self.BACKOFF_BASE * (2 ** attempt)
                    logger.warning("ReccoBeats server error %d. Retry in %.1fs", exc.response.status_code, wait)
                else:
                    raise
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(wait)
            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "ReccoBeats connection error: %s. Retrying in %.1fs (attempt %d/%d)",
                    exc, self.BACKOFF_BASE * (2 ** attempt), attempt + 1, self.MAX_RETRIES
                )
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.BACKOFF_BASE * (2 ** attempt))
        raise RuntimeError(f"ReccoBeats API call failed after {self.MAX_RETRIES} retries")

    # ------------------------------------------------------------------ #
    # Bronze layer storage                                                 #
    # ------------------------------------------------------------------ #

    def _save_to_bronze(self, records: list[dict], data_type: str, partition: str | None = None) -> Path | None:
        """Persist records as JSON under bronze/{data_type}[/{partition}]/YYYY-MM-DD/.

        ``partition`` is used for data types that need an extra sub-directory
        (e.g. the time_range for user_top_tracks).
        """
        if not records:
            return None
        if partition is not None and (
            not re.fullmatch(r'[a-zA-Z0-9_-]+', partition)
        ):
            raise ValueError(f"Invalid partition value: {partition!r}")
        now_utc = datetime.now(timezone.utc)
        timestamp = now_utc.strftime("%Y%m%dT%H%M%SZ")
        date_str = now_utc.strftime("%Y-%m-%d")
        if partition:
            dest_dir = self.bronze_dir / data_type / partition / date_str
            prefix = f"{data_type}_{partition}"
        else:
            dest_dir = self.bronze_dir / data_type / date_str
            prefix = data_type
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{prefix}_{timestamp}.json"
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d %s records -> %s", len(records), data_type, dest_path)
        if self._azure:
            try:
                self._azure.upload_file(dest_path, relative_to=BRONZE_DIR.parent)
            except AzureError as e:
                logger.error("Azure upload failed for %s: %s", dest_path, e)
        return dest_path

    # ------------------------------------------------------------------ #
    # Manifest (incremental loading)                                      #
    # ------------------------------------------------------------------ #

    def _load_manifest(self) -> dict:
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError, ValueError) as e:
                logger.warning("Manifest at %s is invalid (%s); starting fresh.", self._manifest_path, e)
        return {}

    def _save_manifest(self) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._manifest_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2)
        tmp.replace(self._manifest_path)

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


def _merge_artist(by_id: dict[str, dict], artist: dict) -> None:
    """Insert ``artist`` into ``by_id``, unioning ``genres`` if an entry already exists.

    The same artist often appears in multiple time-range top lists and in the
    followed-artists list with slightly different genre tags. A plain
    ``setdefault`` would drop the later genres; this helper preserves them so
    genre derivation sees the full picture.
    """
    existing = by_id.get(artist["id"])
    if existing is None:
        by_id[artist["id"]] = artist
        return
    merged_genres = list(dict.fromkeys((existing.get("genres") or []) + (artist.get("genres") or [])))
    existing["genres"] = merged_genres


def _fallback_genres() -> list[str]:
    return _CONFIG.get("spotify", {}).get("fallback_genres", [])


if __name__ == "__main__":
    client = SpotifyIngestionClient(bronze_dir=BRONZE_DIR)
    result = client.ingest()
    print("Ingestion summary:", result)

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.data_quality import DataQualityReport, DataQualityError, run_quality_checks
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


_CONFIG = _load_config()
BRONZE_DIR = Path(_CONFIG.get("storage", {}).get("bronze_dir", "data/bronze"))
SILVER_DIR = Path(_CONFIG.get("storage", {}).get("silver_dir", "data/silver"))

# Sub-genre patterns → broad category.  Order matters: more specific first.
_GENRE_PATTERNS: list[tuple[str, list[str]]] = [
    ("hip-hop",     ["hip-hop", "hip hop", "rap", "trap", "drill", "grime"]),
    ("electronic",  ["electronic", "edm", "house", "techno", "electro", "dubstep", "trance", "synth"]),
    ("jazz",        ["jazz", "bebop", "swing", "blues"]),
    ("r-n-b",       ["r&b", "rnb", "soul", "funk", "neo soul"]),
    ("country",     ["country", "bluegrass", "americana"]),
    ("classical",   ["classical", "orchestra", "symphony", "chamber", "opera", "baroque"]),
    ("latin",       ["latin", "reggaeton", "salsa", "cumbia", "bossa nova"]),
    ("folk",        ["folk", "singer-songwriter"]),
    ("rock",        ["rock", "metal", "punk", "grunge", "indie", "alternative"]),
    ("pop",         ["pop"]),
]

# Audio features already scaled 0-1 by Spotify
_UNIT_FEATURES = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence",
]
# mode is 0 or 1 — pass through
_BINARY_FEATURES = ["mode"]

# Features requiring explicit normalization (min, max)
_RANGE_FEATURES: dict[str, tuple[float, float]] = {
    "key":            (0, 11),
    "loudness":       (-60, 0),
    "tempo":          (50, 250),
    "time_signature": (3, 7),
}


def _categorize_genres(genres: list[str]) -> str:
    """Map a list of Spotify genre strings to a single broad category."""
    lowered = [g.lower() for g in genres]
    for category, patterns in _GENRE_PATTERNS:
        if any(pat in genre for genre in lowered for pat in patterns):
            return category
    return "other"


def _load_bronze_files(data_type: str, bronze_dir: Path = BRONZE_DIR) -> list[dict]:
    """Read and concatenate all JSON files for a data type from the bronze layer."""
    records: list[dict] = []
    paths = sorted(bronze_dir.glob(f"{data_type}/**/*.json"))
    if not paths:
        logger.warning("No bronze files found for data_type=%s under %s", data_type, bronze_dir)
        return records
    for path in paths:
        with open(path, encoding="utf-8") as f:
            batch = json.load(f)
        if isinstance(batch, list):
            records.extend(batch)
    logger.info("Loaded %d raw %s records from %d file(s)", len(records), data_type, len(paths))
    return records


def transform_tracks(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for t in raw:
        if not t or not t.get("id"):
            continue
        artists = t.get("artists") or []
        album = t.get("album") or {}
        rows.append({
            "track_id": t["id"],
            "name": t.get("name"),
            "popularity": t.get("popularity"),
            "duration_ms": t.get("duration_ms"),
            "explicit": t.get("explicit"),
            "primary_artist_id": artists[0]["id"] if artists and artists[0].get("id") else None,
            "primary_artist_name": artists[0].get("name") if artists else None,
            "album_id": album.get("id"),
            "album_name": album.get("name"),
            "release_date": album.get("release_date"),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["track_id", "name"])
    df = df.drop_duplicates(subset=["track_id"], keep="first")
    return df.reset_index(drop=True)


def transform_audio_features(raw: list[dict]) -> pd.DataFrame:
    all_cols = list(_RANGE_FEATURES) + _UNIT_FEATURES + _BINARY_FEATURES + ["duration_ms", "time_signature"]
    rows = []
    for item in raw:
        if not item or not item.get("id"):
            continue
        row = {"track_id": item["id"]}
        for col in all_cols:
            row[col] = item.get(col)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["track_id"])
    df = df.drop_duplicates(subset=["track_id"], keep="first")

    # Normalize range-bound features
    for col, (lo, hi) in _RANGE_FEATURES.items():
        if col in df.columns:
            df[col] = ((df[col] - lo) / (hi - lo)).clip(0, 1)

    # Clip unit features to [0, 1] as a safety net
    for col in _UNIT_FEATURES:
        if col in df.columns:
            df[col] = df[col].clip(0, 1)

    return df.reset_index(drop=True)


def transform_artists(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for a in raw:
        if not a or not a.get("id"):
            continue
        genres = a.get("genres") or []
        followers = (a.get("followers") or {}).get("total")
        rows.append({
            "artist_id": a["id"],
            "name": a.get("name"),
            "popularity": a.get("popularity"),
            "followers": followers,
            "genres": ",".join(genres),
            "primary_genre": _categorize_genres(genres),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["artist_id", "name"])
    df = df.drop_duplicates(subset=["artist_id"], keep="first")
    return df.reset_index(drop=True)


def _write_parquet(df: pd.DataFrame, table_name: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{table_name}.parquet"
    df.to_parquet(out, engine="pyarrow", index=False)
    logger.info("Wrote %d rows -> %s", len(df), out)
    return out


def run(bronze_dir: Path = BRONZE_DIR, silver_dir: Path = SILVER_DIR) -> dict[str, DataQualityReport]:
    """Bronze → Silver: load, clean, normalize, validate, and persist as Parquet."""
    reports: dict[str, DataQualityReport] = {}

    # --- Tracks ---
    raw = _load_bronze_files("tracks", bronze_dir)
    if raw:
        df = transform_tracks(raw)
        report = run_quality_checks(
            df, "silver/tracks",
            required_cols=["track_id", "name", "popularity"],
            key_cols=["track_id"],
        )
        reports["tracks"] = report
        _write_parquet(df, "tracks", silver_dir)

    # --- Audio features ---
    raw = _load_bronze_files("audio_features", bronze_dir)
    if raw:
        df = transform_audio_features(raw)
        report = run_quality_checks(
            df, "silver/audio_features",
            required_cols=["track_id", "danceability", "energy", "tempo"],
            key_cols=["track_id"],
        )
        reports["audio_features"] = report
        _write_parquet(df, "audio_features", silver_dir)

    # --- Artists ---
    raw = _load_bronze_files("artists", bronze_dir)
    if raw:
        df = transform_artists(raw)
        report = run_quality_checks(
            df, "silver/artists",
            required_cols=["artist_id", "name", "popularity"],
            key_cols=["artist_id"],
        )
        reports["artists"] = report
        _write_parquet(df, "artists", silver_dir)

    return reports


if __name__ == "__main__":
    results = run()
    for table, r in results.items():
        print(f"{table}: {'PASS' if r.passed else 'FAIL'} ({r.row_count} rows)")

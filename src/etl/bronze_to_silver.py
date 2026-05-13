import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.data_quality import DataQualityReport, assert_quality, run_quality_checks
from src.utils.logging_config import get_logger
from src.utils.parquet_io import write_parquet

logger = get_logger(__name__)

# Sub-genre patterns → broad category.  Order matters: more specific first.
# "folk" intentionally precedes "rock" so that compound genres like "folk rock"
# are categorized as folk rather than rock.
# "pop" is intentionally the catch-all tail — any genre not matched above that
# contains "pop" as a substring will land here (e.g. "dream pop", "synth-pop").
_GENRE_PATTERNS: list[tuple[str, list[str]]] = [
    ("hip-hop",     ["hip-hop", "hip hop", "rap", "trap", "drill", "grime"]),
    ("electronic",  ["electronic", "edm", "house", "techno", "electro", "dubstep", "trance", "synth"]),
    ("jazz",        ["jazz", "bebop", "swing", "blues"]),
    ("r-n-b",       ["r&b", "rnb", "soul", "funk", "neo soul"]),
    ("country",     ["country", "bluegrass", "americana"]),
    ("classical",   ["classical", "orchestra", "symphony", "chamber", "opera", "baroque"]),
    ("latin",       ["latin", "reggaeton", "salsa", "cumbia", "bossa nova"]),
    ("folk",        ["folk", "singer-songwriter"]),
    ("rock",        ["rock", "metal", "punk", "grunge", "indie rock", "alternative"]),
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


def _load_bronze_files(data_type: str, bronze_dir: Path) -> list[dict]:
    """Read and concatenate all JSON files for a data type from the bronze layer."""
    records: list[dict] = []
    paths = sorted(bronze_dir.glob(f"{data_type}/**/*.json"))
    if not paths:
        logger.warning("No bronze files found for data_type=%s under %s", data_type, bronze_dir)
        return records
    loaded_count = 0
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                batch = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSON in %s", path)
            continue
        if isinstance(batch, list):
            records.extend(batch)
            loaded_count += 1
        else:
            logger.warning("Skipping non-list JSON in %s", path)
    logger.info("Loaded %d raw %s records from %d file(s)", len(records), data_type, loaded_count)
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
    # time_signature is already in _RANGE_FEATURES so list(_RANGE_FEATURES) covers it
    all_cols = list(_RANGE_FEATURES) + _UNIT_FEATURES + _BINARY_FEATURES + ["duration_ms"]
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

    # Normalize range-bound features; warn when values fall outside the defined bounds
    for col, (lo, hi) in _RANGE_FEATURES.items():
        if col not in df.columns:
            continue
        out_of_range = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        if out_of_range.any():
            logger.warning(
                "%d %s value(s) outside expected range [%s, %s] — will be clipped",
                out_of_range.sum(), col, lo, hi,
            )
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


def run(bronze_dir: Path | None = None, silver_dir: Path | None = None) -> dict[str, DataQualityReport]:
    """Bronze → Silver: load, clean, normalize, validate, and persist as Parquet."""
    cfg = load_config()
    bronze_dir = bronze_dir or Path(cfg.get("storage", {}).get("bronze_dir", "data/bronze"))
    silver_dir = silver_dir or Path(cfg.get("storage", {}).get("silver_dir", "data/silver"))

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
        assert_quality(report)
        write_parquet(df, "tracks", silver_dir)

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
        assert_quality(report)
        write_parquet(df, "audio_features", silver_dir)

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
        assert_quality(report)
        write_parquet(df, "artists", silver_dir)

    return reports


if __name__ == "__main__":
    results = run()
    for table, r in results.items():
        print(f"{table}: {'PASS' if r.passed else 'FAIL'} ({r.row_count} rows)")

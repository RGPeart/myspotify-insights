from pathlib import Path

import pandas as pd
import yaml

from src.utils.data_quality import DataQualityReport, run_quality_checks
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


_CONFIG = _load_config()
SILVER_DIR = Path(_CONFIG.get("storage", {}).get("silver_dir", "data/silver"))
GOLD_DIR = Path(_CONFIG.get("storage", {}).get("gold_dir", "data/gold"))


def _load_silver(table_name: str, silver_dir: Path = SILVER_DIR) -> pd.DataFrame | None:
    path = silver_dir / f"{table_name}.parquet"
    if not path.exists():
        logger.warning("Silver table not found: %s — skipping", path)
        return None
    return pd.read_parquet(path)


def build_dim_tracks(tracks: pd.DataFrame, artists: pd.DataFrame | None) -> pd.DataFrame:
    """Track dimension with composite popularity score and genre annotation."""
    df = tracks.copy()

    if artists is not None and not artists.empty:
        artist_lookup = artists.set_index("artist_id")[["primary_genre", "popularity"]].rename(
            columns={"popularity": "artist_popularity"}
        )
        df = df.join(artist_lookup, on="primary_artist_id")
    else:
        df["primary_genre"] = "unknown"
        df["artist_popularity"] = float("nan")

    df["track_popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    df["composite_popularity"] = (
        0.6 * df["track_popularity"].fillna(0) / 100
        + 0.4 * df["artist_popularity"].fillna(0) / 100
    ).clip(0, 1)

    keep = [
        "track_id", "name", "track_popularity", "duration_ms", "explicit",
        "primary_artist_id", "primary_artist_name", "album_id", "album_name",
        "release_date", "primary_genre", "composite_popularity",
    ]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def build_dim_artists(artists: pd.DataFrame) -> pd.DataFrame:
    """Artist dimension with genre metadata."""
    df = artists.rename(columns={"name": "artist_name"}).copy()
    keep = ["artist_id", "artist_name", "popularity", "followers", "genres", "primary_genre"]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def build_fact_audio_features(
    audio_features: pd.DataFrame, tracks: pd.DataFrame | None
) -> pd.DataFrame:
    """Fact table: normalized audio features joined with track dimension keys."""
    df = audio_features.copy()

    if tracks is not None and not tracks.empty:
        track_keys = tracks[["track_id", "primary_artist_id", "album_id"]].drop_duplicates("track_id")
        df = df.merge(track_keys, on="track_id", how="left")
    else:
        df["primary_artist_id"] = float("nan")
        df["album_id"] = float("nan")

    key_cols = ["track_id", "primary_artist_id", "album_id"]
    feature_cols = [c for c in df.columns if c not in key_cols]
    return df[key_cols + feature_cols].reset_index(drop=True)


def _write_parquet(df: pd.DataFrame, table_name: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{table_name}.parquet"
    df.to_parquet(out, engine="pyarrow", index=False)
    logger.info("Wrote %d rows -> %s", len(df), out)
    return out


def run(silver_dir: Path = SILVER_DIR, gold_dir: Path = GOLD_DIR) -> dict[str, DataQualityReport]:
    """Silver → Gold: build dimensional model and persist as Parquet."""
    tracks = _load_silver("tracks", silver_dir)
    if tracks is None or tracks.empty:
        logger.error("Silver tracks table missing or empty — cannot build gold layer")
        return {}

    audio_features = _load_silver("audio_features", silver_dir)
    artists = _load_silver("artists", silver_dir)

    reports: dict[str, DataQualityReport] = {}

    # dim_tracks
    dim_tracks = build_dim_tracks(tracks, artists)
    reports["dim_tracks"] = run_quality_checks(
        dim_tracks, "gold/dim_tracks",
        required_cols=["track_id", "name", "composite_popularity"],
        key_cols=["track_id"],
    )
    _write_parquet(dim_tracks, "dim_tracks", gold_dir)

    # dim_artists
    if artists is not None and not artists.empty:
        dim_artists = build_dim_artists(artists)
        reports["dim_artists"] = run_quality_checks(
            dim_artists, "gold/dim_artists",
            required_cols=["artist_id", "artist_name"],
            key_cols=["artist_id"],
        )
        _write_parquet(dim_artists, "dim_artists", gold_dir)

    # fact_audio_features
    if audio_features is not None and not audio_features.empty:
        fact_af = build_fact_audio_features(audio_features, tracks)
        reports["fact_audio_features"] = run_quality_checks(
            fact_af, "gold/fact_audio_features",
            required_cols=["track_id", "danceability", "energy"],
            key_cols=["track_id"],
        )
        _write_parquet(fact_af, "fact_audio_features", gold_dir)

    return reports


if __name__ == "__main__":
    results = run()
    for table, r in results.items():
        print(f"{table}: {'PASS' if r.passed else 'FAIL'} ({r.row_count} rows)")

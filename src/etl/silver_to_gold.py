from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.data_quality import DataQualityError, DataQualityReport, assert_quality, run_quality_checks
from src.utils.logging_config import get_logger
from src.utils.parquet_io import write_parquet

logger = get_logger(__name__)


def _load_silver(table_name: str, silver_dir: Path) -> pd.DataFrame | None:
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
        df["primary_genre"] = None
        df["artist_popularity"] = None

    # Fill NaN genre for tracks whose artist didn't match (partial or missing artists table)
    df["primary_genre"] = df["primary_genre"].fillna("unknown")

    df["track_popularity"] = pd.to_numeric(df["popularity"], errors="coerce")

    # Use median artist popularity for unmatched tracks rather than 0, which would
    # unfairly penalise them in the composite score. Falls back to 0 when all are unknown.
    median_ap = df["artist_popularity"].median()
    ap_fill = median_ap if pd.notna(median_ap) else 0.0
    ap = df["artist_popularity"].fillna(ap_fill)

    df["composite_popularity"] = (
        0.6 * df["track_popularity"].fillna(0) / 100
        + 0.4 * ap / 100
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
        df["primary_artist_id"] = None
        df["album_id"] = None

    key_cols = ["track_id", "primary_artist_id", "album_id"]
    feature_cols = [c for c in df.columns if c not in key_cols]
    return df[key_cols + feature_cols].reset_index(drop=True)


def run(silver_dir: Path | None = None, gold_dir: Path | None = None) -> dict[str, DataQualityReport]:
    """Silver → Gold: build dimensional model and persist as Parquet."""
    cfg = load_config()
    silver_dir = silver_dir or Path(cfg.get("storage", {}).get("silver_dir", "data/silver"))
    gold_dir = gold_dir or Path(cfg.get("storage", {}).get("gold_dir", "data/gold"))

    tracks = _load_silver("tracks", silver_dir)
    if tracks is None or tracks.empty:
        logger.error("Silver tracks table missing or empty — cannot build gold layer")
        return {}

    audio_features = _load_silver("audio_features", silver_dir)
    artists = _load_silver("artists", silver_dir)

    reports: dict[str, DataQualityReport] = {}

    # dim_tracks
    dim_tracks = build_dim_tracks(tracks, artists)
    report = run_quality_checks(
        dim_tracks, "gold/dim_tracks",
        required_cols=["track_id", "name", "composite_popularity"],
        key_cols=["track_id"],
    )
    reports["dim_tracks"] = report
    assert_quality(report)
    write_parquet(dim_tracks, "dim_tracks", gold_dir)

    # dim_artists
    if artists is not None and not artists.empty:
        dim_artists = build_dim_artists(artists)
        report = run_quality_checks(
            dim_artists, "gold/dim_artists",
            required_cols=["artist_id", "artist_name"],
            key_cols=["artist_id"],
        )
        reports["dim_artists"] = report
        assert_quality(report)
        write_parquet(dim_artists, "dim_artists", gold_dir)

    # fact_audio_features
    if audio_features is not None and not audio_features.empty:
        fact_af = build_fact_audio_features(audio_features, tracks)
        report = run_quality_checks(
            fact_af, "gold/fact_audio_features",
            required_cols=["track_id", "danceability", "energy"],
            key_cols=["track_id"],
        )
        reports["fact_audio_features"] = report
        assert_quality(report)
        write_parquet(fact_af, "fact_audio_features", gold_dir)

    return reports


if __name__ == "__main__":
    results = run()
    for table, r in results.items():
        print(f"{table}: {'PASS' if r.passed else 'FAIL'} ({r.row_count} rows)")

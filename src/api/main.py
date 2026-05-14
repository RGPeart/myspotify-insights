from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI

from src.utils.config import load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    gold_dir = Path(cfg.get("storage", {}).get("gold_dir", "data/gold"))
    models_dir = Path(cfg.get("models", {}).get("models_dir", "models"))
    app.state.collab_weight = float(cfg.get("models", {}).get("collab_weight", 0.7))
    app.state.n_recommendations = int(cfg.get("models", {}).get("n_recommendations", 10))

    try:
        from src.models.predict import load_artifacts
        app.state.artifacts = load_artifacts(models_dir)
        logger.info("Recommendation model loaded")
    except FileNotFoundError:
        logger.warning("Model artifacts not found — /recommendations will return 503")
        app.state.artifacts = None

    for table in ("dim_tracks", "dim_artists", "fact_audio_features"):
        path = gold_dir / f"{table}.parquet"
        if path.exists():
            setattr(app.state, table, pd.read_parquet(path))
            logger.info("Loaded %s (%d rows)", table, len(getattr(app.state, table)))
        else:
            setattr(app.state, table, None)
            logger.warning("Gold table not found: %s", path)

    # Pre-build O(1) lookup indexes so routes don't rebuild them per request
    dt = app.state.dim_tracks
    da = app.state.dim_artists
    af = app.state.fact_audio_features
    app.state.track_map = dt.set_index("track_id").to_dict("index") if dt is not None else {}
    app.state.tracks_idx = dt.set_index("track_id") if dt is not None else None
    app.state.artists_idx = da.set_index("artist_id") if da is not None else None
    app.state.af_idx = af.set_index("track_id") if af is not None else None

    yield

    for attr in ("artifacts", "dim_tracks", "dim_artists", "fact_audio_features",
                 "track_map", "tracks_idx", "artists_idx", "af_idx"):
        setattr(app.state, attr, None)


app = FastAPI(
    title="MySpotify Insights",
    version="1.0.0",
    description="Music recommendation engine powered by Spotify data",
    lifespan=lifespan,
)

from src.api.routes import router  # noqa: E402 — imported after app to avoid circular import
app.include_router(router)

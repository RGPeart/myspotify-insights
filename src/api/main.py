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

    yield
    app.state.artifacts = None


app = FastAPI(
    title="MySpotify Insights",
    version="1.0.0",
    description="Music recommendation engine powered by Spotify data",
    lifespan=lifespan,
)

from src.api.routes import router  # noqa: E402 — imported after app to avoid circular import
app.include_router(router)

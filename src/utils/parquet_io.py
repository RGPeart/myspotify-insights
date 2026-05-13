from pathlib import Path

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def write_parquet(df: pd.DataFrame, table_name: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{table_name}.parquet"
    df.to_parquet(out, engine="pyarrow", index=False)
    logger.info("Wrote %d rows -> %s", len(df), out)
    return out

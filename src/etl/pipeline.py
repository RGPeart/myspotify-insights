try:
    from prefect import flow, task
except ImportError:
    def flow(fn=None, **_):
        return fn if fn is not None else lambda f: f

    def task(fn=None, **_):
        return fn if fn is not None else lambda f: f

from src.etl import bronze_to_silver, silver_to_gold
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@task
def _run_bronze_to_silver():
    return bronze_to_silver.run()


@task
def _run_silver_to_gold():
    return silver_to_gold.run()


@flow
def spotify_etl_pipeline():
    logger.info("Pipeline started")
    silver_reports = _run_bronze_to_silver()
    gold_reports = _run_silver_to_gold()
    result = {
        "silver": {t: r.row_count for t, r in silver_reports.items()},
        "gold": {t: r.row_count for t, r in gold_reports.items()},
    }
    logger.info("Pipeline complete | %s", result)
    return result


if __name__ == "__main__":
    spotify_etl_pipeline()

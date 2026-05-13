"""
ETL orchestration pipeline.

Run locally (no Prefect required):
    python -m src.etl.pipeline

With Prefect installed, flows and tasks are fully instrumented. Without it,
the decorators are no-ops and the pipeline runs as plain Python functions.

Install Prefect:
    pip install "prefect>=3.0"
"""

try:
    from prefect import flow, task
    _HAS_PREFECT = True
except ImportError:
    _HAS_PREFECT = False

    def flow(fn=None, **_):  # type: ignore[misc]
        return fn if fn is not None else lambda f: f

    def task(fn=None, **_):  # type: ignore[misc]
        return fn if fn is not None else lambda f: f


from src.etl import bronze_to_silver, silver_to_gold
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@task(name="bronze-to-silver", retries=2, retry_delay_seconds=30)
def bronze_to_silver_task() -> dict[str, int]:
    logger.info("Stage 1: Bronze → Silver")
    reports = bronze_to_silver.run()
    return {table: r.row_count for table, r in reports.items()}


@task(name="silver-to-gold", retries=2, retry_delay_seconds=30)
def silver_to_gold_task() -> dict[str, int]:
    logger.info("Stage 2: Silver → Gold")
    reports = silver_to_gold.run()
    return {table: r.row_count for table, r in reports.items()}


@flow(name="spotify-etl-pipeline", log_prints=True)
def spotify_etl_pipeline() -> dict:
    """Full ETL pipeline: Bronze → Silver → Gold."""
    logger.info("Pipeline starting | prefect_available=%s", _HAS_PREFECT)
    silver_summary = bronze_to_silver_task()
    gold_summary = silver_to_gold_task()
    result = {"silver": silver_summary, "gold": gold_summary}
    logger.info("Pipeline complete | %s", result)
    return result


if __name__ == "__main__":
    result = spotify_etl_pipeline()
    print("Pipeline complete:", result)

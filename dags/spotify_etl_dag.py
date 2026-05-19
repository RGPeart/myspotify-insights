from __future__ import annotations
import pendulum
from datetime import timedelta

from airflow.sdk import DAG, task


@task(retries=2, retry_delay=timedelta(minutes=1), execution_timeout=timedelta(hours=1))
def _ingest_data():
    from src.ingestion.spotify_client import SpotifyIngestionClient
    from src.utils.logging_config import get_logger
    log = get_logger(__name__)
    client = SpotifyIngestionClient()
    summary = client.ingest()
    log.info("ingestion_complete", **summary)


@task(retries=2, retry_delay=timedelta(minutes=1), execution_timeout=timedelta(hours=1))
def _bronze_to_silver():
    from src.etl import bronze_to_silver
    from src.utils.logging_config import get_logger
    log = get_logger(__name__)
    reports = bronze_to_silver.run()
    for table, r in reports.items():
        log.info("etl_stage_complete", table=table, status="PASS" if r.passed else "FAIL", row_count=r.row_count)


@task(retries=2, retry_delay=timedelta(minutes=1), execution_timeout=timedelta(hours=1))
def _silver_to_gold():
    from src.etl import silver_to_gold
    from src.utils.logging_config import get_logger
    log = get_logger(__name__)
    reports = silver_to_gold.run()
    for table, r in reports.items():
        log.info("etl_stage_complete", table=table, status="PASS" if r.passed else "FAIL", row_count=r.row_count)


with DAG(
    dag_id="spotify_etl_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2026, 5, 1, tz="UTC"),
    catchup=False,
    tags=["spotify", "etl"],
) as dag:

    ingest_data = _ingest_data()
    transform_bronze_to_silver = _bronze_to_silver()
    transform_silver_to_gold = _silver_to_gold()

    ingest_data >> transform_bronze_to_silver >> transform_silver_to_gold

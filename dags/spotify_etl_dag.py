from __future__ import annotations
import pendulum
from datetime import timedelta

from src.ingestion.spotify_client import SpotifyIngestionClient
from src.etl import bronze_to_silver, silver_to_gold

from airflow.sdk import DAG, task

@task(retries=2, retry_delay=timedelta(minutes=1), execution_timeout=timedelta(hours=1))
def _ingest_data():
    client = SpotifyIngestionClient()
    summary = client.ingest()
    print(f"Ingestion summary: {summary}")

@task(retries=2, retry_delay=timedelta(minutes=1), execution_timeout=timedelta(hours=1))
def _bronze_to_silver():
    reports = bronze_to_silver.run()
    for table, r in reports.items():
        print(f"{table}: {'PASS' if r.passed else 'FAIL'} ({r.row_count} rows)")

@task(retries=2, retry_delay=timedelta(minutes=1), execution_timeout=timedelta(hours=1))
def _silver_to_gold():
    reports = silver_to_gold.run()
    for table, r in reports.items():
        print(f"{table}: {'PASS' if r.passed else 'FAIL'} ({r.row_count} rows)")

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

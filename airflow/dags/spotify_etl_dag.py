from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.decorators import task

from src.ingestion.spotify_client import SpotifyIngestionClient
from src.etl import bronze_to_silver, silver_to_gold

with DAG(
    dag_id="spotify_etl_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2026, 5, 1, tz="UTC"),
    catchup=False,
    tags=["spotify", "etl"],
) as dag:
    @task(task_id="ingest_the_data")
    def _ingest_data(**kwargs):
        client = SpotifyIngestionClient()
        summary = client.ingest()
        print(f"Ingestion summary: {summary}")
    ingest_data = _ingest_data()

    @task(task_id="transform_bronze_to_silver")
    def _bronze_to_silver(**kwargs):
        reports = bronze_to_silver.run()
        for table, r in reports.items():
            print(f"{table}: {'PASS' if r.passed else 'FAIL'} ({r.row_count} rows)")
    transform_bronze_to_silver = _bronze_to_silver()

    @task(task_id="transform_silver_to_gold")
    def _silver_to_gold(**kwargs):
        reports = silver_to_gold.run()
        for table, r in reports.items():
            print(f"{table}: {'PASS' if r.passed else 'FAIL'} ({r.row_count} rows)")
    transform_silver_to_gold = _silver_to_gold()


    ingest_data >> transform_bronze_to_silver >> transform_silver_to_gold

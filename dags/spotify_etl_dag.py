from __future__ import annotations
import pendulum
from datetime import timedelta

from airflow.sdk import DAG, task
from airflow.providers.standard.operators.bash import BashOperator


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


DBT_PROJECT_DIR = "/opt/airflow/dbt"
DBT_VARS = '{"silver_dir": "/opt/airflow/data/silver", "gold_dir": "/opt/airflow/data/gold"}'


with DAG(
    dag_id="spotify_etl_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2026, 5, 1, tz="UTC"),
    catchup=False,
    tags=["spotify", "etl"],
) as dag:

    ingest_data = _ingest_data()
    transform_bronze_to_silver = _bronze_to_silver()

    dbt_deps = BashOperator(
        task_id="dbt_deps",
        bash_command=f"cd {DBT_PROJECT_DIR} && dbt deps --profiles-dir .",
        retries=1,
        execution_timeout=timedelta(minutes=10),
    )

    dbt_run = BashOperator(
        task_id="dbt_run_gold",
        bash_command=(
            f"cd {DBT_PROJECT_DIR} && "
            f"dbt run --profiles-dir . --target prod --vars '{DBT_VARS}'"
        ),
        retries=2,
        retry_delay=timedelta(minutes=1),
        execution_timeout=timedelta(hours=1),
    )

    dbt_test = BashOperator(
        task_id="dbt_test_gold",
        bash_command=(
            f"cd {DBT_PROJECT_DIR} && "
            f"dbt test --profiles-dir . --target prod --vars '{DBT_VARS}'"
        ),
        retries=1,
        execution_timeout=timedelta(minutes=30),
    )

    ingest_data >> transform_bronze_to_silver >> dbt_deps >> dbt_run >> dbt_test

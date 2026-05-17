from __future__ import annotations
import pendulum
from airflow.sdk import DAG, task

@task(task_id="hello_world_task")
def hello_world():
    print("Hello, Airflow!")

with DAG(
    dag_id="test_hello_world",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    schedule=None,
    tags=["test"],
) as dag:
    run_hello_world = hello_world()

    run_hello_world

"""Tests for the spotify_etl_pipeline Airflow DAG structure and task logic."""
import importlib.util
import pathlib
import sys
import types
import warnings
import pytest

warnings.filterwarnings("ignore", category=RuntimeWarning, module="airflow")


# ---------------------------------------------------------------------------
# Load the DAG module with ETL deps mocked so no real data is touched.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dag_module():
    from unittest.mock import MagicMock
    from src.utils.data_quality import DataQualityReport

    # Stub heavy imports before loading the DAG file.
    patched_sys = [
        "src.etl.bronze_to_silver",
        "src.ingestion.spotify_client",
    ]
    originals = {k: sys.modules.get(k) for k in patched_sys}

    b2s_mod = types.ModuleType("src.etl.bronze_to_silver")
    b2s_mod.run = MagicMock(return_value={"tracks": DataQualityReport("tracks", 10)})
    sys.modules["src.etl.bronze_to_silver"] = b2s_mod

    ingestion_mod = types.ModuleType("src.ingestion.spotify_client")
    mock_client = MagicMock()
    mock_client.ingest.return_value = {"tracks": 5, "audio_features": 5, "artists": 3}
    ingestion_mod.SpotifyIngestionClient = MagicMock(return_value=mock_client)
    sys.modules["src.ingestion.spotify_client"] = ingestion_mod

    # The DAG tasks use lazy imports (`from src.etl import bronze_to_silver` inside
    # the function body). Python resolves `from package import submodule` via the
    # package object's attribute, not sys.modules, when the package is already loaded
    # (which happens when pytest collects test_etl.py before test_dag.py runs).
    # Patching sys.modules alone is not enough — we must also set the attribute on
    # the already-loaded src.etl package so the lazy imports hit the mocks.
    import src.etl as _etl_pkg
    _etl_b2s_orig = getattr(_etl_pkg, "bronze_to_silver", None)
    _etl_pkg.bronze_to_silver = b2s_mod

    dag_path = pathlib.Path(__file__).parent.parent / "dags" / "spotify_etl_dag.py"
    spec = importlib.util.spec_from_file_location("spotify_etl_dag", dag_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.bronze_to_silver = b2s_mod
    mod.SpotifyIngestionClient = ingestion_mod.SpotifyIngestionClient

    yield mod

    for key, original in originals.items():
        if original is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = original

    if _etl_b2s_orig is not None:
        _etl_pkg.bronze_to_silver = _etl_b2s_orig
    elif hasattr(_etl_pkg, "bronze_to_silver"):
        delattr(_etl_pkg, "bronze_to_silver")


# ---------------------------------------------------------------------------
# DAG structure tests
# ---------------------------------------------------------------------------

class TestDagStructure:
    def test_dag_id(self, dag_module):
        assert dag_module.dag.dag_id == "spotify_etl_pipeline"

    def test_no_schedule(self, dag_module):
        assert dag_module.dag.schedule is None

    def test_no_catchup(self, dag_module):
        assert dag_module.dag.catchup is False

    def test_tags_present(self, dag_module):
        assert "spotify" in dag_module.dag.tags
        assert "etl" in dag_module.dag.tags

    def test_task_ordering(self, dag_module):
        ingest = dag_module.dag.get_task("_ingest_data")
        b2s = dag_module.dag.get_task("_bronze_to_silver")
        dbt_deps = dag_module.dag.get_task("dbt_deps")
        dbt_run = dag_module.dag.get_task("dbt_run_gold")
        dbt_test = dag_module.dag.get_task("dbt_test_gold")
        assert b2s.task_id in {t.task_id for t in ingest.downstream_list}
        assert dbt_deps.task_id in {t.task_id for t in b2s.downstream_list}
        assert dbt_run.task_id in {t.task_id for t in dbt_deps.downstream_list}
        assert dbt_test.task_id in {t.task_id for t in dbt_run.downstream_list}


# ---------------------------------------------------------------------------
# Task callable tests — call the underlying function directly
# ---------------------------------------------------------------------------

class TestDagTaskCallables:
    def test_ingest_task_callable(self, dag_module):
        dag_module.SpotifyIngestionClient.reset_mock()
        dag_module._ingest_data.function()
        dag_module.SpotifyIngestionClient.assert_called_once()

    def test_bronze_to_silver_task_callable(self, dag_module):
        dag_module.bronze_to_silver.run.reset_mock()
        dag_module._bronze_to_silver.function()
        dag_module.bronze_to_silver.run.assert_called_once()

    def test_dbt_tasks_invoke_dbt(self, dag_module):
        for task_id in ("dbt_deps", "dbt_run_gold", "dbt_test_gold"):
            task = dag_module.dag.get_task(task_id)
            assert "dbt" in task.bash_command

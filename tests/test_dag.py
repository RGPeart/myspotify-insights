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
        "src.etl.silver_to_gold",
        "src.ingestion.spotify_client",
    ]
    originals = {k: sys.modules.get(k) for k in patched_sys}

    b2s_mod = types.ModuleType("src.etl.bronze_to_silver")
    b2s_mod.run = MagicMock(return_value={"tracks": DataQualityReport("tracks", 10)})
    sys.modules["src.etl.bronze_to_silver"] = b2s_mod

    s2g_mod = types.ModuleType("src.etl.silver_to_gold")
    s2g_mod.run = MagicMock(return_value={"dim_tracks": DataQualityReport("dim_tracks", 10)})
    sys.modules["src.etl.silver_to_gold"] = s2g_mod

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
    _etl_s2g_orig = getattr(_etl_pkg, "silver_to_gold", None)
    _etl_pkg.bronze_to_silver = b2s_mod
    _etl_pkg.silver_to_gold = s2g_mod

    dag_path = pathlib.Path(__file__).parent.parent / "dags" / "spotify_etl_dag.py"
    spec = importlib.util.spec_from_file_location("spotify_etl_dag", dag_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.bronze_to_silver = b2s_mod
    mod.silver_to_gold = s2g_mod
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

    if _etl_s2g_orig is not None:
        _etl_pkg.silver_to_gold = _etl_s2g_orig
    elif hasattr(_etl_pkg, "silver_to_gold"):
        delattr(_etl_pkg, "silver_to_gold")


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
        s2g = dag_module.dag.get_task("_silver_to_gold")
        assert b2s.task_id in {t.task_id for t in ingest.downstream_list}
        assert s2g.task_id in {t.task_id for t in b2s.downstream_list}


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

    def test_silver_to_gold_task_callable(self, dag_module):
        dag_module.silver_to_gold.run.reset_mock()
        dag_module._silver_to_gold.function()
        dag_module.silver_to_gold.run.assert_called_once()

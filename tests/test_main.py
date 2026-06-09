from fastapi.testclient import TestClient

from src.api.main import app


# ------------------------------------------------------------------ #
# Application startup (lifespan)                                       #
# ------------------------------------------------------------------ #
# Unlike test_api.py, which builds a minimal stand-in app to isolate the
# routes, these tests boot the *real* application from src/api/main.py.
# Using TestClient as a context manager ("with ... as client") is what
# runs the real lifespan() startup/shutdown, exercising config loading,
# model-artifact loading, and gold-table loading.

class TestAppStartup:
    # Booting the real app and hitting /health proves the lifespan startup
    # runs end-to-end without raising, even when no model or gold data is
    # present (lifespan is written to degrade gracefully to None).
    def test_app_boots_and_health_responds(self):
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200

    # The booted app must report the version declared on the FastAPI object
    # so operators can correlate a running process with a deployment.
    def test_health_reports_version(self):
        with TestClient(app) as client:
            data = client.get("/health").json()
            assert data["version"] == app.version


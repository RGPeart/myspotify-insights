import os

from src.utils.logging_config import get_logger

_log = get_logger(__name__)


def configure_azure_monitor() -> bool:
    """Configure Azure Monitor OpenTelemetry exporter if connection string is present.

    Returns True when successfully configured, False when skipped or unavailable.
    Designed to fail gracefully so the pipeline runs without an Azure workspace.
    """
    connection_string = os.environ.get("AZURE_MONITOR_CONNECTION_STRING")
    if not connection_string:
        _log.info(
            "azure_monitor_skipped",
            reason="AZURE_MONITOR_CONNECTION_STRING not set",
        )
        return False

    from azure.monitor.opentelemetry import configure_azure_monitor as _configure

    try:
        _configure(connection_string=connection_string)
        _log.info("azure_monitor_configured")
        return True
    except Exception as exc:
        _log.warning("azure_monitor_error", error=str(exc))
        return False

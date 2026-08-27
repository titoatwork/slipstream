"""API smoke tests. No GPU required."""

from __future__ import annotations

from fastapi.testclient import TestClient
from slipstream.entrypoints.api_server import create_app


def test_health_and_models() -> None:
    app = create_app(None)
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    models = client.get("/v1/models").json()
    assert models["object"] == "list"
    assert models["data"][0]["owned_by"] == "slipstream"


def test_metrics_without_engine_is_503() -> None:
    client = TestClient(create_app(None))
    assert client.get("/metrics").status_code == 503

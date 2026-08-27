"""WebSocket metrics stream for the live dashboard. Phase 3 (A5)."""

from __future__ import annotations

from slipstream.observability.metrics import MetricsSnapshot


class MetricsStreamer:
    def publish(self, snapshot: MetricsSnapshot) -> None:
        raise NotImplementedError("Phase 3 — S12 MetricsStreamer.publish")

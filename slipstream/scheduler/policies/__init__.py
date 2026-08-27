"""Swappable scheduling policies. Ablations are a one-line config change."""

from __future__ import annotations

from slipstream.core.types import SchedulingPolicy
from slipstream.scheduler.policies.base import SchedulingPolicy as SchedulingPolicyBase
from slipstream.scheduler.policies.fcfs import FCFSPolicy
from slipstream.scheduler.policies.horizon import HorizonPolicy
from slipstream.scheduler.policies.oracle import OraclePolicy

POLICY_REGISTRY: dict[str, type] = {
    "fcfs": FCFSPolicy,
    "horizon": HorizonPolicy,
    "oracle": OraclePolicy,
}


def get_policy(
    name: str,
    *,
    safety_factor: float = 0.95,
    starvation_guard_ms: float = 5_000.0,
    feature_set: object | None = None,
) -> SchedulingPolicy:
    try:
        cls = POLICY_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(POLICY_REGISTRY))
        raise ValueError(f"unknown policy {name!r}; choose one of: {known}") from exc
    if name == "fcfs":
        return cls()  # type: ignore[no-any-return]
    if name == "horizon" and feature_set is not None:
        return cls(  # type: ignore[no-any-return]
            safety_factor=safety_factor,
            starvation_guard_ms=starvation_guard_ms,
            feature_set=feature_set,
        )
    return cls(safety_factor=safety_factor, starvation_guard_ms=starvation_guard_ms)  # type: ignore[no-any-return]


__all__ = [
    "POLICY_REGISTRY",
    "FCFSPolicy",
    "HorizonPolicy",
    "OraclePolicy",
    "SchedulingPolicyBase",
    "get_policy",
]

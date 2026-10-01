"""Coupled maintenance-airworthiness and turnaround-departure decisions.

The release decision (airworthiness) and the departure-plan decision must take
effect at the same point in time.  If maintenance will not release the
aircraft the turnaround plan must not keep its ground-equipment reservations,
and if the plan is withdrawn instead, the release is not published either.
Both branches are evaluated here as one atomic domain transition.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .maintenance import MaintenanceControl
from .models import Decision
from .turnaround import TurnPlan, TurnaroundPlanner


@dataclass(frozen=True)
class MaintenanceDepartureVerdict:
    accepted: bool
    code: str
    reason: str
    release: Decision
    plan: Decision
    details: dict[str, object]


def decide_maintenance_departure(
    maintenance: MaintenanceControl,
    planner: TurnaroundPlanner,
    aircraft_reg: str,
    plan: TurnPlan,
    conditions: Iterable[str],
    withdraw_if_grounded: bool = False,
) -> MaintenanceDepartureVerdict:
    """Settle airworthiness and the departure plan together.

    A positive maintenance release commits the turnaround plan; a negative
    release either leaves the plan uncommitted (``withdraw_if_grounded=False``,
    "wait for maintenance") or withdraws it and frees the reserved equipment
    (``withdraw_if_grounded=True``, "pull the plan").  Either way the
    airworthiness conclusion and the plan change share one decision point: a
    rejected release never coexists with a committed plan.
    """
    release = maintenance.release(aircraft_reg, plan.departure, conditions)
    if release.accepted:
        plan_decision = planner.commit(plan)
        if not plan_decision.accepted:
            return MaintenanceDepartureVerdict(
                False, plan_decision.code, plan_decision.reason, release, plan_decision,
                {"branch": "release-commit"},
            )
        return MaintenanceDepartureVerdict(
            True, "departure-released", "aircraft released and departure plan committed",
            release, plan_decision, {"branch": "release-commit"},
        )

    if withdraw_if_grounded:
        plan_decision = planner.withdraw(plan.flight_id)
        return MaintenanceDepartureVerdict(
            False, "departure-withdrawn",
            "aircraft not released; turnaround plan withdrawn and equipment freed",
            release, plan_decision, {"branch": "withdraw"},
        )

    plan_decision = Decision(
        False, "departure-held",
        "aircraft not released; turnaround plan held without equipment commitment",
        {"flight_id": plan.flight_id},
    )
    return MaintenanceDepartureVerdict(
        False, "departure-held",
        "aircraft not released; waiting for maintenance",
        release, plan_decision, {"branch": "wait"},
    )

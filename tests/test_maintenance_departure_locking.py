import unittest
from datetime import datetime, timezone

from rampflow.ledger import ReservationLedger
from rampflow.maintenance import Defect, MELRule, MaintenanceControl
from rampflow.maintenance_departure import decide_maintenance_departure
from rampflow.models import Decision, ServiceState, TaskSpec
from rampflow.turnaround import TurnTask, TurnaroundPlanner, ResourceRequirement
from rampflow.workflows.control import RampWorkflowControl, WorkflowRejected

UTC = timezone.utc


def dt(y, mo, d, h=0, mi=0, s=0, us=0):
    return datetime(y, mo, d, h, mi, s, us, tzinfo=UTC)


class CalendarDayExpiryTests(unittest.TestCase):
    def _control(self, days):
        return MaintenanceControl([MELRule("52-10-00", "C", days, None)])

    def test_expiry_is_end_of_calendar_day_not_rolling_24h(self):
        mc = self._control(1)
        mc.report(Defect("D1", "B-1", "52-10-00", dt(2026, 3, 10, 8), "minor", "x"))
        deferral = mc.defer("D1", dt(2026, 3, 10, 23, 30), ())
        # Rolling 24h would expire at 23:30 on Mar 11; calendar day expiry is
        # the end of Mar 11, so a morning departure is still legal.
        self.assertEqual(deferral.expiry_at, dt(2026, 3, 11, 23, 59, 59, 999999))
        ok = mc.release("B-1", dt(2026, 3, 11, 6), ())
        self.assertTrue(ok.accepted, ok.reason)

    def test_release_denied_after_last_calendar_day(self):
        mc = self._control(3)
        mc.report(Defect("D1", "B-1", "52-10-00", dt(2026, 3, 10, 8), "minor", "x"))
        mc.defer("D1", dt(2026, 3, 10, 23, 30), ())
        denied = mc.release("B-1", dt(2026, 3, 14, 0, 1), ())
        self.assertFalse(denied.accepted)
        self.assertEqual(denied.code, "mel-expired")


def _make_planner():
    return TurnaroundPlanner({"gpu": ["GPU-1", "GPU-2"]})


def _make_plan(planner, flight_id="CA1", anchor=dt(2026, 3, 10, 8), departure=dt(2026, 3, 10, 9)):
    task = TurnTask(
        TaskSpec("power", 30, earliest_offset_min=0, latest_finish_offset_min=60),
        requirements=(ResourceRequirement("gpu"),),
    )
    return planner.plan(flight_id, [task], anchor, departure)


class StalePlanCommitTests(unittest.TestCase):
    def test_stale_generation_plan_is_rejected_not_accepted(self):
        planner = _make_planner()
        plan = _make_plan(planner)
        # Another plan commits first and advances the resource generation.
        other = _make_plan(planner, "CA2", dt(2026, 3, 10, 12), dt(2026, 3, 10, 13))
        self.assertTrue(planner.commit(other).accepted)
        decision = planner.commit(plan)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "stale-generation")
        self.assertNotIn("CA1", planner.plans)

    def test_fresh_plan_commits_and_withdraw_releases_equipment(self):
        planner = _make_planner()
        plan = _make_plan(planner)
        self.assertTrue(planner.commit(plan).accepted)
        gen = planner.ledger.generation
        decision = planner.withdraw("CA1", gen)
        self.assertTrue(decision.accepted)
        self.assertNotIn("CA1", planner.plans)
        self.assertEqual(planner.ledger.conflicts("GPU-1", plan.runs["power"].planned), ())

    def test_withdraw_stale_generation_rejected(self):
        planner = _make_planner()
        plan = _make_plan(planner)
        planner.commit(plan)
        other = _make_plan(planner, "CA2", dt(2026, 3, 10, 12), dt(2026, 3, 10, 13))
        planner.commit(other)
        decision = planner.withdraw("CA1", 0)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "stale-generation")


class MaintenanceDepartureAtomicTests(unittest.TestCase):
    def _maintenance(self, release_ok):
        if release_ok:
            return MaintenanceControl([])
        mc = MaintenanceControl([MELRule("52-10-00", "C", 1, None)])
        mc.report(Defect("D1", "B-1", "52-10-00", dt(2026, 3, 9, 8), "minor", "x"))
        mc.defer("D1", dt(2026, 3, 9, 10), ())
        return mc  # expires end of Mar 10

    def test_release_and_plan_commit_together(self):
        mc = self._maintenance(True)
        planner = _make_planner()
        plan = _make_plan(planner, departure=dt(2026, 3, 10, 9))
        verdict = decide_maintenance_departure(mc, planner, "B-1", plan, ())
        self.assertTrue(verdict.accepted)
        self.assertIn("CA1", planner.plans)
        self.assertTrue(verdict.plan.accepted)

    def test_wait_branch_leaves_plan_uncommitted_without_equipment(self):
        mc = self._maintenance(False)
        planner = _make_planner()
        plan = _make_plan(planner, departure=dt(2026, 3, 11, 9))
        verdict = decide_maintenance_departure(mc, planner, "B-1", plan, ())
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.code, "departure-held")
        self.assertNotIn("CA1", planner.plans)
        self.assertEqual(planner.ledger.conflicts("GPU-1", plan.runs["power"].planned), ())

    def test_withdraw_branch_frees_equipment(self):
        mc = self._maintenance(False)
        planner = _make_planner()
        plan = _make_plan(planner, departure=dt(2026, 3, 11, 9))
        verdict = decide_maintenance_departure(mc, planner, "B-1", plan, (), withdraw_if_grounded=True)
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.code, "departure-withdrawn")
        self.assertNotIn("CA1", planner.plans)
        self.assertTrue(verdict.plan.accepted)
        self.assertEqual(planner.ledger.conflicts("GPU-1", plan.runs["power"].planned), ())

    def test_stale_plan_means_no_release_effect(self):
        mc = self._maintenance(True)
        planner = _make_planner()
        plan = _make_plan(planner)
        blocker = _make_plan(planner, "CA2", dt(2026, 3, 10, 12), dt(2026, 3, 10, 13))
        planner.commit(blocker)
        verdict = decide_maintenance_departure(mc, planner, "B-1", plan, ())
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.plan.code, "stale-generation")
        self.assertNotIn("CA1", planner.plans)


class WorkflowAtomicBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.at = dt(2026, 3, 10, 9)

    def test_event_not_published_when_action_fails(self):
        control = RampWorkflowControl()

        def boom():
            raise RuntimeError("domain rejected")

        with self.assertRaises(RuntimeError):
            control.release_maintenance_turnaround("op-1", "B-1", self.at, boom)
        self.assertEqual(control.stream.head, 0)
        self.assertEqual(control.receipts, {})

    def test_rejected_requirements_block_release_workflow(self):
        control = RampWorkflowControl()
        denied = Decision(False, "mel-expired", "relief expired")
        with self.assertRaises(WorkflowRejected):
            control.release_maintenance_turnaround(
                "op-2", "B-1", self.at, lambda: "ok", requirements=(denied,)
            )
        self.assertEqual(control.stream.head, 0)

    def test_release_workflow_idempotent(self):
        control = RampWorkflowControl()
        calls = []
        kwargs = dict(subject="B-1", at=self.at, action=lambda: calls.append(1) or "ok")
        first = control.release_maintenance_turnaround("op-3", **kwargs)
        second = control.release_maintenance_turnaround("op-3", **kwargs)
        self.assertIs(first, second)
        self.assertEqual(calls, [1])
        self.assertEqual(control.stream.head, 1)
        self.assertEqual(first.event.topic, "maintenance.turnaround.release.decided")


if __name__ == "__main__":
    unittest.main()

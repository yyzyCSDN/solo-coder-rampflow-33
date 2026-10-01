import unittest
from datetime import datetime, timezone

from rampflow.models import Decision, TaskSpec
from rampflow.maintenance import MaintenanceControl, MELRule, Defect
from rampflow.turnaround import TurnaroundPlanner, TurnTask, ResourceRequirement
from rampflow.workflows.control import RampWorkflowControl, WorkflowRejected

UTC = timezone.utc


def dt(day, hour=0, minute=0):
    return datetime(2026, 3, day, hour, minute, tzinfo=UTC)


class MaintenanceCalendarExpiryTests(unittest.TestCase):
    def _control(self, days):
        mc = MaintenanceControl([MELRule(ata="29-11", category="B", calendar_days=days, flight_cycles=None)])
        mc.report(Defect("D1", "B-1234", "29-11", dt(10, 8, 30), "minor", "hyd light"))
        return mc

    def test_relief_runs_to_end_of_calendar_day_not_rolling_24h(self):
        d = self._control(1).defer("D1", dt(10, 8, 30), [])
        self.assertEqual(d.expiry_at, dt(12))
        mc = self._control(1)
        mc.defer("D1", dt(10, 8, 30), [])
        self.assertTrue(mc.release("B-1234", dt(11, 23, 59), []).accepted)
        self.assertFalse(mc.release("B-1234", dt(12, 0, 1), []).accepted)

    def test_departure_after_24h_but_same_calendar_day_still_released(self):
        mc = self._control(1)
        mc.defer("D1", dt(10, 8, 30), [])
        decision = mc.release("B-1234", dt(11, 9, 0), [])
        self.assertTrue(decision.accepted)

    def test_zero_and_ten_calendar_days_expire_at_midnight_utc(self):
        for days, expiry_day in ((0, 11), (10, 21)):
            mc = MaintenanceControl([MELRule(ata="29", category="X", calendar_days=days, flight_cycles=None)])
            mc.report(Defect(f"D{days}", "B-9", "29", dt(10, 8, 30), "minor", "x"))
            self.assertEqual(mc.defer(f"D{days}", dt(10, 8, 30), []).expiry_at, dt(expiry_day))


class StaleTurnaroundPlanTests(unittest.TestCase):
    def setUp(self):
        self.tasks = [TurnTask(TaskSpec("gpu-on", 20), requirements=(ResourceRequirement("gpu"),))]
        self.planner = TurnaroundPlanner({"gpu": ["G1"]})

    def test_plan_built_on_old_generation_is_rejected(self):
        plan_a = self.planner.plan("A", self.tasks, dt(10, 6), dt(10, 8))
        plan_b = self.planner.plan("B", self.tasks, dt(10, 6, 10), dt(10, 8))
        self.assertTrue(self.planner.commit(plan_a).accepted)
        decision = self.planner.commit(plan_b)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "plan-stale")
        self.assertNotIn("B", self.planner.plans)

    def test_replan_against_current_generation_holds_and_respects_equipment(self):
        plan_a = self.planner.plan("A", self.tasks, dt(10, 6), dt(10, 8))
        stale_b = self.planner.plan("B", self.tasks, dt(10, 6, 10), dt(10, 8))
        self.planner.commit(plan_a)
        self.assertFalse(self.planner.commit(stale_b).accepted)
        held = self.planner.ledger.occupied_at("G1", dt(10, 6, 10))
        self.assertIsNotNone(held)
        self.assertEqual(held.payload, "A:gpu-on")
        fresh_b = self.planner.plan("B", self.tasks, dt(10, 6, 10), dt(10, 8))
        self.assertEqual(fresh_b.runs["gpu-on"].planned.start, dt(10, 6, 20))
        self.assertTrue(self.planner.commit(fresh_b).accepted)


class MaintenanceReleaseWorkflowTests(unittest.TestCase):
    def test_denied_release_commits_no_plan_and_publishes_no_event(self):
        control = RampWorkflowControl()
        calls = []
        denied = Decision(False, "mel-expired", "MEL calendar relief expired")
        with self.assertRaises(WorkflowRejected):
            control.release_maintenance_turnaround(
                "op-1", "B-1234", dt(10, 9),
                lambda: calls.append("departure-locked"),
                requirements=(denied,),
            )
        self.assertEqual(calls, [])
        self.assertEqual(control.stream.head, 0)
        self.assertEqual(control.receipts, {})

    def test_accepted_release_locks_conclusion_and_departure_at_one_instant(self):
        control = RampWorkflowControl()
        calls = []
        receipt = control.release_maintenance_turnaround(
            "op-2", "B-1234", dt(10, 9),
            lambda: calls.append("departure-locked") or "both",
        )
        self.assertEqual(calls, ["departure-locked"])
        self.assertEqual(receipt.event.topic, "maintenance.turnaround.release.decided")
        self.assertEqual(receipt.event.occurred_at, dt(10, 9))
        replayed = control.release_maintenance_turnaround(
            "op-2", "B-1234", dt(10, 9), lambda: calls.append("again"),
        )
        self.assertIs(replayed, receipt)
        self.assertEqual(calls, ["departure-locked"])
        self.assertEqual(control.stream.head, 1)

    def test_failed_joint_action_leaves_no_event_or_receipt(self):
        control = RampWorkflowControl()

        def withdraw():
            raise RuntimeError("plan withdrawn while release fails")

        with self.assertRaises(RuntimeError):
            control.release_maintenance_turnaround("op-3", "B-7", dt(10, 9), withdraw)
        self.assertEqual(control.stream.head, 0)
        self.assertEqual(control.receipts, {})


if __name__ == "__main__":
    unittest.main()

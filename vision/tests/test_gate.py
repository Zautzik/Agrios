"""The gate passes on intact input and fails under both negative controls --
and fails on the criteria each control is supposed to break, not by accident."""

import pytest

from vision.eval.gate import GateReport, run_gate, suites_not_failing
from vision.eval.manifest import Manifest

# A cut-down manifest so the test suite stays fast; CI runs the full
# phase1.json through the CLI as well.
SMALL = Manifest.model_validate(
    {
        "sequences": [
            {"id": "s-surge", "source": {"kind": "scene", "seed": 102, "params": {"surge_s": 8.0}}},
            {"id": "s-calm", "source": {"kind": "scene", "seed": 101, "params": {"duration_s": 10}}},
            {"id": "k", "source": {"kind": "kinematics", "seed": 201, "params": {"duration_s": 21600}}},
            {"id": "g", "source": {"kind": "gait", "seed": 303, "params": {"asymmetry": 0.25}}},
        ],
        "criteria": {
            "scene": {"min_event_recall": 1.0, "max_false_alarms": 0, "min_mota": 0.4},
            "kinematics": {"min_event_recall": 0.95, "max_false_alarms": 0},
            "gait": {"max_gai_error": 2.0},
        },
    }
)


@pytest.fixture(scope="module")
def reports() -> dict[str, GateReport]:
    return {c: run_gate(SMALL, c) for c in ("none", "noise", "shuffle")}


def _failed(report: GateReport) -> set[tuple[str, str]]:
    return {(k, c.metric) for k, s in report.suites.items() for c in s.checks if not c.passed}


def test_gate_passes_on_intact_input(reports: dict[str, GateReport]) -> None:
    assert reports["none"].passed, _failed(reports["none"])
    assert reports["none"].suites["kinematics"].metrics["events"] > 0  # not vacuous


@pytest.mark.parametrize("control", ["noise", "shuffle"])
def test_negative_controls_fail_the_gate(reports: dict[str, GateReport], control: str) -> None:
    failed = _failed(reports[control])
    assert not reports[control].passed
    # Each suite must catch the corruption itself.
    assert ("scene", "event_recall") in failed
    assert ("kinematics", "false_alarms") in failed
    assert ("gait", "max_gai_error") in failed


def test_every_suite_catches_each_control(reports: dict[str, GateReport]) -> None:
    for control in ("noise", "shuffle"):
        assert suites_not_failing(reports[control]) == [], control
    assert suites_not_failing(reports["none"]) == ["scene", "kinematics", "gait"]  # control of the control


GAIT_ONLY = {"sequences": [{"id": "g", "source": {"kind": "gait", "seed": 1}}]}


def test_gate_with_nothing_to_check_fails() -> None:
    report = run_gate(Manifest.model_validate(GAIT_ONLY))  # a suite with no criteria
    assert not report.passed
    assert report.suites["gait"].checks[0].metric == "criteria_defined"


def test_criteria_for_a_suite_that_never_ran_fail() -> None:
    m = Manifest.model_validate(
        GAIT_ONLY | {"criteria": {"gait": {"max_gai_error": 2.0}, "kinematics": {"max_false_alarms": 0}}}
    )
    report = run_gate(m)
    assert not report.passed
    assert report.suites["kinematics"].sequences == 0
    assert report.suites["gait"].checks[0].passed  # control: the suite that did run still passes

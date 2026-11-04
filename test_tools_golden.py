"""What the model sees from the tools -- names, descriptions, argument
schemas, and outputs for fixed calls -- must match the recorded snapshot
byte for byte. See scripts/golden_tool_outputs.py for why."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent
SCRIPT = REPO / "scripts" / "golden_tool_outputs.py"
GOLDEN = REPO / "testdata" / "tool_outputs.json"


def _run(*flags):
    out = subprocess.run([sys.executable, str(SCRIPT), *flags], cwd=REPO, check=True,
                         capture_output=True, encoding="utf-8")
    return out.stdout


def test_tools_match_the_golden_snapshot():
    assert _run() == GOLDEN.read_text(encoding="utf-8")


def test_negative_control_altered_upstream_data_breaks_the_match():
    assert _run("--perturb") != GOLDEN.read_text(encoding="utf-8")

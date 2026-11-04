"""Vendored browser files match the hashes recorded in their README, and each
ships with its licence."""

import hashlib
import re
import shutil
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "static" / "vendor"


def problems(vendor: Path) -> list[str]:
    """Everything wrong with a vendor directory: scripts not in the README's
    table, recorded files that are missing, and hash mismatches."""
    table = (vendor / "README.md").read_text(encoding="utf-8")
    hashes = dict(re.findall(r"^\| `([^`]+)` \|.*\| `([0-9a-f]{64})` \|$", table, flags=re.MULTILINE))
    scripts = {p.name for p in vendor.glob("*.js")}
    found = [f"unrecorded: {name}" for name in sorted(scripts - set(hashes))]
    found += [f"missing: {name}" for name in sorted(set(hashes) - scripts)]
    found += [
        f"hash mismatch: {name}"
        for name, expected in hashes.items()
        if name in scripts and hashlib.sha256((vendor / name).read_bytes()).hexdigest() != expected
    ]
    if not hashes:
        found.append("no hashes recorded")
    return found


def test_every_vendored_script_is_recorded_and_unchanged():
    assert problems(VENDOR) == []


def test_the_check_catches_a_changed_or_unrecorded_file(tmp_path):
    # Negative control: the same check, run on a tampered copy, must fail.
    copy = tmp_path / "vendor"
    shutil.copytree(VENDOR, copy)
    script = copy / "htmx-2.0.11.min.js"
    data = bytearray(script.read_bytes())
    data[0] ^= 1
    script.write_bytes(bytes(data))
    (copy / "extra.js").write_text("// not recorded")
    assert problems(copy) == ["unrecorded: extra.js", "hash mismatch: htmx-2.0.11.min.js"]


def test_each_library_ships_its_licence():
    for script in VENDOR.glob("*.js"):
        library = script.name.split("-")[0]
        assert list(VENDOR.glob(f"{library}-LICENSE*")), library

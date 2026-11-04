"""No real footage in git: nothing media-like is tracked, and data/ is ignored."""

from pathlib import Path

from vision.privacy import is_git_ignored, media_paths, tracked_files

REPO = Path(__file__).resolve().parents[2]


def test_no_media_is_tracked() -> None:
    assert media_paths(tracked_files(REPO)) == []


def test_negative_control_media_paths_are_flagged() -> None:
    sample = ["data/cam1.mp4", "docs/shot.JPG", "vision/x.npz", "README.md", "vision/eval/gate.py"]
    assert media_paths(sample) == ["data/cam1.mp4", "docs/shot.JPG", "vision/x.npz"]


def test_data_dir_is_ignored() -> None:
    assert is_git_ignored(REPO, "data/cam1/2026-11-03.mp4")
    assert not is_git_ignored(REPO, "vision/eval/gate.py")  # control: check-ignore isn't vacuous

"""Privacy guards. Phase 1: real footage may never be tracked by git.

Every fixture this package uses is generated from a seed at test time, so no
image or video file has any reason to be in the repository at all -- the
check is a flat "no tracked media", with no allow-list to grow stale.
Phase 2 adds metadata/EXIF scrubbing on ingest here.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

_VIDEO = (".mp4", ".mkv", ".avi", ".mov", ".ts", ".h264", ".h265", ".hevc", ".webm", ".flv", ".dav")
_STILLS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".gif")
_RAW_FRAMES = (".npy", ".npz")
MEDIA_SUFFIXES = frozenset(_VIDEO + _STILLS + _RAW_FRAMES)


def media_paths(paths: Iterable[str]) -> list[str]:
    return [p for p in paths if PurePosixPath(p).suffix.lower() in MEDIA_SUFFIXES]


def tracked_files(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout
    return [p for p in out.split("\0") if p]


def is_git_ignored(repo: Path, path: str) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", path], cwd=repo).returncode == 0

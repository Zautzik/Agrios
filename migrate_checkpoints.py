"""Copy conversation history from checkpoints.db (SQLite) into the Postgres
database named by DATABASE_URL, so switching to Postgres keeps old threads.

    DATABASE_URL=postgresql+psycopg://user:pass@host/db uv run python migrate_checkpoints.py

Safe to re-run: checkpoints are keyed by id, so copying again overwrites
rather than duplicates.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from main import BASE_DIR, DATABASE_URL, open_checkpointer


def copy_checkpoints(src, dst) -> int:
    """Copy every checkpoint and its pending writes from src to dst; returns
    how many checkpoints were copied."""
    tuples = list(src.list(None))
    # list() is newest-first; write parents before children.
    for t in reversed(tuples):
        configurable = t.config["configurable"]
        parent = t.parent_config or {"configurable": {
            "thread_id": configurable["thread_id"],
            "checkpoint_ns": configurable.get("checkpoint_ns", ""),
        }}
        # Passing every channel version makes the saver store every channel's
        # value, not only the ones that changed in this step.
        dst.put(parent, t.checkpoint, t.metadata, t.checkpoint["channel_versions"])
        writes_by_task = defaultdict(list)
        for task_id, channel, value in t.pending_writes or []:
            writes_by_task[task_id].append((channel, value))
        for task_id, writes in writes_by_task.items():
            dst.put_writes(t.config, writes, task_id)
    return len(tuples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=BASE_DIR / "checkpoints.db",
                        help="SQLite checkpoint file to copy from (default: checkpoints.db)")
    args = parser.parse_args()
    if not DATABASE_URL.startswith("postgresql"):
        print("DATABASE_URL must point at Postgres; it's the destination.", file=sys.stderr)
        return 2
    if not args.source.exists():
        print(f"No checkpoint file at {args.source}.", file=sys.stderr)
        return 2
    with SqliteSaver.from_conn_string(str(args.source)) as src, open_checkpointer() as dst:
        count = copy_checkpoints(src, dst)
        threads = {t.config["configurable"]["thread_id"] for t in src.list(None)}
    print(f"Copied {count} checkpoints across {len(threads)} threads into Postgres.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

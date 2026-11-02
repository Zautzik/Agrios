import os
import tempfile
from pathlib import Path

# Point every test run at a throwaway database before main.py is imported, so
# tests never write to the CLI's agrios.db. An explicit DATABASE_URL (e.g. a
# local Postgres) still wins.
os.environ.setdefault(
    "DATABASE_URL",
    f"sqlite:///{Path(tempfile.mkdtemp(prefix='agrios-test-')) / 'agrios.db'}",
)

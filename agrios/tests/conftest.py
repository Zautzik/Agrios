import pytest

from agrios.db import init_db


@pytest.fixture(scope="session", autouse=True)
def _schema() -> None:
    """Create the schema in the throwaway test database (the root conftest
    points DATABASE_URL at it). Done here rather than relied on from main.py's
    import-time init_db(), so these tests pass on their own, in any order."""
    init_db()

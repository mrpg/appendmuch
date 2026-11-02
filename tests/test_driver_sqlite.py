import sqlite3
from contextlib import contextmanager

import pytest
from driver_contract import SQLDriverContract

from appendmuch import Sqlite3


class UnreliableSqlite3(Sqlite3):
    """Fails commits on demand, as a busy or full database does."""

    fail_commit = False

    @contextmanager
    def _cursor(self):
        with super()._cursor() as cur:
            yield cur

            if self.fail_commit:
                raise sqlite3.OperationalError("database is locked")


@pytest.fixture
def unreliable(tmp_path):
    driver = UnreliableSqlite3(str(tmp_path / "unreliable.db"))
    driver.ensure()
    driver.replace_predicate = lambda namespace, field: field == "replaced"
    yield driver
    driver.fail_commit = False
    driver.close()


@pytest.mark.parametrize("field", ["plain", "replaced"])
def test_failed_write_is_not_stored_and_keeps_earlier_writes(unreliable, field):
    unreliable.append("ns", "kept", b"1", "ctx", 1.0)

    unreliable.fail_commit = True
    with pytest.raises(sqlite3.OperationalError):
        unreliable.append("ns", field, b"2", "ctx", 2.0)

    unreliable.fail_commit = False
    assert [row[2] for row in unreliable.history_all()] == ["kept"]


class TestSqlite3Driver(SQLDriverContract):
    @pytest.fixture
    def driver_factory(self, tmp_path):
        def factory(name="default"):
            return Sqlite3(str(tmp_path / f"{name}.db"))

        return factory

    def corrupt_schema(self, driver):
        conn = driver.get_connection()
        conn.execute(f"CREATE TABLE {driver.table_name} (id INTEGER PRIMARY KEY, other TEXT)")
        conn.commit()

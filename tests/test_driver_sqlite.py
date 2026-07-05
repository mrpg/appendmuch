import pytest
from driver_contract import SQLDriverContract

from appendmuch import Sqlite3


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

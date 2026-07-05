import os
from contextlib import suppress

import pytest
from driver_contract import SQLDriverContract

from appendmuch.drivers import PostgreSQL

CONNINFO = os.environ.get("APPENDMUCH_PG_CONNINFO", "")

pytestmark = pytest.mark.skipif(not CONNINFO, reason="APPENDMUCH_PG_CONNINFO not set")


class TestPostgreSQLDriver(SQLDriverContract):
    @pytest.fixture
    def driver_factory(self):
        created = set()
        drivers = []

        def factory(name="default"):
            driver = PostgreSQL(CONNINFO, table_prefix=f"appendmuch_{name}")
            drivers.append(driver)
            if name not in created:
                # first use of this location in this test: start from scratch
                created.add(name)
                with driver.pool.connection() as conn:
                    conn.execute(f"DROP TABLE IF EXISTS {driver.table_name} CASCADE")
            return driver

        yield factory

        for driver in drivers:
            with suppress(Exception):
                driver.close()

    def corrupt_schema(self, driver):
        with driver.pool.connection() as conn:
            conn.execute(f"CREATE TABLE {driver.table_name} (id BIGSERIAL PRIMARY KEY, other TEXT)")

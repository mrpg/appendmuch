# Copyright Max R. P. Grossmann & Holger Gerhardt, 2026.
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Database driver implementations for the append-only log.

SQL queries use f-strings for table names (self.table_prefix + self.tblextra)
which are validated as Python identifiers at construction time. All user-provided
data uses parameterized queries (%s or ?) to prevent SQL injection.
"""

import sqlite3
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any, cast

import msgpack

from appendmuch.codec import Codec
from appendmuch.types import Value
from appendmuch.utils import ensure

# SQL fragments shared across drivers
_COLS = "(namespace, field, value, created_at, context)"
_INSERT_COLS = f"INSERT INTO {{tbl}} {_COLS}"


def _insert_pg(tbl: str) -> str:
    return f"{_INSERT_COLS.format(tbl=tbl)} VALUES (%s, %s, %s, %s, %s)"  # nosec B608


def _update_pg(tbl: str) -> str:
    return (
        f"UPDATE {tbl} SET value = %s, created_at = %s, context = %s"  # nosec B608
        f" WHERE namespace = %s AND field = %s"
    )


def _insert_sq(tbl: str) -> str:
    return f"{_INSERT_COLS.format(tbl=tbl)} VALUES (?, ?, ?, ?, ?)"  # nosec B608


def _update_sq(tbl: str) -> str:
    return f"UPDATE {tbl} SET value = ?, created_at = ?, context = ? WHERE namespace = ? AND field = ?"  # nosec B608


def _select_all(tbl: str) -> str:
    return f"SELECT namespace, field, value, created_at, context FROM {tbl}"  # nosec B608


def _select_all_ordered(tbl: str) -> str:
    return f"SELECT namespace, field, value, created_at, context FROM {tbl} ORDER BY created_at ASC"  # nosec B608


class DBDriver(ABC):
    def __init__(self) -> None:
        self.now: float = 0.0
        self.codec: Codec = Codec()
        self.replace_predicate: Callable[[str, str], bool] = lambda ns, f: False

    @abstractmethod
    def size(self) -> int | None: ...

    @abstractmethod
    def dump(self) -> Iterator[bytes]: ...

    @abstractmethod
    def restore(self, msgpack_stream: Iterator[bytes]) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def test_connection(self) -> None: ...

    @abstractmethod
    def test_tables(self) -> None: ...

    @abstractmethod
    def insert(self, namespace: str, field: str, data: Any, context: str) -> None: ...

    @abstractmethod
    def delete(self, namespace: str, field: str, context: str) -> None: ...

    @abstractmethod
    def history_all(self) -> Iterator[tuple[str, str, Value]]: ...

    @abstractmethod
    def close(self) -> None:
        pass

    def ensure(self) -> bool:
        try:
            self.test_connection()
        except Exception as exc:
            raise RuntimeError("Cannot connect to database.") from exc

        try:
            self.test_tables()
            return True
        except Exception:
            self.reset()
            return False


class Memory(DBDriver):
    """In-memory implementation for testing and write-through operations."""

    def __init__(self) -> None:
        super().__init__()
        self.log: dict[str, dict[str, list[tuple[float, bool, bytes | None, str]]]] = {}

    def close(self) -> None:
        pass

    def test_connection(self) -> None:
        pass

    def test_tables(self) -> None:
        raise RuntimeError("No tables in memory driver")

    def size(self) -> int | None:
        return None

    def dump(self) -> Iterator[bytes]:
        for namespace, fields in self.log.items():
            for field, values in fields.items():
                for v in values:
                    yield msgpack.packb(
                        {
                            "namespace": namespace,
                            "field": field,
                            "value": v[2],
                            "created_at": v[0],
                            "context": v[3],
                        }
                    )

    def reset(self) -> None:
        self.log.clear()

    def restore(self, msgpack_stream: Iterator[bytes]) -> None:
        unpacker = msgpack.Unpacker(msgpack_stream, raw=False)
        for row_dict in unpacker:
            namespace, field = row_dict["namespace"], row_dict["field"]
            raw_value = row_dict["value"]
            ts = row_dict["created_at"]
            ctx = row_dict["context"]

            if namespace not in self.log:
                self.log[namespace] = {}
            if field not in self.log[namespace]:
                self.log[namespace][field] = []

            self.log[namespace][field].append((ts, raw_value is None, raw_value, ctx))

            if self.replace_predicate(namespace, field):
                self.log[namespace][field] = self.log[namespace][field][-1:]

    def insert(self, namespace: str, field: str, data: Any, context: str) -> None:
        if namespace not in self.log:
            self.log[namespace] = {}
        if field not in self.log[namespace]:
            self.log[namespace][field] = []

        self.log[namespace][field].append((self.now, False, self.codec.encode(data), context))

        if self.replace_predicate(namespace, field):
            self.log[namespace][field] = self.log[namespace][field][-1:]

    def delete(self, namespace: str, field: str, context: str) -> None:
        if namespace not in self.log or field not in self.log[namespace]:
            raise AttributeError(f"Key not found: ({namespace}, {field})")

        self.log[namespace][field].append((self.now, True, None, context))

    def history_all(self) -> Iterator[tuple[str, str, Value]]:
        for namespace, fields in self.log.items():
            for field, values in fields.items():
                for value in values:
                    yield namespace, field, Value(
                        value[0], value[1], self.codec.decode(value[2]) if value[2] is not None else None, value[3]
                    )


class PostgreSQL(DBDriver):
    """PostgreSQL implementation with connection pooling and batched writes."""

    def __init__(
        self,
        conninfo: str = "",
        table_prefix: str = "appendmuch",
        tblextra: str = "",
        min_size: int = 5,
        max_size: int = 50,
        replace_index_specs: list[tuple[str, str]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        import psycopg_pool

        ensure(
            table_prefix.isidentifier(),
            ValueError,
            "table_prefix must be a valid Python identifier",
        )
        ensure(
            tblextra == "" or tblextra.isidentifier(),
            ValueError,
            "tblextra must be empty or valid Python identifier",
        )

        self.pool = psycopg_pool.ConnectionPool(
            conninfo,
            open=True,
            min_size=min_size,
            max_size=max_size,
            **kwargs,
        )
        self.table_prefix = table_prefix
        self.tblextra = tblextra
        self.replace_index_specs = replace_index_specs or []
        self.batch_inserts: list[tuple[Any, ...]] = []
        self.last_batch_time = time.time()
        self.batch_size = 100
        self.batch_timeout = 0.1

    @property
    def table_name(self) -> str:
        return f"{self.table_prefix}{self.tblextra}_values"

    def process_batch(self, conn: Any, cur: Any) -> None:
        """Flush pending batch inserts.

        Warning: On failure, the pending batch is cleared to prevent duplicate
        writes on retry. Callers that need delivery guarantees should implement
        their own write-ahead mechanism.
        """
        if not self.batch_inserts:
            return

        try:
            cur.executemany(_insert_pg(self.table_name), self.batch_inserts)
            self.batch_inserts.clear()
            self.last_batch_time = time.time()
        except Exception:
            self.batch_inserts.clear()
            raise

    def should_flush_batch(self) -> bool:
        return len(self.batch_inserts) >= self.batch_size or time.time() - self.last_batch_time >= self.batch_timeout

    def close(self) -> None:
        if self.batch_inserts and hasattr(self, "pool") and self.pool:
            with (
                self.pool.connection() as conn,
                conn.transaction(),
                conn.cursor() as cur,
            ):
                self.process_batch(conn, cur)

        if hasattr(self, "pool") and self.pool:
            self.pool.close(timeout=5.0)

    def test_connection(self) -> None:
        with (
            self.pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            cur.execute("SELECT 1")

    def test_tables(self) -> None:
        with (
            self.pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            cur.execute(
                "SELECT 1 WHERE EXISTS("
                "SELECT 1 FROM pg_tables "
                "WHERE schemaname = 'public' "
                f"AND tablename = '{self.table_name}'"  # nosec B608
                ")"
            )
            result = cur.fetchone()
            ensure(result and result[0] == 1, RuntimeError, "Table does not exist")

    def size(self) -> int | None:
        with (
            self.pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            cur.execute(
                "SELECT COALESCE("
                f"pg_total_relation_size('{self.table_name}'::regclass)"  # nosec B608
                ", 0)"
            )
            result = cur.fetchone()
            return cast("int", result[0]) if result else None

    def reset(self) -> None:
        with (
            self.pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            cur.execute(f"DROP TABLE IF EXISTS {self.table_name} CASCADE")
            cur.execute(f"""
                CREATE TABLE {self.table_name} (
                    id BIGSERIAL PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    field TEXT NOT NULL,
                    value BYTEA,
                    created_at DOUBLE PRECISION NOT NULL,
                    context TEXT NOT NULL
                )
                """)
            for field_val, ns_pattern in self.replace_index_specs:
                idx_name = f"{self.table_name}_{field_val}_idx"
                cur.execute(
                    f"CREATE UNIQUE INDEX {idx_name} "
                    f"ON {self.table_name} (namespace, field) "
                    f"WHERE field = '{field_val}' "  # nosec B608
                    f"AND namespace LIKE '{ns_pattern}'"
                )

    def dump(self) -> Iterator[bytes]:
        if self.batch_inserts:
            with (
                self.pool.connection() as conn,
                conn.transaction(),
                conn.cursor() as cur,
            ):
                self.process_batch(conn, cur)

        with (
            self.pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            cur.execute(_select_all(self.table_name))
            for namespace, field, value, created_at, ctx in cur:
                yield msgpack.packb(
                    {
                        "namespace": namespace,
                        "field": field,
                        "value": value,
                        "created_at": created_at,
                        "context": ctx,
                    }
                )

    def restore(self, msgpack_stream: Iterator[bytes]) -> None:
        if self.batch_inserts:
            with (
                self.pool.connection() as conn,
                conn.transaction(),
                conn.cursor() as cur,
            ):
                self.process_batch(conn, cur)

        with (
            self.pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            unpacker = msgpack.Unpacker(msgpack_stream, raw=False)

            batch = []
            upsert_batch = []

            for row_dict in unpacker:
                namespace = row_dict["namespace"]
                field = row_dict["field"]

                if self.replace_predicate(namespace, field):
                    upsert_batch.append(
                        (
                            row_dict["value"],
                            row_dict["created_at"],
                            row_dict["context"],
                            namespace,
                            field,
                            namespace,
                            field,
                            row_dict["value"],
                            row_dict["created_at"],
                            row_dict["context"],
                        )
                    )
                else:
                    batch.append(
                        (
                            namespace,
                            field,
                            row_dict["value"],
                            row_dict["created_at"],
                            row_dict["context"],
                        )
                    )

            if upsert_batch:
                for values in upsert_batch:
                    cur.execute(_update_pg(self.table_name), values[:5])
                    if cur.rowcount == 0:
                        cur.execute(_insert_pg(self.table_name), values[5:])

            if batch:
                cur.executemany(_insert_pg(self.table_name), batch)

    def insert(self, namespace: str, field: str, data: Any, context: str) -> None:
        encoded = self.codec.encode(data)

        if self.replace_predicate(namespace, field):
            if self.batch_inserts:
                with (
                    self.pool.connection() as conn,
                    conn.transaction(),
                    conn.cursor() as cur,
                ):
                    self.process_batch(conn, cur)

            with (
                self.pool.connection() as conn,
                conn.transaction(),
                conn.cursor() as cur,
            ):
                cur.execute(
                    _update_pg(self.table_name),
                    (encoded, self.now, context, namespace, field),
                )
                if cur.rowcount == 0:
                    cur.execute(
                        _insert_pg(self.table_name),
                        (namespace, field, encoded, self.now, context),
                    )
        else:
            self.batch_inserts.append((namespace, field, encoded, self.now, context))

            if self.should_flush_batch():
                with (
                    self.pool.connection() as conn,
                    conn.transaction(),
                    conn.cursor() as cur,
                ):
                    self.process_batch(conn, cur)

    def delete(self, namespace: str, field: str, context: str) -> None:
        self.batch_inserts.append((namespace, field, None, self.now, context))

        if self.should_flush_batch():
            with (
                self.pool.connection() as conn,
                conn.transaction(),
                conn.cursor() as cur,
            ):
                self.process_batch(conn, cur)

    def history_all(self) -> Iterator[tuple[str, str, Value]]:
        if self.batch_inserts:
            with (
                self.pool.connection() as conn,
                conn.transaction(),
                conn.cursor() as cur,
            ):
                self.process_batch(conn, cur)

        with (
            self.pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            cur.execute(_select_all_ordered(self.table_name))
            for namespace, field, value, created_at, ctx in cur:
                yield (
                    namespace,
                    field,
                    Value(
                        created_at,
                        value is None,
                        self.codec.decode(value) if value is not None else None,
                        ctx,
                    ),
                )


class Sqlite3(DBDriver):
    """SQLite3 implementation with WAL mode and batched writes.

    When using ``check_same_thread=False``, the caller is responsible for
    serializing concurrent access to the driver.
    """

    def __init__(
        self,
        database: str,
        table_prefix: str = "appendmuch",
        tblextra: str = "",
        replace_index_specs: list[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__()
        ensure(
            table_prefix.isidentifier(),
            ValueError,
            "table_prefix must be a valid Python identifier",
        )
        ensure(
            tblextra == "" or tblextra.isidentifier(),
            ValueError,
            "tblextra must be empty or valid Python identifier",
        )
        self.database = database
        self.table_prefix = table_prefix
        self.tblextra = tblextra
        self.replace_index_specs = replace_index_specs or []
        self.connection: sqlite3.Connection | None = None
        self.batch_inserts: list[tuple[Any, ...]] = []
        self.last_batch_time = time.time()
        self.batch_size = 100
        self.batch_timeout = 0.1

    @property
    def table_name(self) -> str:
        return f"{self.table_prefix}{self.tblextra}_values"

    def get_connection(self) -> sqlite3.Connection:
        if self.connection is None:
            self.connection = sqlite3.connect(self.database, check_same_thread=False)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
            self.connection.execute("PRAGMA cache_size=-64000")
            self.connection.execute("PRAGMA temp_store=MEMORY")
            self.connection.execute("PRAGMA mmap_size=268435456")
        return self.connection

    def process_batch(self, conn: sqlite3.Connection) -> None:
        """Flush pending batch inserts.

        Warning: On failure, the pending batch is cleared to prevent duplicate
        writes on retry. Callers that need delivery guarantees should implement
        their own write-ahead mechanism.
        """
        if not self.batch_inserts:
            return

        try:
            conn.executemany(_insert_sq(self.table_name), self.batch_inserts)
            conn.commit()
            self.batch_inserts.clear()
            self.last_batch_time = time.time()
        except Exception:
            self.batch_inserts.clear()
            raise

    def should_flush_batch(self) -> bool:
        return len(self.batch_inserts) >= self.batch_size or time.time() - self.last_batch_time >= self.batch_timeout

    def close(self) -> None:
        if self.connection:
            if self.batch_inserts:
                self.process_batch(self.connection)
            self.connection.close()
            self.connection = None

    def test_connection(self) -> None:
        conn = self.get_connection()
        conn.execute("SELECT 1")

    def test_tables(self) -> None:
        conn = self.get_connection()
        cursor = conn.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.table_name}'"  # nosec B608
        )
        ensure(cursor.fetchone() is not None, RuntimeError, "Table does not exist")

    def size(self) -> int | None:
        conn = self.get_connection()
        cursor = conn.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
        row = cursor.fetchone()
        return cast("int", row[0]) if row else None

    def reset(self) -> None:
        if self.batch_inserts:
            self.process_batch(self.get_connection())

        conn = self.get_connection()
        conn.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        conn.execute(f"""
            CREATE TABLE {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT NOT NULL,
                field TEXT NOT NULL,
                value BLOB,
                created_at REAL NOT NULL,
                context TEXT NOT NULL
            )
            """)
        for field_val, ns_pattern in self.replace_index_specs:
            idx_name = f"{self.table_name}_{field_val}_idx"
            conn.execute(
                f"CREATE UNIQUE INDEX {idx_name} "
                f"ON {self.table_name} (namespace, field) "
                f"WHERE field = '{field_val}' "  # nosec B608
                f"AND namespace LIKE '{ns_pattern}'"
            )
        conn.commit()

    def dump(self) -> Iterator[bytes]:
        if self.batch_inserts:
            self.process_batch(self.get_connection())

        conn = self.get_connection()
        cursor = conn.execute(_select_all(self.table_name))
        for namespace, field, value, created_at, ctx in cursor:
            yield msgpack.packb(
                {
                    "namespace": namespace,
                    "field": field,
                    "value": value,
                    "created_at": created_at,
                    "context": ctx,
                }
            )

    def restore(self, msgpack_stream: Iterator[bytes]) -> None:
        if self.batch_inserts:
            self.process_batch(self.get_connection())

        conn = self.get_connection()
        unpacker = msgpack.Unpacker(msgpack_stream, raw=False)

        batch = []
        upsert_batch = []

        for row_dict in unpacker:
            namespace = row_dict["namespace"]
            field = row_dict["field"]

            if self.replace_predicate(namespace, field):
                upsert_batch.append(
                    (
                        row_dict["value"],
                        row_dict["created_at"],
                        row_dict["context"],
                        namespace,
                        field,
                        namespace,
                        field,
                        row_dict["value"],
                        row_dict["created_at"],
                        row_dict["context"],
                    )
                )
            else:
                batch.append(
                    (
                        namespace,
                        field,
                        row_dict["value"],
                        row_dict["created_at"],
                        row_dict["context"],
                    )
                )

        if upsert_batch:
            for values in upsert_batch:
                cursor = conn.execute(_update_sq(self.table_name), values[:5])
                if cursor.rowcount == 0:
                    conn.execute(_insert_sq(self.table_name), values[5:])

        if batch:
            conn.executemany(_insert_sq(self.table_name), batch)

        conn.commit()

    def insert(self, namespace: str, field: str, data: Any, context: str) -> None:
        encoded = self.codec.encode(data)

        if self.replace_predicate(namespace, field):
            if self.batch_inserts:
                self.process_batch(self.get_connection())

            conn = self.get_connection()
            cursor = conn.execute(
                _update_sq(self.table_name),
                (encoded, self.now, context, namespace, field),
            )
            if cursor.rowcount == 0:
                conn.execute(
                    _insert_sq(self.table_name),
                    (namespace, field, encoded, self.now, context),
                )
            conn.commit()
        else:
            self.batch_inserts.append((namespace, field, encoded, self.now, context))

            if self.should_flush_batch():
                self.process_batch(self.get_connection())

    def delete(self, namespace: str, field: str, context: str) -> None:
        self.batch_inserts.append((namespace, field, None, self.now, context))

        if self.should_flush_batch():
            self.process_batch(self.get_connection())

    def history_all(self) -> Iterator[tuple[str, str, Value]]:
        if self.batch_inserts:
            self.process_batch(self.get_connection())

        conn = self.get_connection()
        cursor = conn.execute(_select_all_ordered(self.table_name))

        for namespace, field, value, created_at, ctx in cursor:
            yield (
                namespace,
                field,
                Value(
                    created_at,
                    value is None,
                    self.codec.decode(value) if value is not None else None,
                    ctx,
                ),
            )

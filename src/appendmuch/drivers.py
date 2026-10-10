# Copyright Max R. P. Grossmann & Holger Gerhardt, 2026.
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Database driver implementations for the append-only log.

SQL queries use f-strings for table names (self.table_prefix + self.tblextra)
which are validated as Python identifiers at construction time. All user-provided
data uses parameterized queries (%s or ?) to prevent SQL injection.
"""

import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from operator import itemgetter
from typing import Any, Protocol, cast

import msgpack

from appendmuch.utils import SAFE_LIKE_PATTERN, ensure

# (seq, namespace, field, value, created_at, context); value None marks a tombstone
Row = tuple[int, str, str, bytes | None, float, str]


class ReadableBytes(Protocol):
    def read(self, size: int = -1) -> bytes: ...


def _iter_msgpack_dicts(msgpack_stream: Iterable[bytes] | ReadableBytes) -> Iterator[dict[str, Any]]:
    if hasattr(msgpack_stream, "read"):
        yield from msgpack.Unpacker(cast("ReadableBytes", msgpack_stream), raw=False)
        return

    unpacker = msgpack.Unpacker(raw=False)
    for chunk in msgpack_stream:
        unpacker.feed(chunk)
        yield from unpacker


def _iter_msgpack_rows(msgpack_stream: Iterable[bytes] | ReadableBytes) -> Iterator[Row]:
    """Accept either a file-like object or an iterable of msgpack chunks."""
    for row in _iter_msgpack_dicts(msgpack_stream):
        yield row["seq"], row["namespace"], row["field"], row["value"], row["created_at"], row["context"]


def _pack_row(row: Row) -> bytes:
    seq, namespace, field, value, created_at, context = row

    return cast(
        "bytes",
        msgpack.packb(
            {
                "seq": seq,
                "namespace": namespace,
                "field": field,
                "value": value,
                "created_at": created_at,
                "context": context,
            }
        ),
    )


class DBDriver(ABC):
    """Stores encoded rows; encoding and decoding values is up to the caller."""

    def __init__(self) -> None:
        # Serializes writes with flushes and dump reads, so that a dump may be
        # consumed on another thread than the one writing (e.g., streamed by a
        # web server). It is never held while a dump yields.
        self.lock = threading.RLock()
        self.seq: int = 0
        # Fields for which the driver MAY keep only the latest row instead of
        # appending. This is a permission, not a requirement: a driver that
        # always appends is correct, too.
        self.replace_predicate: Callable[[str, str], bool] = lambda ns, f: False

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq

    @abstractmethod
    def size(self) -> int | None: ...

    @abstractmethod
    def dump(self) -> Iterator[bytes]: ...

    @abstractmethod
    def restore(self, msgpack_stream: Iterable[bytes] | ReadableBytes) -> None: ...

    @abstractmethod
    def reset(self) -> None:
        """Destroy all stored data and recreate empty structures."""

    @abstractmethod
    def test_connection(self) -> None: ...

    @abstractmethod
    def table_exists(self) -> bool: ...

    @abstractmethod
    def test_tables(self) -> None:
        """Raise if the existing tables are unusable (e.g. incompatible schema)."""

    @abstractmethod
    def append(self, namespace: str, field: str, value: bytes | None, context: str, now: float) -> int:
        """Append an encoded value (None for a tombstone) and return its seq.

        If replace_predicate(namespace, field) holds, the driver may instead
        replace the field's previous row.
        """

    @abstractmethod
    def history_all(self) -> Iterator[Row]:
        """Yield all rows in seq order."""

    @abstractmethod
    def close(self) -> None: ...

    def flush(self) -> None:
        """Write any buffered rows to durable storage. No-op by default."""
        return None

    def ensure(self) -> bool:
        """Verify connectivity and make sure the schema exists.

        Returns True if the schema already existed, False if it was created.
        Raises RuntimeError if the database is unreachable, or if tables exist
        but are incompatible (never silently destroys existing data; call
        reset() explicitly to start over).
        """
        try:
            self.test_connection()
        except Exception as exc:
            raise RuntimeError("Cannot connect to database.") from exc

        if self.table_exists():
            self.test_tables()
            return True

        self.reset()
        return False


class Memory(DBDriver):
    """In-memory implementation for testing and write-through operations."""

    def __init__(self) -> None:
        super().__init__()
        self.log: dict[tuple[str, str], list[Row]] = {}

    def close(self) -> None:
        pass

    def test_connection(self) -> None:
        pass

    def table_exists(self) -> bool:
        return True

    def test_tables(self) -> None:
        pass

    def size(self) -> int | None:
        return None

    def history_all(self) -> Iterator[Row]:
        with self.lock:
            rows = [row for rows in self.log.values() for row in rows]

        rows.sort(key=itemgetter(0))
        yield from rows

    def dump(self) -> Iterator[bytes]:
        for row in self.history_all():
            yield _pack_row(row)

    def reset(self) -> None:
        with self.lock:
            self.log.clear()

    def _add(self, row: Row) -> None:
        _seq, namespace, field, *_ = row
        rows = self.log.setdefault((namespace, field), [])

        if self.replace_predicate(namespace, field):
            rows.clear()

        rows.append(row)

    def restore(self, msgpack_stream: Iterable[bytes] | ReadableBytes) -> None:
        for row in _iter_msgpack_rows(msgpack_stream):
            with self.lock:
                self.seq = max(self.seq, row[0])
                self._add(row)

    def append(self, namespace: str, field: str, value: bytes | None, context: str, now: float) -> int:
        with self.lock:
            seq = self.next_seq()
            self._add((seq, namespace, field, value, now, context))

            return seq


class SQLDriver(DBDriver):
    """Shared implementation for SQL-backed drivers: batched writes,
    replace-semantics upserts, dump/restore, and history iteration.

    Subclasses provide connection management (``_cursor``) and
    dialect-specific DDL/introspection.
    """

    _param: str  # placeholder style: "?" (sqlite) or "%s" (postgres)

    def __init__(
        self,
        table_prefix: str,
        tblextra: str,
        replace_index_specs: list[tuple[str, str]] | None,
    ) -> None:
        super().__init__()
        ensure(
            table_prefix.isascii() and table_prefix.isidentifier(),
            ValueError,
            "table_prefix must be a valid ASCII identifier",
        )
        ensure(
            tblextra == "" or (tblextra.isascii() and tblextra.isidentifier()),
            ValueError,
            "tblextra must be empty or a valid ASCII identifier",
        )
        for fv, ns in replace_index_specs or []:
            ensure(
                fv.isascii() and fv.isidentifier(),
                ValueError,
                "replace_index_specs field must be a valid ASCII identifier",
            )
            ensure(
                bool(SAFE_LIKE_PATTERN.match(ns)),
                ValueError,
                "replace_index_specs namespace pattern contains disallowed characters",
            )

        self.table_prefix = table_prefix
        self.tblextra = tblextra
        self.replace_index_specs = replace_index_specs or []
        self.batch_inserts: list[Row] = []
        self.last_batch_time = time.time()
        self.batch_size = 100
        self.batch_timeout = 0.1
        self.dump_batch_size = 1000

    @property
    def table_name(self) -> str:
        return f"{self.table_prefix}{self.tblextra}_values"

    # -- connection management (subclass responsibility) --------------------

    @abstractmethod
    def _cursor(self) -> AbstractContextManager[Any]:
        """Context manager yielding a DB-API cursor inside a transaction."""

    # -- SQL builders --------------------------------------------------------

    def _insert_sql(self) -> str:
        p = self._param
        return (
            f"INSERT INTO {self.table_name} (seq, namespace, field, value, created_at, context) "  # nosec B608
            f"VALUES ({p}, {p}, {p}, {p}, {p}, {p})"
        )

    def _update_sql(self) -> str:
        p = self._param
        return (
            f"UPDATE {self.table_name} SET seq = {p}, value = {p}, created_at = {p}, context = {p} "  # nosec B608
            f"WHERE namespace = {p} AND field = {p}"
        )

    def _select_all_sql(self) -> str:
        return (
            f"SELECT seq, namespace, field, value, created_at, context "  # nosec B608
            f"FROM {self.table_name} ORDER BY seq ASC"
        )

    def _select_after_sql(self) -> str:
        p = self._param
        return (
            f"SELECT seq, namespace, field, value, created_at, context "  # nosec B608
            f"FROM {self.table_name} WHERE seq > {p} ORDER BY seq ASC LIMIT {p}"
        )

    def _index_sql(self, field_val: str, ns_pattern: str) -> str:
        # field_val and ns_pattern are validated at construction time
        return (
            f"CREATE UNIQUE INDEX {self.table_name}_{field_val}_idx "
            f"ON {self.table_name} (namespace, field) "
            f"WHERE field = '{field_val}' "  # nosec B608
            f"AND namespace LIKE '{ns_pattern}'"
        )

    @abstractmethod
    def _create_table_sql(self) -> str: ...

    def _drop_table_sql(self) -> str:
        return f"DROP TABLE IF EXISTS {self.table_name}"

    # -- batching ------------------------------------------------------------

    def flush(self) -> None:
        """Write any batched rows to the database immediately.

        On failure, the pending batch is left intact so callers may retry or
        inspect it.
        """
        with self.lock:
            if self.batch_inserts:
                with self._cursor() as cur:
                    self._process_batch(cur)

    def _process_batch(self, cur: Any) -> None:
        if not self.batch_inserts:
            return

        cur.executemany(self._insert_sql(), self.batch_inserts)
        self.batch_inserts.clear()
        self.last_batch_time = time.time()

    def _should_flush_batch(self) -> bool:
        return len(self.batch_inserts) >= self.batch_size or time.time() - self.last_batch_time >= self.batch_timeout

    # -- writes ---------------------------------------------------------------

    def append(self, namespace: str, field: str, value: bytes | None, context: str, now: float) -> int:
        with self.lock:
            seq = self.next_seq()
            row = (seq, namespace, field, value, now, context)

            if self.replace_predicate(namespace, field):
                with self._cursor() as cur:
                    self._process_batch(cur)
                    self._upsert(cur, row)
            else:
                self.batch_inserts.append(row)

                if self._should_flush_batch():
                    with self._cursor() as cur:
                        self._process_batch(cur)

            return seq

    def _upsert(self, cur: Any, row: Row) -> None:
        seq, namespace, field, value, created_at, context = row
        cur.execute(self._update_sql(), (seq, value, created_at, context, namespace, field))

        if cur.rowcount == 0:
            cur.execute(self._insert_sql(), row)

    # -- bulk I/O --------------------------------------------------------------

    def dump(self) -> Iterator[bytes]:
        # Read in batches, each fetched completely under the lock, so that no
        # cursor, transaction or lock is held while the consumer has control.
        # Rows written meanwhile get higher seqs and are included. A replaced
        # row moves to a higher seq and may then appear twice; restore() keeps
        # the later one.
        last = -1

        while True:
            with self.lock:
                self.flush()

                with self._cursor() as cur:
                    cur.execute(self._select_after_sql(), (last, self.dump_batch_size))
                    rows = cur.fetchall()

            if not rows:
                return

            for row in rows:
                yield _pack_row(row)

            last = rows[-1][0]

    def restore(self, msgpack_stream: Iterable[bytes] | ReadableBytes) -> None:
        self.flush()

        batch: list[Row] = []
        upserts: list[Row] = []

        for row in _iter_msgpack_rows(msgpack_stream):
            self.seq = max(self.seq, row[0])
            (upserts if self.replace_predicate(row[1], row[2]) else batch).append(row)

        with self._cursor() as cur:
            for row in upserts:
                self._upsert(cur, row)

            if batch:
                cur.executemany(self._insert_sql(), batch)

    def history_all(self) -> Iterator[Row]:
        self.flush()

        with self._cursor() as cur:
            cur.execute(self._select_all_sql())
            for row in cur:
                self.seq = max(self.seq, row[0])
                yield row

    def reset(self) -> None:
        with self.lock:
            # pending rows would be destroyed together with the table
            self.batch_inserts.clear()

            with self._cursor() as cur:
                cur.execute(self._drop_table_sql())
                cur.execute(self._create_table_sql())
                for field_val, ns_pattern in self.replace_index_specs:
                    cur.execute(self._index_sql(field_val, ns_pattern))


class PostgreSQL(SQLDriver):
    """PostgreSQL implementation with connection pooling and batched writes."""

    _param = "%s"

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
        super().__init__(table_prefix, tblextra, replace_index_specs)
        import psycopg_pool

        self.pool = psycopg_pool.ConnectionPool(
            conninfo,
            open=True,
            min_size=min_size,
            max_size=max_size,
            **kwargs,
        )

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        with (
            self.pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            yield cur

    def close(self) -> None:
        if getattr(self, "pool", None):
            self.flush()
            self.pool.close(timeout=5.0)

    def test_connection(self) -> None:
        with self._cursor() as cur:
            cur.execute("SELECT 1")

    def table_exists(self) -> bool:
        with self._cursor() as cur:
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = %s)",
                (self.table_name,),
            )
            result = cur.fetchone()
            return bool(result and result[0])

    def test_tables(self) -> None:
        with self._cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = %s AND column_name = 'seq'",
                (self.table_name,),
            )
            ensure(
                cur.fetchone() is not None,
                RuntimeError,
                f"Table {self.table_name} exists but is missing the seq column; "
                "call reset() to drop and recreate it (this destroys existing data)",
            )

    def size(self) -> int | None:
        with self._cursor() as cur:
            cur.execute("SELECT pg_total_relation_size(%s::regclass)", (self.table_name,))
            result = cur.fetchone()
            return cast("int", result[0]) if result else None

    def _create_table_sql(self) -> str:
        return f"""
            CREATE TABLE {self.table_name} (
                id BIGSERIAL PRIMARY KEY,
                seq BIGINT NOT NULL UNIQUE,
                namespace TEXT NOT NULL,
                field TEXT NOT NULL,
                value BYTEA,
                created_at DOUBLE PRECISION NOT NULL,
                context TEXT NOT NULL
            )
            """

    def _drop_table_sql(self) -> str:
        return f"DROP TABLE IF EXISTS {self.table_name} CASCADE"


class Sqlite3(SQLDriver):
    """SQLite3 implementation with WAL mode and batched writes."""

    _param = "?"

    def __init__(
        self,
        database: str,
        table_prefix: str = "appendmuch",
        tblextra: str = "",
        replace_index_specs: list[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(table_prefix, tblextra, replace_index_specs)
        self.database = database
        self.connection: sqlite3.Connection | None = None

    def get_connection(self) -> sqlite3.Connection:
        if self.connection is None:
            self.connection = sqlite3.connect(self.database, check_same_thread=False)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
            self.connection.execute("PRAGMA cache_size=-64000")
            self.connection.execute("PRAGMA temp_store=MEMORY")
            self.connection.execute("PRAGMA mmap_size=268435456")
            self.connection.execute("PRAGMA busy_timeout=5000")
        return self.connection

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            cur.close()

    def close(self) -> None:
        if self.connection:
            self.flush()
            self.connection.close()
            self.connection = None

    def test_connection(self) -> None:
        self.get_connection().execute("SELECT 1")

    def table_exists(self) -> bool:
        cursor = self.get_connection().execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (self.table_name,),
        )
        return cursor.fetchone() is not None

    def test_tables(self) -> None:
        conn = self.get_connection()
        ensure(
            any(row[1] == "seq" for row in conn.execute(f"PRAGMA table_info({self.table_name})")),  # nosec B608
            RuntimeError,
            f"Table {self.table_name} exists but is missing the seq column; "
            "call reset() to drop and recreate it (this destroys existing data)",
        )

    def size(self) -> int | None:
        cursor = self.get_connection().execute(
            "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()"
        )
        row = cursor.fetchone()
        return cast("int", row[0]) if row else None

    def _create_table_sql(self) -> str:
        return f"""
            CREATE TABLE {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seq INTEGER NOT NULL UNIQUE,
                namespace TEXT NOT NULL,
                field TEXT NOT NULL,
                value BLOB,
                created_at REAL NOT NULL,
                context TEXT NOT NULL
            )
            """

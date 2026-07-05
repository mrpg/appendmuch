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
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol, cast

import msgpack

from appendmuch.codec import Codec
from appendmuch.types import Value
from appendmuch.utils import SAFE_LIKE_PATTERN, ensure


class ReadableBytes(Protocol):
    def read(self, size: int = -1) -> bytes: ...


def _iter_msgpack_rows(
    msgpack_stream: Iterable[bytes] | ReadableBytes,
) -> Iterator[dict[str, Any]]:
    """Accept either a file-like object or an iterable of msgpack chunks."""
    if hasattr(msgpack_stream, "read"):
        yield from msgpack.Unpacker(cast("ReadableBytes", msgpack_stream), raw=False)
        return

    unpacker = msgpack.Unpacker(raw=False)
    for chunk in msgpack_stream:
        unpacker.feed(chunk)
        yield from unpacker


def _pack_row(
    seq: int,
    namespace: str,
    field: str,
    value: bytes | None,
    created_at: float,
    context: str,
) -> bytes:
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
    def __init__(self) -> None:
        self.seq: int = 0
        self.codec: Codec = Codec()
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
    def insert(self, namespace: str, field: str, data: Any, context: str, now: float) -> int: ...

    @abstractmethod
    def delete(self, namespace: str, field: str, context: str, now: float) -> int: ...

    @abstractmethod
    def history_all(self) -> Iterator[tuple[str, str, Value]]: ...

    @abstractmethod
    def close(self) -> None: ...

    def flush(
        self,
    ) -> None:
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
        self.log: dict[str, dict[str, list[tuple[int, float, bool, bytes | None, str]]]] = {}

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

    def dump(self) -> Iterator[bytes]:
        rows: list[tuple[int, str, str, bytes | None, float, str]] = []
        for namespace, fields in self.log.items():
            for field, values in fields.items():
                for seq, created_at, _unavailable, value, context in values:
                    rows.append((seq, namespace, field, value, created_at, context))

        rows.sort(key=lambda row: row[0])

        for seq, namespace, field, value, created_at, context in rows:
            yield _pack_row(seq, namespace, field, value, created_at, context)

    def reset(self) -> None:
        self.log.clear()

    def restore(self, msgpack_stream: Iterable[bytes] | ReadableBytes) -> None:
        for row_dict in _iter_msgpack_rows(msgpack_stream):
            namespace, field = row_dict["namespace"], row_dict["field"]
            raw_value = row_dict["value"]
            seq = cast("int", row_dict["seq"])
            self.seq = max(self.seq, seq)

            values = self.log.setdefault(namespace, {}).setdefault(field, [])
            values.append(
                (
                    seq,
                    row_dict["created_at"],
                    raw_value is None,
                    raw_value,
                    row_dict["context"],
                )
            )

            if self.replace_predicate(namespace, field):
                self.log[namespace][field] = values[-1:]

    def _append(self, namespace: str, field: str, value: bytes | None, context: str, now: float) -> int:
        seq = self.next_seq()
        values = self.log.setdefault(namespace, {}).setdefault(field, [])
        values.append((seq, now, value is None, value, context))

        if self.replace_predicate(namespace, field):
            self.log[namespace][field] = values[-1:]

        return seq

    def insert(self, namespace: str, field: str, data: Any, context: str, now: float) -> int:
        return self._append(namespace, field, self.codec.encode(data), context, now)

    def delete(self, namespace: str, field: str, context: str, now: float) -> int:
        if namespace not in self.log or field not in self.log[namespace]:
            raise AttributeError(f"Key not found: ({namespace}, {field})")

        return self._append(namespace, field, None, context, now)

    def history_all(self) -> Iterator[tuple[str, str, Value]]:
        for namespace, fields in self.log.items():
            for field, values in fields.items():
                for seq, created_at, unavailable, value, context in values:
                    yield (
                        namespace,
                        field,
                        Value(
                            created_at,
                            unavailable,
                            self.codec.decode(value) if value is not None else None,
                            context,
                            seq=seq,
                        ),
                    )


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
        self.batch_inserts: list[tuple[Any, ...]] = []
        self.last_batch_time = time.time()
        self.batch_size = 100
        self.batch_timeout = 0.1

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

    def _write(
        self,
        namespace: str,
        field: str,
        encoded: bytes | None,
        context: str,
        now: float,
    ) -> int:
        seq = self.next_seq()

        if self.replace_predicate(namespace, field):
            with self._cursor() as cur:
                self._process_batch(cur)
                cur.execute(self._update_sql(), (seq, encoded, now, context, namespace, field))
                if cur.rowcount == 0:
                    cur.execute(
                        self._insert_sql(),
                        (seq, namespace, field, encoded, now, context),
                    )
        else:
            self.batch_inserts.append((seq, namespace, field, encoded, now, context))

            if self._should_flush_batch():
                with self._cursor() as cur:
                    self._process_batch(cur)

        return seq

    def insert(self, namespace: str, field: str, data: Any, context: str, now: float) -> int:
        return self._write(namespace, field, self.codec.encode(data), context, now)

    def delete(self, namespace: str, field: str, context: str, now: float) -> int:
        return self._write(namespace, field, None, context, now)

    # -- bulk I/O --------------------------------------------------------------

    def dump(self) -> Iterator[bytes]:
        self.flush()

        with self._cursor() as cur:
            cur.execute(self._select_all_sql())
            for seq, namespace, field, value, created_at, ctx in cur:
                yield _pack_row(seq, namespace, field, value, created_at, ctx)

    def restore(self, msgpack_stream: Iterable[bytes] | ReadableBytes) -> None:
        self.flush()

        batch = []
        upserts = []

        for row_dict in _iter_msgpack_rows(msgpack_stream):
            namespace = row_dict["namespace"]
            field = row_dict["field"]
            seq = cast("int", row_dict["seq"])
            self.seq = max(self.seq, seq)
            row = (
                seq,
                namespace,
                field,
                row_dict["value"],
                row_dict["created_at"],
                row_dict["context"],
            )

            if self.replace_predicate(namespace, field):
                upserts.append(row)
            else:
                batch.append(row)

        with self._cursor() as cur:
            for seq, namespace, field, value, created_at, ctx in upserts:
                cur.execute(self._update_sql(), (seq, value, created_at, ctx, namespace, field))
                if cur.rowcount == 0:
                    cur.execute(
                        self._insert_sql(),
                        (seq, namespace, field, value, created_at, ctx),
                    )

            if batch:
                cur.executemany(self._insert_sql(), batch)

    def history_all(self) -> Iterator[tuple[str, str, Value]]:
        self.flush()

        with self._cursor() as cur:
            cur.execute(self._select_all_sql())
            for seq, namespace, field, value, created_at, ctx in cur:
                self.seq = max(self.seq, seq)
                yield (
                    namespace,
                    field,
                    Value(
                        created_at,
                        value is None,
                        self.codec.decode(value) if value is not None else None,
                        ctx,
                        seq=seq,
                    ),
                )

    def reset(self) -> None:
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
    """SQLite3 implementation with WAL mode and batched writes.

    When using ``check_same_thread=False``, the caller is responsible for
    serializing concurrent access to the driver.
    """

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

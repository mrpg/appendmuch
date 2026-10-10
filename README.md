# appendmuch

An extensible append-only log with in-memory cache and pluggable storage backends. Ordering is based on a strict monotonic sequence counter, but timestamps are stored for informational purposes.

Appendmuch is intended for synchronous code or async applications that access a store from one event-loop thread. It is not safe for parallel processing: do not share a `Store` or driver across threads, processes, or concurrent workers without external serialization.

## Installation

From PyPI:

```
pip install appendmuch
```

From GitHub:

```
pip install "appendmuch @ git+https://github.com/mrpg/appendmuch.git@master"
```

## Quick start

```python
from appendmuch import Memory, Sqlite3, Store

store = Store(Memory())  # or Store(Sqlite3("db.sqlite3"))

# Storage instances provide attribute-based access to namespaced data
player = store.storage("game", "player1")
player.score = 100
player.nickname = "Alice"

print(player.score)     # 100
print(player.nickname)  # Alice
```

Namespaces and fields form a tree: a name cannot be both a field and a sub-namespace, and writes that would create such a collision raise `ValueError`. The names `along`, `fields`, `flush`, `get`, `history`, `name`, `refresh`, `strict`, `virtual`, and `within` are reserved by `Storage` and cannot be used as field names (`name` holds the last component of the namespace). Deleting a field that does not exist (or was already deleted) raises `AttributeError`, mirroring regular Python attribute semantics.

## Mutable values

Mutable values (lists, dicts, sets) require a context manager. Mutations are detected and persisted automatically on exit:

```python
with player:
    player.items = ["sword"]
    player.items.append("shield")
# Changes are flushed here

with player:
    print(player.items)  # ['sword', 'shield']
```

## Change tracking

A `Store` can call a custom function if changes are made to a `Storage` instance. This can be used to notify other parts of the software of internal state changes.

```python
def update(ns, key, value):
    print(f"{key!r} on {ns!r} is now {value.data!r}")

store2 = Store(Memory(), on_change=update)
customer = store2.storage("firm", "customer")

customer.age = 37
customer.name_ = "John Doe"
```

## Temporal queries

Every write is assigned a monotonically increasing sequence number (and a timestamp for reference). `Storage.within()` returns a reusable, live, read-only `View` of field values as they were when a condition held:

```python
player.app = "my_game"
player.round = 1
player.score = 10
player.round = 2
player.score = 25

round_one = player.within(round=1)
print(round_one.score)  # 10

for round_val, view in player.along("round"):
    print(f"Round {round_val}: score={view.score}")

for round_val, view in player.within(app="my_game").along("round"):
    print(f"Round {round_val}: score={view.score}")

player.round = 1
player.score = 30
print(round_one.score)  # 30; live views reflect later matching writes
```

By default, a view returns the value in effect while its context holds, even if that value was set earlier. Strict queries only return fields written while that context held. Pass `strict=True` to `within()` or `along()`; chained `along()` calls inherit the view's strictness:

```python
player.treatment = "high"
player.round = 1
player.choice = "A"

print(player.within(round=1).treatment)            # high
print(player.within(round=1, strict=True).choice)  # A
print(player.within(round=1, strict=True).get("treatment"))  # None

for round_val, view in player.along("round", strict=True):
    print(f"Round {round_val}: choice={view.get('choice')}")
```

## Inspecting history

Changes are appended to the database, never overwritten, except that drivers may (but need not) overwrite fields configured with replace semantics. The `history()` method on `Storage` instances returns a read-only mapping of field names to tuples of `Value`s, a small frozen dataclass. `fields()` returns the names of the currently available fields:

```python
player.history()["score"]
# Returns:
# (Value(time=1771028451.3298147, unavailable=False, data=10, context='__main__.<module>:14', seq=3), ...)

player.fields()  # ['app', 'round', 'score']
```

A `Value` contains a `context`, indicating the approximate code location that triggered the change. Tombstones have `unavailable=True`. Each history tuple is sorted by `seq` (sequence number), not by `time`.

## Virtual fields

`Storage` instances can be initialized with virtual fields that function similar to `@property`s. This is a simple mechanism to enable more ORM-like behavior.

```python
# Define helper
def get_group(player):
    return store.storage("game", player._group)

# Initialize Storage instances
player2 = store.storage("game", "player2", virtual={"group": get_group})
player3 = store.storage("game", "player3", virtual={"group": get_group})

# Note the underscore before "group"; this is accessed by get_group:
player2._group = player3._group = "group1"

# This is essentially get_group(player2).budget = 42.7:
player2.group.budget = 42.7

# Access from different player with same _group:
print(player3.group.budget)  # Also 42.7
```

Virtual fields can also be added and removed at any time using `storage.virtual`:

```python
player = store.storage("game", "player1")
player.score = 100

# As a decorator:
@player.virtual
def score_doubled(p):
    return p.score * 2

print(player.score_doubled)  # 200

# With an explicit name:
@player.virtual("bonus")
def compute_bonus(p):
    return p.score * 0.1

# Or directly:
player.virtual["penalty"] = lambda p: p.score * -0.05

# Remove when no longer needed:
del player.virtual["penalty"]
```

References to `Storage` instances cannot be stored directly on `Storage` instances, but the following pattern helps with indirection:

```python
def get_members(group):
    return [store.storage("game", p, virtual={"group": get_group}) for p in group._members]

def get_group(player):
    return store.storage("game", player._group, virtual={"members": get_members})

...

with player2.group:
    player2.group._members = ["player2", "player3"]

with player3.group as g:
    print(g.members)
    print(g.members[0].group.budget)  # Ha!
```

## Custom types

The following types can be stored out-of-the-box: `bool`, `int`, `float`, `str`, `tuple`, `bytes`, `complex`, `None`, `decimal.Decimal`, `frozenset`, `datetime.datetime`, `datetime.date`, `datetime.time`, `uuid.UUID`, `list`, `dict`, `bytearray`, `set`, `random.Random`.

**Note**: `orjson` imposes some constraints on some particular values of some types. For example, `math.inf` is unavailable, and so are `dict`s with non-`str` keys. The same applies to certain uncommonly used subtypes of generics; for example, `list[random.Random]` is unavailable. An Exception will be raised if a value cannot be encoded, or if the encoded data does not decode back to the original value (as long as `Codec.vigilant == True`, which is the default). Types without a meaningful `__eq__` can pass `eq=` to `Codec.register` for this check.

Support for other types can be registered using a custom codec. [Example.](https://github.com/mrpg/uproot/blob/main/src/uproot/stable.py) It would also be possible to write a codec that uses `pickle`, or similar, to handle more types.

## Storage backends

- `Memory`: in-memory, ideal for testing
- `Sqlite3`: file-backed via SQLite (stdlib)
- `PostgreSQL`: PostgreSQL with connection pooling (requires `psycopg`; install with `pip install "appendmuch[pg]"`)

The SQL drivers batch writes (flushed after 100 rows or 0.1 s, whichever comes first, and on `close()`). Call `store.flush()` to force pending rows to durable storage at any point.

Opening a `Store` on an existing database reuses it; the schema is only created when absent. If a table exists but has an incompatible schema, a `RuntimeError` is raised — data is never destroyed implicitly. Call `driver.reset()` explicitly to drop and recreate the schema.

Custom SQL backends can subclass `appendmuch.SQLDriver`, which provides batching, replace-semantics upserts, dump/restore, and history iteration; only connection management and DDL are dialect-specific.

## Testing

```
pytest                                    # everything except PostgreSQL
APPENDMUCH_PG_CONNINFO="dbname=mydb" pytest   # includes PostgreSQL driver tests
```

All storage backends are exercised through a shared behavioral contract (`tests/driver_contract.py`); adding a new driver means subclassing the contract and providing a `driver_factory` fixture. PostgreSQL tests are skipped automatically when `APPENDMUCH_PG_CONNINFO` is not set.

## License

Everything in this repository is licensed under the GNU LGPL version 3.0, or, at your option, any later version. See `LICENSE` for details.

© [Max R. P. Grossmann](https://max.pm/), [Holger Gerhardt](https://www.econ.uni-bonn.de/iame/en/team/gerhardt), 2026.

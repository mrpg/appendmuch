# appendmuch

An extensible append-only log with in-memory cache and pluggable storage backends.

## Installation

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
player.name = "Alice"

print(player.score)  # 100
print(player.name)   # Alice
```

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

## Temporal queries with `within`

Every write is timestamped. Use `within` to query field values as they were when a condition held:

```python
from appendmuch import within

player.round = 1
player.score = 10
player.round = 2
player.score = 25

print(within(player, round=1).score)  # 10
print(within(player, round=2).score)  # 25

for round_val, ctx in within.along(player, "round"):
    print(f"Round {round_val}: score={ctx.score}")
```

## Inspecting history

Changes are appended to the database, never overwritten. The `__history__()` method on Storage instances returns a `dict` of `SortedList`s of `Value`s, a special validated type:

```player
player.__history__()["score"]
# Returns:
# SortedKeyList([…, Value(time=1771028451.3298147, unavailable=False, data=10, context='__main__.<module>:14'), …])
```

A `Value` contains a `context`, indicating the approximate code location that triggered the change. Tombstones have `unavailable=True`.

## Virtual fields

`Storage` instances can be initialized with virtual fields that function similar to `@property`s. This is a simple mechanism to enable more ORM-like behavior.

```player
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

**Note**: `orjson` imposes some constraints on some particular values of some types. For example, `math.inf` is unavailable, and so are `dict`s with non-`str` keys. The same applies to certain uncommonly used subtypes of generics; for example, `list[random.Random]` is unavailable.

Support for other types can be registered using a custom codec. [Example.](https://github.com/mrpg/uproot/blob/main/src/uproot/stable.py) It would also be possible to write a codec that uses `pickle`, or similar, to handle more types.

## Storage backends

- `Memory`: in-memory, ideal for testing
- `Sqlite3`: file-backed via SQLite (stdlib)
- `PostgreSQL`: PostgreSQL with connection pooling (requires `psycopg`)

## Testing

```
pytest                                    # core tests (Memory driver)
pytest tests/test_driver_sqlite.py        # SQLite3 driver tests
APPENDMUCH_PG_CONNINFO="dbname=mydb" \
  pytest tests/test_driver_pg.py          # PostgreSQL driver tests
```

PostgreSQL tests are skipped automatically when `APPENDMUCH_PG_CONNINFO` is not set.

## License

Everything in this repository is licensed under the GNU LGPL version 3.0, or, at your option, any later version. See `LICENSE` for details.

© [Max R. P. Grossmann](https://max.pm/), [Holger Gerhardt](https://www.econ.uni-bonn.de/iame/en/team/gerhardt), 2026.

# appendmuch

An extensible append-only log with in-memory cache and pluggable storage backends.

## Installation

```
pip install "appendmuch @ git+https://github.com/mrpg/appendmuch.git@master"
```

## Quick start

```python
from appendmuch import Memory, Sqlite3, Store

store = Store(Memory())  # Or Store(Sqlite3("db.sqlite3"))

# Storage instances provide attribute-based access to namespaced data
player = store.storage("game", "player1")
player.score = 100
player.name = "Alice"

print(player.score)  # 100
print(player.name)   # Alice
```

## Mutable values

Mutable values (lists, dicts, sets) require a context manager. Mutations are
detected and persisted automatically on exit:

```python
with player:
    player.items = ["sword"]
    player.items.append("shield")
# changes are flushed here

with player:
    print(player.items)  # ['sword', 'shield']
```

## Temporal queries with `within`

Every write is timestamped. Use `within` to query field values as they were when
a condition held:

```python
from appendmuch import within

with player:
    player.round = 1
    player.score = 10
    player.round = 2
    player.score = 25

print(within(player, round=1).score)  # 10
print(within(player, round=2).score)  # 25

for round_val, ctx in within.along(player, "round"):
    print(f"Round {round_val}: score={ctx.score}")
```

## Storage backends

- **Memory** -- in-memory, ideal for testing
- **Sqlite3** -- file-backed via SQLite (stdlib)
- **PostgreSQL** -- PostgreSQL with connection pooling (requires `psycopg`)

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

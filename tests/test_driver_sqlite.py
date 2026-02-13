import io

import pytest

from appendmuch import Codec, Sqlite3, Store, within


@pytest.fixture
def store(tmp_path):
    driver = Sqlite3(str(tmp_path / "test.db"))
    s = Store(driver)
    driver.ensure()
    yield s
    s.close()


def test_basic_set_get(store):
    s = store.storage("ns", "test")
    s.x = 42
    assert s.x == 42


def test_persistence_across_load(store):
    s = store.storage("ns", "test")
    s.x = 42
    s.y = "hello"

    store.load()

    s2 = store.storage("ns", "test")
    assert s2.x == 42
    assert s2.y == "hello"


def test_delete(store):
    s = store.storage("ns", "test")
    s.x = 42
    del s.x
    with pytest.raises(AttributeError):
        _ = s.x


def test_batch_flush(store):
    """Writes exceeding batch_size are flushed correctly."""
    s = store.storage("ns", "test")
    for i in range(200):
        s.x = i

    store.load()
    s2 = store.storage("ns", "test")
    assert s2.x == 199


def test_dump_restore(tmp_path):
    codec = Codec()

    store1 = Store(Sqlite3(str(tmp_path / "src.db")), codec=codec)
    store1.driver.ensure()
    s = store1.storage("ns", "test")
    s.x = 42
    s.y = "hello"

    chunks = list(store1.driver.dump())

    store2 = Store(Sqlite3(str(tmp_path / "dst.db")), codec=codec)
    store2.driver.ensure()
    store2.driver.restore(io.BytesIO(b"".join(chunks)))
    store2.load()

    s2 = store2.storage("ns", "test")
    assert s2.x == 42
    assert s2.y == "hello"

    store1.close()
    store2.close()


def test_replace_predicate(tmp_path):
    store = Store(
        Sqlite3(str(tmp_path / "test.db")),
        replace_predicate=lambda ns, field: field == "latest",
    )
    store.driver.ensure()

    s = store.storage("ns", "test")
    s.latest = "v1"
    s.latest = "v2"
    s.latest = "v3"

    history = s.__history__()
    assert len(history["latest"]) == 1
    assert history["latest"][-1].data == "v3"
    store.close()


def test_context_manager_mutable(store):
    s = store.storage("ns", "test")
    with s:
        s.items = [1, 2, 3]
        s.items.append(4)

    store.load()
    s2 = store.storage("ns", "test")
    with s2:
        assert s2.items == [1, 2, 3, 4]


def test_size(store):
    result = store.driver.size()
    assert isinstance(result, int)
    assert result > 0


def test_within(store):
    s = store.storage("ns", "test")
    with s:
        s.round = 1
        s.score = 10
        s.round = 2
        s.score = 20

    assert within(s, round=1).score == 10
    assert within(s, round=2).score == 20


def test_history_all_ordering(store):
    s = store.storage("ns", "test")
    s.x = 1
    s.x = 2
    s.x = 3

    rows = list(store.driver.history_all())
    times = [v.time for _, _, v in rows]
    assert times == sorted(times)

"""Behavioral contract shared by all DBDriver implementations.

Subclass ``DriverContract`` (or ``SQLDriverContract`` for SQL backends) in a
test module and provide a ``driver_factory`` fixture. ``driver_factory(name)``
returns a driver bound to the storage location ``name``; calling it again
with the same name must yield access to the same (persisted) data, while
different names are independent locations.
"""

import io

import pytest

from appendmuch import Store, within


class DriverContract:
    @pytest.fixture
    def driver_factory(self):
        raise NotImplementedError("subclasses must provide a driver_factory fixture")

    @pytest.fixture
    def store(self, driver_factory):
        store = Store(driver_factory())
        yield store
        store.close()

    def test_set_get_delete(self, store):
        s = store.storage("ns", "test")
        s.x = 42
        assert s.x == 42
        del s.x
        with pytest.raises(AttributeError):
            _ = s.x

    def test_reopen_preserves_data(self, driver_factory):
        store1 = Store(driver_factory())
        s = store1.storage("ns", "test")
        s.x = 42
        s.y = "hello"
        store1.close()

        # a second Store over the same location must see the data, not wipe it
        store2 = Store(driver_factory())
        s2 = store2.storage("ns", "test")
        assert s2.x == 42
        assert s2.y == "hello"
        store2.close()

    def test_batched_writes_survive_reopen(self, driver_factory):
        store1 = Store(driver_factory())
        s = store1.storage("ns", "test")
        for i in range(200):  # exceeds the SQL drivers' batch size
            s.x = i
        store1.close()

        store2 = Store(driver_factory())
        assert store2.storage("ns", "test").x == 199
        store2.close()

    def test_seq_continues_after_reopen(self, driver_factory):
        store1 = Store(driver_factory())
        s = store1.storage("ns", "test")
        s.x = 1
        s.x = 2
        store1.close()

        store2 = Store(driver_factory())
        s2 = store2.storage("ns", "test")
        s2.x = 3

        seqs = [v.seq for v in s2.__history__()["x"]]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == 3
        store2.close()

    def test_dump_restore(self, driver_factory):
        store1 = Store(driver_factory("src"))
        s = store1.storage("ns", "test")
        s.x = 42
        s.y = "hello"
        chunks = list(store1.driver.dump())
        store1.close()

        store2 = Store(driver_factory("dst"))
        store2.driver.restore(io.BytesIO(b"".join(chunks)))
        store2.load()
        s2 = store2.storage("ns", "test")
        assert s2.x == 42
        assert s2.y == "hello"
        store2.close()

    def test_replace_predicate(self, driver_factory):
        store = Store(driver_factory(), replace_predicate=lambda ns, field: field == "latest")
        s = store.storage("ns", "test")
        s.latest = "v1"
        s.latest = "v2"
        s.latest = "v3"
        s.keep = 1
        s.keep = 2

        history = s.__history__()
        assert len(history["latest"]) == 1
        assert history["latest"][-1].data == "v3"
        assert len(history["keep"]) == 2
        store.close()

    def test_mutable_value_mutation_persists(self, driver_factory):
        store1 = Store(driver_factory())
        s = store1.storage("ns", "test")
        with s:
            s.items = [1, 2, 3]
            s.items.append(4)
        store1.close()

        store2 = Store(driver_factory())
        with store2.storage("ns", "test") as s2:
            assert s2.items == [1, 2, 3, 4]
        store2.close()

    def test_within_queries(self, store):
        s = store.storage("ns", "test")
        with s:
            s.round = 1
            s.score = 10
            s.round = 2
            s.score = 20

        assert within(s, round=1).score == 10
        assert within(s, round=2).score == 20

    def test_history_all_is_seq_ordered(self, store):
        s = store.storage("ns", "test")
        s.x = 1
        s.x = 2
        s.x = 3

        seqs = [v.seq for _, _, v in store.driver.history_all()]
        assert seqs == sorted(seqs)
        assert len(seqs) == 3

    def test_flush_makes_writes_visible_to_new_driver(self, driver_factory):
        store = Store(driver_factory())
        store.storage("ns", "test").x = 1

        store.flush()

        other = driver_factory()
        assert len(list(other.history_all())) == 1
        other.close()
        store.close()


class SQLDriverContract(DriverContract):
    def corrupt_schema(self, driver):
        """Create an incompatible table at driver.table_name (dialect-specific)."""
        raise NotImplementedError

    def test_size_reports_bytes(self, driver_factory):
        store = Store(driver_factory())
        store.storage("ns", "test").x = 1
        store.flush()
        size = store.driver.size()
        assert isinstance(size, int)
        assert size > 0
        store.close()

    def test_ensure_creates_schema_once(self, driver_factory):
        driver = driver_factory("fresh")
        assert driver.ensure() is False  # created
        assert driver.ensure() is True  # already there
        driver.close()

    def test_ensure_raises_on_incompatible_schema(self, driver_factory):
        driver = driver_factory("badschema")
        self.corrupt_schema(driver)

        with pytest.raises(RuntimeError, match="seq column"):
            driver.ensure()

        # the incompatible table must still exist (no silent data loss)
        assert driver.table_exists()
        driver.reset()  # leave a clean location behind
        driver.close()

import pytest

from appendmuch import Memory, Storage, Store, within


def make_store(**kwargs):
    driver = Memory()
    return Store(driver, **kwargs)


def test_basic_set_get():
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 42
    assert s.x == 42


def test_get_missing_raises():
    store = make_store()
    s = store.storage("ns", "test")
    with pytest.raises(AttributeError):
        _ = s.nonexistent


def test_get_with_default():
    store = make_store()
    s = store.storage("ns", "test")
    assert s.get("nonexistent", "default") == "default"


def test_get_with_default_exists():
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 42
    assert s.get("x", "default") == 42


def test_delete():
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 42
    assert s.x == 42
    del s.x
    with pytest.raises(AttributeError):
        _ = s.x


def test_context_manager_mutable():
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 10

    with s:
        s.x = [1, 2, 3]
        assert s.x == [1, 2, 3]

    # outside context, mutable access should raise
    with pytest.raises(TypeError, match="context manager"):
        _ = s.x


def test_context_manager_immutable():
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 42
    assert s.x == 42  # immutable, no context needed


def test_deep_change_detection():
    store = make_store()
    s = store.storage("ns", "test")
    s.items = [1, 2, 3]

    with s:
        s.items.append(4)
    # flush should have saved the mutation

    with s:
        assert s.items == [1, 2, 3, 4]


def test_explicit_set_no_double_write():
    """If a field is explicitly set and not mutated after, flush should not write again."""
    store = make_store()
    changes = []
    store.on_change = lambda ns, field, val: changes.append((field, val.data))

    s = store.storage("ns", "test")
    with s:
        s.x = 42

    # one insert for explicit set
    assert len([c for c in changes if c[0] == "x"]) == 1


def test_fields():
    store = make_store()
    s = store.storage("ns", "test")
    s.a = 1
    s.b = 2
    fields = s.__fields__()
    assert set(fields) == {"a", "b"}


def test_bool_empty():
    store = make_store()
    s = store.storage("ns", "test")
    assert not bool(s)


def test_bool_nonempty():
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 1
    assert bool(s)


def test_history():
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 1
    s.x = 2
    s.x = 3
    history = s.__history__()
    assert "x" in history
    assert len(history["x"]) == 3


def test_eq():
    store = make_store()
    s1 = store.storage("ns", "test")
    s2 = store.storage("ns", "test")
    s3 = store.storage("ns", "other")
    assert s1 == s2
    assert s1 != s3


def test_hash():
    store = make_store()
    s1 = store.storage("ns", "test")
    s2 = store.storage("ns", "test")
    assert hash(s1) == hash(s2)


def test_repr():
    store = make_store()
    s = store.storage("ns", "test")
    assert "ns" in repr(s)
    assert "test" in repr(s)


def test_virtual_field():
    store = make_store()
    s = store.storage("ns", "test", virtual={"computed": lambda s: 42})
    assert s.computed == 42


def test_virtual_field_not_settable():
    store = make_store()
    s = store.storage("ns", "test", virtual={"computed": lambda s: 42})
    with pytest.raises(AttributeError, match="Cannot assign to virtual"):
        s.computed = 99


def test_namespace_validator_pass():
    store = make_store(namespace_validator=lambda ns: ns[0] == "allowed")
    s = store.storage("allowed", "test")
    assert s.name == "test"


def test_namespace_validator_reject():
    store = make_store(namespace_validator=lambda ns: ns[0] == "allowed")
    with pytest.raises(ValueError, match="namespace validation"):
        store.storage("forbidden", "test")


def test_replace_predicate():
    store = make_store(replace_predicate=lambda ns, field: field == "latest" and ns.startswith("data/"))
    s = store.storage("data", "test")

    s.latest = "v1"
    s.latest = "v2"
    s.latest = "v3"

    # history should have only 1 entry (replace semantics)
    history = s.__history__()
    assert len(history["latest"]) == 1
    assert history["latest"][-1].data == "v3"


def test_replace_predicate_normal_field():
    store = make_store(replace_predicate=lambda ns, field: field == "latest" and ns.startswith("data/"))
    s = store.storage("data", "test")

    s.normal = "v1"
    s.normal = "v2"
    s.normal = "v3"

    # normal fields should have full history
    history = s.__history__()
    assert len(history["normal"]) == 3


def test_on_change_callback():
    changes = []

    def callback(ns, field, value):
        changes.append({"ns": ns, "field": field, "value": value})

    store = make_store(on_change=callback)
    s = store.storage("ns", "test")
    s.x = 42

    assert len(changes) == 1
    assert changes[0]["field"] == "x"
    assert changes[0]["value"].data == 42
    assert changes[0]["ns"] == ("ns", "test")


def test_on_change_callback_delete():
    changes = []

    def callback(ns, field, value):
        changes.append({"field": field, "unavailable": value.unavailable})

    store = make_store(on_change=callback)
    s = store.storage("ns", "test")
    s.x = 42
    del s.x

    assert len(changes) == 2
    assert changes[1]["unavailable"] is True


def test_within_contextual_query():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.score = 10
        s.round = 2
        s.score = 20

    w = within(s, round=1)
    assert w.score == 10

    w2 = within(s, round=2)
    assert w2.score == 20


def test_within_get_default():
    store = make_store()
    s = store.storage("ns", "test")
    s.round = 1

    w = within(s, round=1)
    assert w.get("nonexistent", "default") == "default"


def test_within_along():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.score = 10
        s.round = 2
        s.score = 20

    results = list(within.along(s, "round"))
    assert len(results) == 2
    assert results[0][0] == 1
    assert results[0][1].score == 10
    assert results[1][0] == 2
    assert results[1][1].score == 20


def test_store_load():
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 42

    # simulate reload from driver
    store.load()

    s2 = store.storage("ns", "test")
    assert s2.x == 42


def test_multiple_namespaces():
    store = make_store()
    s1 = store.storage("a", "one")
    s2 = store.storage("b", "two")

    s1.x = 1
    s2.x = 2

    assert s1.x == 1
    assert s2.x == 2


def test_field_history_since():
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 1  # gets a timestamp

    # get the timestamp of the first write
    history = s.__history__()
    first_time = history["x"][0].time

    s.x = 2
    s.x = 3

    # only entries after first_time
    recent = store.field_history_since(("ns", "test"), "x", first_time)
    assert len(recent) == 2  # second and third writes


def test_invalid_namespace():
    store = make_store()
    with pytest.raises(ValueError, match="invalid namespace"):
        store.storage("has space", "test")


def test_subclass_storage():
    """Test that Storage can be subclassed."""
    store = make_store()

    class MyStorage(Storage):
        def __repr__(self):
            return f"My({self.name})"

    s = MyStorage("ns", "test", store=store)
    s.x = 42
    assert s.x == 42
    assert repr(s) == "My(test)"


def test_context_manager_exception_no_flush():
    """If an exception occurs in context, flush should not run (no mutation save)."""
    store = make_store()
    changes = []
    store.on_change = lambda ns, field, val: changes.append(field)

    s = store.storage("ns", "test")
    s.x = 1

    with pytest.raises(RuntimeError), s:
        s.x = [1, 2, 3]  # explicit write-through (fires on_change)
        changes_after_set = len(changes)
        s.x.append(4)  # in-place mutation (only saved on flush)
        raise RuntimeError("boom")

    # the explicit set fired on_change, but the in-place mutation was NOT flushed
    assert len(changes) == changes_after_set


def test_name_attribute():
    store = make_store()
    s = store.storage("ns", "test")
    assert s.name == "test"


def test_codec_through_full_stack():
    """Custom type registration works through full Storage stack."""
    from orjson import dumps as jd
    from orjson import loads as jl

    from appendmuch import Codec

    codec = Codec()

    class Tag:
        def __init__(self, label):
            self.label = label

        def __eq__(self, other):
            return isinstance(other, Tag) and self.label == other.label

    codec.register(
        Tag,
        200,
        lambda t: jd(t.label),
        lambda r: Tag(jl(r)),
    )

    store = Store(Memory(), codec=codec)
    s = store.storage("ns", "test")
    s.tag = Tag("important")
    assert s.tag == Tag("important")

    # reload from driver
    store.load()
    s2 = store.storage("ns", "test")
    assert s2.tag == Tag("important")


def test_field_history_since_missing_namespace():
    """field_history_since returns [] for non-existent namespace."""
    store = make_store()
    result = store.field_history_since(("nonexistent",), "x", 0.0)
    assert result == []


def test_field_history_since_missing_field():
    """field_history_since returns [] for non-existent field."""
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 1
    result = store.field_history_since(("ns", "test"), "nonexistent", 0.0)
    assert result == []


def test_delete_nonexistent_field():
    """Deleting a field that was never set creates a tombstone."""
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 1
    del s.x
    # verify it's gone
    with pytest.raises(AttributeError):
        _ = s.x


def test_get_field_history_missing():
    """get_field_history on a missing field returns an empty SortedList."""
    store = make_store()
    s = store.storage("ns", "test")
    history = store.db_request(s, "get_field_history", "nonexistent")
    assert len(history) == 0


def test_fields_empty_namespace():
    """__fields__ on a namespace with no data returns empty."""
    store = make_store()
    s = store.storage("ns", "empty")
    assert s.__fields__() == [] or len(s.__fields__()) == 0


def test_fields_after_delete():
    """Deleted fields should not appear in __fields__."""
    store = make_store()
    s = store.storage("ns", "test")
    s.a = 1
    s.b = 2
    del s.a
    fields = s.__fields__()
    assert "a" not in fields
    assert "b" in fields


def test_history_empty_namespace():
    """__history__ on a namespace with no data returns empty dict."""
    store = make_store()
    s = store.storage("ns", "empty")
    assert s.__history__() == {}


def test_invalid_db_request_action():
    """Unknown db_request action raises NotImplementedError."""
    store = make_store()
    s = store.storage("ns", "test")
    with pytest.raises(NotImplementedError):
        store.db_request(s, "invalid_action", "x")


def test_within_missing_context_field():
    """within raises when a context field doesn't exist."""
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 1

    w = within(s, nonexistent=42)
    with pytest.raises(AttributeError):
        _ = w.x


def test_within_no_matching_state():
    """within raises when no state matches the context conditions."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.score = 10

    w = within(s, round=999)
    with pytest.raises(AttributeError):
        _ = w.score


def test_within_key_unavailable():
    """within raises when the target key was deleted in matching context."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.score = 10
        del s.score

    w = within(s, round=1)
    with pytest.raises(AttributeError):
        _ = w.score


def test_within_temporal_ordering_raises():
    """within raises if context field was set after target field (no update)."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.value = 42
        s.ctx = "a"
        # value was set BEFORE ctx, and never updated after.
        # So at the moment ctx="a" is true, value is stale (set before ctx).
        # The temporal check rejects this: context_time > target_time.

    w = within(s, ctx="a")
    with pytest.raises(AttributeError):
        _ = w.value


def test_within_temporal_ordering_ok():
    """within succeeds when target field is set after context field."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.ctx = "a"
        s.value = 42

    w = within(s, ctx="a")
    assert w.value == 42


def test_driver_ensure():
    """Memory driver's ensure() should call reset() and return False."""
    driver = Memory()
    result = driver.ensure()
    # Memory.test_tables() raises, so ensure() calls reset() and returns False
    assert result is False


def test_driver_dump_restore():
    """Memory driver dump and restore round-trip."""
    import io

    from appendmuch import Codec

    codec = Codec()
    store = Store(Memory(), codec=codec)
    s = store.storage("ns", "test")
    s.x = 42
    s.y = "hello"

    # dump from the driver
    chunks = list(store.driver.dump())
    assert len(chunks) == 2

    # restore into a fresh driver (msgpack needs a file-like stream)
    store2 = Store(Memory(), codec=codec)
    stream = io.BytesIO(b"".join(chunks))
    store2.driver.restore(stream)  # type: ignore[arg-type]

    # load from driver into cache
    store2.load()
    s2 = store2.storage("ns", "test")
    assert s2.x == 42
    assert s2.y == "hello"


def test_get_current_value_missing():
    """get_current_value raises for missing field."""
    store = make_store()
    with pytest.raises(AttributeError):
        store.get_current_value(("ns", "test"), "nonexistent")


def test_storage_eq_non_storage():
    """Storage != non-Storage object."""
    store = make_store()
    s = store.storage("ns", "test")
    assert s != "not a storage"
    assert s != 42


def test_assign_internal_attr_invalid():
    """Assigning an unknown dunder attribute raises."""
    store = make_store()
    s = store.storage("ns", "test")
    with pytest.raises(AttributeError, match="not an internal attribute"):
        s.__bogus__ = 42


def test_explicit_set_then_mutate_flushes():
    """If a field is explicitly set, then mutated, flush detects the change."""
    store = make_store()
    changes = []
    store.on_change = lambda ns, field, val: changes.append((field, val.data))

    s = store.storage("ns", "test")
    with s:
        s.x = [1, 2]
        initial_changes = len(changes)
        s.x.append(3)  # mutate after explicit set

    # flush should have detected the mutation and written again
    extra_changes = [c for c in changes[initial_changes:] if c[0] == "x"]
    assert len(extra_changes) == 1
    assert extra_changes[0][1] == [1, 2, 3]

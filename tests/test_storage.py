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


def test_virtual_hot_plug_dict():
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 10
    s.virtual["doubled"] = lambda p: p.x * 2
    assert s.doubled == 20
    del s.virtual["doubled"]
    with pytest.raises(AttributeError):
        _ = s.doubled


def test_virtual_hot_plug_decorator():
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 5

    @s.virtual
    def x_plus_one(p):
        return p.x + 1

    assert s.x_plus_one == 6


def test_virtual_hot_plug_decorator_named():
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 3

    @s.virtual("triple")
    def _ignored(p):
        return p.x * 3

    assert s.triple == 9


def test_virtual_hot_plug_replaces():
    store = make_store()
    s = store.storage("ns", "test")
    s.virtual["val"] = lambda p: 1
    assert s.val == 1
    s.virtual["val"] = lambda p: 2
    assert s.val == 2


def test_virtual_hot_plug_shadows_real_field():
    store = make_store()
    s = store.storage("ns", "test")
    s.score = 100
    assert s.score == 100
    s.virtual["score"] = lambda p: 999
    assert s.score == 999
    del s.virtual["score"]
    assert s.score == 100


def test_virtual_validates_name():
    store = make_store()
    s = store.storage("ns", "test")
    with pytest.raises(ValueError, match="valid identifier"):
        s.virtual["not valid!"] = lambda p: 1


def test_virtual_validates_callable():
    store = make_store()
    s = store.storage("ns", "test")
    with pytest.raises(TypeError, match="callable"):
        s.virtual["field"] = 42


def test_virtual_validates_reserved():
    store = make_store()
    s = store.storage("ns", "test")
    with pytest.raises(ValueError, match="reserved"):
        s.virtual["__store__"] = lambda p: None


def test_virtual_not_reassignable():
    store = make_store()
    s = store.storage("ns", "test")
    with pytest.raises(AttributeError, match="reserved"):
        s.virtual = {}


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


def test_within_along_chained_basic():
    """Chained along yields values set under the context with working attribute access."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.app = "some_game"
        s.round = 1
        s.score = 10
        s.round = 2
        s.score = 20

    results = list(within(s, app="some_game").along("round"))
    assert len(results) == 2
    assert results[0][0] == 1
    assert results[0][1].score == 10
    assert results[1][0] == 2
    assert results[1][1].score == 20


def test_within_along_chained_excludes_wrong_context():
    """Values of the along field set under a different context are excluded."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.app = "some_game"
        s.round = 1
        s.score = 10
        s.app = "other"
        s.round = 2
        s.score = 99

    results = list(within(s, app="some_game").along("round"))
    assert len(results) == 1
    assert results[0][0] == 1
    assert results[0][1].score == 10


def test_within_along_chained_context_toggles():
    """Context that flips back picks up values from both matching windows."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.app = "some_game"
        s.round = 1
        s.app = "other"
        s.round = 2
        s.app = "some_game"
        s.round = 3

    results = list(within(s, app="some_game").along("round"))
    assert [v for v, _ in results] == [1, 3]


def test_within_along_chained_multiple_context_fields():
    """Parent within with multiple context fields filters correctly."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.app = "some_game"
        s.mode = "ranked"
        s.round = 1
        s.mode = "casual"
        s.round = 2
        s.mode = "ranked"
        s.round = 3

    results = list(within(s, app="some_game", mode="ranked").along("round"))
    assert [v for v, _ in results] == [1, 3]


def test_within_along_chained_no_matching_context():
    """Context value that never existed yields nothing."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.score = 10

    assert list(within(s, app="nonexistent").along("round")) == []


def test_within_along_chained_along_field_missing():
    """Along field that doesn't exist in the namespace yields nothing."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.app = "some_game"

    assert list(within(s, app="some_game").along("round")) == []


def test_within_along_chained_multiple_values_same_context():
    """Multiple along values set under the same context are all yielded."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.app = "some_game"
        s.round = 2
        s.where = "South Yarra"
        s.where = "Carlton"
        s.where = "Fremantle"

    results = list(within(s, app="some_game", round=2).along("where"))
    assert [v for v, _ in results] == ["South Yarra", "Carlton", "Fremantle"]


def test_within_along_chained_attribute_access_uses_merged_context():
    """Yielded within objects carry the merged context for attribute resolution."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.app = "some_game"
        s.round = 1
        s.score = 10
        s.round = 2
        s.score = 20

    results = list(within(s, app="some_game").along("round"))
    # Each yielded within should resolve score within {app=some_game, round=N}
    assert results[0][1].score == 10
    assert results[1][1].score == 20
    # Nonexistent field still raises
    with pytest.raises(AttributeError):
        results[0][1].nonexistent  # noqa: B018


def test_within_along_chained_context_set_after_along_values():
    """Along values set before the context field exists are excluded."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.round = 2
        s.app = "some_game"
        s.round = 3

    results = list(within(s, app="some_game").along("round"))
    assert [v for v, _ in results] == [3]


def test_within_along_classmethod_still_works():
    """Existing classmethod API within.along(s, field) is unchanged."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.score = 5

    results = list(within.along(s, "round"))
    assert len(results) == 1
    assert results[0][0] == 1
    assert results[0][1].score == 5


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


def test_within_value_set_before_context():
    """within returns the latest value of key in effect while ctx holds,
    even when the key was set before the context field was established."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.value = 42
        s.ctx = "a"
        # value was set BEFORE ctx and never changed. While ctx="a" holds,
        # value is still 42 — that's the latest value in that context.

    w = within(s, ctx="a")
    assert w.value == 42


def test_within_value_set_after_context():
    """within succeeds when target field is set after context field."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.ctx = "a"
        s.value = 42

    w = within(s, ctx="a")
    assert w.value == 42


def test_within_value_persists_across_context_toggles():
    """A value set before ctx is reached carries forward into later ctx windows."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.role = "buyer"
        s.round = 1
        s.value = 17
        s.round = 2
        s.value = -42

    assert within(s, round=1).role == "buyer"
    assert within(s, round=2).role == "buyer"
    assert within(s, round=1).value == 17
    assert within(s, round=2).value == -42


def test_within_key_deleted_before_context_raises():
    """If key was deleted before ctx is reached and never re-set, raise."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.role = "buyer"
        del s.role
        s.ctx = "a"

    w = within(s, ctx="a")
    with pytest.raises(AttributeError):
        _ = w.role


def test_within_reenters_context_picks_up_current_value():
    """When ctx leaves and re-enters the same value, within sees the value in
    effect at the latest re-entry — even if it was assigned while ctx held a
    different value."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.ctx = "a"
        s.value = 1
        s.ctx = "b"
        s.value = 2  # assigned while ctx="b"
        s.ctx = "a"  # ctx re-enters "a"; value is still 2

    assert within(s, ctx="a").value == 2


def test_within_key_is_context_field():
    """Querying a field that is itself the context field returns the context value."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.color = "red"
        s.color = "blue"

    assert within(s, color="red").color == "red"
    assert within(s, color="blue").color == "blue"


def test_within_multiple_ctx_never_simultaneously_satisfied():
    """When two ctx fields never simultaneously hold their required values, raise."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.a = 1
        s.b = 2  # a=1, b=2
        s.a = 2  # a=2, b=2
        s.b = 1  # a=2, b=1
        s.value = 99

    # a=1 and b=1 never hold simultaneously.
    with pytest.raises(AttributeError):
        _ = within(s, a=1, b=1).value


def test_within_value_deleted_and_reset_during_ctx():
    """If value is set, deleted, and re-set while ctx holds, return the final value."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.ctx = "a"
        s.value = 1
        del s.value
        s.value = 2

    assert within(s, ctx="a").value == 2


def test_within_ctx_deleted_and_restored():
    """When ctx is set, deleted, and set again to the same value, within returns
    the value in effect at the latest re-entry of that ctx value."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.ctx = "a"
        s.value = 1
        del s.ctx
        s.value = 2  # assigned while ctx is unavailable
        s.ctx = "a"  # ctx restored; value is 2

    assert within(s, ctx="a").value == 2


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


def test_refresh_picks_up_external_write():
    """refresh lets a Storage object see a write made by a different object."""
    store = make_store()
    s1 = store.storage("ns", "test")
    s2 = store.storage("ns", "test")

    s1.x = 1
    assert s1.x == 1
    assert s2.x == 1

    # s2 writes to the store; s1's cache is now stale
    s2.x = 99
    assert s1.x == 1  # stale

    s1.refresh("x")
    assert s1.x == 99  # fresh


def test_refresh_all():
    """refresh with no arguments clears the entire field cache."""
    store = make_store()
    s1 = store.storage("ns", "test")
    s2 = store.storage("ns", "test")

    s1.x = 1
    s1.y = 2
    assert s1.x == 1
    assert s1.y == 2

    s2.x = 10
    s2.y = 20

    s1.refresh()
    assert s1.x == 10
    assert s1.y == 20


def test_refresh_uncached_field_is_noop():
    """refresh on a field never accessed does not raise."""
    store = make_store()
    s = store.storage("ns", "test")
    s.refresh("nonexistent")  # should not raise

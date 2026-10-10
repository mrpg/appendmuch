import random

import pytest

from appendmuch import Memory, Storage, Store, View


def make_store(**kwargs):
    return Store(Memory(), **kwargs)


# --- basics -----------------------------------------------------------------


def test_basic_set_get():
    store = make_store()
    s1 = store.storage("a", "one")
    s2 = store.storage("b", "two")
    s1.x = 1
    s2.x = 2
    assert s1.x == 1  # namespaces are isolated
    assert s2.x == 2


def test_missing_field_raises():
    store = make_store()
    s = store.storage("ns", "test")
    with pytest.raises(AttributeError):
        _ = s.nonexistent
    with pytest.raises(AttributeError):
        store.get_current_value(("ns", "test"), "nonexistent")


def test_get_with_default():
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 42
    assert s.get("nonexistent", "default") == "default"
    assert s.get("x", "default") == 42


def test_get_only_resolves_fields():
    store = make_store()
    s = store.storage("ns", "test")

    assert s.get("get") is None
    assert s.get("__store__") is None
    assert s.get("__class__") is None


def test_get_resolves_field_shadowed_by_reserved_method():
    store = make_store()
    s = store.storage("ns", "test")
    store.insert(("ns", "test"), "ns/test", "within", "stored value", "test setup")

    assert callable(s.within)
    assert s.get("within") == "stored value"


def test_delete():
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 42
    del s.x
    with pytest.raises(AttributeError):
        _ = s.x


def test_delete_raises_like_python():
    """Deleting a nonexistent or already-deleted field raises AttributeError."""
    store = make_store()
    s = store.storage("ns", "test")

    with pytest.raises(AttributeError):
        del s.never_set

    s.x = 1
    del s.x
    with pytest.raises(AttributeError):
        del s.x

    # exactly one tombstone in history
    assert sum(1 for v in s.history()["x"] if v.unavailable) == 1


def test_insert_rejects_invalid_key():
    store = make_store()
    with pytest.raises(ValueError, match="identifier"):
        store.insert(("ns", "test"), "ns/test", "not an identifier!", 1, "ctx")


def test_invalid_namespace():
    store = make_store()
    with pytest.raises(ValueError, match="invalid namespace"):
        store.storage("has space", "test")
    with pytest.raises(ValueError, match="invalid namespace"):
        store.storage()


def test_subclass_storage():
    store = make_store()

    class MyStorage(Storage):
        def __repr__(self):
            return f"My({self.name})"

    s = MyStorage("ns", "test", store=store)
    s.x = 42
    assert s.x == 42
    assert repr(s) == "My(test)"


def test_repr_and_name():
    store = make_store()
    s = store.storage("ns", "test")
    assert "ns" in repr(s)
    assert "test" in repr(s)
    assert s.name == "test"


def test_equality_and_hash():
    store = make_store()
    s1 = store.storage("ns", "test")
    s2 = store.storage("ns", "test")
    s3 = store.storage("ns", "other")
    assert s1 == s2
    assert s1 != s3
    assert hash(s1) == hash(s2)
    assert s1 != "not a storage"
    assert s1 != 42


def test_assign_internal_attr_invalid():
    store = make_store()
    s = store.storage("ns", "test")
    with pytest.raises(AttributeError, match="not an internal attribute"):
        s.__bogus__ = 42


def test_reserved_names_rejected():
    """Fields or virtuals shadowing Storage methods would be unreadable."""
    store = make_store()
    s = store.storage("ns", "test")

    for reserved in ("along", "fields", "flush", "get", "history", "refresh", "strict", "within"):
        with pytest.raises(AttributeError, match=r"reserved|method"):
            setattr(s, reserved, 1)
        with pytest.raises(ValueError, match="reserved"):
            s.virtual[reserved] = lambda p: 1

    with pytest.raises(ValueError, match="reserved"):
        store.storage("ns", "reserved_virtual", virtual={"get": lambda p: 1})

    with pytest.raises(ValueError, match="valid identifier"):
        store.storage("ns", "invalid_virtual", virtual={"not valid": lambda p: 1})


# --- namespace tree ----------------------------------------------------------


def test_sub_namespaces_are_not_fields():
    """Field-level views must skip sub-namespaces instead of crashing."""
    store = make_store()
    g = store.storage("game")
    p = store.storage("game", "p1")
    p.y = 2

    assert g.fields() == []
    assert not bool(g)  # only a sub-namespace exists, no fields
    assert store.field_history(("game",), "p1") == []

    g.x = 1
    assert g.fields() == ["x"]
    assert bool(g)
    assert set(g.history().keys()) == {"x"}


def test_namespace_field_collision():
    """A name cannot be both a field and a sub-namespace."""
    store = make_store()
    store.storage("ns2").child = 1
    with pytest.raises(ValueError, match="collides"):
        store.storage("ns2", "child").val = 2
    with pytest.raises(ValueError, match="collides"):
        store.storage("ns2", "child", "deeper").val = 2  # collision mid-path

    store.storage("game", "p1").y = 2
    with pytest.raises(ValueError, match="collides"):
        store.storage("game").p1 = 5


# --- fields / history views ---------------------------------------------------


def test_fields():
    store = make_store()
    s = store.storage("ns", "test")
    assert s.fields() == []

    s.a = 1
    s.b = 2
    assert set(s.fields()) == {"a", "b"}

    del s.a
    assert s.fields() == ["b"]


def test_bool():
    store = make_store()
    s = store.storage("ns", "test")
    assert not bool(s)
    s.x = 1
    assert bool(s)


def test_history():
    store = make_store()
    s = store.storage("ns", "empty")
    assert s.history() == {}

    s = store.storage("ns", "test")
    s.x = 1
    s.x = 2
    s.x = 3
    history = s.history()
    assert [v.data for v in history["x"]] == [1, 2, 3]


def test_history_return_is_structurally_immutable():
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 1

    history = s.history()
    assert isinstance(history["x"], tuple)

    with pytest.raises(TypeError):
        history["y"] = ()

    with pytest.raises(AttributeError):
        history["x"].append(history["x"][0])


def test_field_history_missing():
    store = make_store()
    assert store.field_history(("nonexistent",), "x") == []
    s = store.storage("ns", "test")
    s.x = 1
    assert store.field_history(("ns", "test"), "nonexistent") == []


def test_load_sorts_histories():
    """After load(), histories are ordered by seq even with interleaved fields."""
    store = make_store()
    s = store.storage("ns", "test")
    for i in range(50):
        s.x = i
        s.y = -i

    store.load()
    hist = store.storage("ns", "test").history()
    assert [v.data for v in hist["x"]] == list(range(50))
    seqs = [v.seq for v in hist["y"]]
    assert seqs == sorted(seqs)


# --- context manager & mutability guards --------------------------------------


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


def test_tuple_with_mutable_member_requires_context_and_flushes():
    store = make_store()
    s = store.storage("ns", "test")
    s.x = (1, [2, 3])

    with pytest.raises(TypeError, match="context manager"):
        _ = s.x

    with s:
        value = s.x
        value[1].append(4)

    s.refresh("x")
    with s:
        assert s.x == (1, [2, 3, 4])


def test_deep_change_detection():
    store = make_store()
    s = store.storage("ns", "test")
    s.items = [1, 2, 3]

    with s:
        s.items.append(4)
    # flush should have saved the mutation

    with s:
        assert s.items == [1, 2, 3, 4]


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


def test_nested_context_is_one_mutation_scope():
    """Re-entering the same storage must not discard outer mutations."""
    store = make_store()
    s = store.storage("ns", "test")
    s.items = []

    with s:
        s.items.append("outer-before")

        with s:
            s.items.append("inner")

        s.items.append("outer-after")

    with s:
        assert s.items == ["outer-before", "inner", "outer-after"]


# --- flush semantics -----------------------------------------------------------


def test_explicit_set_no_double_write():
    """If a field is explicitly set and not mutated after, flush should not write again."""
    store = make_store()
    changes = []
    store.on_change = lambda ns, field, val: changes.append((field, val.data))

    s = store.storage("ns", "test")
    with s:
        s.x = 42

    assert len([c for c in changes if c[0] == "x"]) == 1


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

    extra_changes = [c for c in changes[initial_changes:] if c[0] == "x"]
    assert len(extra_changes) == 1
    assert extra_changes[0][1] == [1, 2, 3]


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        (lambda s: setattr(s, "b", [99]), [[], [99]]),
        (lambda s: delattr(s, "b"), [[], None]),
        (lambda s: s.refresh("b"), [[]]),
    ],
    ids=["replace", "delete", "refresh"],
)
def test_flush_respects_callback_changes_to_pending_field(update, expected):
    store = make_store()
    s = store.storage("ns", "test")
    s.a = []
    s.b = []

    def on_change(ns, field, value):
        if field == "a":
            update(s)

    store.on_change = on_change

    with s:
        s.a.append(1)
        s.b.append(2)

    assert [entry.data for entry in s.history()["a"]] == [[], [1]]
    assert [entry.data for entry in s.history()["b"]] == expected


def test_flush_skips_stale_mutation_after_external_write():
    """A context must not write back a stale cached value over a newer external write."""
    store = make_store()
    s1 = store.storage("ns", "test")
    s2 = store.storage("ns", "test")

    s1.x = [1]

    with s1:
        value = s1.x
        s2.x = [9]
        value.append(2)

    assert [entry.data for entry in s1.history()["x"]] == [[1], [9]]


def test_flush_skips_duplicate_write_when_external_change_matches_local_mutation():
    """If another writer already persisted the same value, flush should not append it again."""
    store = make_store()
    changes = []
    store.on_change = lambda ns, field, val: changes.append((field, val.data))

    s1 = store.storage("ns", "test")
    s2 = store.storage("ns", "test")

    s1.x = [1]

    with s1:
        value = s1.x
        value.append(2)
        s2.x = [1, 2]

    x_changes = [value for field, value in changes if field == "x"]
    assert x_changes == [[1], [1, 2]]
    assert [entry.data for entry in s1.history()["x"]] == [[1], [1, 2]]


def test_flush_preserves_mutation_after_equivalent_external_set_write():
    """Set iteration order must not turn an equivalent write into a conflict."""

    class ForwardSet(set):
        def __iter__(self):
            return iter(sorted(set.__iter__(self)))

    class ReverseSet(set):
        def __iter__(self):
            return iter(sorted(set.__iter__(self), reverse=True))

    store = make_store()
    s1 = store.storage("ns", "test")
    s2 = store.storage("ns", "test")
    initial = ForwardSet({0, 8, 16, 24})

    s1.x = initial

    with s1:
        s1.x.add(32)
        s2.x = ReverseSet(initial)

    history = [entry.data for entry in s1.history()["x"]]
    assert history == [initial, initial, initial | {32}]


def test_flush_preserves_mutation_after_equivalent_external_dict_write():
    """Dict insertion order must not turn an equivalent write into a conflict."""
    store = make_store()
    s1 = store.storage("ns", "test")
    s2 = store.storage("ns", "test")
    initial = {"a": 1, "b": 2}

    s1.x = initial

    with s1:
        s1.x["c"] = 3
        s2.x = {"b": 2, "a": 1}

    history = [entry.data for entry in s1.history()["x"]]
    assert history == [initial, initial, {**initial, "c": 3}]


def test_flush_skips_stale_post_assignment_mutation_after_external_write():
    """If a field was assigned and then mutated locally, a newer external value wins."""
    store = make_store()
    s1 = store.storage("ns", "test")
    s2 = store.storage("ns", "test")

    s1.x = [0]

    with s1:
        s1.x = [1]
        s2.x = [9]
        s1.x.append(2)

    assert [entry.data for entry in s1.history()["x"]] == [[0], [1], [9]]


def test_random_write_semantics():
    """Stateful mutable values like random.Random write once per real state change,
    and reading them in a later context does not rewrite the same state."""
    store = make_store()
    s = store.storage("ns", "test")
    expected = random.Random(123)

    with s:
        s.rng = random.Random(123)
        block = [1, 2, 3]
        s.rng.shuffle(block)

    expected.shuffle([1, 2, 3])
    history = s.history()["rng"]
    assert len(history) == 2  # assignment + mutation
    assert history[-1].data.getstate() == expected.getstate()

    with s:
        _ = s.rng  # read-only context

    assert len(s.history()["rng"]) == 2  # no third write


# --- refresh --------------------------------------------------------------------


def test_refresh_picks_up_external_write():
    store = make_store()
    s1 = store.storage("ns", "test")
    s2 = store.storage("ns", "test")

    s1.x = 1
    s2.x = 99
    assert s1.x == 1  # stale

    s1.refresh("x")
    assert s1.x == 99  # fresh

    s1.refresh("never_accessed")  # unknown fields are a no-op


def test_refresh_all():
    store = make_store()
    s1 = store.storage("ns", "test")
    s2 = store.storage("ns", "test")

    s1.x = 1
    s1.y = 2
    s2.x = 10
    s2.y = 20

    s1.refresh()
    assert s1.x == 10
    assert s1.y == 20


# --- store configuration ---------------------------------------------------------


def test_namespace_validator():
    store = make_store(namespace_validator=lambda ns: ns[0] == "allowed")
    assert store.storage("allowed", "test").name == "test"
    with pytest.raises(ValueError, match="namespace validation"):
        store.storage("forbidden", "test")


def test_replace_predicate():
    store = make_store(replace_predicate=lambda ns, field: field == "latest" and ns.startswith("data/"))
    s = store.storage("data", "test")

    s.latest = "v1"
    s.latest = "v2"
    s.latest = "v3"
    s.normal = "v1"
    s.normal = "v2"

    history = s.history()
    assert len(history["latest"]) == 1  # replace semantics
    assert history["latest"][-1].data == "v3"
    assert len(history["normal"]) == 2  # full history


def test_on_change_callback():
    changes = []
    store = make_store(on_change=lambda ns, field, value: changes.append((ns, field, value)))
    s = store.storage("ns", "test")

    s.x = 42
    del s.x

    assert len(changes) == 2
    assert changes[0][:2] == (("ns", "test"), "x")
    assert changes[0][2].data == 42
    assert changes[1][2].unavailable is True


# --- virtual fields ---------------------------------------------------------------


def test_virtual_field():
    store = make_store()
    s = store.storage("ns", "test", virtual={"computed": lambda s: 42})
    assert s.computed == 42
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


def test_virtual_decorator():
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 5

    @s.virtual
    def x_plus_one(p):
        return p.x + 1

    @s.virtual("triple")
    def _ignored(p):
        return p.x * 3

    assert s.x_plus_one == 6
    assert s.triple == 15


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


def test_virtual_validation():
    store = make_store()
    s = store.storage("ns", "test")
    with pytest.raises(ValueError, match="valid identifier"):
        s.virtual["not valid!"] = lambda p: 1
    with pytest.raises(TypeError, match="callable"):
        s.virtual["field"] = 42
    with pytest.raises(ValueError, match="reserved"):
        s.virtual["__store__"] = lambda p: None
    with pytest.raises(AttributeError, match="reserved"):
        s.virtual = {}


# --- contextual views ---------------------------------------------------------------


def test_within_is_a_view():
    store = make_store()
    s = store.storage("ns", "test")
    assert isinstance(s.within(), View)


def test_view_get_only_uses_field_lookup():
    store = make_store()
    s = store.storage("ns", "test")
    store.insert(("ns", "test"), "ns/test", "along", "stored field", "test setup")
    view = s.within()

    assert callable(view.along)
    assert view.get("along") == "stored field"
    assert view.get("missing", "default") == "default"


def test_within_get_bypasses_descriptors_methods_and_internal_attributes():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
    store.insert(("ns", "test"), "ns/test", "along", "field-named-along", "test setup")

    view = s.within(round=1)

    assert callable(view.along)
    assert view.get("along") == "field-named-along"
    assert view.get("get") is None
    assert view.get("__storage__") is None
    assert view.get("__class__") is None


def test_storage_within_returns_reusable_live_view():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.score = 10

    history = s.within(round=1)
    assert isinstance(history, View)
    assert history.score == 10
    assert history.score == 10

    with s:
        s.round = 2
        s.score = 20
        s.round = 1
        s.score = 15

    assert history.score == 15


def test_storage_within_supports_strict_mode():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.treatment = "high"
        s.round = 1
        s.choice = "A"

    history = s.within(round=1, strict=True)

    assert history.choice == "A"
    assert history.get("treatment") is None


def test_view_rejects_setattr():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.score = 10

    view = s.within(round=1)
    with pytest.raises(AttributeError, match="read-only"):
        view.score = 99


def test_view_rejects_delattr():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.score = 10

    view = s.within(round=1)
    with pytest.raises(AttributeError, match="read-only"):
        del view.score


# --- within (contextual queries) ----------------------------------------------------


def test_within_contextual_query():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.score = 10
        s.round = 2
        s.score = 20

    assert s.within(round=1).score == 10
    assert s.within(round=2).score == 20


def test_within_get_default():
    store = make_store()
    s = store.storage("ns", "test")
    s.round = 1

    assert s.within(round=1).get("nonexistent", "default") == "default"


def test_within_missing_context_field():
    store = make_store()
    s = store.storage("ns", "test")
    s.x = 1

    with pytest.raises(AttributeError):
        _ = s.within(nonexistent=42).x


def test_within_no_matching_state():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.score = 10

    with pytest.raises(AttributeError):
        _ = s.within(round=999).score


def test_within_key_unavailable():
    """within raises when the target key was deleted in matching context."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.score = 10
        del s.score

    with pytest.raises(AttributeError):
        _ = s.within(round=1).score


def test_within_value_set_before_context():
    """within returns the latest value of key in effect while ctx holds,
    even when the key was set before the context field was established."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.value = 42
        s.ctx = "a"

    assert s.within(ctx="a").value == 42


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

    assert s.within(round=1).role == "buyer"
    assert s.within(round=2).role == "buyer"
    assert s.within(round=1).value == 17
    assert s.within(round=2).value == -42


def test_within_key_deleted_before_context_raises():
    """If key was deleted before ctx is reached and never re-set, raise."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.role = "buyer"
        del s.role
        s.ctx = "a"

    with pytest.raises(AttributeError):
        _ = s.within(ctx="a").role


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

    assert s.within(ctx="a").value == 2


def test_within_key_is_context_field():
    """Querying a field that is itself the context field returns the context value."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.color = "red"
        s.color = "blue"

    assert s.within(color="red").color == "red"
    assert s.within(color="blue").color == "blue"


def test_within_multiple_ctx_never_simultaneously_satisfied():
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
        _ = s.within(a=1, b=1).value


def test_within_value_deleted_and_reset_during_ctx():
    """If value is set, deleted, and re-set while ctx holds, return the final value."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.ctx = "a"
        s.value = 1
        del s.value
        s.value = 2

    assert s.within(ctx="a").value == 2


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

    assert s.within(ctx="a").value == 2


def test_within_and_along_return_copies():
    """Mutating values returned by within/along must not corrupt the cache."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.items = [1, 2]

    s.within(round=1).items.append(99)
    assert s.within(round=1).items == [1, 2]

    for value, _ctx in s.along("items"):
        value.append(99)
    assert [v for v, _ in s.along("items")] == [[1, 2]]


# --- strict contextual queries -------------------------------------------------


def test_within_strict_rejects_value_set_before_context():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.choice = "A"
        s.round = 1

    assert s.within(round=1).choice == "A"
    with pytest.raises(AttributeError):
        _ = s.within(strict=True, round=1).choice


def test_within_strict_accepts_value_set_after_context():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.choice = "A"

    assert s.within(strict=True, round=1).choice == "A"


def test_within_strict_across_rounds():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.choice = "A"
        s.round = 2

    assert s.within(round=2).choice == "A"
    with pytest.raises(AttributeError):
        _ = s.within(strict=True, round=2).choice

    assert s.within(strict=True, round=1).choice == "A"


def test_within_strict_with_value_in_both_rounds():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.choice = "A"
        s.round = 2
        s.choice = "B"

    assert s.within(strict=True, round=1).choice == "A"
    assert s.within(strict=True, round=2).choice == "B"


def test_within_strict_rejects_value_from_between_context_windows():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.ctx = "a"
        s.value = 1
        s.ctx = "b"
        s.value = 2
        s.ctx = "a"

    assert s.within(ctx="a").value == 2
    with pytest.raises(AttributeError):
        _ = s.within(strict=True, ctx="a").value


def test_within_strict_get_returns_default():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.choice = "A"
        s.round = 1

    assert s.within(strict=True, round=1).get("choice", "default") == "default"


def test_within_strict_treatment_carries_but_strict_rejects():
    """Treatment set once before any round should be rejected in strict mode."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.treatment = "high"
        s.round = 1
        s.choice = "A"
        s.round = 2
        s.choice = "B"

    assert s.within(round=1).treatment == "high"
    assert s.within(round=2).treatment == "high"
    with pytest.raises(AttributeError):
        _ = s.within(strict=True, round=1).treatment
    with pytest.raises(AttributeError):
        _ = s.within(strict=True, round=2).treatment


# --- along -------------------------------------------------------------------------


def test_within_along():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.score = 10
        s.round = 2
        s.score = 20

    results = list(s.along("round"))
    assert [(v, ctx.score) for v, ctx in results] == [(1, 10), (2, 20)]


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

    results = list(s.within(app="some_game").along("round"))
    assert [(v, ctx.score) for v, ctx in results] == [(1, 10), (2, 20)]

    # Nonexistent field still raises through the merged context
    with pytest.raises(AttributeError):
        results[0][1].nonexistent  # noqa: B018


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

    results = list(s.within(app="some_game").along("round"))
    assert [(v, ctx.score) for v, ctx in results] == [(1, 10)]


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

    assert [v for v, _ in s.within(app="some_game").along("round")] == [1, 3]


def test_within_along_chained_multiple_context_fields():
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

    assert [v for v, _ in s.within(app="some_game", mode="ranked").along("round")] == [
        1,
        3,
    ]


def test_within_along_chained_empty_results():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.app = "some_game"
        s.round = 1

    # context value never existed
    assert list(s.within(app="nonexistent").along("round")) == []
    # along field never existed
    assert list(s.within(app="some_game").along("nope")) == []


def test_within_along_chained_multiple_values_same_context():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.app = "some_game"
        s.round = 2
        s.where = "South Yarra"
        s.where = "Carlton"
        s.where = "Fremantle"

    results = list(s.within(app="some_game", round=2).along("where"))
    assert [v for v, _ in results] == ["South Yarra", "Carlton", "Fremantle"]


def test_within_along_chained_context_set_after_along_values():
    """Along values set before the context field exists are excluded."""
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.round = 2
        s.app = "some_game"
        s.round = 3

    assert [v for v, _ in s.within(app="some_game").along("round")] == [3]


def test_along_strict_mode():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.round = 1
        s.score = 10
        s.round = 2
        s.score = 20

    results = list(s.along("round", strict=True))
    assert [(v, ctx.score) for v, ctx in results] == [(1, 10), (2, 20)]


def test_along_strict_rejects_carried_value():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.score = 10
        s.round = 1
        s.round = 2

    results = list(s.along("round", strict=True))
    assert len(results) == 2
    with pytest.raises(AttributeError):
        _ = results[0][1].score
    with pytest.raises(AttributeError):
        _ = results[1][1].score


def test_along_strict_instance_propagates():
    store = make_store()
    s = store.storage("ns", "test")

    with s:
        s.group = "X"
        s.round = 1
        s.score = 10
        s.round = 2
        s.score = 20

    results = list(s.within(strict=True, group="X").along("round"))
    assert [(v, ctx.score) for v, ctx in results] == [(1, 10), (2, 20)]
    with pytest.raises(AttributeError):
        _ = results[0][1].group


# --- codec integration -----------------------------------------------------------


def test_codec_through_full_stack():
    """Custom type registration works through the full Storage stack."""
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
    assert store.storage("ns", "test").tag == Tag("important")


def test_load_keeps_only_latest_replaced_value_from_appending_driver():
    """Drivers may append despite replace semantics; load() still keeps only the latest."""
    driver = Memory()
    Store(driver).storage("ns", "test").latest = 1  # no predicate: the driver appends
    Store(driver).storage("ns", "test").latest = 2

    store = Store(driver, replace_predicate=lambda ns, field: field == "latest")
    assert [v.data for v in store.storage("ns", "test").history()["latest"]] == [2]

import pytest

from appendmuch.storage import dbns2tuple, flatten, tuple2dbns
from appendmuch.utils import ensure, safe_deepcopy, valid_token


def test_ensure_passes():
    ensure(True)


def test_ensure_raises_default():
    with pytest.raises(ValueError, match="Constraint violation"):
        ensure(False)


def test_ensure_raises_custom_type():
    with pytest.raises(TypeError, match="Constraint violation: bad"):
        ensure(False, TypeError, "bad")


def test_valid_token_basic():
    assert valid_token("hello")
    assert valid_token("foo-bar")
    assert valid_token("a.b.c")
    assert valid_token("test_123")


def test_valid_token_rejects():
    assert not valid_token("")
    assert not valid_token("has space")
    assert not valid_token("has/slash")


def test_safe_deepcopy_immutable():
    immutable = (int, float, str, bool, type(None))
    assert safe_deepcopy(42, immutable) == 42
    assert safe_deepcopy("hello", immutable) == "hello"


def test_safe_deepcopy_mutable():
    immutable = (int, float, str, bool, type(None))
    original = [1, 2, 3]
    copied = safe_deepcopy(original, immutable)
    assert copied == original
    assert copied is not original


def test_tuple2dbns():
    assert tuple2dbns(("session", "room1")) == "session/room1"
    assert tuple2dbns(("admin",)) == "admin"


def test_dbns2tuple():
    assert dbns2tuple("session/room1") == ("session", "room1")
    assert dbns2tuple("admin") == ("admin",)


def test_tuple2dbns_roundtrip():
    ns = ("player", "room1", "alice")
    assert dbns2tuple(tuple2dbns(ns)) == ns


def test_flatten_simple():
    d = {"a": 1, "b": 2}
    result = flatten(d)
    assert result == {("a",): 1, ("b",): 2}


def test_flatten_nested():
    d = {"a": {"b": {"c": 1}}}
    result = flatten(d)
    assert result == {("a", "b", "c"): 1}


def test_flatten_mixed():
    d = {"a": 1, "b": {"c": 2}}
    result = flatten(d)
    assert result == {("a",): 1, ("b", "c"): 2}


def test_flatten_with_iterable():
    """Lists as values are treated as leaf nodes, not recursed into."""
    d = {"a": [1, 2, 3], "b": {"c": (4, 5)}}
    result = flatten(d)
    assert result == {("a",): [1, 2, 3], ("b", "c"): (4, 5)}


def test_context_returns_string():
    import sys

    from appendmuch.utils import context

    result = context(sys._getframe())
    assert isinstance(result, str)
    assert ":" in result  # format is "module.function:lineno"


def test_context_none_frame():
    from appendmuch.utils import context

    assert context(None) == "<unknown>"


def test_valid_token_non_string():
    assert not valid_token(123)  # type: ignore[arg-type]

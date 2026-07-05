import sys

import pytest

from appendmuch.storage import dbns2tuple, flatten, tuple2dbns
from appendmuch.utils import context, ensure, safe_deepcopy, valid_token


def test_ensure():
    ensure(True)  # no-op
    with pytest.raises(ValueError, match="Constraint violation"):
        ensure(False)
    with pytest.raises(TypeError, match="Constraint violation: bad"):
        ensure(False, TypeError, "bad")


def test_valid_token():
    assert valid_token("hello")
    assert valid_token("foo-bar")
    assert valid_token("a.b.c")
    assert valid_token("test_123")
    assert not valid_token("")
    assert not valid_token("has space")
    assert not valid_token("has/slash")
    assert not valid_token(123)  # type: ignore[arg-type]


def test_safe_deepcopy():
    immutable = (int, float, str, bool, type(None))
    assert safe_deepcopy(42, immutable) == 42
    assert safe_deepcopy("hello", immutable) == "hello"

    original = [1, 2, 3]
    copied = safe_deepcopy(original, immutable)
    assert copied == original
    assert copied is not original


def test_dbns_conversions():
    assert tuple2dbns(("session", "room1")) == "session/room1"
    assert tuple2dbns(("admin",)) == "admin"
    assert dbns2tuple("session/room1") == ("session", "room1")
    assert dbns2tuple("admin") == ("admin",)

    ns = ("player", "room1", "alice")
    assert dbns2tuple(tuple2dbns(ns)) == ns


def test_flatten():
    assert flatten({"a": 1, "b": 2}) == {("a",): 1, ("b",): 2}
    assert flatten({"a": {"b": {"c": 1}}}) == {("a", "b", "c"): 1}
    assert flatten({"a": 1, "b": {"c": 2}}) == {("a",): 1, ("b", "c"): 2}
    # lists/tuples are leaf nodes, not recursed into
    assert flatten({"a": [1, 2, 3], "b": {"c": (4, 5)}}) == {
        ("a",): [1, 2, 3],
        ("b", "c"): (4, 5),
    }


def test_context():
    result = context(sys._getframe())
    assert isinstance(result, str)
    assert ":" in result  # format is "module.function:lineno"
    assert context(None) == "<unknown>"

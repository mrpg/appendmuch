import random
from datetime import UTC, date, datetime, time, timedelta, timezone
from decimal import Decimal
from uuid import UUID

import pytest
from orjson import dumps as jd
from orjson import loads as jl

from appendmuch.codec import Codec


@pytest.fixture
def codec():
    return Codec()


ROUNDTRIP_VALUES = [
    42,
    -1,
    0,
    3.14,
    0.0,
    -1.5,
    "hello",
    "",
    True,
    False,
    None,
    (1, 2, 3),
    (),
    b"hello",
    b"",
    complex(1, 2),
    Decimal("3.14159"),
    frozenset([1, 2, 3]),
    date(2013, 9, 17),
    date(1, 1, 1),
    date(9999, 12, 31),
    date(2000, 2, 29),  # leap day
    time(14, 30, 45, 123456),
    time(14, 30, 45),
    time(0, 0, 0),
    time(0, 0, 0, 1),
    time(23, 59, 59, 999999),
    datetime(2013, 9, 17, 14, 30, 45, 123456),
    datetime(2013, 9, 17, 14, 30, 45, 123456, timezone(timedelta(hours=5, minutes=30))),
    datetime(2013, 9, 17, 14, 30, 45, tzinfo=UTC),
    UUID("12345678-1234-5678-1234-567812345678"),
    [1, 2, 3],
    [],
    {"a": 1, "b": 2},
    bytearray(b"hello"),
    {1, 2, 3},
]


@pytest.mark.parametrize("value", ROUNDTRIP_VALUES, ids=repr)
def test_roundtrip(codec, value):
    decoded = codec.decode(codec.encode(value))
    assert decoded == value
    assert type(decoded) is type(value)  # e.g. bool vs int, date vs datetime


def test_datetime_timezone_preserved(codec):
    aware = datetime(2013, 9, 17, 14, 30, 45, tzinfo=UTC)
    naive = datetime(2013, 9, 17, 14, 30, 45)
    assert codec.decode(codec.encode(aware)).tzinfo == UTC
    assert codec.decode(codec.encode(naive)).tzinfo is None


def test_wire_type_ids(codec):
    """Subclass relationships must not change wire IDs: bool is not int,
    datetime is not date."""
    assert codec.encode(True)[0] == 5
    assert codec.encode(1)[0] == 0
    assert codec.encode(datetime(2013, 9, 17, 14, 30, 45))[0] == 12
    assert codec.encode(date(2013, 9, 17))[0] == 10


def test_random_roundtrip(codec):
    rng = random.Random(99999)
    for _ in range(100):
        rng.random()
    rng.gauss(0, 1)  # sets internal gauss state

    decoded = codec.decode(codec.encode(rng))
    assert isinstance(decoded, random.Random)
    assert decoded.getstate() == rng.getstate()
    assert decoded.random() == rng.random()


def test_type_classification(codec):
    immutable = set(codec.immutable_types())
    assert {int, str, bool, date, time, datetime} <= immutable
    assert not {list, dict, set, random.Random} & immutable


def test_predicate_registration_keeps_base_mutability(codec):
    codec.register(list, 201, jd, jl, predicate=lambda d: False)
    assert list not in codec.immutable_types()


def test_custom_type_registration(codec):
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __eq__(self, other):
            return isinstance(other, Point) and self.x == other.x and self.y == other.y

    codec.register(
        Point,
        200,
        lambda p: jd([p.x, p.y]),
        lambda r: Point(*jl(r)),
    )

    p = Point(3, 4)
    decoded = codec.decode(codec.encode(p))
    assert decoded == p
    assert isinstance(decoded, Point)


def test_custom_type_with_predicate(codec):
    """Predicate-based registration dispatches on content, not just type."""
    codec.register(
        list,
        201,
        lambda d: jd(d),
        lambda r: jl(r),
        mutable=True,
        predicate=lambda d: bool(d) and all(isinstance(e, str) for e in d),
    )

    assert codec.encode(["a", "b", "c"])[0] == 201  # predicate matches
    assert codec.encode([1, 2, 3])[0] == 128  # falls back to plain list
    assert codec.encode([])[0] == 128


def test_encode_raw(codec):
    type_id, raw = codec.encode_raw(42)
    assert type_id == 0
    assert codec.decode(bytes([type_id]) + raw) == 42


@pytest.mark.parametrize("base", [set, frozenset])
def test_unordered_container_encoding_is_canonical(codec, base):
    class Forward(base):
        def __iter__(self):
            return iter(sorted(base.__iter__(self)))

    class Reverse(base):
        def __iter__(self):
            return iter(sorted(base.__iter__(self), reverse=True))

    forward = Forward({0, 8, 16, 24})
    reverse = Reverse(forward)

    assert list(forward) == [0, 8, 16, 24]
    assert list(reverse) == [24, 16, 8, 0]
    assert codec.encode_raw(forward) == codec.encode_raw(reverse)


@pytest.mark.parametrize(
    ("forward", "reverse"),
    [
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
        ([{"a": 1, "b": 2}], [{"b": 2, "a": 1}]),
        (({"a": 1, "b": 2},), ({"b": 2, "a": 1},)),
    ],
)
def test_mapping_encoding_is_canonical(codec, forward, reverse):
    assert forward == reverse
    assert codec.encode_raw(forward) == codec.encode_raw(reverse)


def test_unknown_type_raises(codec):
    with pytest.raises(NotImplementedError):
        codec.encode(object())


def test_unknown_typeid_raises(codec):
    with pytest.raises(NotImplementedError):
        codec.decode(bytes([255, 0]))


def test_type_id_collision_raises(codec):
    """Registering an already used type ID raises."""
    with pytest.raises(ValueError, match="Type ID 8 already registered"):
        codec.register(UUID, 8, lambda d: jd(str(d)), lambda r: UUID(jl(r)))


def test_type_reregistration_raises(codec):
    with pytest.raises(ValueError, match="Decimal already registered"):
        codec.register(Decimal, 250, lambda d: jd(str(d)), lambda r: Decimal(jl(r)))


def test_isinstance_slow_path(codec):
    """Subclass encoding uses the isinstance slow path."""

    class MyInt(int):
        pass

    encoded = codec.encode(MyInt(42))
    assert encoded[0] == 0  # falls through to int via isinstance
    assert codec.decode(encoded) == 42


def test_predicate_cannot_hijack_type_id(codec):
    with pytest.raises(ValueError, match="Type ID 3 already registered"):
        codec.register(list, 3, jd, jl, predicate=lambda d: False)

    assert codec.decode(codec.encode((1, 2))) == (1, 2)


@pytest.mark.parametrize("type_id", [-1, 256])
def test_type_id_out_of_range_raises(codec, type_id):
    with pytest.raises(ValueError, match="outside the range"):
        codec.register(object, type_id, jd, jl)


def test_custom_round_trip_equality(codec):
    class Opaque:
        def __init__(self, x):
            self.x = x

    codec.register(
        Opaque,
        202,
        lambda o: jd(o.x),
        lambda r: Opaque(jl(r)),
        eq=lambda left, right: isinstance(right, Opaque) and left.x == right.x,
    )
    assert codec.decode(codec.encode(Opaque(1))).x == 1

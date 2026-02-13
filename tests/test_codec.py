import random
from datetime import UTC, date, datetime, time, timedelta, timezone
from decimal import Decimal
from uuid import UUID

import pytest

from appendmuch.codec import Codec


@pytest.fixture
def codec():
    return Codec()


def test_int(codec):
    assert codec.decode(codec.encode(42)) == 42
    assert codec.decode(codec.encode(-1)) == -1
    assert codec.decode(codec.encode(0)) == 0


def test_float(codec):
    assert codec.decode(codec.encode(3.14)) == 3.14
    assert codec.decode(codec.encode(0.0)) == 0.0
    assert codec.decode(codec.encode(-1.5)) == -1.5


def test_str(codec):
    assert codec.decode(codec.encode("hello")) == "hello"
    assert codec.decode(codec.encode("")) == ""


def test_bool(codec):
    assert codec.decode(codec.encode(True)) is True
    assert codec.decode(codec.encode(False)) is False


def test_bool_vs_int_dispatch(codec):
    """bool must encode as type 5, not int type 0."""
    encoded_true = codec.encode(True)
    assert encoded_true[0] == 5
    encoded_int = codec.encode(1)
    assert encoded_int[0] == 0


def test_none(codec):
    assert codec.decode(codec.encode(None)) is None


def test_tuple(codec):
    assert codec.decode(codec.encode((1, 2, 3))) == (1, 2, 3)
    assert codec.decode(codec.encode(())) == ()


def test_bytes(codec):
    assert codec.decode(codec.encode(b"hello")) == b"hello"
    assert codec.decode(codec.encode(b"")) == b""


def test_complex(codec):
    assert codec.decode(codec.encode(complex(1, 2))) == complex(1, 2)


def test_decimal(codec):
    d = Decimal("3.14159")
    assert codec.decode(codec.encode(d)) == d


def test_frozenset(codec):
    fs = frozenset([1, 2, 3])
    assert codec.decode(codec.encode(fs)) == fs


def test_date(codec):
    d = date(2013, 9, 17)
    assert codec.decode(codec.encode(d)) == d
    assert isinstance(codec.decode(codec.encode(d)), date)


def test_time(codec):
    t = time(14, 30, 45, 123456)
    assert codec.decode(codec.encode(t)) == t


def test_time_no_microseconds(codec):
    t = time(14, 30, 45)
    assert codec.decode(codec.encode(t)) == t


def test_datetime(codec):
    dt = datetime(2013, 9, 17, 14, 30, 45, 123456)
    assert codec.decode(codec.encode(dt)) == dt


def test_datetime_vs_date_dispatch(codec):
    """datetime must encode as type 12, not date type 10."""
    dt = datetime(2013, 9, 17, 14, 30, 45)
    encoded = codec.encode(dt)
    assert encoded[0] == 12

    d = date(2013, 9, 17)
    encoded = codec.encode(d)
    assert encoded[0] == 10


def test_datetime_with_timezone(codec):
    tz = timezone(timedelta(hours=5, minutes=30))
    dt = datetime(2013, 9, 17, 14, 30, 45, 123456, tz)
    decoded = codec.decode(codec.encode(dt))
    assert decoded == dt
    assert decoded.tzinfo is not None


def test_datetime_utc(codec):
    dt = datetime(2013, 9, 17, 14, 30, 45, tzinfo=UTC)
    decoded = codec.decode(codec.encode(dt))
    assert decoded == dt
    assert decoded.tzinfo == UTC


def test_datetime_naive(codec):
    dt = datetime(2013, 9, 17, 14, 30, 45)
    decoded = codec.decode(codec.encode(dt))
    assert decoded == dt
    assert decoded.tzinfo is None


def test_uuid(codec):
    u = UUID("12345678-1234-5678-1234-567812345678")
    assert codec.decode(codec.encode(u)) == u


def test_list(codec):
    assert codec.decode(codec.encode([1, 2, 3])) == [1, 2, 3]
    assert codec.decode(codec.encode([])) == []


def test_dict(codec):
    d = {"a": 1, "b": 2}
    assert codec.decode(codec.encode(d)) == d


def test_bytearray(codec):
    ba = bytearray(b"hello")
    assert codec.decode(codec.encode(ba)) == ba


def test_set(codec):
    s = {1, 2, 3}
    assert codec.decode(codec.encode(s)) == s


def test_random(codec):
    rng = random.Random(42)
    for _ in range(10):
        rng.random()

    encoded = codec.encode(rng)
    decoded = codec.decode(encoded)

    assert isinstance(decoded, random.Random)
    assert rng.random() == decoded.random()
    assert rng.random() == decoded.random()


def test_random_state_preservation(codec):
    rng = random.Random(99999)
    for _ in range(1000):
        rng.random()

    original_state = rng.getstate()
    decoded = codec.decode(codec.encode(rng))
    assert decoded.getstate() == original_state


def test_random_gauss_state(codec):
    rng = random.Random(42)
    rng.gauss(0, 1)

    original_state = rng.getstate()
    decoded = codec.decode(codec.encode(rng))
    assert decoded.getstate() == original_state


def test_immutable_types(codec):
    immutable = codec.immutable_types()
    assert int in immutable
    assert str in immutable
    assert bool in immutable
    assert date in immutable
    assert time in immutable
    assert datetime in immutable
    assert list not in immutable
    assert dict not in immutable


def test_mutable_types(codec):
    mutable = codec.mutable_types()
    assert list in mutable
    assert dict in mutable
    assert set in mutable
    assert random.Random in mutable
    assert int not in mutable
    assert str not in mutable


def test_custom_type_registration(codec):
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __eq__(self, other):
            return isinstance(other, Point) and self.x == other.x and self.y == other.y

    from orjson import dumps as jd
    from orjson import loads as jl

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
    """Test predicate-based registration for subtype matching."""
    from orjson import dumps as jd
    from orjson import loads as jl

    codec.register(
        list,
        201,
        lambda d: jd(d),
        lambda r: jl(r),
        mutable=True,
        predicate=lambda d: d and all(isinstance(e, str) for e in d) and len(d) > 0,
    )

    # list of strings should use predicate type
    encoded = codec.encode(["a", "b", "c"])
    assert encoded[0] == 201

    # plain list should use default type 128
    encoded = codec.encode([1, 2, 3])
    assert encoded[0] == 128

    # empty list should use default type 128
    encoded = codec.encode([])
    assert encoded[0] == 128


def test_encode_raw(codec):
    type_id, raw = codec.encode_raw(42)
    assert type_id == 0
    assert codec.decode(bytes([type_id]) + raw) == 42


def test_unknown_type_raises(codec):
    with pytest.raises(NotImplementedError):
        codec.encode(object())


def test_unknown_typeid_raises(codec):
    with pytest.raises(NotImplementedError):
        codec.decode(bytes([255, 0]))


def test_date_edge_cases(codec):
    cases = [
        date(1, 1, 1),
        date(9999, 12, 31),
        date(2000, 2, 29),
    ]
    for d in cases:
        decoded = codec.decode(codec.encode(d))
        assert decoded == d
        assert isinstance(decoded, date)
        assert not isinstance(decoded, datetime)


def test_time_edge_cases(codec):
    cases = [
        time(0, 0, 0),
        time(23, 59, 59),
        time(23, 59, 59, 999999),
        time(0, 0, 0, 1),
    ]
    for t in cases:
        decoded = codec.decode(codec.encode(t))
        assert decoded == t


def test_type_id_collision_raises(codec):
    """Registering same type ID for a different type without predicate raises."""
    from orjson import dumps as jd
    from orjson import loads as jl

    codec.register(
        Decimal,
        250,
        lambda d: jd(str(d)),
        lambda r: Decimal(jl(r)),
    )
    with pytest.raises(ValueError, match="Type ID 250 already registered"):
        codec.register(
            UUID,
            250,
            lambda d: jd(str(d)),
            lambda r: UUID(jl(r)),
        )


def test_isinstance_slow_path(codec):
    """Subclass encoding uses the isinstance slow path."""

    class MyInt(int):
        pass

    val = MyInt(42)
    encoded = codec.encode(val)
    assert encoded[0] == 0  # falls through to int via isinstance
    assert codec.decode(encoded) == 42


def test_type_map(codec):
    """type_map() returns a dict of registered types."""
    tm = codec.type_map()
    assert isinstance(tm, dict)
    assert tm[int] == 0
    assert tm[bool] == 5
    assert tm[str] == 2

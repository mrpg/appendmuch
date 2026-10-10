# Copyright Max R. P. Grossmann & Holger Gerhardt, 2026.
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Extensible type codec for append-only log serialization.

Wire format: 1-byte type ID prefix + orjson payload.
This format MUST remain backward compatible across versions.
"""

import base64
import random
from collections.abc import Callable
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any
from uuid import UUID

from orjson import OPT_SORT_KEYS
from orjson import dumps as orjson_dumps
from orjson import loads as jl


def jd(data: Any) -> bytes:
    """Encode JSON with a stable representation for equal mappings."""
    return orjson_dumps(data, option=OPT_SORT_KEYS)


class Codec:
    def __init__(self) -> None:
        # ordered list for isinstance dispatch (order matters: bool before int, etc.)
        self.ordered_encoders: list[tuple[type, int, Callable[[Any], bytes]]] = []
        # predicate-based entries (checked first; empty for the built-in codec)
        self.predicated_encoders: list[tuple[type, int, Callable[[Any], bytes], Callable[[Any], bool]]] = []
        # fast path for exact type match
        self.exact_encoders: dict[type, tuple[int, Callable[[Any], bytes]]] = {}
        # decoder dispatch by type id
        self.decoders: dict[int, Callable[[bytes], Any]] = {}
        # mutability of each registered type
        self.mutability: dict[type, bool] = {}
        self.vigilant = True
        self.vigilant_skip: Callable[[Any], bool] | None = None
        self._immutable_types_cache: tuple[type, ...] | None = None

        self.register_builtins()

    def register(
        self,
        cls: type,
        type_id: int,
        encoder: Callable[[Any], bytes],
        decoder: Callable[[bytes], Any],
        *,
        mutable: bool = False,
        predicate: Callable[[Any], bool] | None = None,
    ) -> None:
        """Register a type for encoding/decoding.

        Args:
            cls: The Python type to register.
            type_id: Unique integer ID (0-255) for wire format.
            encoder: Function that takes a value and returns raw bytes (without type ID prefix).
            decoder: Function that takes raw bytes and returns a value.
            mutable: Whether this type is mutable (affects context manager guards).
            predicate: Optional content-based predicate for disambiguation (e.g., list subtypes).
        """
        if not 0 <= type_id <= 255:
            raise ValueError(f"Type ID {type_id} is outside the range 0-255")

        if type_id in self.decoders:
            raise ValueError(f"Type ID {type_id} already registered")

        self.decoders[type_id] = decoder

        if predicate is not None:
            # a predicate refines an already registered type, whose mutability prevails
            self.mutability.setdefault(cls, mutable)
            self.predicated_encoders.append((cls, type_id, encoder, predicate))
        else:
            self.mutability[cls] = mutable
            self.ordered_encoders.append((cls, type_id, encoder))
            self.exact_encoders.setdefault(cls, (type_id, encoder))

        self._immutable_types_cache = None

    def encode(self, data: Any) -> bytes:
        """Encode a value to bytes with type ID prefix."""
        type_id, raw = self.encode_raw(data)
        result = bytes((type_id,)) + raw

        if self.vigilant and not (self.vigilant_skip is not None and self.vigilant_skip(data)):
            decoded = self.decode(result)

            if decoded != data and not (
                isinstance(data, random.Random)
                and isinstance(decoded, random.Random)
                and data.getstate() == decoded.getstate()
            ):
                raise ValueError(f"Value {data!r} failed round-trip encoding.")

        return result

    def encode_raw(self, data: Any) -> tuple[int, bytes]:
        """Encode a value, returning (type_id, raw_bytes) separately."""
        # predicate-based entries override everything else
        # (e.g. list[PlayerIdentifier] vs plain list); empty unless registered
        for cls, type_id, encoder, predicate in self.predicated_encoders:
            if isinstance(data, cls) and predicate(data):
                return type_id, encoder(data)

        # fast path: exact type match (handles most common cases)
        exact = self.exact_encoders.get(type(data))
        if exact is not None:
            return exact[0], exact[1](data)

        # slow path: ordered isinstance chain (subclasses of registered types)
        for cls, type_id, encoder in self.ordered_encoders:
            if isinstance(data, cls):
                return type_id, encoder(data)

        raise NotImplementedError(f"{data} has invalid type: {type(data)}")

    def decode(self, allbytes: bytes) -> Any:
        """Decode bytes (with type ID prefix) back to a value."""
        decoder = self.decoders.get(allbytes[0])
        if decoder is None:
            raise NotImplementedError(f"Invalid typeid: {allbytes[0]}")

        return decoder(allbytes[1:])

    def immutable_types(self) -> tuple[type, ...]:
        if self._immutable_types_cache is None:
            self._immutable_types_cache = tuple(cls for cls, mutable in self.mutability.items() if not mutable)

        return self._immutable_types_cache

    def register_builtins(self) -> None:
        """Register all built-in Python types. Order matters for isinstance dispatch."""

        def encode_unordered(items: set[Any] | frozenset[Any]) -> bytes:
            """Encode unordered containers independently of iteration order."""
            encoded_items = sorted(jd(item) for item in items)
            return b"[" + b",".join(encoded_items) + b"]"

        # bool MUST come before int (bool is subclass of int)
        self.register(bool, 5, lambda d: jd(d), lambda r: jl(r))

        # numeric types
        self.register(int, 0, lambda d: jd(d), lambda r: jl(r))
        self.register(float, 1, lambda d: jd(d), lambda r: jl(r))

        # string
        self.register(str, 2, lambda d: jd(d), lambda r: jl(r))

        # tuple
        self.register(tuple, 3, lambda d: jd(d), lambda r: tuple(jl(r)))

        # bytes
        self.register(
            bytes,
            4,
            lambda d: jd(base64.b64encode(d).decode("ascii")),
            lambda r: base64.b64decode(jl(r).encode("ascii")),
        )

        # complex
        self.register(
            complex,
            6,
            lambda d: jd([d.real, d.imag]),
            lambda r: complex(*jl(r)),
        )

        # None
        self.register(type(None), 7, lambda d: b"null", lambda r: None)

        # Decimal
        self.register(
            Decimal,
            8,
            lambda d: jd(str(d)),
            lambda r: Decimal(jl(r)),
        )

        # frozenset
        self.register(
            frozenset,
            9,
            encode_unordered,
            lambda r: frozenset(jl(r)),
        )

        # datetime MUST come before date (datetime is subclass of date)
        self.register(
            datetime,
            12,
            lambda d: jd(d.isoformat()),
            lambda r: datetime.fromisoformat(jl(r)),
        )

        # date
        self.register(
            date,
            10,
            lambda d: jd(d.isoformat()),
            lambda r: date.fromisoformat(jl(r)),
        )

        # time
        self.register(
            time,
            11,
            lambda d: jd(d.isoformat()),
            lambda r: time.fromisoformat(jl(r)),
        )

        # UUID
        self.register(
            UUID,
            13,
            lambda d: jd(base64.b64encode(d.bytes).decode("ascii")),
            lambda r: UUID(bytes=base64.b64decode(jl(r).encode("ascii"))),
        )

        # mutable types (id >= 128)
        self.register(list, 128, lambda d: jd(d), lambda r: jl(r), mutable=True)
        self.register(dict, 129, lambda d: jd(d), lambda r: jl(r), mutable=True)
        self.register(
            bytearray,
            130,
            lambda d: jd(base64.b64encode(d).decode("ascii")),
            lambda r: bytearray(base64.b64decode(jl(r).encode("ascii"))),
            mutable=True,
        )
        self.register(set, 131, encode_unordered, lambda r: set(jl(r)), mutable=True)

        # random.Random
        def encode_random(d: random.Random) -> bytes:
            return jd(d.getstate())

        def decode_random(r: bytes) -> random.Random:
            state = jl(r)
            state = (state[0], tuple(state[1]), state[2])
            rng = random.Random()  # noqa: S311  # nosec B311
            rng.setstate(state)

            return rng

        self.register(random.Random, 133, encode_random, decode_random, mutable=True)

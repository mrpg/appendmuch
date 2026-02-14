# Copyright Max R. P. Grossmann, Holger Gerhardt, et al., 2026.
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

from orjson import dumps as jd
from orjson import loads as jl


class Codec:
    def __init__(self) -> None:
        # ordered list for isinstance dispatch (order matters: bool before int, etc.)
        self.ordered_encoders: list[tuple[type, int, Callable[[Any], bytes], Callable[[Any], bool] | None]] = []
        # fast path for exact type match
        self.exact_encoders: dict[type, tuple[int, Callable[[Any], bytes]]] = {}
        # decoder dispatch by type id
        self.decoders: dict[int, Callable[[bytes], Any]] = {}
        # type metadata
        self.type_to_id: dict[type, int] = {}
        self.id_to_type: dict[int, type] = {}
        self.mutable_ids: set[int] = set()
        self.vigilant = True

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
        if type_id in self.id_to_type and self.id_to_type[type_id] is not cls:
            existing = self.id_to_type[type_id]
            if predicate is None:
                raise ValueError(f"Type ID {type_id} already registered for {existing.__name__}")

        self.type_to_id[cls] = type_id
        self.id_to_type[type_id] = cls
        self.decoders[type_id] = decoder

        if mutable:
            self.mutable_ids.add(type_id)

        self.ordered_encoders.append((cls, type_id, encoder, predicate))

        # only add to exact fast-path if no predicate (predicate entries need ordered check)
        if predicate is None and cls not in self.exact_encoders:
            self.exact_encoders[cls] = (type_id, encoder)

    def encode(self, data: Any) -> bytes:
        """Encode a value to bytes with type ID prefix."""
        type_id, raw = self.encode_raw(data)
        result = bytes([type_id]) + raw

        if self.vigilant and self.decode(result) != data:
            raise ValueError(f"Value {data!r} failed round-trip encoding.")

        return result

    def encode_raw(self, data: Any) -> tuple[int, bytes]:
        """Encode a value, returning (type_id, raw_bytes) separately."""
        # fast path: exact type match (handles most common cases)
        exact = self.exact_encoders.get(type(data))
        if exact is not None:
            type_id, encoder = exact
            # check if there's a predicate-based entry that should override
            # (e.g. list[PlayerIdentifier] vs plain list)
            for cls, tid, enc, pred in self.ordered_encoders:
                if pred is not None and isinstance(data, cls) and pred(data):
                    return tid, enc(data)
            return type_id, encoder(data)

        # slow path: ordered isinstance chain
        for cls, type_id, encoder, predicate in self.ordered_encoders:
            if isinstance(data, cls) and (predicate is None or predicate(data)):
                return type_id, encoder(data)

        raise NotImplementedError(f"{data} has invalid type: {type(data)}")

    def decode(self, allbytes: bytes) -> Any:
        """Decode bytes (with type ID prefix) back to a value."""
        type_id = allbytes[0]
        raw = allbytes[1:]

        decoder = self.decoders.get(type_id)
        if decoder is None:
            raise NotImplementedError(f"Invalid typeid: {type_id}")

        return decoder(raw)

    def immutable_types(self) -> tuple[type, ...]:
        return tuple(cls for cls, tid in self.type_to_id.items() if tid not in self.mutable_ids)

    def mutable_types(self) -> tuple[type, ...]:
        return tuple(cls for cls, tid in self.type_to_id.items() if tid in self.mutable_ids)

    def type_map(self) -> dict[type, int]:
        return dict(self.type_to_id)

    def register_builtins(self) -> None:
        """Register all built-in Python types. Order matters for isinstance dispatch."""

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
            lambda d: jd(list(d)),
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
        self.register(set, 131, lambda d: jd(list(d)), lambda r: set(jl(r)), mutable=True)

        # random.Random
        def encode_random(d: random.Random) -> bytes:
            return jd(d.getstate())

        def decode_random(r: bytes) -> random.Random:
            state = jl(r)
            state = (state[0], tuple(state[1]), state[2])
            rng = random.Random()  # noqa: S311  # nosec B311
            rng.setstate(state)

            if self.vigilant:
                # HACK
                random.Random.__eq__ = lambda self, other: self.getstate() == other.getstate()  # type: ignore

            return rng

        self.register(random.Random, 133, encode_random, decode_random, mutable=True)

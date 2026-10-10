# Copyright Max R. P. Grossmann & Holger Gerhardt, 2026.
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Core storage module: Store (config + state), Storage (namespace accessor),
and View (contextual queries).
"""

import weakref
from bisect import insort
from collections.abc import Callable, Iterator
from contextlib import suppress
from inspect import currentframe
from operator import attrgetter, itemgetter
from time import time
from types import MappingProxyType
from typing import Any, Literal, Self, cast

from appendmuch.codec import Codec
from appendmuch.drivers import DBDriver
from appendmuch.types import Value
from appendmuch.utils import context, ensure, is_immutable, safe_deepcopy, valid_token

_SEQ_KEY = attrgetter("seq")
_BY_SEQ = itemgetter(0)
_MISSING: Any = object()


def tuple2dbns(ns: tuple[str, ...]) -> str:
    return "/".join(ns)


def dbns2tuple(dbns: str) -> tuple[str, ...]:
    if not dbns:
        return ()
    return tuple(dbns.split("/"))


def flatten(d: dict[str, Any], trail: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    result = {}

    for k, v in d.items():
        new_trail = (*trail, k)

        if hasattr(v, "__iter__") and not isinstance(v, (dict, str)):
            result[new_trail] = v
        elif isinstance(v, dict):
            result.update(flatten(v, new_trail))
        else:
            result[new_trail] = v

    return result


def values_equal(codec: Codec, left: Any, right: Any) -> bool:
    if left is right:
        return True

    try:
        return codec.encode_raw(left) == codec.encode_raw(right)
    except Exception:
        return bool(left == right)


def _append_by_seq(hist: list[Value], value: Value) -> None:
    if hist and hist[-1].seq >= value.seq:
        insort(hist, value, key=_SEQ_KEY)
    else:
        hist.append(value)


def _sort_histories(node: dict[str, Any]) -> None:
    for child in node.values():
        if isinstance(child, list):
            child.sort(key=_SEQ_KEY)
        else:
            _sort_histories(child)


class VirtualFields(dict[str, Callable[..., Any]]):
    """A dict subclass that doubles as a decorator for registering virtual fields.

    Supports three styles of registration::

        # 1. Decorator (uses function name)
        @player.virtual
        def score_doubled(p):
            return p.score * 2


        # 2. Decorator with explicit name
        @player.virtual("bonus")
        def compute_bonus(p):
            return p.score * 0.1


        # 3. Dict-style
        player.virtual["computed"] = lambda p: p.score + 10

    Removal works via ``del player.virtual["name"]``.
    """

    _RESERVED: frozenset[str] | None = None

    def __init__(self, initial: dict[str, Callable[..., Any]] | None = None) -> None:
        super().__init__()
        for name, func in (initial or {}).items():
            self[name] = func

    @staticmethod
    def _reserved() -> frozenset[str]:
        if VirtualFields._RESERVED is None:
            VirtualFields._RESERVED = Storage.INTERNAL_ATTRS | Storage.RESERVED_METHODS
        return VirtualFields._RESERVED

    def __setitem__(self, name: str, func: Callable[..., Any]) -> None:
        ensure(
            isinstance(name, str) and name.isidentifier(),
            ValueError,
            f"Virtual field name must be a valid identifier, got {name!r}",
        )
        ensure(
            callable(func),
            TypeError,
            f"Virtual field must be callable, got {type(func).__name__}",
        )
        ensure(
            name not in self._reserved(),
            ValueError,
            f"Cannot use reserved name {name!r} as a virtual field",
        )
        super().__setitem__(name, func)

    def __call__(self, func_or_name: Callable[..., Any] | str) -> Any:
        if callable(func_or_name):
            self[func_or_name.__name__] = func_or_name
            return func_or_name

        ensure(
            isinstance(func_or_name, str),
            TypeError,
            f"Expected a callable or string, got {type(func_or_name).__name__}",
        )

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self[func_or_name] = func
            return func

        return decorator


class Store:
    """Central configuration and state holder for an append-only log."""

    def __init__(
        self,
        driver: DBDriver,
        *,
        codec: Codec | None = None,
        replace_predicate: Callable[[str, str], bool] | None = None,
        on_change: Callable[[tuple[str, ...], str, Value], None] | None = None,
        namespace_validator: Callable[[tuple[str, ...]], bool] | None = None,
    ) -> None:
        self.codec = codec or Codec()
        self.replace_predicate: Callable[[str, str], bool] = replace_predicate or (lambda ns, field: False)
        self.on_change = on_change
        self.namespace_validator = namespace_validator
        self.cache: dict[str, Any] = {}

        # wire codec and replace_predicate into driver
        self.driver = driver
        self.driver.codec = self.codec
        self.driver.replace_predicate = self.replace_predicate

        self.driver.ensure()
        self.load()
        self._finalizer = weakref.finalize(self, self._cleanup, driver)

    @staticmethod
    def _cleanup(driver: DBDriver) -> None:
        with suppress(Exception):
            driver.close()

    def storage(
        self,
        *namespace: str,
        virtual: dict[str, Callable[["Storage"], Any]] | None = None,
    ) -> "Storage":
        return Storage(*namespace, store=self, virtual=virtual)

    def flush(self) -> None:
        """Force any batched driver writes to durable storage."""
        self.driver.flush()

    def close(self) -> None:
        """Close the underlying driver and release resources."""
        if self._finalizer.detach():
            self.driver.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        self.close()
        return False

    def load(self) -> None:
        """Load the entire database history into memory."""
        self.cache.clear()

        for dbns, field, value in self.driver.history_all():
            namespace = dbns2tuple(dbns)
            hist = self.field_history(namespace, field, create=True)
            hist.append(value)

        _sort_histories(self.cache)

    def get_namespace(
        self,
        namespace: tuple[str, ...],
        create: bool = False,
    ) -> dict[str, Any] | None:
        current: Any = self.cache

        for i, part in enumerate(namespace):
            if not isinstance(current, dict):
                if create:
                    raise ValueError(f"Namespace {namespace!r} collides with the field {namespace[i - 1]!r}")
                return None

            if part not in current:
                if not create:
                    return None
                current[part] = {}

            current = current[part]

        if not isinstance(current, dict):
            if create:
                raise ValueError(f"Namespace {namespace!r} collides with the field {namespace[-1]!r}")
            return None

        return cast("dict[str, Any]", current)

    def field_history(
        self,
        namespace: tuple[str, ...],
        field: str,
        *,
        create: bool = False,
    ) -> list[Value]:
        """Return the history list for a field, optionally creating it.

        Raises ValueError when the field name collides with a sub-namespace.
        """
        nested = self.get_namespace(namespace, create=create)
        if nested is None:
            return []

        hist = nested.get(field)

        if hist is None:
            if not create:
                return []
            hist = nested[field] = []
        elif not isinstance(hist, list):
            # the name refers to a sub-namespace, not a field
            if create:
                raise ValueError(f"Field {field!r} collides with a sub-namespace of {namespace!r}")
            return []

        return hist

    def get_current_value(self, namespace: tuple[str, ...], field: str) -> Any:
        hist = self.field_history(namespace, field)

        if hist:
            latest = hist[-1]
            if not latest.unavailable:
                return safe_deepcopy(latest.data, self.codec.immutable_types())

        raise AttributeError(f"Key not found: ({namespace}, {field})")

    def insert(
        self,
        namespace: tuple[str, ...],
        dbns: str,
        key: str,
        value: Any,
        ctx: str,
    ) -> None:
        ensure(key.isidentifier(), ValueError, "Key must be a valid identifier")

        hist = self.field_history(namespace, key, create=True)
        now = time()
        seq = self.driver.insert(dbns, key, value, ctx, now)

        new_value = Value(
            now,
            False,
            safe_deepcopy(value, self.codec.immutable_types()),
            ctx,
            seq=seq,
        )

        if self.replace_predicate(dbns, key):
            hist.clear()
            hist.append(new_value)
        else:
            _append_by_seq(hist, new_value)

        if self.on_change is not None:
            self.on_change(namespace, key, new_value)

    def delete(
        self,
        namespace: tuple[str, ...],
        dbns: str,
        key: str,
        ctx: str,
    ) -> None:
        ensure(key.isidentifier(), ValueError, "Key must be a valid identifier")

        hist = self.field_history(namespace, key)
        if not hist or hist[-1].unavailable:
            raise AttributeError(f"Key not found: ({dbns}, {key})")

        now = time()
        seq = self.driver.delete(dbns, key, ctx, now)

        tombstone = Value(now, True, None, ctx, seq=seq)

        if self.replace_predicate(dbns, key):
            hist.clear()
            hist.append(tombstone)
        else:
            _append_by_seq(hist, tombstone)

        if self.on_change is not None:
            self.on_change(namespace, key, tombstone)

    def list_fields(
        self,
        namespace: tuple[str, ...],
    ) -> list[str]:
        current = self.get_namespace(namespace)
        if not current:
            return []

        return [
            field
            for field, values in current.items()
            if isinstance(values, list) and values and not values[-1].unavailable
        ]

    def check_has_fields(
        self,
        namespace: tuple[str, ...],
    ) -> bool:
        current = self.get_namespace(namespace)
        if not current:
            return False

        return any(isinstance(values, list) and values and not values[-1].unavailable for values in current.values())

    def get_full_history(
        self,
        namespace: tuple[str, ...],
    ) -> MappingProxyType[str, tuple[Value, ...]]:
        current = self.get_namespace(namespace)
        if not current:
            return MappingProxyType({})

        return MappingProxyType({field: tuple(values) for field, values in current.items() if isinstance(values, list)})

    def resolve_within_context(
        self,
        namespace: tuple[str, ...],
        key: str,
        ctx: dict[str, Any],
        *,
        strict: bool = False,
    ) -> Any:
        def _not_found() -> AttributeError:
            return AttributeError(f"No value found for {key} within the specified context in namespace {namespace}")

        ns_ = self.get_namespace(namespace)

        if not ns_ or not isinstance(ns_.get(key), list):
            raise _not_found()

        for cf in ctx:
            if not isinstance(ns_.get(cf), list):
                raise _not_found()

        events: list[tuple[int, str, bool, Any]] = []
        for field in {key, *ctx}:
            for val in ns_[field]:
                events.append((val.seq, field, val.unavailable, val.data))

        events.sort(key=_BY_SEQ)

        # Single ordered pass: track whether each context condition currently
        # holds, plus the current state of the queried key.
        satisfied = dict.fromkeys(ctx, False)
        unmet = len(ctx)
        key_state: Any = _MISSING  # (unavailable, data) of key at this point
        candidate: Any = _MISSING
        prev_met = False

        for _seq, field, unavailable, data in events:
            if field in satisfied:
                holds = not unavailable and data == ctx[field]
                if holds != satisfied[field]:
                    satisfied[field] = holds
                    unmet += -1 if holds else 1

            if field == key:
                key_state = (unavailable, data)

            met = unmet == 0

            if strict:
                if met:
                    if not prev_met:
                        candidate = _MISSING
                    if field == key:
                        candidate = key_state
                prev_met = met
            elif met:
                candidate = key_state

        if candidate is _MISSING or candidate[0]:
            raise _not_found()

        return safe_deepcopy(candidate[1], self.codec.immutable_types())


class View:
    """Live, read-only contextual view over a Storage namespace."""

    __slots__ = ("__context_fields__", "__storage__", "__strict__")

    def __init__(self, storage: "Storage", *, strict: bool = False, **context_fields: Any) -> None:
        object.__setattr__(self, "__storage__", storage)
        object.__setattr__(self, "__context_fields__", context_fields)
        object.__setattr__(self, "__strict__", strict)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is a read-only view")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is a read-only view")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        return self.__storage__.__store__.resolve_within_context(
            self.__storage__.__namespace__,
            name,
            self.__context_fields__,
            strict=self.__strict__,
        )

    def get(self, name: str, default: Any = None) -> Any:
        """Return a field, falling back to default when it is absent."""
        if name.startswith("__") and name.endswith("__"):
            return default

        try:
            return self.__getattr__(name)
        except AttributeError:
            return default

    def along(self, field: str) -> Iterator[tuple[Any, "View"]]:
        """Iterate over values of field that occur within this view's context."""
        storage = self.__storage__
        ctx = self.__context_fields__
        store: Store = storage.__store__
        ns_ = store.get_namespace(storage.__namespace__)
        if not ns_:
            return

        ctx_changes: list[tuple[int, str, bool, Any]] = []
        for cf in ctx:
            hist = ns_.get(cf)
            if isinstance(hist, list):
                for val in hist:
                    ctx_changes.append((val.seq, cf, val.unavailable, val.data))
        ctx_changes.sort(key=_BY_SEQ)

        immutable = store.codec.immutable_types()
        current_ctx: dict[str, Any] = {}
        i = 0
        n = len(ctx_changes)

        for value in store.field_history(storage.__namespace__, field):
            if value.unavailable:
                continue

            # advance the context state up to (and including) this value's seq
            while i < n and ctx_changes[i][0] <= value.seq:
                _, cf, unavailable, cd = ctx_changes[i]
                if unavailable:
                    current_ctx.pop(cf, None)
                else:
                    current_ctx[cf] = cd
                i += 1

            if all(cf in current_ctx and current_ctx[cf] == cv for cf, cv in ctx.items()):
                next_context = {**ctx, field: value.data}
                yield (
                    safe_deepcopy(value.data, immutable),
                    View(storage, strict=self.__strict__, **next_context),
                )


class Storage:
    INTERNAL_ATTRS = frozenset(
        {
            "__accessed_fields__",
            "__contexts__",
            "__field_cache__",
            "name",
            "virtual",
            "__namespace__",
            "__dbns__",
            "__store__",
        }
    )

    # method names that regular fields and virtual fields must not shadow
    RESERVED_METHODS = frozenset({"along", "fields", "flush", "get", "history", "refresh", "strict", "within"})

    _PLAIN_ATTRS = frozenset({"name", "virtual"}) | RESERVED_METHODS

    def __init__(
        self,
        *namespace: str,
        store: Store,
        virtual: dict[str, Callable[[Self], Any]] | None = None,
    ) -> None:
        ensure(
            bool(namespace) and all(type(t) is str and valid_token(t) for t in namespace),
            ValueError,
            f"{namespace!r} is an invalid namespace",
        )

        if store.namespace_validator is not None:
            ensure(
                store.namespace_validator(namespace),
                ValueError,
                f"{namespace!r} failed namespace validation",
            )

        object.__setattr__(self, "name", namespace[-1])
        object.__setattr__(self, "__namespace__", namespace)
        object.__setattr__(self, "__dbns__", tuple2dbns(namespace))
        object.__setattr__(self, "__store__", store)
        object.__setattr__(self, "__contexts__", 0)
        object.__setattr__(self, "__accessed_fields__", {})
        object.__setattr__(self, "__field_cache__", {})
        object.__setattr__(self, "virtual", VirtualFields(virtual or {}))

    def __hash__(self) -> int:
        return hash(self.__namespace__)

    def __setattr__(self, name: str, value: Any) -> None:
        ensure(
            name.isidentifier(),
            ValueError,
            "Attribute name must be a valid identifier",
        )

        if name == "__class__":
            return object.__setattr__(self, name, value)

        if name in ("name", "virtual") or (name.startswith("__") and name.endswith("__")):
            ensure(
                name in type(self).INTERNAL_ATTRS,
                AttributeError,
                f"Attribute '{name}' is not an internal attribute",
            )
            ensure(
                name
                not in (
                    "name",
                    "virtual",
                    "__namespace__",
                    "__dbns__",
                    "__store__",
                )
                or not hasattr(self, name),
                AttributeError,
                f"Attribute '{name}' is reserved and cannot be changed",
            )
            return object.__setattr__(self, name, value)

        if name in type(self).RESERVED_METHODS:
            raise AttributeError(f"'{name}' is a Storage method and cannot be used as a field name")

        virtual = object.__getattribute__(self, "virtual")

        if name in virtual:
            raise AttributeError(f"Cannot assign to virtual field '{name}'")

        cls = type(self)

        if cls is not Storage:
            for klass in cls.__mro__:
                if klass is Storage:
                    break
                if name in klass.__dict__:
                    attr = klass.__dict__[name]

                    if hasattr(attr, "__set__"):
                        attr.__set__(self, value)
                        return

                    raise AttributeError(f"Cannot assign to '{name}'")

        # object.__getattribute__ avoids re-entering our __getattribute__ hook
        getattribute = object.__getattribute__
        store: Store = getattribute(self, "__store__")

        store.insert(
            getattribute(self, "__namespace__"),
            getattribute(self, "__dbns__"),
            name,
            value,
            context(currentframe()),
        )

        getattribute(self, "__field_cache__")[name] = value
        getattribute(self, "__accessed_fields__")[name] = safe_deepcopy(value, store.codec.immutable_types())

    def __guarded_return__(self, name: str, value: Any) -> Any:
        store: Store = object.__getattribute__(self, "__store__")

        ensure(
            is_immutable(value, store.codec.immutable_types()) or self.__contexts__ > 0,
            TypeError,
            f"This {self!r} must be wrapped in a context manager (use 'with') "
            f"because the field '{name}' is of a mutable type ({type(value).__name__}).",
        )

        return value

    def __get_field_value__(self, name: str) -> Any:
        """Resolve a field without exposing methods or internal attributes."""
        accessed_fields = object.__getattribute__(self, "__accessed_fields__")
        field_cache = object.__getattribute__(self, "__field_cache__")
        virtual = object.__getattribute__(self, "virtual")

        if name in virtual:
            return virtual[name](self)

        if name in field_cache:
            return self.__guarded_return__(name, field_cache[name])

        try:
            store: Store = object.__getattribute__(self, "__store__")
            value = store.get_current_value(object.__getattribute__(self, "__namespace__"), name)
            field_cache[name] = value
            if name not in accessed_fields:
                accessed_fields[name] = safe_deepcopy(value, store.codec.immutable_types())
            return self.__guarded_return__(name, value)
        except AttributeError as e:
            raise AttributeError(f"{self} has no .{name}") from e

    def __getattribute__(self, name: str) -> Any:
        if name in Storage._PLAIN_ATTRS or (name.startswith("__") and name.endswith("__")):
            return object.__getattribute__(self, name)

        cls = type(self)

        if cls is not Storage:
            for klass in cls.__mro__:
                if klass is Storage:
                    break
                if name in klass.__dict__:
                    attr = klass.__dict__[name]

                    if hasattr(attr, "__get__"):
                        return attr.__get__(self, cls)

                    return attr

        return object.__getattribute__(self, "__get_field_value__")(name)

    def __delattr__(self, name: str) -> None:
        getattribute = object.__getattribute__
        store: Store = getattribute(self, "__store__")
        store.delete(
            getattribute(self, "__namespace__"),
            getattribute(self, "__dbns__"),
            name,
            context(currentframe()),
        )
        getattribute(self, "__field_cache__").pop(name, None)
        getattribute(self, "__accessed_fields__").pop(name, None)

    def __enter__(self) -> Self:
        if self.__contexts__ == 0:
            self.__field_cache__.clear()
            self.__accessed_fields__.clear()

        self.__contexts__ += 1

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        self.__contexts__ -= 1

        if self.__contexts__ == 0:
            if exc_type is None:
                self.flush()

            self.__field_cache__.clear()

        return False

    def flush(self) -> None:
        """Persist in-place mutations of fields read or written in this context.

        A field is only written if the store still holds the value it was
        read (or last written) as; otherwise the newer stored value wins.
        """
        store: Store = object.__getattribute__(self, "__store__")
        accessed_fields = object.__getattribute__(self, "__accessed_fields__")
        immutable = store.codec.immutable_types()

        # copied, as on_change callbacks may write to this Storage
        for field, current_value in list(object.__getattribute__(self, "__field_cache__").items()):
            original_value = accessed_fields.get(field, _MISSING)

            if original_value is _MISSING or values_equal(store.codec, current_value, original_value):
                continue

            try:
                latest_value = store.get_current_value(self.__namespace__, field)
            except AttributeError:
                accessed_fields.pop(field, None)
                continue

            if values_equal(store.codec, latest_value, original_value):
                store.insert(self.__namespace__, self.__dbns__, field, current_value, context(currentframe()))
                latest_value = current_value

            accessed_fields[field] = safe_deepcopy(latest_value, immutable)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Storage):
            return False

        return cast("bool", self.__namespace__ == other.__namespace__)

    def fields(self) -> list[str]:
        """Return the names of the currently available fields."""
        store: Store = object.__getattribute__(self, "__store__")
        return store.list_fields(self.__namespace__)

    def __bool__(self) -> bool:
        store: Store = object.__getattribute__(self, "__store__")
        return store.check_has_fields(self.__namespace__)

    def history(self) -> MappingProxyType[str, tuple[Value, ...]]:
        """Return a read-only snapshot of this namespace's field histories."""
        store: Store = object.__getattribute__(self, "__store__")
        return store.get_full_history(self.__namespace__)

    def __repr__(self) -> str:
        return f"Storage{self.__namespace__}"

    def refresh(self, *fields: str) -> None:
        """Clear cached values for the given fields so the next read
        fetches from the store.  If no fields are given, the entire
        field cache is cleared."""
        field_cache = object.__getattribute__(self, "__field_cache__")
        if fields:
            for field in fields:
                field_cache.pop(field, None)
        else:
            field_cache.clear()

    def within(self, *, strict: bool = False, **ctx: Any) -> View:
        """Return a reusable contextual view over this namespace."""
        return View(self, strict=strict, **ctx)

    def along(self, field: str, *, strict: bool = False) -> Iterator[tuple[Any, View]]:
        """Iterate over a field's history and its corresponding contextual views."""
        return View(self, strict=strict).along(field)

    def get(self, name: str, default: Any = None) -> Any:
        """Return a field, falling back to default when it is absent."""
        if name.startswith("__") and name.endswith("__"):
            return default

        try:
            return object.__getattribute__(self, "__get_field_value__")(name)
        except AttributeError:
            return default

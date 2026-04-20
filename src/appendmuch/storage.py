# Copyright Max R. P. Grossmann & Holger Gerhardt, 2026.
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Core storage module: Store (config + state), Storage (namespace accessor),
within (contextual queries).
"""

import copy
import functools
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import suppress
from inspect import currentframe
from time import time
from typing import Any, Literal, Self, cast

from sortedcontainers import SortedList

from appendmuch.codec import Codec
from appendmuch.drivers import DBDriver
from appendmuch.types import Value
from appendmuch.utils import context, ensure, safe_deepcopy, valid_token


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

    @staticmethod
    def _reserved() -> frozenset[str]:
        if VirtualFields._RESERVED is None:
            VirtualFields._RESERVED = frozenset(Storage.INTERNAL_ATTRS)
        return VirtualFields._RESERVED

    def __setitem__(self, name: str, func: Callable[..., Any]) -> None:
        ensure(
            isinstance(name, str) and name.isidentifier(),
            ValueError,
            f"Virtual field name must be a valid identifier, got {name!r}",
        )
        ensure(callable(func), TypeError, f"Virtual field must be callable, got {type(func).__name__}")
        ensure(name not in self._reserved(), ValueError, f"Cannot use reserved name {name!r} as a virtual field")
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
            nested_dict = self.get_namespace(namespace, create=True)

            if nested_dict is not None and field not in nested_dict:
                nested_dict[field] = SortedList(key=lambda v: v.seq)

            if nested_dict is not None:
                nested_dict[field].add(value)

    def get_namespace(
        self,
        namespace: tuple[str, ...],
        create: bool = False,
    ) -> dict[str, Any] | None:
        current: Any = self.cache

        for part in namespace:
            if not isinstance(current, dict):
                return None

            if create and part not in current:
                current[part] = {}
            elif part not in current:
                return None

            current = current[part]

        return cast("dict[str, Any] | None", current)

    def field_history_since(
        self,
        namespace: tuple[str, ...],
        field: str,
        since: float,
    ) -> list[Value]:
        current = self.get_namespace(namespace)
        if not current or not isinstance(current, dict) or field not in current:
            return []

        return [v for v in current[field] if v.time is not None and cast("float", v.time) > since]

    def get_current_value(self, namespace: tuple[str, ...], field: str) -> Any:
        current = self.get_namespace(namespace)
        if (
            current
            and isinstance(current, dict)
            and field in current
            and hasattr(current[field], "__iter__")
            and current[field]
        ):
            latest = current[field][-1]
            if not latest.unavailable:
                return safe_deepcopy(latest.data, self.codec.immutable_types())

        raise AttributeError(f"Key not found: ({namespace}, {field})")

    def db_request(
        self,
        caller: "Storage | None",
        action: str,
        key: str = "",
        value: Any | None = None,
        *,
        ctx: str | None = None,
        extra: (
            str
            | tuple[list[str], str]
            | tuple[Sequence[tuple[str, ...]], str]
            | tuple[str, float]
            | dict[str, Any]
            | None
        ) = None,
    ) -> Any:
        ensure(
            key == "" or key.isidentifier(),
            ValueError,
            "Key must be empty or a valid identifier",
        )
        rval = None
        namespace: tuple[str, ...] = ()

        if caller is not None:
            namespace = caller.__namespace__

        dbns = tuple2dbns(namespace) if namespace else ""

        match action, key, value:
            case "insert", _, _ if isinstance(ctx, str):
                rval = self.perform_insert(namespace, dbns, key, value, ctx)
            case "delete", _, None if isinstance(ctx, str):
                self.perform_delete(namespace, dbns, key, ctx)
            case "get", _, None:
                rval = self.get_current_value(namespace, key)
            case "get_field_history", _, None:
                rval = self.get_field_history(namespace, key)
            case "fields", "", None:
                rval = self.list_fields(namespace)
            case "has_fields", "", None:
                rval = self.check_has_fields(namespace)
            case "history", "", None:
                rval = self.get_full_history(namespace)
            case "get_within_context", _, None if isinstance(extra, dict):
                rval = self.resolve_within_context(namespace, key, extra)
            case _, _, _:
                raise NotImplementedError

        return rval

    def perform_insert(
        self,
        namespace: tuple[str, ...],
        dbns: str,
        key: str,
        value: Any,
        ctx: str,
    ) -> Any:
        self.driver.now = time()
        immutable = self.codec.immutable_types()
        seq = self.driver.insert(dbns, key, value, ctx)
        nested_dict = self.get_namespace(namespace, create=True)

        if nested_dict is not None and key not in nested_dict:
            nested_dict[key] = SortedList(key=lambda v: v.seq)

        new_value = Value(
            self.driver.now,
            False,
            safe_deepcopy(value, immutable),
            ctx,
            seq=seq,
        )

        if nested_dict is not None:
            if self.replace_predicate(dbns, key):
                nested_dict[key] = SortedList([new_value], key=lambda v: v.time)
            else:
                nested_dict[key].add(new_value)

        if self.on_change is not None:
            self.on_change(namespace, key, new_value)

        return value

    def perform_delete(
        self,
        namespace: tuple[str, ...],
        dbns: str,
        key: str,
        ctx: str,
    ) -> None:
        ns_dict = self.get_namespace(namespace)
        if ns_dict is None or key not in ns_dict:
            raise AttributeError(f"Key not found: ({dbns}, {key})")

        self.driver.now = time()
        seq = self.driver.delete(dbns, key, ctx)
        nested_dict = self.get_namespace(namespace, create=True)
        if nested_dict is not None and key not in nested_dict:
            nested_dict[key] = SortedList(key=lambda v: v.seq)

        tombstone = Value(self.driver.now, True, None, ctx, seq=seq)
        if nested_dict is not None:
            nested_dict[key].add(tombstone)

        if self.on_change is not None:
            self.on_change(namespace, key, tombstone)

    def get_field_history(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> Any:
        current = self.get_namespace(namespace)
        if current and isinstance(current, dict) and key in current:
            return current[key]
        return SortedList(key=lambda v: v.seq)

    def list_fields(
        self,
        namespace: tuple[str, ...],
    ) -> list[str]:
        current = self.get_namespace(namespace)
        if current and isinstance(current, dict):
            result: list[str] = []
            for field in current:
                if hasattr(current[field], "__iter__") and current[field] and not current[field][-1].unavailable:
                    result.append(field)
            return result
        return []

    def check_has_fields(
        self,
        namespace: tuple[str, ...],
    ) -> bool:
        current = self.get_namespace(namespace)
        if current and isinstance(current, dict):
            return any(
                hasattr(values, "__iter__") and values and not values[-1].unavailable for values in current.values()
            )
        return False

    def get_full_history(
        self,
        namespace: tuple[str, ...],
    ) -> dict[str, Any]:
        current = self.get_namespace(namespace)
        return current if current and isinstance(current, dict) else {}

    def resolve_within_context(
        self,
        namespace: tuple[str, ...],
        key: str,
        ctx: dict[str, Any],
    ) -> Any:
        def _not_found() -> AttributeError:
            return AttributeError(f"No value found for {key} within the specified context in namespace {namespace}")

        ns_ = self.get_namespace(namespace)

        if not ns_ or not isinstance(ns_, dict) or key not in ns_:
            raise _not_found()

        for cf in ctx:
            if cf not in ns_:
                raise _not_found()

        changes = []
        for field in {key, *ctx}:
            if field in ns_:
                for val in ns_[field]:
                    changes.append(
                        {
                            "seq": val.seq,
                            "field": field,
                            "unavailable": val.unavailable,
                            "data": val.data,
                        }
                    )

        changes.sort(key=lambda x: x["seq"])

        current_state: dict[str, Any] = {}
        latest_valid_state = None

        for change in changes:
            current_state[change["field"]] = {
                "unavailable": change["unavailable"],
                "data": change["data"],
            }

            all_conditions_met = True
            if ctx:
                for cond_field, cond_value in ctx.items():
                    if (
                        cond_field not in current_state
                        or current_state[cond_field]["unavailable"]
                        or current_state[cond_field]["data"] != cond_value
                    ):
                        all_conditions_met = False
                        break

            if all_conditions_met:
                latest_valid_state = copy.deepcopy(current_state)

        if latest_valid_state is None:
            raise _not_found()

        if key not in latest_valid_state or latest_valid_state[key]["unavailable"]:
            raise _not_found()

        result_data = latest_valid_state[key]["data"]

        if result_data is None:
            raise _not_found()

        return result_data


class within:  # noqa: N801
    __slots__ = (
        "__context_fields__",
        "__storage__",
    )

    def __init__(self, storage: "Storage", **ctx: Any) -> None:
        self.__storage__ = storage
        self.__context_fields__ = ctx

    def __getattr__(self, name: str) -> Any:
        return self.__storage__.__store__.db_request(
            self.__storage__,
            "get_within_context",
            name,
            extra=self.__context_fields__,
        )

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return self.__storage__.__store__.db_request(
                self.__storage__,
                "get_within_context",
                name,
                extra=self.__context_fields__,
            )
        except AttributeError:
            return default

    along: "AlongDescriptor"


class AlongDescriptor:
    """Descriptor enabling both ``within.along(s, field)`` and ``within(s, …).along(field)``."""

    def __get__(self, obj: "within | None", objtype: "type[within] | None" = None) -> Any:
        if obj is None:
            return self._class_along
        return functools.partial(self._instance_along, obj)

    @staticmethod
    def _class_along(storage: "Storage", field: str) -> Iterator[tuple[Any, "within"]]:
        for value in cast(
            "list[Value]",
            storage.__store__.db_request(storage, "get_field_history", field),
        ):
            if value.data is not None:
                yield value.data, within(storage, **{field: value.data})

    @staticmethod
    def _instance_along(self: "within", field: str) -> Iterator[tuple[Any, "within"]]:
        storage = self.__storage__
        ctx = self.__context_fields__
        store = storage.__store__
        ns_ = store.get_namespace(storage.__namespace__)
        if not ns_ or not isinstance(ns_, dict):
            return

        ctx_changes: list[tuple[int, str, Any]] = []
        for cf in ctx:
            if cf in ns_:
                for val in ns_[cf]:
                    if not val.unavailable:
                        ctx_changes.append((val.seq, cf, val.data))
        ctx_changes.sort()

        for value in cast(
            "list[Value]",
            store.db_request(storage, "get_field_history", field),
        ):
            if value.data is None:
                continue
            s = value.seq
            current_ctx: dict[str, Any] = {}
            for cs, cf, cd in ctx_changes:
                if cs > s:
                    break
                current_ctx[cf] = cd
            if all(current_ctx.get(cf) == cv for cf, cv in ctx.items()):
                yield value.data, within(storage, **{**ctx, field: value.data})


within.along = AlongDescriptor()


class Storage:
    INTERNAL_ATTRS = (
        "__accessed_fields__",
        "__contexts__",
        "__field_cache__",
        "__explicitly_set__",
        "__assigned_values__",
        "name",
        "virtual",
        "__namespace__",
        "__store__",
    )

    def __init__(
        self,
        *namespace: str,
        store: Store,
        virtual: dict[str, Callable[[Self], Any]] | None = None,
    ) -> None:
        ensure(
            all(type(t) is str and valid_token(t) for t in namespace),
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
        object.__setattr__(self, "__store__", store)
        object.__setattr__(self, "__contexts__", 0)
        object.__setattr__(self, "__accessed_fields__", {})
        object.__setattr__(self, "__field_cache__", {})
        object.__setattr__(self, "__explicitly_set__", set())
        object.__setattr__(self, "__assigned_values__", {})
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
                    "__store__",
                )
                or not hasattr(self, name),
                AttributeError,
                f"Attribute '{name}' is reserved and cannot be changed",
            )
            return object.__setattr__(self, name, value)

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

        store: Store = object.__getattribute__(self, "__store__")
        immutable = store.codec.immutable_types()

        newval = store.db_request(
            self,
            "insert",
            name,
            value,
            ctx=context(currentframe()),
        )

        self.__field_cache__[name] = newval
        if self.__contexts__ == 0:
            self.__accessed_fields__[name] = safe_deepcopy(newval, immutable)
        self.__explicitly_set__.add(name)
        self.__assigned_values__[name] = safe_deepcopy(newval, immutable)

    def __guarded_return__(self, name: str, value: Any) -> Any:
        store: Store = object.__getattribute__(self, "__store__")
        immutable = store.codec.immutable_types()

        ensure(
            isinstance(value, immutable) or self.__contexts__ > 0,
            TypeError,
            f"This {self!r} must be wrapped in a context manager (use 'with') "
            f"because the field '{name}' is of a mutable type ({type(value).__name__}).",
        )

        return value

    def __getattribute__(self, name: str) -> Any:
        if name in ("name", "virtual", "flush", "get", "refresh") or (name.startswith("__") and name.endswith("__")):
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

        accessed_fields = object.__getattribute__(self, "__accessed_fields__")
        field_cache = object.__getattribute__(self, "__field_cache__")
        virtual = object.__getattribute__(self, "virtual")

        if name in virtual:
            return virtual[name](self)

        if name in field_cache:
            return self.__guarded_return__(name, field_cache[name])

        try:
            store: Store = object.__getattribute__(self, "__store__")
            value = store.db_request(self, "get", name)
            field_cache[name] = value
            if name not in accessed_fields:
                accessed_fields[name] = safe_deepcopy(value, store.codec.immutable_types())
            return self.__guarded_return__(name, value)
        except AttributeError as e:
            raise AttributeError(f"{self} has no .{name}") from e

    def __delattr__(self, name: str) -> None:
        store: Store = object.__getattribute__(self, "__store__")
        store.db_request(self, "delete", name, None, ctx=context(currentframe()))
        self.__field_cache__.pop(name, None)
        self.__accessed_fields__.pop(name, None)
        self.__explicitly_set__.discard(name)
        self.__assigned_values__.pop(name, None)

    def __enter__(self) -> Self:
        self.__contexts__ += 1
        self.__field_cache__.clear()
        self.__accessed_fields__.clear()
        self.__explicitly_set__.clear()
        self.__assigned_values__.clear()

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        self.__contexts__ -= 1

        if exc_type is None:
            self.flush()

        self.__field_cache__.clear()

        return False

    def flush(self) -> None:
        try:
            accessed_fields = object.__getattribute__(self, "__accessed_fields__")
            field_cache = object.__getattribute__(self, "__field_cache__")
            explicitly_set = object.__getattribute__(self, "__explicitly_set__")
            assigned_values = object.__getattribute__(self, "__assigned_values__")
            store: Store = object.__getattribute__(self, "__store__")
        except AttributeError:
            return

        immutable = store.codec.immutable_types()
        all_fields = set(accessed_fields.keys()) | set(field_cache.keys())

        for field in all_fields:
            if field in field_cache:
                current_value = field_cache[field]
                original_value = accessed_fields.get(field)

                if field in explicitly_set:
                    assigned_value = assigned_values.get(field)
                    if current_value != assigned_value:
                        store.db_request(
                            self,
                            "insert",
                            field,
                            current_value,
                            ctx=context(currentframe()),
                        )
                    accessed_fields[field] = safe_deepcopy(current_value, immutable)
                    continue

                if original_value is not None and current_value != original_value:
                    store.db_request(
                        self,
                        "insert",
                        field,
                        current_value,
                        ctx=context(currentframe()),
                    )
                    accessed_fields[field] = safe_deepcopy(current_value, immutable)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Storage):
            return False

        return cast("bool", self.__namespace__ == other.__namespace__)

    def __fields__(self) -> list[str]:
        store: Store = object.__getattribute__(self, "__store__")
        return cast("list[str]", store.db_request(self, "fields"))

    def __bool__(self) -> bool:
        store: Store = object.__getattribute__(self, "__store__")
        return cast("bool", store.db_request(self, "has_fields"))

    def __history__(self) -> dict[str, list[Value]]:
        store: Store = object.__getattribute__(self, "__store__")
        return cast("dict[str, list[Value]]", store.db_request(self, "history"))

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

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return getattr(self, name)
        except AttributeError:
            return default

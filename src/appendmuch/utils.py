# Copyright Max R. P. Grossmann & Holger Gerhardt, 2026.
# SPDX-License-Identifier: LGPL-3.0-or-later

import copy
import re
import string
from types import FrameType
from typing import Any

TOKEN_CHARS = set(string.ascii_letters + string.digits + "-._")
SAFE_LIKE_PATTERN = re.compile(r"^[a-zA-Z0-9_%.\-/]+\Z")


def ensure(
    condition: bool,
    exctype: type[Exception] = ValueError,
    msg: str | None = None,
) -> None:
    if not condition:
        msg = "Constraint violation: " + msg if msg else "Constraint violation"

        raise exctype(msg)


def valid_token(x: str) -> bool:
    if not isinstance(x, str) or len(x) == 0:
        return False

    return all(ch in TOKEN_CHARS for ch in x)


def is_immutable(value: Any, immutable_types: tuple[type, ...]) -> bool:
    if not isinstance(value, immutable_types):
        return False

    if isinstance(value, (tuple, frozenset)):
        try:
            hash(value)  # rules out containers with mutable members
        except TypeError:
            return False

    return True


def safe_deepcopy(value: Any, immutable_types: tuple[type, ...]) -> Any:
    return value if is_immutable(value, immutable_types) else copy.deepcopy(value)


def context(frame: FrameType | None) -> str:
    try:
        if frame is None:
            return "<unknown>"

        caller_frame = frame.f_back
        if caller_frame is None:
            return "<unknown>"

        module_name = caller_frame.f_globals.get("__name__", "<unknown>")

        return f"{module_name}.{caller_frame.f_code.co_name}:{caller_frame.f_lineno}"
    except Exception:
        return "<unknown>"

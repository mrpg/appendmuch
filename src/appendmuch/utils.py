# Copyright Max R. P. Grossmann, Holger Gerhardt, et al., 2026.
# SPDX-License-Identifier: LGPL-3.0-or-later

import copy
import inspect
import string
from types import FrameType
from typing import Any

TOKEN_CHARS = set(string.ascii_letters + string.digits + "-._")


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


def safe_deepcopy(value: Any, immutable_types: tuple[type, ...]) -> Any:
    if isinstance(value, immutable_types):
        return value

    return copy.deepcopy(value)


def context(frame: FrameType | None) -> str:
    try:
        if frame is None:
            return "<unknown>"

        caller_frame = frame.f_back
        if caller_frame is None:
            return "<unknown>"

        caller_function = caller_frame.f_code.co_name
        caller_lineno = caller_frame.f_lineno

        caller_module = inspect.getmodule(caller_frame)
        module_name = caller_module.__name__ if caller_module else "<unknown>"

        return f"{module_name}.{caller_function}:{caller_lineno}"
    except Exception:
        return "<unknown>"

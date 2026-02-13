# Copyright Max R. P. Grossmann, Holger Gerhardt, et al., 2026.
# SPDX-License-Identifier: LGPL-3.0-or-later

from typing import Any

from pydantic.dataclasses import dataclass as validated_dataclass


@validated_dataclass(frozen=True)
class Value:
    time: float | None = None
    unavailable: bool = True
    data: Any | None = None
    context: str = ""

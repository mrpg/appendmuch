# Copyright Max R. P. Grossmann & Holger Gerhardt, 2026.
# SPDX-License-Identifier: LGPL-3.0-or-later

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Value:
    time: float
    unavailable: bool
    data: Any
    context: str
    seq: int

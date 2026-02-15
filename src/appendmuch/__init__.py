# Copyright Max R. P. Grossmann & Holger Gerhardt, 2026.
# SPDX-License-Identifier: LGPL-3.0-or-later

from appendmuch.codec import Codec
from appendmuch.drivers import DBDriver, Memory, PostgreSQL, Sqlite3
from appendmuch.storage import Storage, Store, dbns2tuple, flatten, tuple2dbns, within
from appendmuch.types import Value
from appendmuch.utils import safe_deepcopy

__all__ = [
    "Codec",
    "DBDriver",
    "Memory",
    "PostgreSQL",
    "Sqlite3",
    "Storage",
    "Store",
    "Value",
    "dbns2tuple",
    "flatten",
    "safe_deepcopy",
    "tuple2dbns",
    "within",
]

__version_info__ = 0, 0, 1
__version__ = ".".join(map(str, __version_info__))
__author__ = "Max R. P. Grossmann, Holger Gerhardt"
__email__ = "appendmuch@grossmann.nexus"

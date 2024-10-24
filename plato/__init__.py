"""Plato.
Copyright 2023-2024 Thomas Schouten
"""
from plato import (
    globe,
    plates,
    points,
    plot,
    settings,
    slabs,
    utils_data,
    utils_calc,
    utils_init,
)

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

name = "plato"
__all__ = [
    "globe",
    "plates",
    "points",
    "plot",
    "settings",
    "slabs",
    "utils_data",
    "utils_calc",
    "utils_init",
]
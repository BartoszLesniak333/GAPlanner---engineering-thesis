# vrptw/__init__.py

from .data import load_instance
from .model import Instance, Route
from .split import split_routes
from . import fitness

__all__ = [
    "load_instance",
    "Instance",
    "Route",
    "split_routes",
    "fitness",
]

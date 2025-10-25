"""Compatibility shim: tests expect a module named `grep` exposing `grep()`.
This file re-exports the implementation from `main.py` to avoid changing test code.
"""
from main import grep

__all__ = ["grep"]

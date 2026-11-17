"""Base operations for values in the closed interval [0, 1]."""
from __future__ import annotations
from collections.abc import Iterable
from typing import TypeAlias
import numpy as np
Scalar: TypeAlias = float | np.floating
Array: TypeAlias = np.ndarray
Value: TypeAlias = Scalar | Array

def N(x: Value) -> Value: return 1.0 - x
def A(x: Value, y: Value) -> Value: return np.minimum(1.0, x + y)
def M(x: Value, y: Value) -> Value: return x * y
def _as_output(value):
    if isinstance(value, np.ndarray) and value.ndim == 0: return float(value)
    if isinstance(value, np.generic): return float(value)
    return value
def A_many(items: Iterable[Value] | np.ndarray, axis: int | None = None) -> Value:
    if isinstance(items, np.ndarray):
        if axis is None:
            iterator = iter(items.flat)
            try: result = next(iterator)
            except StopIteration: return 0.0
            for item in iterator: result = A(result, item)
            return _as_output(np.asarray(result))
        moved = np.moveaxis(items, axis, 0)
        if moved.shape[0] == 0: return np.zeros(moved.shape[1:], dtype=float)
        result = np.zeros(moved.shape[1:], dtype=float)
        for item in moved: result = A(result, item)
        return _as_output(np.asarray(result))
    iterator = iter(items)
    try: result = next(iterator)
    except StopIteration: return 0.0
    for item in iterator: result = A(result, item)
    if isinstance(result, np.ndarray): return result
    return _as_output(np.asarray(result))
def D(x: Value, y: Value) -> Value: return N(A(N(x), y))
def B(x: Value, k: int = 1) -> Value:
    if not isinstance(k, int) or k < 0: raise ValueError("k must be a non-negative integer")
    result = x
    for _ in range(k): result = A(result, result)
    return result
def update_Q(Q: Value, confirm: Value, falsify: Value, eta_plus: float, eta_minus: float) -> Value:
    return A(D(Q, M(eta_minus, falsify)), M(eta_plus, confirm))
__all__ = ["N", "A", "M", "A_many", "D", "B", "update_Q"]

"""Closed operations for scalar and array values in the unit interval."""
from __future__ import annotations

from typing import TypeAlias

import numpy as np

Scalar: TypeAlias = float | np.floating
Array: TypeAlias = np.ndarray
Value: TypeAlias = Scalar | Array


def N(x: Value) -> Value:
    """Complement in [0, 1]."""
    return 1.0 - x


def A(x: Value, y: Value) -> Value:
    """Saturating addition in [0, 1]."""
    return np.minimum(1.0, x + y)


def M(x: Value, y: Value) -> Value:
    """Multiplication in [0, 1]."""
    return x * y


def D(x: Value, y: Value) -> Value:
    """Bounded difference max(0, x-y)."""
    return N(A(N(x), y))


def update_Q(Q: Value, confirm: Value, falsify: Value,
             eta_plus: float, eta_minus: float) -> Value:
    """Increase on confirmation and decay on falsification, remaining bounded."""
    if not 0.0 <= eta_plus <= 1.0 or not 0.0 <= eta_minus <= 1.0:
        raise ValueError("learning rates must lie in [0, 1]")
    return A(D(Q, M(eta_minus, falsify)), M(eta_plus, confirm))


__all__ = ["N", "A", "M", "D", "update_Q"]

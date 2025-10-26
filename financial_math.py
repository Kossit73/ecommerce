"""Lightweight financial helper functions used across the app.

These helpers replicate the subset of ``numpy_financial`` behaviour that the
analytics engine requires so the application can run even when the optional
package is unavailable.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def npv(rate: float | np.ndarray, cashflows: Iterable[float]) -> float:
    """Compute the net present value for ``cashflows`` discounted by ``rate``."""

    values = np.asarray(list(cashflows), dtype=float)
    if values.size == 0:
        return 0.0

    if np.isscalar(rate):
        rates = np.full(values.size, float(rate), dtype=float)
    else:
        rates = np.asarray(rate, dtype=float)
        if rates.size == 1:
            rates = np.full(values.size, float(rates[0]), dtype=float)
        if rates.size != values.size:
            raise ValueError("Rate array must match cashflows length or be scalar")

    periods = np.arange(values.size, dtype=float)
    discount = np.power(1 + rates, periods)
    discount[discount == 0] = np.finfo(float).eps
    return float(np.sum(values / discount))


def _npv_derivative(rate: float, cashflows: np.ndarray) -> float:
    periods = np.arange(cashflows.size, dtype=float)
    denom = np.power(1 + rate, periods + 1)
    denom[denom == 0] = np.finfo(float).eps
    return float(np.sum(-periods * cashflows / denom))


def irr(cashflows: Iterable[float], *, guess: float = 0.1) -> float:
    """Estimate the internal rate of return for ``cashflows``.

    The implementation combines a Newton refinement with a bounded bisection
    search so results remain stable even when the cashflow series is irregular.
    """

    values = np.asarray(list(cashflows), dtype=float)
    if values.size < 2:
        return np.nan

    def npv_at(rate: float) -> float:
        return npv(rate, values)

    rate = float(guess)
    for _ in range(50):
        value = npv_at(rate)
        if abs(value) < 1e-8:
            return rate
        derivative = _npv_derivative(rate, values)
        if abs(derivative) < 1e-10:
            break
        step = value / derivative
        rate -= step
        if abs(step) < 1e-8:
            return rate
        if rate <= -0.9999:
            rate = -0.9999
        if rate > 1e6:
            break

    low, high = -0.9999, 1.0
    f_low, f_high = npv_at(low), npv_at(high)
    iteration = 0
    while f_low * f_high > 0 and iteration < 50:
        high *= 2
        f_high = npv_at(high)
        iteration += 1
        if high > 1e6:
            break

    if f_low * f_high > 0:
        return np.nan

    for _ in range(100):
        mid = (low + high) / 2
        f_mid = npv_at(mid)
        if abs(f_mid) < 1e-8 or abs(high - low) < 1e-8:
            return mid
        if f_low * f_mid < 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid

    return mid


def pmt(rate: float, nper: int, pv: float, fv: float = 0.0, when: str | int = "end") -> float:
    """Replicate ``numpy_financial.pmt`` for the limited use cases in the app."""

    rate = float(rate)
    nper = int(nper)
    fv = float(fv)
    pv = float(pv)
    when = 1 if when in {1, "begin", "start"} else 0

    if nper <= 0:
        raise ValueError("nper must be positive")

    if abs(rate) < 1e-12:
        return -(pv + fv) / nper

    rate_term = (1 + rate) ** nper
    denominator = (1 + rate * when) * (1 - rate_term ** -1)
    if abs(denominator) < 1e-12:
        denominator = np.finfo(float).eps

    payment = -(rate * (fv + pv * rate_term)) / denominator
    return float(payment)


__all__ = ["irr", "npv", "pmt"]

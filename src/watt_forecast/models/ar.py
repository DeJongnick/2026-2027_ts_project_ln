"""Seasonal normal and AR model of the daily consumption anomaly.

The daily consumption is split into a "normal" (yearly cycle + day of the week)
and an anomaly, which is modelled by an AR(p). Setup: on day D before 12:00 the
last complete day is D-1, so the target day T = D+1 is 2 steps ahead of T-2.
"""

import numpy as np
import polars as pl

YEAR_DAYS = 365.25


def _design_matrix(daily: pl.DataFrame, k: int = 2) -> np.ndarray:
    """Columns: k pairs of yearly Fourier harmonics + 7 day-of-week indicators.

    The 7 indicators replace the constant (one intercept per day of the week).
    """
    t = (daily["date"] - daily["date"].min()).dt.total_days().to_numpy()
    cols = []
    for j in range(1, k + 1):
        cols += [np.cos(2 * np.pi * j * t / YEAR_DAYS), np.sin(2 * np.pi * j * t / YEAR_DAYS)]
    weekday = daily["date"].dt.weekday().to_numpy()  # 1 = Monday ... 7 = Sunday
    cols += [(weekday == d).astype(float) for d in range(1, 8)]
    return np.column_stack(cols)


def seasonal_normal(daily: pl.DataFrame, train_end) -> pl.Series:
    """Normal daily consumption in GW: yearly cycle (2 Fourier pairs) + day of the week.

    Fitted by least squares on the days up to train_end only (no leakage), then
    applied to every day of `daily` (columns date and target).
    """
    X = _design_matrix(daily)
    train = (daily["date"] <= train_end).to_numpy()
    beta = np.linalg.lstsq(X[train], daily["target"].to_numpy()[train], rcond=None)[0]
    return pl.Series("normal_gw", X @ beta)


def ar_forecast(params, history, h: int) -> list:
    """Recursive h-step forecast of an AR(p).

    params = [c, phi_1, ..., phi_p]; history = past anomalies, the last value is
    the most recent. Each forecast is appended to the history for the next step.
    """
    p = len(params) - 1
    hist = list(history)
    out = []
    for _ in range(h):
        nxt = params[0] + sum(params[j] * hist[-j] for j in range(1, p + 1))
        hist.append(nxt)
        out.append(nxt)
    return out

"""Reference forecasts: naive lag 7d and lag 2d.

Setup: on day D before 12:00, we forecast the 24 hours of D+1. Consumption is
only known up to D 11:00, so the last complete day is D-1, i.e. T-2 for a
target day T = D+1.
"""

import polars as pl


def naive_forecast(df: pl.DataFrame, lag_days: int) -> pl.DataFrame:
    """Forecast hour h of day T with the consumption of hour h on day T - lag_days.

    Respects the cutoff as soon as lag_days >= 2: day T - 2 is complete when T is
    forecast. The join is on (date, hour) in local time; the duplicated October
    hour is averaged. Adds the column f"pred_lag{lag_days}d".
    """
    if lag_days < 2:
        raise ValueError("lag_days must be at least 2: day T-1 is not known at the cutoff.")

    col = f"pred_lag{lag_days}d"
    past = (
        df.group_by("date", "hour")
        .agg(pl.col("target").mean().alias(col))
        .with_columns(pl.col("date").dt.offset_by(f"{lag_days}d"))
    )
    return df.join(past, on=["date", "hour"], how="left", maintain_order="left")
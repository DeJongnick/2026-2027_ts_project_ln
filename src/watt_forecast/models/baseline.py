"""Reference forecasts: naive lag 7d / lag 2d, and debiased RTE (flat or hourly, used in ntb/extension_rte_debiased).

Setup: on day D before 12:00, we forecast the 24 hours of D+1. Consumption is
only known up to D 11:00, so the last complete day is D-1, i.e. T-2 for a
target day T = D+1.
"""

from datetime import timedelta

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


def debias_rte(df: pl.DataFrame, window_days: int = 28) -> pl.DataFrame:
    """RTE day-ahead forecast corrected by its recent mean bias.

    For a target day T, the bias is the mean of (rte_forecast_j1_mw - target)
    over days T-29 to T-2: these are complete days, known at the cutoff on day
    T-1. Adds the column "rte_debiased_mw".
    """
    daily = (
        df.group_by("date")
        .agg((pl.col("rte_forecast_j1_mw") - pl.col("target")).mean().alias("daily_bias"))
        .sort("date")
    )
    # Shifting by 2 rows only means 2 days if no date is missing
    if (daily["date"].diff().drop_nulls() != timedelta(days=1)).any():
        raise ValueError("The date series is not continuous.")

    daily = daily.with_columns(
        pl.col("daily_bias").rolling_mean(window_days).shift(2).alias("bias")
    )
    return (
        df.join(daily.select("date", "bias"), on="date", how="left", maintain_order="left")
        .with_columns((pl.col("rte_forecast_j1_mw") - pl.col("bias")).alias("rte_debiased_mw"))
        .drop("bias")
    )


def debias_rte_hourly(df: pl.DataFrame, window_days: int = 28) -> pl.DataFrame:
    """RTE day-ahead forecast corrected by its recent mean bias at the same local hour.

    For a target hour h of day T, the bias is the mean of (rte_forecast_j1_mw -
    target) at hour h over days T-29 to T-2: complete days, known at the cutoff
    on day T-1. Days are put on a full date x hour grid, so a shift of 2 rows is
    2 days; a window with missing hours (switch to summer time, missing RTE)
    averages the available days (at least half of them). The duplicated October
    hour is averaged. Adds the column "rte_debiased_hourly_mw".
    """
    hourly = df.group_by("date", "hour").agg(
        (pl.col("rte_forecast_j1_mw") - pl.col("target")).mean().alias("hourly_bias")
    )
    dates = pl.DataFrame({"date": pl.date_range(df["date"].min(), df["date"].max(), "1d", eager=True)})
    hours = pl.DataFrame({"hour": pl.Series(range(24), dtype=df["hour"].dtype)})
    grid = (
        dates.join(hours, how="cross")
        .join(hourly, on=["date", "hour"], how="left")
        .sort("hour", "date")
        .with_columns(
            pl.col("hourly_bias")
            .rolling_mean(window_days, min_samples=window_days // 2)
            .shift(2)
            .over("hour")
            .alias("bias")
        )
    )
    return (
        df.join(grid.select("date", "hour", "bias"), on=["date", "hour"], how="left", maintain_order="left")
        .with_columns((pl.col("rte_forecast_j1_mw") - pl.col("bias")).alias("rte_debiased_hourly_mw"))
        .drop("bias")
    )

"""Forecast evaluation metrics."""

import polars as pl
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

PEAK_HOURS = [18, 19, 20]  # local start hours: 18:00 to 21:00
COLD_DAY_C = 5.0


def compute_metrics(df: pl.DataFrame, pred_col: str, actual_col: str = "target") -> dict:
    """MAE, RMSE, MAPE (%) and bias, with error = forecast - actual.

    Rows where the forecast is null are ignored.
    """
    df = df.drop_nulls(pred_col)
    y_true, y_pred = df[actual_col], df[pred_col]
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "bias": df.select((pl.col(pred_col) - pl.col(actual_col)).mean()).item(),
    }


def evaluate(df: pl.DataFrame, pred_col: str) -> pl.DataFrame:
    """Metrics on three scopes: global, evening peak, and evening peak on cold days.

    A day is cold when its mean temperature is below 5 °C.
    """
    df = df.with_columns(pl.col("temperature_c").mean().over("date").alias("daily_temp"))
    peak = df.filter(pl.col("hour").is_in(PEAK_HOURS))
    cold_peak = peak.filter(pl.col("daily_temp") < COLD_DAY_C)

    scopes = {
        "global": df,
        "evening_peak_18-20h": peak,
        "evening_peak_18-20h_cold_days": cold_peak,
    }
    return pl.DataFrame(
        [{"scope": name, **compute_metrics(data, pred_col)} for name, data in scopes.items()]
    )


def scorecard(df: pl.DataFrame, forecasts: dict, reference: str = "RTE raw") -> pl.DataFrame:
    """MAE (MW) of each forecast on the three scopes of `evaluate`, one row per forecast.

    `forecasts` maps a display name to a column. "vs_reference_(%)" is the relative gap of the global MAE
    to the forecast named `reference` (negative = better).
    """
    names = {
        "global": "global",
        "evening_peak_18-20h": "evening_peak_18-20h",
        "evening_peak_18-20h_cold_days": "evening_peak_cold_days",
    }
    rows = []
    for name, col in forecasts.items():
        mae = dict(evaluate(df, col).select("scope", "MAE").iter_rows())
        rows.append({"forecast": name, **{label: mae[scope] for scope, label in names.items()}})
    table = pl.DataFrame(rows)
    ref = table.filter(pl.col("forecast") == reference)["global"].item()
    return table.with_columns(
        ((pl.col("global") - ref) / ref * 100).round(1).alias("vs_reference_(%)"),
        pl.col(list(names.values())).round(0),
    )

def scorecard_daily(df: pl.DataFrame, forecasts: dict, reference: str = "RTE raw") -> pl.DataFrame:
    """MAE and bias (GW) of each daily forecast, one row per forecast, with the same "vs_reference_(%)"."""
    rows = []
    for name, col in forecasts.items():
        m = compute_metrics(df, col, actual_col="target")
        rows.append({"forecast": name, "MAE": m["MAE"], "bias": m["bias"]})
    table = pl.DataFrame(rows)
    ref = table.filter(pl.col("forecast") == reference)["MAE"].item()
    return table.with_columns(
        ((pl.col("MAE") - ref) / ref * 100).round(1).alias("vs_reference_(%)"),
        pl.col("MAE", "bias").round(2),
    )
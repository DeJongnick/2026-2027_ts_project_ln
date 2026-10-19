"""Forecast evaluation metrics."""

import polars as pl
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)


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

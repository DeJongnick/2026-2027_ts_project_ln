"""LightGBM day-ahead model, walk-forward validation, intervals and perturbation test.

One model for the 24 hours of D+1, trained on the feature table of
watt_forecast.features.make_features.
"""

import numpy as np
import polars as pl
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from watt_forecast.features import KEY_COLS, feature_cols


def fit_lgbm(X: pl.DataFrame, y: pl.Series, params: dict) -> LGBMRegressor:
    """Fit a LightGBM regressor (L2 loss) with the given hyperparameters.

    Fitted on NumPy arrays: the feature names are the columns of X, in order.
    """
    model = LGBMRegressor(**params, random_state=0, verbose=-1)
    return model.fit(X.to_numpy(), y.to_numpy())


def predict(model, X: pl.DataFrame) -> np.ndarray:
    """Predictions of any fitted model with a scikit-learn predict method."""
    return model.predict(X.to_numpy())


def fit_residual(X: pl.DataFrame, y: pl.Series, params: dict) -> LGBMRegressor:
    """Fit LightGBM on the correction of RTE's forecast: target = y - rte_forecast.

    The peaks come from RTE's forecast, the trees only learn the correction, so
    they do not have to extrapolate above the loads seen in training. Rows
    without RTE forecast are dropped. X must contain the column rte_forecast.
    """
    correction = y - X["rte_forecast"]
    keep = correction.is_not_null()
    return fit_lgbm(X.filter(keep), correction.filter(keep), params)


def predict_residual(model, X: pl.DataFrame) -> np.ndarray:
    """Final forecast of a residual model: rte_forecast + predicted correction.

    NaN where RTE's forecast is missing.
    """
    return X["rte_forecast"].to_numpy() + predict(model, X)


def walk_forward_predictions(df_features: pl.DataFrame, folds, params: dict, fit=fit_lgbm, predict_fn=predict) -> pl.DataFrame:
    """Walk-forward forecasts of every row of the folds.

    folds is a list of (start, end) dates, both included. Each fold trains on
    every row of df_features before start (rows without lag_336 are dropped)
    and predicts the block start-end. fit(X, y, params) returns a fitted model
    (default: LightGBM) and predict_fn(model, X) its forecasts. Returns
    datetime_utc, date, target, fold (index) and pred (null if missing).
    """
    cols = feature_cols(df_features)
    blocks = []
    for k, (start, end) in enumerate(folds):
        train = df_features.filter(pl.col("date") < start).drop_nulls("lag_336")
        test = df_features.filter(pl.col("date").is_between(start, end))
        model = fit(train.select(cols), train["target"], params)
        blocks.append(
            test.select("datetime_utc", "date", "target").with_columns(
                pl.lit(k).alias("fold"),
                pl.Series("pred", predict_fn(model, test.select(cols))).fill_nan(None),
            )
        )
    return pl.concat(blocks)


def walk_forward(df_features: pl.DataFrame, folds, params: dict, fit=fit_lgbm, predict_fn=predict) -> list[float]:
    """MAE (MW) of each chronological fold, from walk_forward_predictions.

    Hours without forecast are ignored. Pass the same folds to every model.
    """
    preds = walk_forward_predictions(df_features, folds, params, fit, predict_fn).drop_nulls("pred")
    return [
        mean_absolute_error(block["target"], block["pred"])
        for block in (preds.filter(pl.col("fold") == k) for k in range(len(folds)))
    ]


def conformal_intervals(residuals_valid, hours_valid, alpha: float, cold_valid=None) -> pl.DataFrame:
    """Split conformal interval: empirical quantiles of validation errors by local hour.

    residuals_valid = actual - forecast on the validation period (walk-forward
    forecasts, never the test). Returns, per hour (and per cold flag if
    cold_valid is given), q_low and q_high, the alpha/2 and 1 - alpha/2
    quantiles: the interval of a new forecast is [pred + q_low, pred + q_high].
    """
    data = pl.DataFrame({"hour": hours_valid, "residual": residuals_valid})
    keys = ["hour"]
    if cold_valid is not None:
        data = data.with_columns(pl.Series("cold", cold_valid))
        keys.append("cold")
    return (
        data.drop_nulls()
        .group_by(keys)
        .agg(
            pl.col("residual").quantile(alpha / 2, "linear").alias("q_low"),
            pl.col("residual").quantile(1 - alpha / 2, "linear").alias("q_high"),
            pl.len().alias("n"),
        )
        .sort(keys)
    )


def first_changed_row(make, df: pl.DataFrame, s: int):
    """Perturbation test: add 10,000 MW to target at row s, rebuild the features
    with make(df), and return the first row whose features change (None if none).

    With features that respect the cutoff, the answer must be s + 48.
    """
    bumped = df.with_columns(
        pl.when(pl.int_range(pl.len()) == s).then(pl.col("target") + 10_000).otherwise(pl.col("target"))
    )
    before = make(df).drop(KEY_COLS).to_numpy().astype(float)
    after = make(bumped).drop(KEY_COLS).to_numpy().astype(float)
    changed = np.flatnonzero(~np.isclose(before, after, equal_nan=True).all(axis=1))
    return int(changed[0]) if len(changed) else None

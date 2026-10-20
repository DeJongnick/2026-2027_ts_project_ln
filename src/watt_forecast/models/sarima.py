"""Day-ahead rolling forecasts of SARIMA(X) and Holt-Winters models.

The parameters are estimated once on the train period. For each target day T
(position i), the model is re-run with these frozen parameters on the data up
to T-2 (position i - h, h = 2), the last complete day at the cutoff on D = T-1,
and forecasts h steps ahead. Only the last forecast (day T) is kept.
"""

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def rolling_forecast(result, endog, exog, test_pos, h: int = 2, exog_future=None) -> np.ndarray:
    """h-step forecast of each test day with a fitted SARIMAX, without re-estimation.

    For position i, result.apply filters endog[:i-h+1] (days up to i - h) with the
    train parameters, so no value after the cutoff enters the state; then
    forecast(h) gives days i-h+1 to i. exog (or None) holds the observed
    regressors of the past days. exog_future holds, for each day, the regressors
    as known h days before (default: exog, i.e. perfectly known). Only the
    regressors of day i matter for its forecast: in a regression with SARIMA
    errors, y_i = x_i beta + u_i and u_i is forecast from past errors only.
    """
    exog_future = exog if exog_future is None else exog_future
    preds = []
    for i in test_pos:
        o = i - h + 1  # number of known days
        past = None if exog is None else exog[:o]
        future = None if exog is None else exog_future[o:i + 1]
        res = result.apply(endog[:o], exog=past)
        preds.append(res.forecast(h, exog=future)[-1])
    return np.array(preds)


def hw_rolling_forecast(result, endog, test_pos, h: int = 2) -> np.ndarray:
    """Same protocol for a fitted Holt-Winters (ExponentialSmoothing) model.

    The smoothing parameters and the initial states estimated on the train are
    frozen (initialization_method="known", optimized=False); the smoothing is
    re-run on endog[:i-h+1] and forecast h steps ahead.
    """
    model, p = result.model, result.params
    has_trend = model.trend is not None

    def frozen(name):
        return p[name] if has_trend else None

    preds = []
    for i in test_pos:
        refit = ExponentialSmoothing(
            endog[:i - h + 1],
            trend=model.trend,
            damped_trend=model.damped_trend,
            seasonal=model.seasonal,
            seasonal_periods=model.seasonal_periods,
            initialization_method="known",
            initial_level=p["initial_level"],
            initial_trend=frozen("initial_trend"),
            initial_seasonal=p["initial_seasons"],
        ).fit(
            smoothing_level=p["smoothing_level"],
            smoothing_trend=frozen("smoothing_trend"),
            smoothing_seasonal=p["smoothing_seasonal"],
            damping_trend=p["damping_trend"] if model.damped_trend else None,
            optimized=False,
        )
        preds.append(refit.forecast(h)[-1])
    return np.array(preds)
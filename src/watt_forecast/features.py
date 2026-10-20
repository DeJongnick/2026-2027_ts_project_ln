"""Calendar, temperature and lag features.

Daily regressors for SARIMAX (daily_exog) and the hourly feature table for
LightGBM (make_features).
"""

from datetime import date, timedelta

import polars as pl

HEATING_C = 15.0  # below: heating (course L3)
COOLING_C = 22.0  # above: air conditioning (course L3)

FIXED_HOLIDAYS = [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]

LAGS_H = [48, 72, 168, 336]  # all >= 48 h: known at the cutoff for every hour of D+1
TEMP_MODES = ["none", "persistence", "observed"]
KEY_COLS = ["datetime_utc", "date", "target"]  # identifiers and target, not features


def _easter(year: int) -> date:
    """Date of Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 19 * l) // 433
    month = (h + l - 7 * m + 90) // 25
    day = (h + l - 7 * m + 33 * month + 19) % 32
    return date(year, month, day)


def french_holidays(years) -> list[date]:
    """Sorted dates of the 11 French public holidays of the given years.

    8 fixed dates plus Easter Monday, Ascension (Easter + 39 days) and
    Whit Monday (Easter + 50 days).
    """
    days = []
    for year in years:
        easter = _easter(year)
        days += [date(year, m, d) for m, d in FIXED_HOLIDAYS]
        days += [easter + timedelta(days=n) for n in (1, 39, 50)]
    return sorted(days)


def daily_exog(daily: pl.DataFrame) -> pl.DataFrame:
    """Regressors of each day: heating and cooling degrees, public holiday (0/1).

    heating = max(0, 15 - T) and cooling = max(0, T - 22), with T the daily mean
    temperature (temperature_c). A missing temperature gives missing degrees.
    """
    years = range(daily["date"].min().year, daily["date"].max().year + 1)
    return daily.select(
        "date",
        (HEATING_C - pl.col("temperature_c")).clip(lower_bound=0).alias("heating"),
        (pl.col("temperature_c") - COOLING_C).clip(lower_bound=0).alias("cooling"),
        pl.col("date").is_in(french_holidays(years)).cast(pl.Int8).alias("holiday"),
    )


def make_features(df: pl.DataFrame, temp_mode: str = "persistence", with_rte: bool = False) -> pl.DataFrame:
    """Hourly feature table for the day-ahead model, from load_dataset().

    Returns KEY_COLS (datetime_utc, date, target) and the features.
    Cutoff: on day D before 12:00, load is known up to D 11:00 and we forecast
    D+1 00:00 to 23:00, so with one model for the 24 hours every past value must
    be at least 36 h old. Why each feature respects it:

    - lag_48, lag_72, lag_168, lag_336: target shifted by 48 h or more (rows are
      consecutive UTC hours, checked below);
    - roll_mean_24, roll_mean_168: means of target.shift(48), so the window ends
      48 h before the target hour (never a rolling mean without shift);
    - hour, weekday, month, holiday, day_after_holiday, day_before_holiday: local
      calendar of the target hour, known in advance;
    - temp_mode "persistence": temp_lag_48, temperature of the same hour 48 h
      before (known at the cutoff); "observed": temp_observed, temperature of the
      target hour, NOT available at the cutoff (perfect weather forecast, upper
      bound only); "none": no temperature;
    - with_rte: rte_forecast, RTE's day-ahead forecast of the target hour.
      Assumption to verify: RTE publishes it on D before 12:00.
    """
    if temp_mode not in TEMP_MODES:
        raise ValueError(f"temp_mode must be one of {TEMP_MODES}")
    if (df["datetime_utc"].diff().drop_nulls() != timedelta(hours=1)).any():
        raise ValueError("The rows must be consecutive UTC hours: shifting by k rows must mean k hours.")

    # One extra year on each side for the day before 1 January and after 31 December
    holidays = french_holidays(range(df["date"].min().year - 1, df["date"].max().year + 2))
    load = pl.col("target")

    features = [load.shift(k).alias(f"lag_{k}") for k in LAGS_H]
    features += [load.shift(48).rolling_mean(w).alias(f"roll_mean_{w}") for w in (24, 168)]
    features += [
        pl.col("hour", "weekday", "month"),
        pl.col("date").is_in(holidays).cast(pl.Int8).alias("holiday"),
        pl.col("date").dt.offset_by("-1d").is_in(holidays).cast(pl.Int8).alias("day_after_holiday"),
        pl.col("date").dt.offset_by("1d").is_in(holidays).cast(pl.Int8).alias("day_before_holiday"),
    ]
    if temp_mode == "persistence":
        features.append(pl.col("temperature_c").shift(48).alias("temp_lag_48"))
    elif temp_mode == "observed":
        features.append(pl.col("temperature_c").alias("temp_observed"))
    if with_rte:
        features.append(pl.col("rte_forecast_j1_mw").alias("rte_forecast"))
    return df.select(*KEY_COLS, *features)


def feature_cols(table: pl.DataFrame) -> list[str]:
    """Feature columns of a make_features table (everything but KEY_COLS)."""
    return [c for c in table.columns if c not in KEY_COLS]

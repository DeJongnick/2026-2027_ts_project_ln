"""Calendar and temperature features.

Daily regressors for SARIMAX (daily_exog): heating and cooling degrees and
French public holidays.
"""

from datetime import date, timedelta

import polars as pl

HEATING_C = 15.0  # below: heating (course L3)
COOLING_C = 22.0  # above: air conditioning (course L3)

FIXED_HOLIDAYS = [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]


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
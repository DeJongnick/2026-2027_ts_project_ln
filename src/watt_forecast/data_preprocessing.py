"""In-memory loading and preprocessing of the electricity consumption data."""

from pathlib import Path
import polars as pl

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
NATIONAL_CSV = DATA_DIR / "france_electricity_hourly.csv"
PREDICTIONS_DIR = DATA_DIR / "predictions"

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S%z"
LOCAL_TZ = "Europe/Paris"


def _parse_utc(col: str = "datetime_utc") -> pl.Expr:
    """Parse a string column as a UTC Datetime."""
    return pl.col(col).str.to_datetime(DATETIME_FORMAT, time_zone="UTC")


def _add_calendar(df: pl.DataFrame) -> pl.DataFrame:
    """Add local time and calendar variables (Europe/Paris)."""
    local = pl.col("datetime_local")
    return df.with_columns(
        pl.col("datetime_utc").dt.convert_time_zone(LOCAL_TZ).alias("datetime_local")
    ).with_columns(
        local.dt.date().alias("date"),
        local.dt.hour().alias("hour"),
        local.dt.weekday().alias("weekday"),
        local.dt.month().alias("month"),
    )


def load_national() -> pl.DataFrame:
    """Load the hourly national data (2019-2026) with RTE's day-ahead forecast.

    The variable to predict, the national load in MW (load_mw in the CSV), is
    renamed target. It is interpolated at the missing hour of each switch to
    winter time; the RTE forecast is left missing there.
    """
    return _add_calendar(
        pl.read_csv(NATIONAL_CSV)
        .rename({"load_mw": "target", "rte_forecast_d1_mw": "rte_forecast_j1_mw"})
        .with_columns(_parse_utc())
        .sort("datetime_utc")
        .with_columns(pl.col("target").interpolate())
    )


def load_dataset() -> pl.DataFrame:
    """Single entry point: hourly national series + RTE day-ahead forecast."""
    return (
        load_national()
        .select(
            "datetime_utc", "datetime_local", "date", "hour", "weekday", "month",
            "target", "temperature_c", "rte_forecast_j1_mw",
        )
    )


def load_daily() -> pl.DataFrame:
    """Daily series in local time: mean consumption, temperature and RTE forecast.

    target (the load) and rte_gw are daily means in GW. The first day is dropped if it has
    fewer than 23 hours (23 is a normal day at the switch to summer time).
    """
    daily = (
        load_dataset()
        .group_by("date")
        .agg(
            (pl.col("target").mean() / 1000).alias("target"),
            pl.col("temperature_c").mean(),
            (pl.col("rte_forecast_j1_mw").mean() / 1000).alias("rte_gw"),
            pl.len().alias("n_hours"),
        )
        .sort("date")
    )
    if daily["n_hours"][0] < 23:
        daily = daily.slice(1)
    return daily.drop("n_hours")


HOURLY_KEYS = ["datetime_fr", "target"]
DAILY_KEYS = ["date", "target"]
CSV_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def save_predictions(df: pl.DataFrame, forecasts: dict, name: str, keys: list = HOURLY_KEYS) -> None:
    """Write test forecasts to data/predictions/<name>.csv, read back by ntb/06_comparison.

    Only the keys and the forecasts are written. datetime_fr is the French local time with its UTC offset
    (taken from datetime_local), so the two 02:00 of the switch to winter time stay distinct. `forecasts`
    maps a display name to a column; each forecast is saved under its display name, with the spaces
    replaced by "_" (no column name contains a space).
    """
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    if "datetime_fr" in keys and "datetime_fr" not in df.columns:
        df = df.with_columns(pl.col("datetime_local").alias("datetime_fr"))
    df.select(*keys, *[pl.col(col).alias(n.replace(" ", "_")) for n, col in forecasts.items()]).write_csv(
        PREDICTIONS_DIR / f"{name}.csv", datetime_format=CSV_DATETIME_FORMAT
    )


def load_predictions(names: list, keys: list = HOURLY_KEYS) -> pl.DataFrame:
    """Join the forecasts saved by several notebooks on their first key (datetime_fr or date).

    For the hourly files, datetime_fr is read back in Europe/Paris and the columns needed by
    metrics.evaluate are rebuilt: date and hour in local time (as in _add_calendar), and temperature_c
    joined from load_dataset() on the instant (not on date + hour, ambiguous at the time changes).
    """
    tables = [pl.read_csv(PREDICTIONS_DIR / f"{n}.csv", try_parse_dates=True) for n in names]
    out = tables[0]
    for t in tables[1:]:
        out = out.join(t.drop(keys[1:]), on=keys[0], how="inner")
    if "datetime_fr" not in out.columns:
        return out
    local = pl.col("datetime_fr")
    temperature = load_dataset().select(pl.col("datetime_local").alias("datetime_fr"), "temperature_c")
    return (
        out.with_columns(local.dt.convert_time_zone(LOCAL_TZ))  # the CSV offsets are read back in UTC
        .with_columns(local.dt.date().alias("date"), local.dt.hour().alias("hour"))
        .join(temperature, on="datetime_fr", how="left", maintain_order="left")
        .select("datetime_fr", "date", "hour", "target", "temperature_c", pl.exclude(*HOURLY_KEYS, "date", "hour", "temperature_c"))
    )

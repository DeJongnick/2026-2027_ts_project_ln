"""Numbers shared by report.qmd and one_pager.qmd.

Nothing is fitted here: we read the dataset and the test forecasts saved by the notebooks (run 01 to 05 and the
extension first), and compute the same scorecards as 06_comparison and 07_executive_summary. Figures come from
07_executive_summary (reports/figures/).
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import polars as pl

from watt_forecast.data_preprocessing import DAILY_KEYS, HOURLY_KEYS, load_dataset, load_predictions
from watt_forecast.metrics import COLD_DAY_C, PEAK_HOURS, compute_metrics, scorecard, scorecard_daily

TEST_START, TEST_END = date(2025, 7, 1), date(2026, 6, 30)
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}

FORECASTS = {  # display name -> column saved in data/predictions/, same as 07
    "Lag 7d": "Lag_7d",
    "LightGBM, no RTE": "LightGBM_persistence",
    "RTE raw": "RTE_raw",
    "LightGBM residual": "LightGBM_residual",
    "RTE debiased by hour": "RTE_debiased_by_hour",
}
FINAL, CANDIDATE = "LightGBM residual", "RTE debiased by hour"
KPI = ["RTE raw", FINAL, CANDIDATE]

# Walk-forward folds (Jul 2024 to Jun 2025): mean MAE in MW, not saved to disk, read from the notebooks
FOLDS_MAE = {"RTE raw": 1362, "LightGBM residual": 860, "RTE debiased by hour": 830}

df = load_dataset()
test = load_predictions(["01_baseline", "04_lightgbm", "05_final_model", "extension_rte_debiased", "05_intervals"])
assert test.height == 365 * 24, "the hourly test period should have 8,760 hours"
assert (test["date"].min(), test["date"].max()) == (TEST_START, TEST_END), "test year of 01 to 06"

test = test.with_columns(
    pl.col("temperature_c").mean().over("date").alias("daily_temp"),
    pl.col("date").dt.month().replace_strict(SEASONS).alias("season"),
)
COLD_DAY = pl.col("daily_temp") < COLD_DAY_C  # observed: for evaluation only
PEAK = pl.col("hour").is_in(PEAK_HOURS)
N_COLD_PEAK_HOURS = test.filter(PEAK & COLD_DAY).height
N_COLD_DAYS = test.filter(COLD_DAY)["date"].n_unique()

score = scorecard(test, FORECASTS)
mae = {row["forecast"]: row for row in score.iter_rows(named=True)}
gain = {name: -mae[name]["vs_reference_(%)"] for name in FORECASTS}  # % below RTE raw
cold_gain = {name: 100 * (1 - mae[name]["evening_peak_cold_days"] / mae["RTE raw"]["evening_peak_cold_days"]) for name in KPI}

# All hourly forecasts of 01 to 05 and the extension (table of 06_comparison)
EXTRA = ["date", "hour", "temperature_c", "daily_temp", "season", "lo80", "hi80", "lo95", "hi95"]
all_hourly = scorecard(test, {c: c for c in test.columns if c not in HOURLY_KEYS + EXTRA}, reference="RTE_raw").sort("global")

daily = load_predictions(["02_ar", "03_sarima"], keys=DAILY_KEYS)
daily_score = scorecard_daily(daily, {c: c for c in daily.columns if c not in DAILY_KEYS}, reference="RTE_raw").sort("MAE")

# RTE's bias
since_2023 = df.filter(pl.col("date") >= date(2023, 1, 1)).drop_nulls("rte_forecast_j1_mw")
bias_2023 = (since_2023["rte_forecast_j1_mw"] - since_2023["target"]).mean()
rte_test = compute_metrics(test, FORECASTS["RTE raw"])
final_test = compute_metrics(test, FORECASTS[FINAL])

# Coverage of the conformal intervals of the final model (as in 05 and 07)
COVERAGE_SCOPES = {
    "All hours": pl.lit(True),
    **{season.capitalize(): pl.col("season") == season for season in ["summer", "autumn", "winter", "spring"]},
    "Cold days": COLD_DAY,
    "Cold evening peaks": PEAK & COLD_DAY,
}
_rows = []
for _name, _condition in COVERAGE_SCOPES.items():
    _part = test.filter(_condition).drop_nulls(FORECASTS[FINAL])
    _row = {"scope": _name, "hours": _part.height}
    for _tag in (80, 95):
        _row[f"coverage_{_tag}_%"] = 100 * _part["target"].is_between(_part[f"lo{_tag}"], _part[f"hi{_tag}"]).mean()
        _row[f"width_{_tag}_(MW)"] = (_part[f"hi{_tag}"] - _part[f"lo{_tag}"]).mean()
    _rows.append(_row)
coverage = pl.DataFrame(_rows)
cov = {row["scope"]: row for row in coverage.iter_rows(named=True)}


def gw(x, digits=2):
    return f"{x / 1000:.{digits}f} GW"


def mw(x):
    return f"{x:,.0f} MW"


def pct(x):
    return f"{x:.0f} %"


def md_table(table):
    """Markdown table of a polars DataFrame (values already formatted)."""
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * table.width]
    return "\n".join(lines + ["| " + " | ".join(str(v) for v in row) + " |" for row in table.iter_rows()])


def kpi_table():
    """MAE of RTE raw, the final model and the candidate on the three scopes."""
    return md_table(pl.DataFrame([
        {
            "Forecast": name,
            "All hours": mw(mae[name]["global"]),
            "Evening peak": mw(mae[name]["evening_peak_18-20h"]),
            "Cold evening peaks": mw(mae[name]["evening_peak_cold_days"]),
            "vs RTE raw": "reference" if name == "RTE raw" else f"{-gain[name]:+.1f} %",
        }
        for name in KPI
    ]))

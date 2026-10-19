# Day-ahead forecasting of French electricity consumption

Final project for the Time Series Analysis & Forecasting course. Every day D before 12:00, a utility's trading desk
needs the consumption of each hour of D+1 to buy energy on the day-ahead market. The goal is to build a forecast for
that and compare it with RTE's public forecast.

The manager's three questions:

1. Can we do better than RTE's public forecast, and by how much?
2. How uncertain is the forecast, and when does it fail?
3. What should we deploy, and what are the risks?

## Setup

- **Cutoff**: D at 12:00 (Paris time). Consumption is known up to D 11:00, so for a target day T = D+1 the last
  complete day is T-2.
- **Test year**: 1 July 2025 to 30 June 2026. All choices are made before, on 4 quarterly walk-forward folds
  (July 2024 to June 2025).
- **Metric**: MAE in MW, overall, on the evening peak (18-20h) and on the evening peak of cold days (below 5 °C).
- **Temperature**: the dataset has the observed temperature, not a forecast. Using the temperature of D+1 would mean
  a perfect weather forecast, so it can only be shown as an upper bound.

## Installation

Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download `france_electricity_hourly.csv` from the
[course Google Drive folder](https://drive.google.com/drive/folders/1GY9IOY_7zdZpAm5xfbzSkFa-DunAA3Ag?usp=sharing)
and put it in `data/`. It already includes RTE's forecast, no other data is needed.

# Day-ahead forecasting of French electricity consumption

Final project for the Time Series Analysis & Forecasting course. Every day D before 12:00, a utility's trading desk
needs the consumption of each hour of D+1 to buy energy on the day-ahead market. We built a forecast for that and
compared it with RTE's public forecast.

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
  a perfect weather forecast, so we only show it as an upper bound.

## Results

MAE in MW on the test year:

| Forecast | Global | Evening peak | Cold evening peak | vs RTE raw |
|---|---|---|---|---|
| Lag 7d (naive) | 3,425 | 3,612 | 9,225 | +138.0 % |
| LightGBM, no RTE (temperature of T-2) | 2,010 | 2,036 | 4,466 | +39.7 % |
| RTE raw | 1,439 | 1,348 | 2,307 | 0 % |
| LightGBM residual (load - RTE) | 910 | 987 | 1,906 | -36.8 % |
| **RTE debiased by hour (extension)** | **819** | **943** | **1,528** | **-43.1 %** |

What we learnt along the way:

- **Classical models.** AR, SARIMA, Holt-Winters and SARIMAX work on daily means and stay well behind RTE. The best
  one is the SARIMAX with T-2 temperature and holidays, at 1.65 GW of daily MAE against 1.16 GW for RTE. Even with
  the observed temperature it only matches RTE.
- **Without RTE**, our best model is a LightGBM at 2,010 MW. To beat RTE we have to correct its forecast, not
  replace it.
- **RTE's forecast is biased.** It underestimates by about 1,100 MW since 2023, and the bias depends on the hour.
  The LightGBM residual learns this correction. The extension (`extension_rte_debiased`) shows that a simple rule does it even better:
  RTE minus its mean error at the same hour over T-29 to T-2. It is also the best on the folds (830 MW vs 860 MW).

## Answers

1. **Yes, by correcting RTE.** The LightGBM residual is 36.8 % below RTE raw on the test, and the hourly debiasing
   rule 43.1 % below.
2. **Uncertainty.** Our conformal intervals at 80 % and 95 % are about 2.4 and 4.2 GW wide. They cover 76 % and 94 %
   of test hours, but only 51 % and 73 % on cold evening peaks, so they need to be wider on cold days. The model
   fails when RTE fails: Christmas week, holidays, cold days, some weekends.
3. **Deploy.** The LightGBM residual is our final model (`05`). The hourly debiasing rule is the simplest and best
   candidate, to be validated in production alongside it. The selection rule of `05` (best mean MAE on the folds)
   picks among the LightGBM models. The debiasing rule came later, as an extension. It does better on the folds
   (830 vs 860 MW) and on the test, but it has no prediction intervals. That is why we propose it as a candidate
   to validate in production alongside `05`, not as the final model. Risks:
   - We assumed RTE's forecast is published before 12:00 on D. This is not checked yet. If it is late, fall back
     to the LightGBM without RTE.
   - RTE's bias can change suddenly. The rule takes about a month (28-day window) to follow it, and the LightGBM
     needs regular retraining. Monitor the bias.
   - We used consolidated data, but at the cutoff only real-time data is available.
   - Extreme cold and unusual holidays are rare in the data. Keep a manual override and a bigger margin there.

## Repository

```
data/                          france_electricity_hourly.csv and predictions/ (test forecasts, CSV), both not versioned
src/watt_forecast/
  data_preprocessing.py        loading, hourly and daily series, save/load of test forecasts
  features.py                  holidays, SARIMAX regressors, LightGBM features
  metrics.py                   MAE, RMSE, MAPE, bias, evening-peak scopes, scorecard
  models/                      baseline.py (naive, RTE debiasing), ar.py, sarima.py, gbm.py (LightGBM, conformal)
ntb/
  00_eda                       exploration of the series and of RTE's error
  01_baseline                  naive baselines vs RTE
  02_ar, 03_sarima             daily models: AR, SARIMA, Holt-Winters, SARIMAX
  04_lightgbm                  hourly LightGBM, walk-forward
  05_final_model               final model, prediction intervals, failures, deployment risks
  06_comparison                all models of 01 to 05 and the extension in one table
  extension_rte_debiased       extension (bonus): debiasing RTE's forecast, best result
reports/
  report.qmd                   full report of the project (Quarto source of the HTML)
  one_pager.qmd                one-page summary for management (Quarto source of the PDF)
  _quarto.yml                  Quarto project: what to render, where, how
  code/07_executive_summary    storyline for management, writes the figures below
  code/report_data.py          numbers of the two reports, read from data/predictions/
  figures/                     figures of 07_executive_summary (PNG)
  output/                      report.html and one_pager.pdf, written by quarto render
```

Notebooks 01 to 05 follow the same layout: a verdict, then setup, model, results, where it fails, conclusion and an
appendix. Run 01 to 05 and the extension before 06 and 07 (in reports/code/), since they read the saved forecasts.

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

## Reports

`reports/output/report.html` explains the whole work, `reports/output/one_pager.pdf` sums it up on one page. Both are built with
[Quarto](https://quarto.org), installed by `requirements.txt` (`quarto-cli`, which ships Typst for the PDF, so no
LaTeX is needed). The numbers are computed at render time from the saved forecasts, so run the notebooks 01 to 05,
the extension and 07 first, then:

```bash
cd reports
quarto render            # output/report.html and output/one_pager.pdf
```

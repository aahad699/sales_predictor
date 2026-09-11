# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "a89b98b2-b7b2-477d-bc9c-9005e3b396aa",
# META       "default_lakehouse_name": "Gold",
# META       "default_lakehouse_workspace_id": "1dca1c65-3ce2-483e-a7fb-baf7fca28e9f",
# META       "known_lakehouses": [
# META         { "id": "a89b98b2-b7b2-477d-bc9c-9005e3b396aa" }
# META       ]
# META     }
# META   }
# META }
# MARKDOWN ********************

#  # Revenue forecasting on the Gold layer (weekly + monthly)
#
#  ```
#  Bronze (sales csv) Ã¢â€â‚¬Silver_NotebookÃ¢â€â‚¬Ã¢â€“Â¶ Silver.sales_silver Ã¢â€â‚¬Gold_NotebookÃ¢â€â‚¬Ã¢â€“Â¶ Gold.factsales_gold + dims
#                                                                                     Ã¢â€â€š  this notebook
#                                                                                     Ã¢â€“Â¼
#          Gold.revenue_forecast_gold   daily forecast (latest run)          Gold.revenue_weekly_gold    actual + forecast per ISO week
#          Gold.forecast_backtest_gold  benchmark of every candidate model   Gold.revenue_monthly_gold   actual + forecast per month
#          Gold.revenue_forecast_history_gold  monthly forecast of every run (append) for accuracy tracking
#          MLflow experiment `sales_revenue_forecast` + registered model(s)
#  ```
#
#  **Two ways to run it Ã¢â‚¬â€œ both read the *live* lakehouse tables, nothing is copied**
#
#  | | How | What happens |
#  |---|---|---|
#  | **Fabric** (portal, pipeline, or VS Code with the *Fabric Data Engineering* extension and the **Microsoft Fabric Runtime** kernel) | `spark` is available Ã¢â€ â€™ `RUN_MODE = "fabric-spark"` | aggregation runs in Spark, tables are written with `saveAsTable`, MLflow logs to the workspace experiment |
#  | **Local VS Code** (any Python kernel, no Spark) | `spark` is missing Ã¢â€ â€™ `RUN_MODE = "local"` | reads `factsales_gold` straight from OneLake with `deltalake` + `azure-identity` (Delta protocol, always the latest committed version), writes the result tables back to OneLake, logs MLflow to Fabric through `synapseml-mlflow` (falls back to `./mlruns`) |
#
#  Local setup: `pip install -r requirements-local.txt`, sign in once with `az login` (or let the browser prompt appear), open this file in VS Code, pick your Python kernel and run all. Details in `README_VSCode.md`.
#
#  **What it does**
#  1. Builds the daily revenue series (`Quantity Ãƒâ€” UnitPrice + Tax`).
#  2. Benchmarks 9 candidate forecasters with rolling-origin back-tests (train up to a month-end cutoff, forecast the next 3 months) and scores them on **weekly** and **monthly** totals: MAE, RMSE, MAPE, sMAPE, WAPE, Accuracy (= 100 Ã¢Ë†â€™ WAPE) and bias.
#  3. Picks the champion (lowest average weekly sMAPE), retrains on all history, forecasts the rest of the current month + `HORIZON_MONTHS` full months.
#  4. Writes daily / weekly / monthly tables + the benchmark table, logs runs, metrics and models to MLflow.
#
#  **Pipeline note** Ã¢â‚¬â€œ Prophet is not in the default Fabric runtime. Interactively the first cell installs it; for pipeline runs attach an *Environment* that contains `prophet`, or add the notebook-activity parameter `_inlineInstallationEnabled = true` (Boolean).
#
#
#

# PARAMETERS CELL ********************

# ---------------- Parameters (mark as "parameter cell" to override from a pipeline) ----------------
WORKSPACE_ID      = "1dca1c65-3ce2-483e-a7fb-baf7fca28e9f"   # used by local mode (OneLake path) and MLflow-to-Fabric
GOLD_LAKEHOUSE_ID = "a89b98b2-b7b2-477d-bc9c-9005e3b396aa"
LAKEHOUSE_NAME    = "Gold"
SCHEMA            = "dbo"

FACT_TABLE        = "factsales_gold"          # input  (Gold_Notebook)
FORECAST_TABLE    = "revenue_forecast_gold"   # output: daily forecast of the latest run
WEEKLY_TABLE      = "revenue_weekly_gold"     # output: actual + forecast revenue per ISO week
MONTHLY_TABLE     = "revenue_monthly_gold"    # output: actual + forecast revenue per month
BACKTEST_TABLE    = "forecast_backtest_gold"  # output: benchmark metrics per model / fold / grain
HISTORY_TABLE     = "revenue_forecast_history_gold"  # output: monthly forecast of EVERY run (append) -> track accuracy over time
KEEP_FORECAST_HISTORY = True

EXPERIMENT_NAME   = "sales_revenue_forecast"
REGISTERED_MODEL  = "sales_revenue_forecaster"   # prefix of the registered ML model item(s)

INCLUDE_TAX       = True    # Revenue = Quantity * UnitPrice + Tax
HORIZON_MONTHS    = 3       # forecast the rest of the current month + this many full months
BACKTEST_FOLDS    = 3       # rolling-origin folds (each = train to a month-end, forecast the next BACKTEST_HORIZON_MONTHS)
BACKTEST_HORIZON_MONTHS = 3
INTERVAL_WIDTH    = 0.90    # prediction-interval width

# local (VS Code python kernel) only
LOCAL_MLFLOW_TO_FABRIC = True           # log locally by default; set True only after Fabric MLflow workspace access is granted
LOCAL_MLFLOW_FALLBACK  = "file:./mlruns" # used when the plugin is not installed



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

#  ## 1. Connect to the Gold lakehouse (Fabric Spark or OneLake from your laptop)
#
#

# CELL ********************

# ---- MLflow tracking: Fabric experiment in both modes when possible ----
import mlflow

if RUN_MODE == "local":
    tracking = None
    if LOCAL_MLFLOW_TO_FABRIC:
        try:
            from fabric.analytics.environment.credentials import SetFabricAnalyticsDefaultTokenCredentialsGlobally
            from azure.identity import DefaultAzureCredential
            SetFabricAnalyticsDefaultTokenCredentialsGlobally(
                credential=DefaultAzureCredential(exclude_interactive_browser_credential=False))
            tracking = f"sds://api.fabric.microsoft.com/v1/workspaces/{WORKSPACE_ID}/mlflow"
        except ImportError:
            print("synapseml-mlflow not installed -> logging to", LOCAL_MLFLOW_FALLBACK)
    if tracking is None:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        tracking = LOCAL_MLFLOW_FALLBACK
    mlflow.set_tracking_uri(tracking)

# Connecting to Fabric does not prove the current identity can access its MLflow API.
# In a local kernel, preserve the forecast run by falling back when that API rejects us.
try:
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception as exc:
    if RUN_MODE != "local" or mlflow.get_tracking_uri() == LOCAL_MLFLOW_FALLBACK:
        raise
    print(f"Fabric MLflow unavailable ({exc}); logging locally to {LOCAL_MLFLOW_FALLBACK}")
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(LOCAL_MLFLOW_FALLBACK)
    mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.autolog(disable=True)          # everything is logged explicitly
print("MLflow tracking URI:", mlflow.get_tracking_uri())



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

#  ## 2. Daily, weekly and monthly revenue history
#
#

# CELL ********************

 [markdown]
# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import logging
from scipy.stats import norm
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").disabled = True
Z = float(norm.ppf(1 - (1 - INTERVAL_WIDTH) / 2))

PROPHET_PARAMS = dict(growth="linear", yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                      changepoint_prior_scale=0.5, changepoint_range=0.9, interval_width=INTERVAL_WIDTH)
# Lagged growth only: level-vs-trend and calendar features were tested and hurt (they are learned from a single prior year).
# growth_mean4 is left out on purpose: it is exactly the mean of growth_lag1..4 (perfect collinearity, identical forecasts without it).
# Remaining features are only mildly correlated (adjacent lags about -0.4, VIF <= 1.7).
ML_FEATURE_NAMES = ["growth_lag1", "growth_lag2", "growth_lag3", "growth_lag4", "growth_lag8"]


def weekly_bins(series):
    # complete 7-day bins ending on the last date of the series (used for modelling, not for reporting)
    anchor = "W-" + ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][series.index[-1].dayofweek]
    w = series.resample(anchor).sum()
    if (w.index[0] - pd.Timedelta(days=6)) < series.index[0]:
        w = w.iloc[1:]
    return w

def weekly_to_daily(point, lower, upper, start, horizon):
    idx = pd.date_range(start, periods=horizon, freq="D")
    spread = lambda x: np.repeat(np.asarray(x, float) / 7.0, 7)[:horizon]
    return pd.DataFrame({"yhat": spread(point), "yhat_lower": spread(lower), "yhat_upper": spread(upper)}, index=idx).clip(lower=0.0)

def horizon_log_std(weekly, max_h=52, min_obs=10):
    # empirical std of the h-week change in log revenue, h = 1..max_h (a random walk would grow like sqrt(h);
    # weekly totals are noisy but that noise is transient, so the measured growth is much slower)
    l = np.log1p(weekly); out = []
    for h in range(1, max_h + 1):
        d = (l.shift(-h) - l).dropna()
        out.append(float(d.std()) if len(d) >= min_obs else (out[-1] * np.sqrt(h / len(out)) if out else 0.3))
    return np.maximum.accumulate(np.array(out))            # non-decreasing in h

def growth_band(point, hstd, horizon_weeks):
    s = np.resize(hstd, horizon_weeks) if horizon_weeks > len(hstd) else hstd[:horizon_weeks]
    return point * np.exp(-Z * s), point * np.exp(Z * s)


# ---- baselines --------------------------------------------------------------------------------
def make_naive(n_weeks):
    def fit(series):
        w = weekly_bins(series)
        return {"level": float(w.iloc[-n_weeks:].mean()), "hstd": horizon_log_std(w)}
    def predict(model, start, horizon):
        n = int(np.ceil(horizon / 7)); point = np.repeat(model["level"], n)
        lo, hi = growth_band(point, model["hstd"], n)
        return weekly_to_daily(point, lo, hi, start, horizon)
    return fit, predict


# ---- Holt damped trend (weekly) -----------------------------------------------------------------
def fit_holt(series):
    return ExponentialSmoothing(weekly_bins(series).values, trend="add", damped_trend=True).fit(optimized=True)

def predict_holt(model, start, horizon):
    n = int(np.ceil(horizon / 7))
    point = np.clip(model.forecast(n), 0, None)
    sims = model.simulate(n, repetitions=500, error="add", random_state=42)
    lo = np.clip(np.quantile(sims, (1 - INTERVAL_WIDTH) / 2, axis=1), 0, None)
    hi = np.quantile(sims, 1 - (1 - INTERVAL_WIDTH) / 2, axis=1)
    return weekly_to_daily(point, lo, hi, start, horizon)


# ---- Prophet (daily) ------------------------------------------------------------------------------
def fit_prophet(series):
    m = Prophet(**PROPHET_PARAMS)
    m.fit(pd.DataFrame({"ds": series.index, "y": series.values}))
    return m

def predict_prophet(model, start, horizon):
    future = pd.DataFrame({"ds": pd.date_range(start, periods=horizon, freq="D")})
    return model.predict(future).set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]].clip(lower=0.0)


# ---- feature-based ML models (weekly log-growth target, StandardScaler + estimator pipeline) ----------
def ml_features(logy):
    g = logy.diff()
    X = pd.DataFrame(index=logy.index)
    for lag in (1, 2, 3, 4, 8):
        X[f"growth_lag{lag}"] = g.shift(lag)
    X["growth_mean4"] = g.shift(1).rolling(4).mean()
    X["growth_mean13"] = g.shift(1).rolling(13).mean()
    X["level_vs_13w"] = logy.shift(1) - logy.shift(1).rolling(13).mean()       # available for experiments
    woy = X.index.isocalendar().week.astype(float).values
    X["woy_sin"], X["woy_cos"] = np.sin(2 * np.pi * woy / 52.18), np.cos(2 * np.pi * woy / 52.18)
    X["month"] = X.index.month
    return X[ML_FEATURE_NAMES], g

def make_ml(estimator_factory):
    def fit(series):
        logy = np.log1p(weekly_bins(series))
        X, g = ml_features(logy)
        mask = X.notna().all(axis=1) & g.notna()
        pipe = Pipeline([("scale", StandardScaler()), ("model", estimator_factory())])
        pipe.fit(X[mask], g[mask])
        return {"pipe": pipe, "logy": logy, "hstd": horizon_log_std(np.expm1(logy))}
    def predict(model, start, horizon):
        logy, pipe = model["logy"].copy(), model["pipe"]
        n = int(np.ceil(horizon / 7)); preds = []
        for _ in range(n):                                   # recursive multi-step
            nxt = logy.index[-1] + pd.Timedelta(weeks=1)
            logy.loc[nxt] = np.nan
            X, _ = ml_features(logy)
            logy.loc[nxt] = logy.iloc[-2] + float(pipe.predict(X.iloc[[-1]])[0])
            preds.append(logy.iloc[-1])
        point = np.expm1(np.array(preds))
        lo, hi = growth_band(point, model["hstd"], n)
        return weekly_to_daily(point, lo, hi, start, horizon)
    return fit, predict


COMPONENTS = {
    "naive_last_week": make_naive(1),
    "moving_avg_4w":   make_naive(4),
    "holt_damped":     (fit_holt, predict_holt),
    "prophet":         (fit_prophet, predict_prophet),
    "ridge_lags":      make_ml(lambda: Ridge(alpha=1.0)),
    "lightgbm_lags":   make_ml(lambda: LGBMRegressor(n_estimators=100, learning_rate=0.02, num_leaves=3,
                                                     min_child_samples=10, verbose=-1, random_state=42)),
}
CANDIDATES = {**{name: [name] for name in COMPONENTS},
              "prophet_holt_blend":  ["prophet", "holt_damped"],
              "prophet_ridge_blend": ["prophet", "ridge_lags"],
              "prophet_holt_ridge_blend": ["prophet", "holt_damped", "ridge_lags"]}

def blend(component_preds, names):
    return sum(component_preds[n] for n in names) / len(names)


def score(actual, forecast):
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    err = f - a
    wape = 100 * np.sum(np.abs(err)) / max(np.sum(np.abs(a)), 1e-9)
    return {"MAE":  float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAPE": float(100 * np.mean(np.abs(err) / np.maximum(np.abs(a), 1e-9))),
            "sMAPE": float(100 * np.mean(2 * np.abs(err) / np.maximum(np.abs(a) + np.abs(f), 1e-9))),
            "WAPE": float(wape),
            "AccuracyPct": float(max(0.0, 100 - wape)),
            "BiasPct": float(100 * np.sum(err) / max(np.sum(a), 1e-9))}



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

#  ## 4. Benchmark: rolling-origin back-test on weekly and monthly totals
#  Each fold trains on everything up to a month-end cutoff and forecasts the next 3 calendar months; the folds are the last three quarters of history. The champion is the lowest average **weekly sMAPE** (weekly has ~13 points per fold, monthly only 3, so it is the more stable criterion).
#
#

# CELL ********************

 [markdown]
# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from datetime import datetime, timezone

RUN_TS = datetime.now(timezone.utc)
FORECAST_START = LAST_DATE + pd.Timedelta(days=1)

def log_component_model(comp, model):
    try:
        name = f"{REGISTERED_MODEL}_{comp}"
        if comp == "prophet":
            mlflow.prophet.log_model(model, artifact_path=comp, registered_model_name=name)
        elif comp == "holt_damped":
            mlflow.statsmodels.log_model(model, artifact_path=comp, registered_model_name=name)
        elif comp.endswith("_lags"):
            mlflow.sklearn.log_model(model["pipe"], artifact_path=comp, registered_model_name=name)
        else:
            mlflow.log_dict({"level": model["level"]}, f"{comp}.json")
    except Exception as e:      # a flavour problem must not fail the pipeline
        print(f"WARNING: could not log {comp} to MLflow: {e}")

with mlflow.start_run(run_name=f"champion_{CHAMPION}") as run:
    fitted, preds = {}, {}
    for comp in CANDIDATES[CHAMPION]:
        fit_fn, predict_fn = COMPONENTS[comp]
        fitted[comp] = fit_fn(daily)
        preds[comp]  = predict_fn(fitted[comp], FORECAST_START, HORIZON_DAYS)
    fc = blend(preds, CANDIDATES[CHAMPION]).rename(columns={"yhat": "ForecastRevenue", "yhat_lower": "ForecastLower", "yhat_upper": "ForecastUpper"})

    # daily table
    forecast_daily = fc.reset_index().rename(columns={"index": "ForecastDate", "ds": "ForecastDate"})
    forecast_daily.insert(1, "HorizonDay", np.arange(1, HORIZON_DAYS + 1))
    forecast_daily["Year"], forecast_daily["Month"] = forecast_daily["ForecastDate"].dt.year, forecast_daily["ForecastDate"].dt.month
    forecast_daily["yyyymm"]  = forecast_daily["ForecastDate"].dt.strftime("%Y%m")
    forecast_daily["mmmyyyy"] = forecast_daily["ForecastDate"].dt.strftime("%b-%Y")

    # weekly / monthly tables = actual history + forecast, one row per period (a period can hold both if it straddles LAST_DATE)
    def combined(grain):
        f = to_periods(fc, grain).rename(columns={"Days": "ForecastDays"})
        out = actual_weekly.join(f, how="outer") if grain == "weekly" else actual_monthly.join(f, how="outer")
        out = out.fillna({"ActualRevenue": 0.0, "ActualDays": 0, "ForecastRevenue": 0.0, "ForecastLower": 0.0, "ForecastUpper": 0.0, "ForecastDays": 0})
        out["ActualDays"], out["ForecastDays"] = out["ActualDays"].astype(int), out["ForecastDays"].astype(int)
        out = out.reset_index()
        if grain == "weekly":
            out["PeriodEnd"] = out["PeriodStart"] + pd.Timedelta(days=6)
            iso = out["PeriodStart"].dt.isocalendar()
            out["IsoYear"], out["IsoWeek"] = iso["year"].astype(int), iso["week"].astype(int)
            out["YearWeek"] = out["IsoYear"].astype(str) + "-W" + out["IsoWeek"].astype(str).str.zfill(2)
        else:
            out["PeriodEnd"] = out["PeriodStart"] + pd.offsets.MonthEnd(0)
            out["Year"], out["Month"] = out["PeriodStart"].dt.year, out["PeriodStart"].dt.month
            out["yyyymm"], out["mmmyyyy"] = out["PeriodStart"].dt.strftime("%Y%m"), out["PeriodStart"].dt.strftime("%b-%Y")
        out["TotalRevenue"] = out["ActualRevenue"] + out["ForecastRevenue"]          # continuous series for charts
        out["RecordType"] = np.select([out["ForecastDays"].eq(0), out["ActualDays"].eq(0)], ["actual", "forecast"], "mixed")
        return out

    forecast_weekly, forecast_monthly = combined("weekly"), combined("monthly")

    for df_ in (forecast_daily, forecast_weekly, forecast_monthly):
        df_["Model"], df_["HistoryEndDate"], df_["ModelRunTS"], df_["MLflowRunId"], df_["RunMode"] = CHAMPION, LAST_DATE, RUN_TS, run.info.run_id, RUN_MODE

    mlflow.log_params({"model": CHAMPION, "components": "+".join(CANDIDATES[CHAMPION]), "include_tax": INCLUDE_TAX,
                       "history_start": str(daily.index.min().date()), "history_end": str(LAST_DATE.date()),
                       "forecast_end": str(FORECAST_END.date()), "horizon_days": HORIZON_DAYS, "interval_width": INTERVAL_WIDTH, "run_mode": RUN_MODE})
    for grain in ("weekly", "monthly"):
        mlflow.log_metrics({f"backtest_{grain}_{m}": float(v) for m, v in leaderboard.loc[CHAMPION, grain].items()})
    mlflow.log_metric("forecast_total_revenue", float(fc["ForecastRevenue"].sum()))
    for name, df_ in (("forecast_daily", forecast_daily), ("forecast_weekly", forecast_weekly), ("forecast_monthly", forecast_monthly), ("backtest", backtest)):
        mlflow.log_text(df_.to_csv(index=False), f"{name}.csv")
    for comp, model in fitted.items():
        log_component_model(comp, model)

print(f"champion = {CHAMPION} | forecast {FORECAST_START.date()} -> {FORECAST_END.date()} | expected revenue {fc['ForecastRevenue'].sum():,.0f}\n")
cols = ["PeriodStart", "PeriodEnd", "RecordType", "ActualRevenue", "ForecastRevenue", "ForecastLower", "ForecastUpper", "TotalRevenue"]
print("Weekly (last 4 actual weeks + forecast):");  print(forecast_weekly[forecast_weekly["PeriodEnd"] >= LAST_DATE - pd.Timedelta(weeks=4)][cols].round(0).to_string(index=False))
print("\nMonthly (last 3 actual months + forecast):"); print(forecast_monthly[forecast_monthly["PeriodEnd"] >= LAST_DATE - pd.DateOffset(months=3)][cols].round(0).to_string(index=False))



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

#  ## 6. Write everything to the Gold lakehouse
#  The four main tables are overwritten on each run (stamped with `ModelRunTS`); `revenue_forecast_history_gold` is appended to, so every run's monthly forecast is kept and can later be compared with the actual months (forecast accuracy over time).
#
#

# CELL ********************

 [markdown]
# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))

w = forecast_weekly.set_index("PeriodStart")
w = w[(w["ActualDays"] + w["ForecastDays"]) == 7]          # complete weeks only (partial edge weeks would show as dips)
hist = w[w["RecordType"] == "actual"]; fut = w[w["ForecastDays"] > 0]   # a mixed week is plotted as actual + forecast days
ax1.plot(hist.index, hist["ActualRevenue"], color="#1f77b4", label="actual")
ax1.plot(fut.index, fut["TotalRevenue"], color="#d62728", label=f"forecast ({CHAMPION})")
ax1.fill_between(fut.index, fut["ActualRevenue"] + fut["ForecastLower"], fut["ActualRevenue"] + fut["ForecastUpper"], color="#d62728", alpha=0.15, label=f"{int(INTERVAL_WIDTH*100)}% interval")
ax1.axvline(LAST_DATE, color="grey", ls="--", lw=1); ax1.set_title("Weekly revenue (ISO weeks)"); ax1.legend(loc="upper left"); ax1.grid(alpha=0.3)

m = forecast_monthly.set_index("PeriodStart")
ax2.bar(m.index, m["ActualRevenue"], width=20, color="#1f77b4", label="actual")
ax2.bar(m.index, m["ForecastRevenue"], width=20, bottom=m["ActualRevenue"], color="#d62728", alpha=0.7, label="forecast")
fut_m = m[m["ForecastDays"] > 0]
ax2.errorbar(fut_m.index, fut_m["ActualRevenue"] + fut_m["ForecastRevenue"],
             yerr=[fut_m["ForecastRevenue"] - fut_m["ForecastLower"], fut_m["ForecastUpper"] - fut_m["ForecastRevenue"]], fmt="none", ecolor="black", capsize=3)
ax2.set_title("Monthly revenue"); ax2.legend(loc="upper left"); ax2.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.show()



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

#  ## 8. Next steps
#  * **Pipeline** Ã¢â‚¬â€œ add this notebook as a third activity after `Gold_Notebook` (On success). Handle the Prophet dependency (Environment or `_inlineInstallationEnabled`).
#  * **Power BI** Ã¢â‚¬â€œ `revenue_weekly_gold` / `revenue_monthly_gold` already contain actual + forecast per period (`TotalRevenue` for a continuous line, `ForecastLower/Upper` for the band); `forecast_backtest_gold` gives the model leaderboard (filter `Grain`); join `revenue_forecast_history_gold` to `revenue_monthly_gold` on `yyyymm` to report how accurate past forecasts turned out.
#  * **VS Code** Ã¢â‚¬â€œ see `README_VSCode.md`: Fabric extension + *Microsoft Fabric Runtime* kernel to run on Fabric Spark, or a plain Python kernel that reads OneLake directly (`RUN_MODE = local`).
#  * **Extensions** Ã¢â‚¬â€œ per-category forecasts (bikes vs. accessories), holidays/promotions as Prophet regressors or ML features, hyper-parameter search on the ML pipelines, more folds as history grows.
#
#
#
#

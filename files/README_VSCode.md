# Running `Revenue_Forecast_Notebook.ipynb` from VS Code

The notebook never works on a copy of the data: every run reads the current version of
`Gold.dbo.factsales_gold` from the lakehouse and writes the result tables back to it.
There are two ways to run it from VS Code; the notebook detects which one it is in
(`RUN_MODE`) and nothing has to be edited.

## Option A – run on Fabric Spark from VS Code (recommended)

1. Install the **Fabric Data Engineering** extension in VS Code and sign in with your Fabric account.
2. Select your workspace in the extension side bar (or click **Open in VS Code** on the notebook page in the Fabric portal).
3. Open the notebook and choose the **Microsoft Fabric Runtime** kernel (Runtime 1.3, Spark).
4. Run all. The code executes on the Fabric Spark pool: `spark` exists, so `RUN_MODE = "fabric-spark"`,
   tables are read/written with Spark and MLflow logs straight into the workspace experiment.
   The first cell installs Prophet in that remote session.

Use this when you want exactly the same behaviour as the portal / pipeline, with VS Code as the editor.

## Option B – run on your laptop with a plain Python kernel

The notebook connects to OneLake directly (Delta Lake protocol over `abfss://…@onelake.dfs.fabric.microsoft.com`),
so it always fetches the latest committed data without exporting anything.

```bash
python -m venv .venv && .\.venv\Scripts\activate      # (Linux/macOS: source .venv/bin/activate)
pip install -r requirements-local.txt
az login                                             # Azure CLI sign-in; a browser prompt is used as fallback
```

Then open the notebook in VS Code, pick the `.venv` kernel and run all:

* no `spark` → `RUN_MODE = "local"`
* `read_fact_daily()` reads `Tables/dbo/factsales_gold` from OneLake with `deltalake` (delta-rs) and an Entra ID token
  from `azure-identity` (`DefaultAzureCredential`, scope `https://storage.azure.com/.default`).
* `write_table()` writes `revenue_forecast_gold`, `revenue_weekly_gold`, `revenue_monthly_gold` and
  `forecast_backtest_gold` back to OneLake with `write_deltalake` – they appear in the Gold lakehouse like any other table.
* MLflow: with `synapseml-mlflow` installed and `LOCAL_MLFLOW_TO_FABRIC = True` the runs, metrics and models land in
  the workspace experiment `sales_revenue_forecast` (tracking URI `sds://api.fabric.microsoft.com/v1/workspaces/<workspace-id>/mlflow`).
  Without the plugin the notebook falls back to a local `./mlruns` folder.

Requirements: you need write access to the workspace (Member/Contributor), and your network must reach
`onelake.dfs.fabric.microsoft.com`. Tokens expire after about an hour – just re-run the cell if a long session fails on auth.
For unattended local runs use a service principal (`AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`
environment variables are picked up by `DefaultAzureCredential`).

## Which option when?

| | Option A (Fabric runtime) | Option B (local kernel) |
|---|---|---|
| compute | Fabric capacity (CU) | your laptop |
| data access | Spark, tables by name | delta-rs over OneLake, path by workspace/lakehouse id |
| freshness | live | live |
| libraries | Fabric runtime + `%pip` / Environment | `requirements-local.txt` |
| best for | production-identical runs, debugging pipeline notebooks | fast iteration, offline modelling, no capacity usage |

Both modes produce the same tables; `RunMode` is stored in every output row so you can tell them apart.

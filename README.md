# Sales Revenue Forecast

Revenue forecasting notebook for a Microsoft Fabric Gold lakehouse. It reads the current `Gold.dbo.factsales_gold` Delta table, evaluates multiple forecasting models, selects a champion model, and writes daily, weekly, monthly, backtest, and forecast-history tables.

## Run locally in VS Code

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r files\requirements-local.txt
az login
```

Open `files/Revenue_Forecast_Notebook.ipynb` in VS Code, select the `.venv` kernel, and run the cells in order. See [the VS Code guide](files/README_VSCode.md) for Fabric-runtime and local-run details.

## Data and credentials

The notebook accesses OneLake using your Entra ID credentials. Do not commit credentials, local MLflow runs, Fabric caches, or exported lakehouse data. The included `.gitignore` excludes these machine-specific files.

## Documentation

The accompanying [forecasting manual](files/Revenue_Forecasting_Manual.pdf) describes the model and outputs.

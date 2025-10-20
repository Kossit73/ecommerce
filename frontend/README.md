# Ecommerce Financial Console (Static)

This directory hosts a zero-dependency HTML/JavaScript console that talks to the FastAPI ecommerce model.
All assets are plain text, so no Node.js build or `react-scripts` runtime is required.

## Running locally

1. Start the FastAPI service from the project root:

   ```bash
   uvicorn main:app --reload --port 8000
   ```

2. Serve the static frontend (any HTTP server works). For example:

   ```bash
   python -m http.server 3000 --directory frontend
   ```

3. Visit `http://localhost:3000/index.html`. Set the API base URL in the top-right corner to `http://localhost:8000` and click **Apply**.

The console will now drive data ingestion, scenario management, visualization, reporting, and advanced analytics through the FastAPI endpoints. No package installation steps are necessary.

## Features

- **Input & Assumptions** – Upload or initialize the Excel workbook, edit yearly assumptions across dedicated demand, cost, staffing, asset, and financing tables, filter time periods, and run the base analysis workflow.
- **Key Financial Metrics** – View summary KPIs, operational metrics, valuation snapshots, scenario comparisons, and Plotly-powered charts.
- **Financial Performance** – Inspect income statements and customer metrics derived from the active scenario.
- **Financial Position** – Review balance sheet and capital asset schedules.
- **Cash Flow Statement** – Explore detailed cash flow schedules and debt amortization tables.
- **Sensitivity Analysis** – Trigger top-rank sensitivity, what-if overlays, and goal seek calculations.
- **Advanced Analysis** – Launch Monte Carlo, schedule risk, neural tools, decision trees, statistical forecasts, and Evolver optimization routines.

## Customization

The frontend JavaScript lives in `assets/js/`. `app.js` wires UI events to the API, `tables.js` renders grid views, and `charts.js` wraps Plotly visualizations. Styles reside in `assets/css/styles.css`.

Because everything is static, adjustments only require refreshing the browser—no build step is involved.

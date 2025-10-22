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

3. Visit `http://localhost:3000/index.html`. The console will automatically
   target the FastAPI service based on the page origin. You can optionally add
   a query parameter such as `?apiBase=http://localhost:8000` or define
   `window.ECOM_API_BASE` in `index.html` to hard-code another backend. The
   final base URL is persisted in `localStorage` for subsequent visits.

The console will now drive data ingestion, scenario management, visualization, reporting, and advanced analytics through the FastAPI endpoints. No package installation steps are necessary.

## Features

- **Input & Assumptions** – Upload or initialize the Excel workbook, or rebuild the grouped assumption tables from the bundled template before editing demand, cost, staffing, asset, and financing values. Filter time periods and run the base analysis workflow once the backend is connected.
- **Key Financial Metrics** – View summary KPIs, operational metrics, valuation snapshots, scenario comparisons, narrative implications, and Plotly-powered revenue/traffic/cash charts. Recompute scenarios with editable multipliers and rates.
- **Financial Performance** – Inspect income statements, customer metrics, and waterfall, breakeven, funnel, and margin-trend visualizations derived from the active scenario.
- **Financial Position** – Review balance sheet and capital asset schedules.
- **Cash Flow Statement** – Explore detailed cash flow schedules, debt amortization tables, and DCF/cash-flow forecast charts.
- **Sensitivity Analysis** – Trigger top-rank sensitivity, what-if overlays, and goal seek calculations.
- **Advanced Analysis** – Launch Monte Carlo, schedule risk, neural tools, decision trees, statistical forecasts, Evolver optimization routines, and export the consolidated Excel workbook.

## Customization

The frontend JavaScript lives in `assets/js/`. `app.js` wires UI events to the API, `tables.js` renders grid views, and `charts.js` wraps Plotly visualizations. Styles reside in `assets/css/styles.css`.

Because everything is static, adjustments only require refreshing the browser—no build step is involved. To hard-code a backend target, set `window.ECOM_API_BASE` in `index.html` before `assets/js/app.js` is loaded.

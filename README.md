# Financial Management API


## Prerequisites

- **Python**: Version 3.8 or higher
- **pip**: Python package manager
- **Virtualenv** (optional, for environment management)
- **Excel File**: `financial_assumptions.xlsx` (required for data loading)

## Setup Instructions

### 1. Create a Virtual Environment

Create a virtual environment to isolate project dependencies:

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

Activate the virtual environment based on your operating system:

- **Windows**:
  ```bash
  venv\Scripts\activate
  ```

- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

Choose the dependency set that matches the component you plan to run:

- **Streamlit dashboard (default on Streamlit Cloud):**

  ```bash
  pip install -r requirements.txt
  ```

  The streamlined `requirements.txt` only contains the packages needed to run
  `streamlit_app.py`, keeping hosted deployments lightweight and avoiding
  native-build failures.

- **FastAPI backend and analytics services:**

  ```bash
  pip install -r backend-requirements.txt
  ```

  Install this superset when you are running the API locally or on a server.
  It includes FastAPI, Uvicorn, scientific libraries, and Excel tooling used by
  the underlying financial model.


### 4. Run the FastAPI Server

Start the FastAPI server with Uvicorn, using debug logging and port 8002:

```bash
uvicorn main:app --reload --log-level debug --port 8002
```

- `--reload`: Automatically reloads the server on code changes (development only).
- `--log-level debug`: Provides detailed logs for debugging.
- `--port 8002`: Runs the server on port 8002.

The API will be available at `http://localhost:8002`.

### 5. Launch a User Interface

You can work with the ecommerce financial model through either the static web
console or the Streamlit app.

#### Option A: Static HTML console

1. From the project root, start the FastAPI service (see step 4 above).
2. Serve the `frontend/` directory with any HTTP server. For example:

   ```bash
   python -m http.server 3000 --directory frontend
   ```

3. Visit `http://localhost:3000/index.html`. The console automatically targets
   the backend based on the page origin. You can optionally override the
   FastAPI base via the `?apiBase=` query parameter, a global `window.ECOM_API_BASE`
   assignment, or previously saved settings in `localStorage`. The browser
   console now provides the Input & Assumptions, Key Financial Metrics,
   Financial Performance, Financial Position, Cash Flow Statement, Sensitivity
   Analysis, and Advanced Analysis workflows showcased in the screenshot below.
   If the backend is offline, the Input & Assumptions page seeds its tables with
   a bundled template so you can rebuild the drivers manually before reconnecting.

#### Option B: Streamlit console

1. In a new terminal, launch Streamlit with the packaged dashboard. You can use
   the standard CLI or run the script directly—both options start the same
   server:

   ```bash
   streamlit run streamlit_app.py
   # or
   python streamlit_app.py
   ```

2. Your browser will open to a multi-tab experience mirroring the static web
   console. All calculations now run directly in the Streamlit session, so no
   FastAPI backend or API base configuration is required.

3. Each assumption schedule now includes inline “Add line”/“Remove line”
   controls, so you can insert or delete rows directly inside Streamlit before
   saving the rebuilt workbook inputs. A guidance panel inside every schedule
   outlines the best way to edit each line item so you know which fields to
   tweak when assumptions change.

4. Use the **Production horizon** selectors at the top of the Input &
   Assumptions tab to choose the start and end years for the model. All
   schedules, statements, and analytics stay within that range, so forecasts no
   longer extend beyond your selected production horizon.

5. Review the **Model consistency diagnostics** section on the Key Financial
   Metrics tab after each update. The Streamlit app now reconciles income
   statement, balance sheet, and cash-flow tie-outs automatically and flags any
   differences that exceed a 0.01 tolerance so you can correct issues before
   moving on to scenario or risk analysis.

#### Showing asset additions in Streamlit

- Navigate to **Input & Assumptions → Asset Register** and add a row for each
  purchase. Provide the acquisition year, a descriptive asset name, the amount
  invested, and its depreciation rate (as a percentage or decimal).
- After you apply the assumptions, the **Key Financial Metrics** tab surfaces an
  **Asset additions roll-forward** table. This summary shows the beginning
  balance, new additions, annual depreciation, cumulative depreciation, and
  ending balance for every year.
- The same totals feed straight into the depreciation matrix, the income
  statement, the cash-flow schedule, and the balance sheet, so you can verify
  that new assets immediately flow through the entire model.
## Project Structure

```
financial_management_api/
│
├── main.py                       # FastAPI application entry point
├── ecommerce.py                  # Core ecommerce model implementation
├── requirements.txt              # Streamlit dashboard dependencies
├── backend-requirements.txt      # FastAPI/analytics dependency set
├── frontend/                     # Static HTML/JS console assets
├── streamlit_app.py              # Streamlit dashboard
├── financial_assumptions.xlsx    # Input Excel file for data loading
├── ecommerce_api_usage.xlsx      # Documentation of API endpoints and purposes
└── README.md                     # This file
```

## Model computation notes

- The Streamlit experience now runs entirely on the schedules you input—apply assumptions and every tab refreshes without relying on the FastAPI backend.
- The static HTML console still supports calling the API if you host it, but it also seeds the same bundled assumption template so you can experiment offline.
- Valuation metrics (NPV/IRR) are derived from **free cash flow**—operating cash flow plus investing cash flow—so financing inflows/outflows no longer distort the internal rate of return. Make sure your inputs include at least one outflow (e.g., capex or equity distributions) for a finite IRR.
- If you plan to extend the backend, keep `backend-requirements.txt` handy for local development; Streamlit deployments only need the lightweight `requirements.txt` bundle.

## Troubleshooting

- **Port Conflict**: If port 8002 is in use, change the port in the Uvicorn command (e.g., `--port 8003`).
- **Missing Dependencies**: Ensure all packages in `requirements.txt` are installed. Run `pip install -r requirements.txt` again if errors occur.
- **Streamlit Cloud install failures**: The hosted environment only needs the
  lightweight `requirements.txt`. If you deploy the backend elsewhere, install
  `backend-requirements.txt` outside of Streamlit Cloud to avoid native build
  steps that the platform blocks.
- **File Not Found**: Verify `financial_assumptions.xlsx` exists in the project root or update the filename/path in the load API call.
- **API Errors**: Check server logs (enabled with `--log-level debug`) for detailed error messages.


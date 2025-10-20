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

Install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```


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

3. Visit `http://localhost:3000/index.html`, set the **API Base URL** at the
   top-right of the page to `http://localhost:8002`, and click **Apply**. The
   browser console now provides the Input & Assumptions, Key Financial Metrics,
   Financial Performance, Financial Position, Cash Flow Statement, Sensitivity
   Analysis, and Advanced Analysis workflows showcased in the screenshot
   below.

#### Option B: Streamlit console

1. Ensure the backend is running (step 4).
2. In a new terminal, launch Streamlit with the packaged dashboard:

   ```bash
   streamlit run streamlit_app.py -- --api-base http://localhost:8002
   ```

3. Your browser will open to a multi-tab experience mirroring the static web
   console. Use the sidebar to adjust the API base URL if needed.

## Important Note

The APIs for retrieving financial schedules and running analysis depend on data loaded into the backend. You **must** first call the `/api/ecommerce/file_action?action=Load%20Existing` API to load the financial data from `financial_assumptions.xlsx` before other APIs can function correctly.

## API Endpoints

### 1. Load Data (`GET /api/ecommerce/file_action?action=Load%20Existing`)

Loads financial data from an Excel file into the backend. This API must be called first to populate `filtered_df` and `debt_schedules`.

- **Method**: GET
- **Path**: `/api/ecommerce/file_action?action=Load%20Existing`
- **Path Parameter**:
  - `filename`: Name of the Excel file (e.g., `financial_assumptions.xlsx`).
- **Response**: JSON object confirming successful data loading or error details.
- **Example Request**:
  ```bash
  curl -X 'GET' \
  'http://127.0.0.1:8001/api/ecommerce/file_action?action=Load%20Existing' \
  -H 'accept: application/json'
  ```

- **Purpose**: Initializes the backend with financial data required for other APIs.

### 3. Other APIs

The project includes additional APIs for retrieving financial schedules (e.g., Income Statement, Customer Metrics). These APIs depend on the data loaded via `/api/ecommerce/file_action?action=Load%20Existing`. For a complete list of APIs and their purposes, refer to the attached Excel file (`ecommerce_api_usage.xlsx`), which details each API, its endpoint, method, and usage.

## Project Structure

```
financial_management_api/
│
├── main.py                 # FastAPI application entry point
├── models/ecommerce.py     # FastAPI application models
├── requirements.txt        # Python dependencies
├── data/financial_assumptions.xlsx  # Input Excel file for data loading
├── ecommerce_api_usage.xlsx          # Documentation of API endpoints and purposes
├── venv/                   # Virtual environment (created after setup)
└── README.md               # This file
```

## API Usage Notes

- **Data Dependency**: Always call `/api/ecommerce/file_action?action=Load%20Existing` first to load the financial data. Other APIs
- **Excel File**: Ensure `financial_assumptions.xlsx` is in the project root or specify the correct path in the load API.
- **Authentication**: The APIs are currently public. To add authentication, implement OAuth2 or JWT and add `Depends(get_current_user)` to endpoints (see previous examples).
- **Error Handling**: APIs return HTTP 500 for processing errors and HTTP 400 for invalid inputs. Check error messages for debugging.

## Troubleshooting

- **Port Conflict**: If port 8002 is in use, change the port in the Uvicorn command (e.g., `--port 8003`).
- **Missing Dependencies**: Ensure all packages in `requirements.txt` are installed. Run `pip install -r requirements.txt` again if errors occur.
- **File Not Found**: Verify `financial_assumptions.xlsx` exists in the project root or update the filename/path in the load API call.
- **API Errors**: Check server logs (enabled with `--log-level debug`) for detailed error messages.


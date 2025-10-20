import traceback
from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File,Query,Body
from fastapi.responses import JSONResponse,StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field,validator
from typing import Dict, List, Any,Optional
import pandas as pd
import uuid
from fastapi.encoders import jsonable_encoder
from ecommerce import EcommerceModel
import logging
import numpy as np
from io import BytesIO
from pathlib import Path
from pydantic import BaseModel, field_validator
from enum import Enum
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="Industry Financial Models")
# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Directory to store Excel files
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# In-memory storage (replace with database in production)
session_state = {
    "df":None,
    "years_data": {},
    "start_year": None,
    "current_file": "financial_assumptions.xlsx",
    "filtered_df":None,
    "valuation_details":None,
    "scenarios":None,
    "forecast_df":None
}



# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://13.61.58.105/ecommerce", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

STATIC_CREDENTIALS = {"username": "admin", "password": "admin"}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the valid options for sensitivity variables
VALID_SENSITIVITY_VARIABLES = [
    'Average Item Value',
    'COGS Percentage',
    'Email Traffic',
    'Paid Search Traffic',
    'Email Conversion Rate',
    'Organic Search Conversion Rate',
    'Paid Search Conversion Rate',
    'Affiliates Conversion Rate'
]

class EvolverOptimizationInput(BaseModel):
    budget_dict: Dict[str, float]
    forecast_years: int = 10


    @validator('budget_dict')
    def validate_budget_dict(cls, v):
        if len(v) != 1:
            raise ValueError("budget_dict must contain exactly one key-value pair")
        budget_line, budget_amount = list(v.items())[0]
        valid_lines = [
            'Total Marketing Budget', 'Freight/Shipping per Order', 'Labor/Handling per Order',
            'General Warehouse Rent', 'Office Rent', 'Salaries, Wages & Benefits', 'Professional Fees'
        ]
        if budget_line not in valid_lines:
            raise ValueError(f"Invalid budget_line: {budget_line}. Must be one of {valid_lines}")
        if budget_amount <= 0:
            raise ValueError("Budget amount must be positive")
        return v

    @validator('forecast_years')
    def validate_forecast_years(cls, v):
        if v < 1 or v > 20:
            raise ValueError("forecast_years must be between 1 and 20")
        return v

class OptimizedVariable(BaseModel):
    year: float
    variables: Dict[str, float]
    changes: Dict[str, float]

class EvolverOptimizationResponse(BaseModel):
    status: str
    optimized_data: List[OptimizedVariable]
    original_ebitda: float
    optimized_ebitda: float
    ebitda_change_percent: float
    message: str


class PrecisionTreeResponse(BaseModel):
    status: str
    decision_outcomes: Dict[str, float]
    decision_tree_image: str  # Base64-encoded PNG
    message: str

# Pydantic model for output metric
class OperationalMetric(BaseModel):
    metric: str
    current: float

# Pydantic model for output
class OperationalMetricsResponse(BaseModel):
    metrics: List[OperationalMetric]

# Pydantic Models
class RentCategory(BaseModel):
    name: str
    square_meters: float
    cost_per_sqm: float

class FeeType(BaseModel):
    name: str
    cost: float

class Asset(BaseModel):
    name: str
    amount: float
    rate: float

class Debt(BaseModel):
    name: str
    amount: float
    interest_rate: float
    duration: int

class StaffCategory(BaseModel):
    name: str
    hours_per_year: int
    number: int
    hourly_rate: float

class ExecutiveRole(BaseModel):
    name: str
    salary: float

class BenefitType(BaseModel):
    name: str
    cost_per_staff: float

class YearData1(BaseModel):
    year: int
    data: Dict[str, float | str | List[str] | List[Asset] | List[Debt]]

class FileActionResponse(BaseModel):
    action: str
    filename: Optional[str] = None
    exists: bool = False
    data: Optional[Dict[int, Dict]] = None

class SaveDataRequest(BaseModel):
    filename: str
    years_data: Dict[int, Dict]

class YearData(BaseModel):
    year: int
    email_traffic: float
    organic_search_traffic: float
    paid_search_traffic: float
    affiliates_traffic: float
    email_conversion_rate: float
    organic_search_conversion_rate: float
    paid_search_conversion_rate: float
    affiliates_conversion_rate: float
    average_item_value: float
    number_of_items_per_order: float
    average_markdown: float
    average_promotion_discount: float
    cogs_percentage: float
    churn_rate: float
    email_cost_per_click: float
    organic_search_cost_per_click: float
    paid_search_cost_per_click: float
    affiliates_cost_per_click: float
    freight_shipping_per_order: float
    labor_handling_per_order: float
    general_warehouse_rent: float
    other: float
    interest: float
    tax_rate: float
    staff_categories: List[StaffCategory]
    executive_roles: List[ExecutiveRole]
    benefit_types: List[BenefitType]
    rent_categories: List[RentCategory]
    professional_fee_types: List[FeeType]
    assets: List[Asset]
    accounts_receivable_days: float
    inventory_days: float
    accounts_payable_days: float
    technology_development: float
    office_equipment: float
    technology_depreciation_years: float
    office_equipment_depreciation_years: float
    interest_rate: float
    equity_raised: float
    dividends_paid: float
    debts: List[Debt]


# Pydantic model for output metric row
class CustomerMetric(BaseModel):
    Year: Optional[float]
    NPV: Optional[float]
    CAC: Optional[float]
    Contribution_Margin_Per_Order: Optional[float]
    LTV: Optional[float]
    LTV_CAC_Ratio: Optional[float]
    Payback_Orders: Optional[float]
    Burn_Rate: Optional[float]

# Pydantic model for output
class CustomerMetricsResponse(BaseModel):
    irr: Optional[float]
    metrics: List[CustomerMetric]

# Pydantic Models
class Adjustment(BaseModel):
    year: int
    variable: str
    multiplier: float

class DecisionToolParams(BaseModel):
    forecast_years: int
    num_simulations: int
    confidence_level: float
    variables_to_test: List[str]
    change_percentage: float
    traffic_increase_percentage: float
    selected_budget_line: str
    budget_amount: int

# Pydantic model for output
class SummaryMetric(BaseModel):
    Scenario: str
    Net_Income_M: float
    EBITDA_M: float
    IRR: float
    NPV_M: float
    Payback_Period_Orders: float
    Gross_Profit_Margin: float
    Net_Profit_Margin: float
    Net_Cash_Flow_M: float

class AnalysisRequest(BaseModel):
    input_data: Optional[List[Dict[str, Any]]] = None
    discount_rate: float
    wacc: float
    perpetual_growth: float
    tax_rate: float
    inflation_rate: float
    direct_labor_rate_increase: float
    selected_scenario: str
    start_year: int
    end_year: int
    monte_carlo_forecast_years: int
    normal_forecast_years: int
    scenario_type: str
    num_simulations: int
    confidence_level: float
    distribution: str
    what_if_adjustments: List[Dict[str, Any]] #List[Adjustment]
    goal_year: int
    profit_margin_increase: float
    variable_to_adjust: str
    decision_tools: Optional[DecisionToolParams] = None
    scenario_params: Dict[str, float] = None
   # Sensitivity analysis parameters
    sensitivity_variables_to_test: List[str] = Field(
        default=['Average Item Value', 'COGS Percentage'],
        description="List of variables to test in sensitivity analysis"
    )
    sensitivity_change_percentage: float = Field(
        default=10.0,
        ge=5.0,
        le=20.0,
        description="Percentage change for sensitivity analysis (between 5.0 and 20.0)"
    )

    # Validator for sensitivity_variables_to_test
    @validator('sensitivity_variables_to_test')
    def validate_sensitivity_variables(cls, v):
        if not v:
            raise ValueError("At least one variable must be selected for sensitivity analysis")
        invalid_vars = [var for var in v if var not in VALID_SENSITIVITY_VARIABLES]
        if invalid_vars:
            raise ValueError(
                f"Invalid sensitivity variables: {invalid_vars}. Must be one of {VALID_SENSITIVITY_VARIABLES}"
            )
        return v

    # Validator for sensitivity_change_percentage (already handled by Field constraints, but adding for clarity)
    @validator('sensitivity_change_percentage')
    def validate_change_percentage(cls, v):
        if not 5.0 <= v <= 20.0:  # Redundant with Field(ge, le), but kept for explicit validation
            raise ValueError("Sensitivity change percentage must be between 5.0 and 20.0")
        return v        
   
# Pydantic model for output
class ValuationResponse(BaseModel):
    enterprise_value_m: float
    equity_value_m: float
  
class ScheduleType(str, Enum):
    INCOME_STATEMENT = "Income Statement"
    CASH_FLOW_STATEMENT = "Cash Flow Statement"
    BALANCE_SHEET = "Balance Sheet"
    CAPITAL_ASSETS = "Capital Assets"
    VALUATION = "Valuation"
    CUSTOMER_METRICS = "Customer Metrics"
    DEBT_PAYMENT_SCHEDULE = "Debt Payment Schedule"

# Pydantic models for output
class ScheduleRecord(BaseModel):
    Year: Optional[float]
    # Generic fields to cover all schedules
    Net_Revenue: Optional[float] = None
    Gross_Profit: Optional[float] = None
    EBITDA: Optional[float] = None
    Net_Income: Optional[float] = None
    Total_Orders: Optional[float] = None
    Cash_from_Operations: Optional[float] = None
    Cash_from_Investing: Optional[float] = None
    Cash_from_Financing: Optional[float] = None
    Net_Cash_Flow: Optional[float] = None
    Total_Assets: Optional[float] = None
    Total_Liabilities: Optional[float] = None
    Total_Equity: Optional[float] = None
    Balance_Sheet_Check: Optional[float] = None
    Total_Opening_Balance: Optional[float] = None
    Total_Additions: Optional[float] = None
    Total_Depreciation: Optional[float] = None
    Total_Closing_Balance: Optional[float] = None
    EBIT: Optional[float] = None
    Unlevered_FCF: Optional[float] = None
    PV_of_FCF: Optional[float] = None
    Total_Enterprise_Value: Optional[float] = None
    CAC: Optional[float] = None
    Contribution_Margin_Per_Order: Optional[float] = None
    LTV: Optional[float] = None
    LTV_CAC_Ratio: Optional[float] = None
    Payback_Orders: Optional[float] = None

class DebtScheduleRecord(BaseModel):
    Year: Optional[float]
    Principal: Optional[float]
    Interest: Optional[float]
    Payment: Optional[float]

class DebtDetail(BaseModel):
    Debt_Name: str
    Amount: float
    Interest_Rate: float
    Duration: int
    Schedule: List[DebtScheduleRecord]

class DebtYear(BaseModel):
    Year: int
    Schedules: List[DebtDetail]

class ScheduleData(BaseModel):
    schedule: str
    data: List[ScheduleRecord] | List[DebtYear]

class SchedulesResponse(BaseModel):
    schedules: List[ScheduleData]

# Enum for variables
class WhatIfVariable(str, Enum):
    TOTAL_ORDERS = "Total Orders"
    AVERAGE_ITEM_VALUE = "Average Item Value"
    EMAIL_CONVERSION_RATE = "Email Conversion Rate"
    PAID_SEARCH_TRAFFIC = "Paid Search Traffic"
    COGS_PERCENTAGE = "COGS Percentage"
    LABOR_HANDLING_PER_ORDER = "Labor/Handling per Order"
    FREIGHT_SHIPPING_PER_ORDER = "Freight/Shipping per Order"
    MARKETING_EXPENSES = "Marketing Expenses"
    INTEREST_RATE = "Interest Rate"

# Pydantic models for input
class WhatIfAdjustment(BaseModel):
    year: int
    variable: WhatIfVariable
    multiplier: float

class WhatIfInput(BaseModel):
    num_adjustments: int
    adjustments: List[WhatIfAdjustment]
    discount_rate: float = 0.1  # Default discount rate

# Pydantic models for output
class MetricSummaryWhatIf(BaseModel):
    year: int
    net_revenue: float
    gross_profit: float
    ebitda: float
    net_income: float
    total_orders: float

class WhatIfResponse(BaseModel):
    status: str
    results: List[MetricSummaryWhatIf]
    adjustments: List[WhatIfAdjustment]
    warnings: List[str]

# Enum for distribution types
class DistributionType(str, Enum):
    NORMAL = "Normal"
    LOGNORMAL = "Lognormal"
    UNIFORM = "Uniform"
    EXPONENTIAL = "Exponential"
    BINOMIAL = "Binomial"
    POISSON = "Poisson"
    GEOMETRIC = "Geometric"
    BERNOULLI = "Bernoulli"
    CHI_SQUARE = "Chi-square"
    GAMMA = "Gamma"
    WEIBULL = "Weibull"
    HYPERGEOMETRIC = "Hypergeometric"
    MULTINOMIAL = "Multinomial"
    BETA = "Beta"
    F_DISTRIBUTION = "F-distribution"
    DISCRETE = "Discrete"
    CONTINUOUS = "Continuous"
    CUMULATIVE = "Cumulative"

# Pydantic model for input
class MonteCarloInput(BaseModel):
    forecast_years: int
    num_simulations: int
    confidence_level: float
    distribution_type: DistributionType
    discount_rate: float
    wacc: float
    perpetual_growth: float

# Pydantic model for output
class MetricSummary(BaseModel):
    mean: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    unit: str
    final_year_values: Optional[List[float]] = None  # For histograms

class MonteCarloResponse(BaseModel):
    status: str
    metrics: Dict[str, MetricSummary]
    confidence_level: float
    final_year_income: List[float]
    npv_values: List[float]
    mean_income: float
    ci_values: List[float]
    mean_npv: float
    ci_npv: List[float]

# Enum for variables
class GoalSeekVariable(str, Enum):
    NET_REVENUE = "Net Revenue"
    TOTAL_ORDERS = "Total Orders"
    AVERAGE_ITEM_VALUE = "Average Item Value"
    PAID_SEARCH_TRAFFIC = "Paid Search Traffic"
    COGS_PERCENTAGE = "COGS Percentage"
    LABOR_HANDLING_PER_ORDER = "Labor/Handling per Order"
    FREIGHT_SHIPPING_PER_ORDER = "Freight/Shipping per Order"
    MARKETING_EXPENSES = "Marketing Expenses"

# Pydantic models
class GoalSeekInput(BaseModel):
    target_profit_margin: float
    variable_to_adjust: GoalSeekVariable
    year_to_adjust: int
    max_iterations: int = 100
    tolerance: float = 0.001
    discount_rate: float = 0.1

class MetricSummary_GS(BaseModel):
    year: int
    net_revenue: float
    gross_profit: float
    ebitda: float
    net_income: float
    total_orders: float

class GoalSeekResponse(BaseModel):
    status: str
    goal_seek_year:int
    current_profit_margin:float
    target_profit_margin:float
    multiplier: Optional[float] = None
    results: Optional[List[MetricSummary_GS]] = None
    message: Optional[str] = None


class ScheduleRiskInput(BaseModel):
    num_simulations: int
    confidence_level: float

class ConfidenceInterval(BaseModel):
    lower: float
    upper: float
    confidence_level: float

class Task(BaseModel):
    name: str
    base_duration: float
    std_dev: float

class ScheduleRiskResponse(BaseModel):
    status: str
    simulation_durations: List[float]
    mean_duration: float
    confidence_interval: ConfidenceInterval
    tasks: List[Task]
    message: str

# Pydantic models
class NeuralToolsInput(BaseModel):
    traffic_increase_percentage: float

class FeatureImportance(BaseModel):
    feature: str
    importance: float

class NeuralToolsResponse(BaseModel):
    status: str
    traffic_increase_percentage: float
    predicted_revenue: float
    feature_importance: List[FeatureImportance]
    message: str

# Pydantic models
class TopRankSensitivityInput(BaseModel):
    variables_to_test: List[str]
    change_percentage: float
    discount_rate: float = 0.20

class SensitivityResult(BaseModel):
    variable: str
    direction: str
    net_income_change: float
    ebitda_change: float
    net_cash_flow_change: float
    equity_value_change: float

class TopRankSensitivityResponse(BaseModel):
    status: str
    sensitivity_results: List[SensitivityResult]
    message: str
    sensitivity_insights: List[str]

# Pydantic models
class StatToolsForecastingInput(BaseModel):
    forecast_years: int
    confidence_level: float

class ChartTrace(BaseModel):
    x: List[float]
    y: List[float]
    name: str
    type: str = "scatter"
    mode: str
    line: Dict[str, Any]
    fill: str = None
    fillcolor: str = None
    showlegend: bool = True
    marker: Dict[str, float] = None

class ChartConfig(BaseModel):
    metric: str
    traces: List[ChartTrace]
    layout: Dict[str, Any]

class StatToolsForecastingResponse(BaseModel):
    status: str
    forecast_data: List[Dict[str, Any]]
    summary_statistics: Dict[str, Dict[str, float]]
    format_dict: Dict[str, str]
    chart_data: List[ChartConfig]
    message: str

# # Pydantic models
# class EvolverOptimizationInput(BaseModel):
#     budget_dict: Dict[str, float]
#     forecast_years: int = 10

# class OptimizedVariable(BaseModel):
#     year: float
#     variables: Dict[str, float]
#     changes: Dict[str, float]

# class EvolverOptimizationResponse(BaseModel):
#     status: str
#     optimized_data: List[OptimizedVariable]
#     original_ebitda: float
#     optimized_ebitda: float
#     ebitda_change_percent: float
#     message: str

class TimePeriodRequest(BaseModel):
    # session_id: str
    scenario_type: str
    start_year: int
    end_year: int
    discount_rate: float
    wacc: float
    perpetual_growth: float
    tax_rate: float
    inflation_rate: float
    direct_labor_rate_increase: float

class BaseAnalysisRequest(BaseModel):
    discount_rate: float
    wacc: float
    perpetual_growth: float
    tax_rate: float
    inflation_rate: float
    direct_labor_rate_increase: float
    normal_forecast_years: int

class BaseAnalysisResponse(BaseModel):
    status: str
    base_df: List[Dict[str, Any]]
    forecast_df: List[Dict[str, Any]]
    base_scenario: List[Dict[str, Any]]
    session_id: str
    message: str

class ScenarioParams(BaseModel):
    conversion_rate_mult: float
    aov_mult: float
    cogs_mult: float
    interest_mult: float
    labor_mult: float
    material_mult: float
    markdown_mult: float
    political_risk: float
    env_impact: float

    @field_validator('*')
    @classmethod
    def check_positive(cls, v, info):
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return v

class ScenarioRequest(BaseModel):
    scenario_type: str
    scenario_params: Optional[ScenarioParams] = None
    discount_rate: float
    tax_rate: float
    inflation_rate: float
    direct_labor_rate_increase: float

    @field_validator('scenario_type')
    @classmethod
    def validate_scenario_type(cls, v):
        valid_scenarios = ["Base Case", "Best Case", "Worst Case"]
        if v not in valid_scenarios:
            raise ValueError(f"scenario_type must be one of {valid_scenarios}")
        return v

class ScenarioResponse(BaseModel):
    status: str
    scenario_df: Optional[List[Dict[str, Any]]] = None
    message: str

ecommerce_model = EcommerceModel()

async def get_current_user(request: Request):
    if not request.session.get("username"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return request.session["username"]

async def create_new_file():
    filename="financial_assumptions.xlsx"
    
    file_path = DATA_DIR / filename
    if file_path.exists():
        raise HTTPException(status_code=400, detail=f"File {filename} already exists")
    
    try:
        # Create an empty DataFrame with a single row to initialize the file
        df = pd.DataFrame(columns=['Year'])
        df.to_excel(file_path, index=False)
        return {"filename": filename, "message": f"New file {filename} created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating file {filename}: {str(e)}")
    
# POST /upload-file: Upload and read an Excel file
@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    filename = "financial_assumptions.xlsx"
    file_path = DATA_DIR / filename    
    try:
        # Save uploaded file
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)   
        session_state["df"]=None
        session_state["years_data"]={}
        session_state["start_year"]=None
        session_state["filtered_df"]=None
        session_state["valuation_details"]=None
        session_state["scenarios"]=None
        session_state["forecast_df"]=None
        # Read and process data
        df = pd.read_excel(file_path)
        session_state['df']=df
        data =ecommerce_model.process_excel_data(df)
        if filename:
            file_path = DATA_DIR / filename
            if file_path.exists():
                try:
                    load_existing_data_response = ecommerce_model.load_existing_data()
                    session_state['years_data']=load_existing_data_response["data"]
                    session_state['debts_data']=load_existing_data_response['debts_data']
                    await run_base_analysis(BaseAnalysisRequest(
                                        discount_rate= 0.20,
                                        wacc= 0.10,
                                        perpetual_growth= 0.02,
                                        tax_rate= 0,
                                        inflation_rate= 0,
                                        direct_labor_rate_increase= 0,
                                        normal_forecast_years= 10
                    )
                                        )
                    return FileActionResponse(action="Select File", filename=filename, exists=True, data=data)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error loading file {filename}: {str(e)}")
            else:
                return FileActionResponse(action="Select File", filename=filename, exists=False)
        return {"filename": filename, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing uploaded file: {str(e)}")    

# GET /file-action: Check file existence or handle file action
@app.get("/file_action")
async def check_file_action(action: str = "Load Existing"):

    filename="financial_assumptions.xlsx"
    file_path = DATA_DIR / filename    
    if action not in ["Load Existing", "Start New", "Select File"]:
        raise HTTPException(status_code=400, detail="Invalid action")
    
    if action == "Start New":
        session_state["df"]=None
        session_state["years_data"]={}
        session_state["start_year"]=None
        session_state["filtered_df"]=None
        session_state["valuation_details"]=None
        session_state["scenarios"]=None
        session_state["forecast_df"]=None

        response=await create_new_file()
        df = pd.read_excel(file_path)
        session_state['df']=df
        data = ecommerce_model.process_excel_data(df)
        load_existing_data_response = ecommerce_model.load_existing_data()
        session_state['years_data']=load_existing_data_response["data"]
        session_state['debts_data']=load_existing_data_response['debts_data']
        return response
    
    if action == "Select File":
        return {"Use upload file API /ecommerce/upload_file"}
    
    if filename:
        file_path = DATA_DIR / filename
        if file_path.exists():
            try:
                df = pd.read_excel(file_path)
                print("df_new :" , df.columns.to_list())
                session_state['df']=df
                data = ecommerce_model.process_excel_data(df)
                load_existing_data_response = ecommerce_model.load_existing_data()
                session_state['years_data']=load_existing_data_response["data"]
                session_state['debts_data']=load_existing_data_response['debts_data']
                await run_base_analysis(BaseAnalysisRequest(
                                    discount_rate= 20,
                                    wacc= 10,
                                    perpetual_growth= 2,
                                    tax_rate= 0,
                                    inflation_rate= 0,
                                    direct_labor_rate_increase= 0,
                                    normal_forecast_years= 10
                )
                                    )
                return FileActionResponse(action=action, filename=filename, exists=True, data=data)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error loading file {filename}: {str(e)}")
        else:
            return FileActionResponse(action=action, filename=filename, exists=False)
    
    return FileActionResponse(action=action, exists=False)
# (Assume ecommerce_model.process_excel_data and ecommerce_model.load_existing_data exist,
#  and session_state is a dict where you store your “df” and related state.)

@app.post("/save_assumptions")
async def save_assumptions(years_data: List[Dict[str, Any]]):
    """
    Receives a list of “year‐objects” (each dict must have a "Year" key plus
    all the other columns). For each dict:
      - If Year already exists in the existing Excel, overwrite that row.
      - Otherwise, append as a new row.
    Then write the updated DataFrame back to `financial_assumptions.xlsx`
    and refresh session_state.
    """
    print("years_data ",years_data)
    # 1) Validate input payload has at least one element and each has a "Year" key
    if not isinstance(years_data, list) or len(years_data) == 0:
        raise HTTPException(status_code=400, detail="Payload must be a non‐empty list of year‐objects.")

    for row in years_data:
        if "Year" not in row:
            raise HTTPException(status_code=400, detail="Every object must contain a 'Year' key.")

    # 2) Load existing Excel into a DataFrame (or create empty DF with correct columns if not present)
    filename="financial_assumptions.xlsx"
    file_path = DATA_DIR / filename
    if file_path.exists():
        try:
            df_existing = pd.read_excel(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not read {file_path}: {e}")
    else:
        # If the file doesn't exist yet, start with an empty DataFrame
        df_existing = pd.DataFrame()

    # 3) Convert incoming years_data into a DataFrame
    try:
        df_incoming = pd.DataFrame(years_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload format: {e}")

    # 4) Merge Logic: overwrite rows where Year matches, append new Years
    if "Year" not in df_existing.columns:
        # No “Year” column yet, so just write incoming as brand‐new file
        df_updated = df_incoming.copy()
    else:
        # Set index to Year on both for easy upsert
        df_existing_indexed = df_existing.set_index("Year")
        df_incoming_indexed = df_incoming.set_index("Year")

        # For each Year in incoming, overwrite or append
        # pandas’ combine_first will fill missing years in existing, but we want incoming to override existing
        # so: drop incoming Years from existing, then concat.
        overlapping_years = df_existing_indexed.index.intersection(df_incoming_indexed.index)
        if len(overlapping_years) > 0:
            # Drop those rows in existing
            df_existing_indexed = df_existing_indexed.drop(index=overlapping_years)

        # Now append all incoming
        df_updated_indexed = pd.concat([df_existing_indexed, df_incoming_indexed], axis=0, sort=False)

        # Reset index so "Year" is a column again
        df_updated = df_updated_indexed.reset_index()

    # 5) (Optional) Re‐order columns so that "Year" comes first, then all other columns
    cols = list(df_updated.columns)
    if "Year" in cols:
        cols.remove("Year")
        df_updated = df_updated[["Year"] + cols]

    # 6) Write df_updated back to Excel
    try:
        # ensure directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_updated.to_excel(file_path, index=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not write to {file_path}: {e}")

    # 7) Update session_state so that your in‐memory versions match the new file
    try:
        # Reload into session_state['df']
        session_state["df"] = df_updated.copy()

        # Re‐compute years_data (dictionary of per‐year dicts) via your existing helper
        processed = ecommerce_model.process_excel_data(df_updated)
        session_state["years_data"] = processed

        # Re‐compute debts_data (if you need it)
        load_resp = ecommerce_model.load_existing_data()
        session_state["debts_data"] = load_resp["debts_data"]

        # If you have to re‐run any “base analysis,” do it here:
        # await run_base_analysis(BaseAnalysisRequest(...))
        # (only if needed—omit if unnecessary)
    except Exception:
        # even if re‐computing session_state fails, we still consider the save itself successful
        pass
    try:
        last_year = int(df_updated["Year"].max())
    except Exception:
        last_year = None
    return JSONResponse({"status": "success", "message": "Assumptions saved to Excel.","last_year": last_year })


@app.get("/get_scenario_parameters/{scenario_type}", dependencies=[Depends(get_current_user)])
async def get_scenario_parameters(scenario_type: str):
    try:
        if scenario_type not in ["Base Case", "Best Case", "Worst Case"]:
            raise HTTPException(status_code=400, detail="Invalid scenario type")
        params = ecommerce_model._scenario_parameters(scenario_type)
        return JSONResponse(content={"status": "success", "parameters": params})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/revenue_chart_data")
async def revenue_chart_data():
    try:
        revenue_chart_data_response = ecommerce_model.create_revenue_chart_data(session_state['filtered_df'])
        return JSONResponse(content={"status": "success", "revenue_chart_data": revenue_chart_data_response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get('/traffic_chart_data')
async def traffic_chart_data():
    try:
       traffic_chart_data_response=ecommerce_model.create_traffic_chart_data(session_state['filtered_df'])
       return JSONResponse(content={"status":"success","traffic_chart_data":traffic_chart_data_response})
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    
@app.get('/profitability_chart_data')
async def profitability_chart_data():
    try:
        profitability_chart_data_response=ecommerce_model.create_profitability_chart_data(session_state['filtered_df'])
        return JSONResponse(content={"status":"success","data":profitability_chart_data_response})
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

@app.get('/waterfall_chart_data')
async def waterfall_chart_data():
    try:
        waterfall_chart_data_response=ecommerce_model.create_waterfall_chart_data(session_state['filtered_df'])
        return JSONResponse(content={"status":"success","data":waterfall_chart_data_response})
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    
@app.get('/breakeven_chart_data')
async def breakeven_chart_data():
    try:
        breakeven_chart_data_response=ecommerce_model.create_break_even_chart_data(session_state['filtered_df'])
        return JSONResponse(content={"status":"success","data":breakeven_chart_data_response})
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    
@app.get('/consideration_chart_data')
async def consideration_chart_data():
    try:
        consideration_chart_data_response=ecommerce_model.create_consideration_chart_data(session_state['filtered_df'])
        return JSONResponse(content={"status":"success","data":consideration_chart_data_response})
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    
@app.get('/margin_safety_chart')
async def margin_safety_chart_data():
    try:
        margin_safety_chart_data_response=ecommerce_model.create_margin_safety_chart_data(session_state['filtered_df'])
        return JSONResponse(content={"status":"success","data":margin_safety_chart_data_response})
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    
@app.get('/dcf_summary_chart_data')
async def dcf_summary_chart_data():
    try:
        dcf_summary_chart_data_response=ecommerce_model.create_dcf_summary_chart_data(session_state['valuation_details'])
        return JSONResponse(content={"status":"success","data":dcf_summary_chart_data_response})
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    
@app.get('/cashflow_forecast_chart_data')
async def cashflow_forecast_chart_data():
    try:
        cashflow_forecast_chart_data_response=ecommerce_model.create_cash_flow_forecast_chart(session_state['filtered_df'])
        return JSONResponse(content={"status":"success","data":cashflow_forecast_chart_data_response})
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

@app.get('/profitability_margin_trends_chart_data')
async def profitability_margin_trends_chart_data():
    try:
        profitability_margin_trends_chart_data_response=ecommerce_model.create_profitability_margin_trends_chart(session_state['filtered_df'])
        return JSONResponse(content={"status":"success","data":profitability_margin_trends_chart_data_response})
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
# Need To check from this API

@app.get('/key_implications',dependencies=[Depends(get_current_user)])
async def key_implications_of_three_cases():
    try:
        implications = {
                        'Base Case': "Stable performance baseline with moderate growth and risk.",
                        'Best Case': "Optimistic growth with improved efficiencies and reduced costs.",
                        'Worst Case': "Challenging conditions with higher costs and reduced revenue."
                    }
        return JSONResponse(content={"status":"success","data":implications})
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

@app.get("/display_metrics_scenario_analysis")
async def display_metrics_scenario_analysis(
    selected_scenario: str = Query("Base Case", description="One of: Base Case, Best Case, Worst Case")
):
    try:
        scenarios = session_state.get("scenarios", {})
        if selected_scenario not in scenarios:
            raise HTTPException(
                status_code=400,
                detail=f"Scenario '{selected_scenario}' not found"
            )
        full_df = scenarios[selected_scenario]
        if full_df is None or full_df.empty:
            raise HTTPException(
                status_code=400,
                detail=f"No data for scenario '{selected_scenario}'"
            )

        # Pass the entire DataFrame into the updated function
        
        result = ecommerce_model.display_metrics_scenario_analysis(full_df)
        return JSONResponse(content={"status": "success", "data": result})
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
# FastAPI endpoint
@app.get("/export_excel")
async def export_excel():
    try:        
        # Generate Excel file
        df        = session_state.get("filtered_df")
        scenarios = session_state.get("scenarios")
        val_det   = session_state.get("valuation_details")

        if df is None or scenarios is None or val_det is None:
            raise HTTPException(400, "Insufficient data in session to build report")

        try:
            # This returns a bytes object containing the XLSX file.
            excel_data = ecommerce_model.combine_exports_to_excel(df, scenarios, val_det)
            
        # Return as downloadable file
            filename = "output_financial_assumptions.xlsx"
            headers = {
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            }
            return StreamingResponse(
                BytesIO(excel_data),
                headers=headers,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception:
            logger.exception("Failed to generate Excel report")
            # return the actual error to the client in dev; in production you might hide details
            raise HTTPException(500, "Error generating Excel file; see server logs")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Excel file: {str(e)}")
    
@app.get('/display_metrics_summary_of_analysis')
async def display_metrics_summary_of_analysis():
    try:
        df_length = len(session_state["years_data"])

        # Call your existing model method to get a DataFrame back:
        summary_df = ecommerce_model.display_summary_of_analysis(
            session_state["scenarios"],
            df_length,
        )

        result: List[SummaryMetric] = []

        # Helper that turns NaN or infinite → 0.0, otherwise rounds normally
        def safe_round(val: float, digits: int) -> float:
            if pd.isna(val) or not np.isfinite(val):
                return 0.0
            return round(val, digits)

        for _, row in summary_df.iterrows():
            result.append(
                SummaryMetric(
                    Scenario=row.get("Scenario", ""),
                    Net_Income_M=safe_round(row.get("Net Income ($M)"), 1),
                    EBITDA_M=safe_round(row.get("EBITDA ($M)"), 1),
                    IRR=safe_round(row.get("IRR (%)"), 1),
                    NPV_M=safe_round(row.get("NPV ($M)"), 1),
                    Payback_Period_Orders=safe_round(
                        row.get("Payback Period (Orders)"), 2
                    ),
                    Gross_Profit_Margin=safe_round(
                        row.get("Gross Profit Margin (%)"), 1
                    ),
                    Net_Profit_Margin=safe_round(
                        row.get("Net Profit Margin (%)"), 1
                    ),
                    Net_Cash_Flow_M=safe_round(
                        row.get("Net Cash Flow ($M)"), 1
                    ),
                )
            )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error computing summary metrics: {str(e)}"
        )

@app.get("/dcf_valuation", response_model=ValuationResponse)
async def dcf_valuation():
    try:
        # Convert values to millions and round to 1 decimal place
        enterprise_value_m = round(session_state['valuation_details']["enterprise_value"] / 1e6, 1) if session_state['valuation_details']["enterprise_value"] is not None and not pd.isna(session_state['valuation_details']["enterprise_value"]) else 0.0
        equity_value_m = round(session_state['valuation_details']["equity_value"] / 1e6, 1) if session_state['valuation_details']["equity_value"] is not None and not pd.isna(session_state['valuation_details']["equity_value"]) else 0.0

        return ValuationResponse(
            enterprise_value_m=enterprise_value_m,
            equity_value_m=equity_value_m
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing valuation details: {str(e)}")
    
@app.get("/customer_metrics",response_model=CustomerMetricsResponse)
async def customer_metrics():
    try:
        display_df,irr_value=ecommerce_model.display_customer_metrics(session_state['filtered_df'])
        # Convert to list of dictionaries with formatted values
        metrics = []
        columns = ['Year', 'NPV', 'CAC', 'Contribution Margin Per Order', 'LTV', 'LTV/CAC Ratio', 'Payback Orders', 'Burn Rate']
        for _, row in display_df[columns].iterrows():
            metric = {
                'Year': float(row['Year']) if pd.notna(row['Year']) else None,
                'NPV': float(row['NPV']) if pd.notna(row['NPV']) else None,
                'CAC': round(float(row['CAC']), 2) if pd.notna(row['CAC']) else None,
                'Contribution_Margin_Per_Order': float(row['Contribution Margin Per Order']) if pd.notna(row['Contribution Margin Per Order']) else None,
                'LTV': float(row['LTV']) if pd.notna(row['LTV']) else None,
                'LTV_CAC_Ratio': round(float(row['LTV/CAC Ratio']), 2) if pd.notna(row['LTV/CAC Ratio']) else None,
                'Payback_Orders': round(float(row['Payback Orders']), 2) if pd.notna(row['Payback Orders']) else None,
                'Burn_Rate': float(row['Burn Rate']) if pd.notna(row['Burn Rate']) else None
            }
            metrics.append(CustomerMetric(**metric))
        
        return CustomerMetricsResponse(
            irr=irr_value,
            metrics=metrics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing customer metrics: {str(e)}")    

@app.get("/operational_metrics", response_model=OperationalMetricsResponse)
async def operational_metrics():
    try:
        # Convert input to pandas DataFrame
        df = session_state['filtered_df']

        # Initialize metrics list
        metrics = [
            {'Metric': 'Revenue Growth', 'column': 'Net Revenue', 'calc': lambda x: x.pct_change().iloc[-1] * 100 if len(x) > 1 else 0.0, 'decimals': 1},
            {'Metric': 'Gross Margin', 'column': 'Gross Margin', 'calc': lambda x: x.iloc[-1] * 100, 'decimals': 1},
            {'Metric': 'EBITDA Margin', 'column': 'EBITDA Margin', 'calc': lambda x: x.iloc[-1] * 100, 'decimals': 1},
            {'Metric': 'Return on Equity', 'column': 'Return on Equity', 'calc': lambda x: x.iloc[-1] * 100, 'decimals': 1},
            {'Metric': 'Asset Turnover', 'column': 'Asset Turnover', 'calc': lambda x: x.iloc[-1], 'decimals': 2}
        ]

        # Compute metrics
        result = []
        for metric in metrics:
            column = metric['column']
            if column in df.columns:
                value = df[column]
                try:
                    calc_value = metric['calc'](value)
                    if pd.isna(calc_value) or np.isinf(calc_value):
                        calc_value = 0.0
                    formatted_value = round(float(calc_value), metric['decimals'])
                except (ValueError, TypeError, IndexError):
                    formatted_value = 0.0
            else:
                formatted_value = 0.0

            result.append(OperationalMetric(
                metric=metric['Metric'],
                current=formatted_value
            ))

        return OperationalMetricsResponse(metrics=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing operational metrics: {str(e)}")

@app.get("/financial_schedules", response_model=SchedulesResponse)
async def get_financial_schedules(schedules: List[ScheduleType] = Query([], description="Select one or more financial schedule types")):
    try:
        filtered_df=session_state['filtered_df'].copy()
        filtered_df = filtered_df.reset_index(drop=True)
        debt_schedules = ecommerce_model.calculate_cash_flow_statement_with_debt_schedules(filtered_df,session_state['debts_data'])
        if not schedules:
            return SchedulesResponse(schedules=[])
        print("filtered_df :", filtered_df.columns.to_list())
        # Define column mappings and formatting
        schedule_configs = {
            ScheduleType.INCOME_STATEMENT: {
                'cols': ['Year', 'Net Revenue', 'Gross Profit', 'EBITDA', 'Net Income', 'Total Orders'],
                'format': {
                    'Year': lambda x: float(x) if pd.notna(x) else None,
                    'Net Revenue': lambda x: float(x) if pd.notna(x) else None,
                    'Gross Profit': lambda x: float(x) if pd.notna(x) else None,
                    'EBITDA': lambda x: float(x) if pd.notna(x) else None,
                    'Net Income': lambda x: float(x) if pd.notna(x) else None,
                    'Total Orders': lambda x: float(x) if pd.notna(x) else None
                }
            },
            ScheduleType.CASH_FLOW_STATEMENT: {
                'cols': ['Year', 'Cash from Operations', 'Cash from Investing', 'Cash from Financing', 'Net Cash Flow'],
                'format': {
                    'Year': lambda x: float(x) if pd.notna(x) else None,
                    'Cash from Operations': lambda x: float(x) if pd.notna(x) else None,
                    'Cash from Investing': lambda x: float(x) if pd.notna(x) else None,
                    'Cash from Financing': lambda x: float(x) if pd.notna(x) else None,
                    'Net Cash Flow': lambda x: float(x) if pd.notna(x) else None
                }
            },
            ScheduleType.BALANCE_SHEET: {
                'cols': ['Year', 'Total Assets', 'Total Liabilities', 'Total Equity', 'Balance Sheet Check'],
                'format': {
                    'Year': lambda x: float(x) if pd.notna(x) else None,
                    'Total Assets': lambda x: float(x) if pd.notna(x) else None,
                    'Total Liabilities': lambda x: float(x) if pd.notna(x) else None,
                    'Total Equity': lambda x: float(x) if pd.notna(x) else None,
                    'Balance Sheet Check': lambda x: float(x) if pd.notna(x) else None
                }
            },
            ScheduleType.CAPITAL_ASSETS: {
                'cols': ['Year', 'Total Opening Balance', 'Total Additions', 'Total Depreciation', 'Total Closing Balance'],
                'format': {
                    'Year': lambda x: float(x) if pd.notna(x) else None,
                    'Total Opening Balance': lambda x: float(x) if pd.notna(x) else None,
                    'Total Additions': lambda x: float(x) if pd.notna(x) else None,
                    'Total Depreciation': lambda x: float(x) if pd.notna(x) else None,
                    'Total Closing_Balance': lambda x: float(x) if pd.notna(x) else None
                }
            },
            ScheduleType.VALUATION: {
                'cols': ['Year', 'EBIT', 'Unlevered FCF', 'PV of FCF', 'Total Enterprise Value'],
                'format': {
                    'Year': lambda x: float(x) if pd.notna(x) else None,
                    'EBIT': lambda x: float(x) if pd.notna(x) else None,
                    'Unlevered FCF': lambda x: float(x) if pd.notna(x) else None,
                    'PV of FCF': lambda x: float(x) if pd.notna(x) else None,
                    'Total Enterprise Value': lambda x: float(x) if pd.notna(x) else None
                }
            },
            ScheduleType.CUSTOMER_METRICS: {
                'cols': ['Year', 'CAC', 'Contribution Margin Per Order', 'LTV', 'LTV/CAC Ratio', 'Payback Orders'],
                'format': {
                    'Year': lambda x: float(x) if pd.notna(x) else None,
                    'CAC': lambda x: round(float(x), 2) if pd.notna(x) else None,
                    'Contribution Margin Per Order': lambda x: round(float(x), 2) if pd.notna(x) else None,
                    'LTV': lambda x: round(float(x), 2) if pd.notna(x) else None,
                    'LTV/CAC Ratio': lambda x: round(float(x), 2) if pd.notna(x) else None,
                    'Payback Orders': lambda x: round(float(x), 2) if pd.notna(x) else None
                }
            }
        }

        result = []
        for schedule in schedules:
            # Handle Debt Payment Schedule
            if schedule == ScheduleType.DEBT_PAYMENT_SCHEDULE:
                debt_data = []
                for year, schedules in debt_schedules.items():
                    year_schedules = []
                    for debt in schedules:
                        schedule_df = debt['Schedule'].replace([np.inf, -np.inf, np.nan], None)
                        debt_schedule = [
                            DebtScheduleRecord(
                                Year=float(row['Year']) if pd.notna(row['Year']) else None,
                                Principal=round(float(row['Principal']), 2) if pd.notna(row['Principal']) else None,
                                Interest=round(float(row['Interest']), 2) if pd.notna(row['Interest']) else None,
                                Payment=round(float(row['Payment']), 2) if pd.notna(row['Payment']) else None
                            )
                            for _, row in schedule_df.iterrows()
                        ]
                        year_schedules.append(DebtDetail(
                            Debt_Name=debt['Debt Name'],
                            Amount=float(debt['Amount']),
                            Interest_Rate=float(debt['Interest Rate']),
                            Duration=int(debt['Duration']),
                            Schedule=debt_schedule
                        ))
                    debt_data.append(DebtYear(Year=int(year), Schedules=year_schedules))
                result.append(ScheduleData(schedule=schedule, data=debt_data))
                continue

            # Handle other schedules
            schedule_info = schedule_configs.get(schedule)
            if not schedule_info:
                continue  # Skip invalid schedule types

            cols = schedule_info['cols']
            format_funcs = schedule_info['format']

            # Select available columns
            available_cols = [col for col in cols if col in filtered_df.columns]
            if not available_cols:
                continue  # Skip schedules with no valid columns

            # Create display DataFrame
            display_df = filtered_df[available_cols].copy()
            display_df = display_df.replace([np.inf, -np.inf, np.nan], None)

            # Convert to list of records
            data = []
            for _, row in display_df.iterrows():
                record = {}
                for col in cols:
                    value = row[col] if col in row else None
                    if value is not None and col in format_funcs:
                        value = format_funcs[col](value)
                    record[col.replace(' ', '_')] = value
                data.append(ScheduleRecord(**record))
            print("schedule ",schedule)
            print("data ", data)
            result.append(ScheduleData(schedule=schedule, data=data))

        return SchedulesResponse(schedules=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing schedules: {str(e)}")


@app.post("/monte_carlo", response_model=MonteCarloResponse)
async def run_monte_carlo_simulation(params: MonteCarloInput):
    try:
        # 1) VALIDATE payload ranges
        if params.forecast_years < 1:
            raise HTTPException(400, "Forecast years must be at least 1")
        if params.num_simulations < 100:
            raise HTTPException(400, "Number of simulations must be at least 100")
        if not 80 <= params.confidence_level <= 99:
            raise HTTPException(400, "Confidence level must be between 80 and 99")
        if params.discount_rate <= 0 or params.wacc <= 0:
            raise HTTPException(400, "Discount rate and WACC must be positive")

        # 2) PULL & CLEAN the DataFrame
        raw = session_state.get("filtered_df")
        if raw is None:
            raise HTTPException(400, "No filtered_df in session_state")
        cleaned_df = pd.DataFrame(raw).copy()

        # convert only the columns we’ll actually use
        numeric_cols = [
            "Net Revenue", "COGS", "Total Variable Costs",
            "Total Fixed Costs", "Interest",
            "Tax Rate", "Depreciation", "Capital Expenditures"
        ]
        for col in numeric_cols:
            try:
                cleaned_df[col] = (
                    pd.to_numeric(cleaned_df.get(col, 0), errors="coerce")
                      .fillna(0)
                      .clip(lower=0)
                )
            except TypeError:
                logger.warning(f"Skipping non-numeric column '{col}'")

        # 3) ENSURE Tax Rate exists
        if "Tax Rate" not in cleaned_df.columns:
            cleaned_df["Tax Rate"] = 0.3

        # 4) RUN the simulation
        #    convert rates from percentages to proportions
        dr = params.discount_rate / 100.0
        w  = params.wacc            / 100.0
        pg = params.perpetual_growth/ 100.0

        sim_results, confidence_bounds = ecommerce_model.run_monte_carlo_simulation(
            cleaned_df,
            params.forecast_years,
            params.num_simulations,
            params.confidence_level,
            dr, w, pg,
            params.distribution_type.value
        )
        if sim_results is None:
            raise HTTPException(500, "Simulation failed")

        # 5) EXTRACT & SCALE final-year arrays → Python lists
        raw_income = sim_results["Net Income"][:, -1] / 1e6
        raw_npv    = sim_results["NPV"]             / 1e6

        final_year_income = raw_income.tolist()
        npv_values        = raw_npv.tolist()

        mean_income = float(np.nanmean(final_year_income))
        mean_npv    = float(np.nanmean(npv_values))

        # 6) SAFELY compute top-level confidence intervals
        try:
            ci_values = np.nanpercentile(final_year_income, confidence_bounds).tolist()
        except Exception as e:
            logger.error("Top-level percentile error (income):\n" + traceback.format_exc())
            ci_values = [0.0, 0.0]

        try:
            ci_npv = np.nanpercentile(npv_values, confidence_bounds).tolist()
        except Exception as e:
            logger.error("Top-level percentile error (NPV):\n" + traceback.format_exc())
            ci_npv = [0.0, 0.0]

        # 7) BUILD per-metric summaries
        metrics = {}
        available = [
            "Net Income", "NPV", "EBITDA",
            "Net Revenue", "Cash from Operations",
            "discount_rate", "wacc", "perpetual_growth"
        ]
        for metric in available:
            if metric not in sim_results:
                continue

            arr = sim_results[metric]
            # pick last-year slice if 2D, else use 1D
            vals = (arr[:, -1] if arr.ndim == 2 else arr)

            # scale & unit
            if metric in ("discount_rate", "wacc", "perpetual_growth"):
                scaled = vals * 100.0
                unit = "%"
            else:
                scaled = vals / 1e6
                unit = "$M"

            # filter out zeros & NaNs
            mask = np.isfinite(scaled) & (scaled != 0)
            valid = scaled[mask]

            # percentile for this metric
            try:
                lower = float(np.nanpercentile(valid, confidence_bounds[0]))
                upper = float(np.nanpercentile(valid, confidence_bounds[1]))
            except Exception:
                logger.error(f"Percentile error for metric {metric}:\n" + traceback.format_exc())
                lower = upper = 0.0

            metrics[metric] = MetricSummary(
                mean=float(np.nanmean(valid)) if valid.size else 0.0,
                ci_lower=lower,
                ci_upper=upper,
                unit=unit,
                final_year_values=valid.tolist()
            )

        # 8) RETURN only pure-Python types
        return MonteCarloResponse(
            status="success",
            metrics=metrics,
            confidence_level=params.confidence_level,
            final_year_income=final_year_income,
            npv_values=npv_values,
            mean_income=mean_income,
            ci_values=ci_values,
            mean_npv=mean_npv,
            ci_npv=ci_npv
        )

    except HTTPException:
        # re-raise FastAPI HTTPExceptions unchanged
        raise
    except Exception:
        logger.error("Monte Carlo endpoint exception:\n" + traceback.format_exc())
        raise HTTPException(500, "Internal server error")


@app.post("/what_if", response_model=WhatIfResponse)
async def run_what_if_analysis(params: WhatIfInput):
    try:
        # Validate inputs
        if params.num_adjustments < 1 or params.num_adjustments > 5:
            raise HTTPException(status_code=400, detail="Number of adjustments must be between 1 and 5")
        if len(params.adjustments) != params.num_adjustments:
            raise HTTPException(status_code=400, detail="Number of adjustments does not match adjustment list")
        if params.discount_rate <= 0:
            raise HTTPException(status_code=400, detail="Discount rate must be positive")      
       
        
        for adj in params.adjustments:
            
            if adj.multiplier < 0.5 or adj.multiplier > 1.5:                
                raise HTTPException(status_code=400, detail="Multiplier must be between 0.5 and 1.5")
            if adj.year < session_state['forecast_df']['Year'].min() or adj.year > session_state['forecast_df']['Year'].max():
                
                raise HTTPException(status_code=400, detail=f"Year {adj.year} is outside forecast range {session_state['forecast_df']['Year'].min()}-{session_state['forecast_df']['Year'].max()}")
            
        adjusted_forecast = ecommerce_model.apply_what_if_adjustments(session_state['forecast_df'], [
            {'year': adj.year, 'variable': adj.variable.value, 'multiplier': adj.multiplier}
            for adj in params.adjustments
        ])        
        
        # Combine historical and adjusted forecast (assuming historical df is empty for now)
        # what_if_df = pd.concat([session_state['df'], adjusted_forecast]) if not session_state['df'].empty else adjusted_forecast
        what_if_df=adjusted_forecast
        # Perform financial calculations
        ecommerce_model.calculate_income_statement(what_if_df)
        ecommerce_model.calculate_cash_flow_statement(what_if_df)
        ecommerce_model.calculate_supporting_schedules(what_if_df)
        ecommerce_model.calculate_balance_sheet(what_if_df)
        what_if_df = ecommerce_model.calculate_valuation(what_if_df, params.discount_rate)
        what_if_df.fillna(0, inplace=True)
        
        # Prepare results
        display_cols = ['Year', 'Net Revenue', 'Gross Profit', 'EBITDA', 'Net Income', 'Total Orders']
        missing_cols = [col for col in display_cols if col not in what_if_df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in results: {missing_cols}")
        
        results = [
            MetricSummaryWhatIf(
                year=int(row['Year']),
                net_revenue=float(row['Net Revenue']),
                gross_profit=float(row['Gross Profit']),
                ebitda=float(row['EBITDA']),
                net_income=float(row['Net Income']),
                total_orders=float(row['Total Orders'])
            )
            for _, row in what_if_df[display_cols].iterrows()
        ]
        
        # Check for warnings
        warnings = []
        if what_if_df['Net Income'].iloc[-1] < 0 and any(adj.multiplier > 1 and adj.variable in [
            WhatIfVariable.TOTAL_ORDERS, WhatIfVariable.AVERAGE_ITEM_VALUE, WhatIfVariable.PAID_SEARCH_TRAFFIC
        ] for adj in params.adjustments):
            warnings.append("Negative Net Income despite positive adjustments. Verify adjustment logic or input data.")
        
        return WhatIfResponse(
            status="success",
            results=results,
            adjustments=params.adjustments,
            warnings=warnings
        )
    
    except ValueError as e:
        logger.error(f"Endpoint error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/precision_tree", response_model=PrecisionTreeResponse)
async def run_precision_tree_analysis():
    try:
        logger.info("Received PrecisionTree analysis request")
        
        # Run precision tree analysis
        result = ecommerce_model.run_precision_tree(session_state['filtered_df'])
        
        # Prepare response
        return PrecisionTreeResponse(
            status="success",
            decision_outcomes=result["decision_outcomes"],
            decision_tree_image=result["decision_tree_image"],
            message=result["message"]
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/goal_seek", response_model=GoalSeekResponse)
async def run_goal_seek_analysis(params: GoalSeekInput):
    try:
       
        if params.target_profit_margin < 0 or params.target_profit_margin > 50:
            raise HTTPException(status_code=400, detail="Target profit margin must be between 0 and 50")
        if params.max_iterations < 1:
            raise HTTPException(status_code=400, detail="Max iterations must be at least 1")
        if params.tolerance <= 0:
            raise HTTPException(status_code=400, detail="Tolerance must be positive")
        if params.discount_rate <= 0:
            raise HTTPException(status_code=400, detail="Discount rate must be positive")
        if session_state['filtered_df'] is None or session_state['filtered_df'].empty:
            raise HTTPException(status_code=500, detail="DataFrame is not initialized")
        if params.year_to_adjust not in session_state['filtered_df']['Year'].values:
            raise HTTPException(status_code=400, detail=f"Year {params.year_to_adjust} not in DataFrame")
        if params.variable_to_adjust.value not in session_state['filtered_df'].columns:
            raise HTTPException(status_code=400, detail=f"Variable {params.variable_to_adjust.value} not in DataFrame")
        
        logger.info(f"Running Goal Seek: target_profit_margin={params.target_profit_margin}, variable={params.variable_to_adjust.value}, year={params.year_to_adjust}")

        # Calculate current profit margin for the selected year
        year_data = session_state['filtered_df'][session_state['filtered_df']['Year'] == params.year_to_adjust]
        if not year_data.empty:
            current_net_income = year_data['Net Income'].iloc[0]
            current_net_revenue = year_data['Net Revenue'].iloc[0]
            current_profit_margin = current_net_income / current_net_revenue if current_net_revenue != 0 else 0
            target_profit_margin = current_profit_margin * (1 + params.target_profit_margin)
            
            # st.write(f"Current Profit Margin for {params.year_to_adjust}: {current_profit_margin:.2%}")
            # st.write(f"Target Profit Margin: {target_profit_margin:.2%}")
        
            # Run goal seek
            #filtered_df,session_state['years_data'], request.profit_margin_increase, request.variable_to_adjust, request.goal_year
            multiplier, adjusted_df = ecommerce_model.run_goal_seek(
                session_state['filtered_df'],
                session_state['years_data'],
                params.target_profit_margin,
                params.variable_to_adjust.value,
                params.year_to_adjust,
                params.max_iterations,
                params.tolerance,
                # params.discount_rate
            )
            
            if multiplier is None or adjusted_df is None:
                return GoalSeekResponse(
                    status="error",
                    message="Goal Seek failed to find a solution"
                )
            
            # Prepare results
            display_cols = ['Year', 'Net Revenue', 'Gross Profit', 'EBITDA', 'Net Income', 'Total Orders']
            missing_cols = [col for col in display_cols if col not in adjusted_df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in results: {missing_cols}")
            
            results = [
                MetricSummary_GS(
                    year=int(row['Year']),
                    net_revenue=float(row['Net Revenue']),
                    gross_profit=float(row['Gross Profit']),
                    ebitda=float(row['EBITDA']),
                    net_income=float(row['Net Income']),
                    total_orders=float(row['Total Orders'])
                )
                for _, row in adjusted_df[display_cols].iterrows()
            ]
            
            return GoalSeekResponse(
                status="success",
                goal_seek_year=params.year_to_adjust,
                current_profit_margin=current_profit_margin,
                target_profit_margin=target_profit_margin,
                multiplier=multiplier,
                results=results,
                message=f"Required multiplier for {params.variable_to_adjust.value}: {multiplier:.2f}x"
            )
    
    except ValueError as e:
        logger.error(f"Endpoint error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")   

@app.post("/schedule_risk_analysis", response_model=ScheduleRiskResponse)
async def run_schedule_risk_analysis(params: ScheduleRiskInput):
    try:
        # Validate inputs
        if params.num_simulations < 100:
            raise HTTPException(status_code=400, detail="Number of simulations must be at least 100")
        if params.confidence_level < 80 or params.confidence_level > 99:
            raise HTTPException(status_code=400, detail="Confidence level must be between 80 and 99")
        
        logger.info(f"Received ScheduleRiskAnalysis request: num_simulations={params.num_simulations}, confidence_level={params.confidence_level}")
        
        # Run schedule risk analysis
        # Note: df is not used in the method, but passed for compatibility
        result = ecommerce_model.run_schedule_risk_analysis(
            session_state['filtered_df'],  # Empty DataFrame since not used
            num_simulations=params.num_simulations,
            confidence_level=params.confidence_level
        )
        
        # Prepare response
        return ScheduleRiskResponse(
            status="success",
            simulation_durations=result["simulation_durations"],
            mean_duration=result["mean_duration"],
            confidence_interval=ConfidenceInterval(
                lower=result["confidence_interval"]["lower"],
                upper=result["confidence_interval"]["upper"],
                confidence_level=result["confidence_interval"]["confidence_level"]
            ),
            tasks=[Task(**task) for task in result["tasks"]],
            message=f"ScheduleRiskAnalysis completed: Mean Completion Time = {result['mean_duration']:.1f} days, {params.confidence_level}% CI = [{result['confidence_interval']['lower']:.1f}, {result['confidence_interval']['upper']:.1f}] days"
        )
    
    except ValueError as e:
        logger.error(f"Endpoint error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")    

@app.post("/neural_tools", response_model=NeuralToolsResponse)
async def run_neural_tools_prediction(params: NeuralToolsInput):
    try:
        # Validate input
        if params.traffic_increase_percentage < 0:
            raise HTTPException(status_code=400, detail="Traffic increase percentage must be non-negative")
        if session_state['filtered_df'] is None or session_state['filtered_df'].empty:
            raise HTTPException(status_code=500, detail="DataFrame is not initialized")
        
        logger.info(f"Received NeuralTools prediction request: traffic_increase_percentage={params.traffic_increase_percentage}%")
        
        # Run neural tools prediction
        
        result = await run_in_threadpool(
                lambda:ecommerce_model.run_neural_tools_prediction(
                    df=session_state['filtered_df'],
                    traffic_increase_percentage=params.traffic_increase_percentage
                )
            )
        # Prepare response
        return NeuralToolsResponse(
            status="success",
            traffic_increase_percentage=result["traffic_increase_percentage"],
            predicted_revenue=result["predicted_revenue"],
            feature_importance=[
                FeatureImportance(feature=item["feature"], importance=item["importance"])
                for item in result["feature_importance"]
            ],
            message=result["message"]
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")    

@app.post("/top_rank_sensitivity", response_model=TopRankSensitivityResponse)
async def run_top_rank_sensitivity(params: TopRankSensitivityInput):
    try:
        # Validate input
        if not params.variables_to_test:
            raise HTTPException(status_code=400, detail="At least one variable must be provided")
        if params.change_percentage <= 0:
            raise HTTPException(status_code=400, detail="Change percentage must be positive")
        if params.discount_rate <= 0:
            raise HTTPException(status_code=400, detail="Discount rate must be positive")
        if session_state['filtered_df'] is None or session_state['filtered_df'].empty:
            raise HTTPException(status_code=500, detail="DataFrame is not initialized")
        
        logger.info(f"Received TopRank sensitivity request: variables={params.variables_to_test}, change_percentage={params.change_percentage}%")
        
        # Run sensitivity analysis
        result = ecommerce_model.run_top_rank_sensitivity(
            df=session_state['filtered_df'],
            variables_to_test=params.variables_to_test,
            change_percentage=params.change_percentage,
            discount_rate=params.discount_rate
        )
        
        # Prepare response
        return TopRankSensitivityResponse(
            status=result["status"],
            sensitivity_results=[
                SensitivityResult(
                    variable=item["variable"],
                    direction=item["direction"],
                    net_income_change=item["net_income_change"],
                    ebitda_change=item["ebitda_change"],
                    net_cash_flow_change=item["net_cash_flow_change"],
                    equity_value_change=item["equity_value_change"]
                )
                for item in result["sensitivity_results"]
            ],
            message=result["message"],
            sensitivity_insights=result['sensitivity_insights']
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/stat_tools_forecasting", response_model=StatToolsForecastingResponse)
async def run_stat_tools_forecasting(params: StatToolsForecastingInput):
    try:
        # Validate input
        if params.forecast_years < 1:
            raise HTTPException(status_code=400, detail="Number of forecast years must be at least 1")
        if params.confidence_level < 80 or params.confidence_level > 99:
            raise HTTPException(status_code=400, detail="Confidence level must be between 80 and 99")
        
        logger.info(f"Received StatTools forecasting request: forecast_years={params.forecast_years}, confidence_level={params.confidence_level}%")
        
        # Run forecasting
        result = ecommerce_model.run_stat_tools_forecasting(
            filtered_df=pd.DataFrame(),  # Dummy DataFrame since unused
            forecast_years=params.forecast_years,
            confidence_level=params.confidence_level
        )
        
        # Prepare response
        return StatToolsForecastingResponse(
            status="success",
            forecast_data=result["forecast_data"],
            summary_statistics=result["summary_statistics"],
            format_dict=result["format_dict"],
            chart_data=result["chart_data"],
            message=result["message"]
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")





@app.post("/evolver_optimization", response_model=EvolverOptimizationResponse)
async def run_evolver_optimization_endpoint(params: EvolverOptimizationInput):
    try:
        logger.info(f"Received Evolver optimization request: budget={params.budget_dict}, years={params.forecast_years}")

        if 'filtered_df' not in session_state:
            logger.error("Filtered DataFrame not found in session")
            raise HTTPException(status_code=400, detail="Filtered DataFrame not found")

        filtered_df = session_state['filtered_df']

        # Log DataFrame info
        logger.info(f"filtered_df columns: {filtered_df.columns.tolist()}")
        logger.info(f"filtered_df head: {filtered_df.head().to_dict()}")

        # Run optimization
        result = ecommerce_model.run_evolver_optimization(
            df=filtered_df,
            budget_dict=params.budget_dict,
            forecast_years=params.forecast_years
        )

        # Prepare response
        return EvolverOptimizationResponse(
            status="success",
            optimized_data=[OptimizedVariable(**item) for item in result["optimized_data"]],
            original_ebitda=result["original_ebitda"],
            optimized_ebitda=result["optimized_ebitda"],
            ebitda_change_percent=result["ebitda_change_percent"],
            message=result["message"]
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/filter_time_period")
async def filter_time_period(request: TimePeriodRequest):
    try:        
        scenarios = session_state.get('scenarios', {})
        scenario_df = scenarios.get(request.scenario_type, scenarios['Base Case'])
        
        # Filter by time period
        filtered_df = scenario_df[(scenario_df['Year'] >= request.start_year) & (scenario_df['Year'] <= request.end_year)].copy()
        filtered_df= filtered_df.reset_index(drop=True)
        if filtered_df.empty:   
            logger.error("No data in selected time period")
            raise HTTPException(status_code=400, detail="No data in selected time period")     
        if request.scenario_type not in ['Base Case','Best Case','Worst Case']:
            raise HTTPException(status_code=400, detail="Invalid Scenario Type") 
        filtered_df = filtered_df.reset_index(drop=True)
        # Recalculate metrics
        filtered_df = ecommerce_model.calculate_income_statement(filtered_df, request.tax_rate, request.inflation_rate, request.direct_labor_rate_increase)    
        ecommerce_model.calculate_cash_flow_statement(filtered_df)       
        ecommerce_model.calculate_supporting_schedules(filtered_df)       
        filtered_df = ecommerce_model.calculate_balance_sheet(filtered_df)    
        filtered_df = ecommerce_model.calculate_customer_metrics(filtered_df, request.discount_rate)       
        filtered_df = ecommerce_model.calculate_valuation(filtered_df, request.discount_rate)       
        filtered_df['IRR'] = ecommerce_model.calculate_project_irr(filtered_df)       
        filtered_df = ecommerce_model.calculate_additional_metrics(filtered_df)      
        filtered_df = ecommerce_model.calculate_break_even(filtered_df)      
        filtered_df = ecommerce_model.calculate_margin_of_safety(filtered_df)    
        filtered_df = ecommerce_model.calculate_consideration(filtered_df)
 
        session_state['filtered_df']=filtered_df    

        # Calculate valuation
        valuation_details = ecommerce_model.calculate_dcf_valuation(filtered_df, request.wacc, request.perpetual_growth)
        session_state['valuation_details']=valuation_details        

        return JSONResponse(content={"status":"success"})

    except Exception as e:
        logger.error(f"Error in time period filtering: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in time period filtering: {str(e)}") 

@app.post("/run_base_analysis", response_model=BaseAnalysisResponse)
async def run_base_analysis(request: BaseAnalysisRequest):
    try:
        # Create DataFrame
        df = ecommerce_model.create_dataframe()
        logger.info(f"DataFrame created with shape: {df.shape}, columns: {df.columns.tolist()}")
        if df.empty:
            logger.error("No data available for analysis.")
            raise HTTPException(status_code=400, detail="No data available")

        # Validate required metrics
        required_metrics = [
            'Email Traffic', 'Email Conversion Rate', 'Average Item Value',
            'Number of Items per Order', 'COGS Percentage',
            'Email Cost per Click', 'Churn Rate',
            'Accounts Receivable Days', 'Inventory Days', 'Accounts Payable Days',
            'Depreciation', 'Technology Development', 'Office Equipment',
            'Debt Issued', 'Equity Raised', 'Dividends Paid'
        ]
        missing_metrics = [m for m in required_metrics if m not in df.columns]
        if missing_metrics:
            logger.error(f"Missing required input metrics: {missing_metrics}")
            raise HTTPException(status_code=400, detail=f"Missing required input metrics: {missing_metrics}")

        df = ecommerce_model.calculate_income_statement(df, request.tax_rate, request.inflation_rate, request.direct_labor_rate_increase)
        df = ecommerce_model.calculate_cash_flow_statement(df)
        df = ecommerce_model.calculate_supporting_schedules(df)
        df = ecommerce_model.calculate_balance_sheet(df)
        df = ecommerce_model.calculate_customer_metrics(df, request.discount_rate)
        df = ecommerce_model.calculate_valuation(df, request.discount_rate)
        df['IRR'] = ecommerce_model.calculate_project_irr(df)
        df = ecommerce_model.calculate_additional_metrics(df)
        df = ecommerce_model.calculate_break_even(df)
        df = ecommerce_model.calculate_margin_of_safety(df)
        df = ecommerce_model.calculate_consideration(df)

        # 2) Forecast *only* the future years
        forecast_only = ecommerce_model.forecast_metrics(df, request.normal_forecast_years)
        forecast_only = ecommerce_model.calculate_income_statement(
            forecast_only, request.tax_rate, request.inflation_rate, request.direct_labor_rate_increase
        )
        forecast_only = ecommerce_model.calculate_cash_flow_statement(forecast_only)
        forecast_only = ecommerce_model.calculate_supporting_schedules(forecast_only)
        forecast_only = ecommerce_model.calculate_balance_sheet(forecast_only)
        forecast_only = ecommerce_model.calculate_valuation(forecast_only, request.discount_rate)
        forecast_only['IRR'] = ecommerce_model.calculate_project_irr(forecast_only)

        # 3) Build the Base Case by concatenating exactly once
        base_case_df = pd.concat([df, forecast_only], ignore_index=True)

        # 4) Now build Best & Worst Case using the same pattern
        scenarios = {"Base Case": base_case_df}
        for scen in ["Best Case", "Worst Case"]:
            params = ecommerce_model._scenario_parameters(scen)
            raw = ecommerce_model._create_scenarios(df, forecast_only,params).reset_index(drop=True)
            raw = ecommerce_model.calculate_income_statement(raw, request.tax_rate, request.inflation_rate, request.direct_labor_rate_increase)
            raw = ecommerce_model.calculate_cash_flow_statement(raw)
            raw = ecommerce_model.calculate_supporting_schedules(raw)
            raw = ecommerce_model.calculate_balance_sheet(raw)
            raw = ecommerce_model.calculate_valuation(raw, request.discount_rate)
            raw['IRR'] = ecommerce_model.calculate_project_irr(raw)
            scenarios[scen] = raw

        # 5) Store everything
        session_state['forecast_df']   = forecast_only
        session_state['filtered_df']   = base_case_df.copy()
        session_state['scenarios']     = scenarios
        session_state['full_base']     = base_case_df.reset_index(drop=True)
        valuation_details = ecommerce_model.calculate_dcf_valuation(
            session_state['filtered_df'], request.discount_rate, request.perpetual_growth
        )
        session_state['valuation_details'] = valuation_details
        return JSONResponse(content={"status":"success"})

    except Exception as e:
        logger.error(f"Error in base analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/select_scenario", response_model=ScenarioResponse)
async def select_scenario(
    scenario_type: str = Body(..., embed=True),
    scenario_params: ScenarioParams | None = Body(None),
    discount_rate: float = Body(..., embed=True),
    tax_rate: float = Body(..., embed=True),
    inflation_rate: float = Body(..., embed=True),
    direct_labor_rate_increase: float = Body(..., embed=True),
):
    """
    Recompute Best/Worst Case scenario on demand.
    """
    # 1) Grab and reset your existing base & forecast slices
    historical_df = session_state["filtered_df"].copy().reset_index(drop=True)
    forecast_df   = session_state["forecast_df"].copy().reset_index(drop=True)

    # 2) Drop any duplicate column names (Excel import can sometimes introduce them)
    historical_df = historical_df.loc[:, ~historical_df.columns.duplicated()]
    forecast_df   = forecast_df.loc[:, ~forecast_df.columns.duplicated()]

    # 3) Turn the Pydantic model into a plain dict if provided
    if scenario_params is not None:
        params: dict = scenario_params.dict()
    else:
        params = ecommerce_model._scenario_parameters(scenario_type)

    # 4) Validate you got all the keys you need
    required = {
        "conversion_rate_mult",
        "aov_mult",
        "cogs_mult",
        "interest_mult",
        "labor_mult",
        "material_mult",
        "markdown_mult",
        "political_risk",
        "env_impact",
    }
    missing = required - params.keys()
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing scenario parameters: {sorted(missing)}"
        )

    # 5) Build the new scenario Time‐Series
    raw_df = ecommerce_model._create_scenarios(
        historical_df,
        forecast_df,
        params
    ).reset_index(drop=True)

    # 6) Re‐run all downstream calculations on that fresh DataFrame
    df_scenario = ecommerce_model.calculate_income_statement(
        raw_df,
        tax_rate=tax_rate,
        inflation_rate=inflation_rate,
        direct_labor_rate_increase=direct_labor_rate_increase,
    )
    df_scenario = ecommerce_model.calculate_cash_flow_statement(df_scenario)
    df_scenario = ecommerce_model.calculate_supporting_schedules(df_scenario)
    df_scenario = ecommerce_model.calculate_balance_sheet(df_scenario)
    df_scenario = ecommerce_model.calculate_valuation(df_scenario, discount_rate)
    df_scenario["IRR"] = ecommerce_model.calculate_project_irr(df_scenario)

    # 7) Overwrite it in session_state
    session_state["scenarios"][scenario_type] = df_scenario

    # 8) Return exactly the two fields your ScenarioResponse expects
    return ScenarioResponse(
        status="success",
        message=f"{scenario_type} scenario generated successfully"
    )

import pandas as pd
import numpy as np
import numpy_financial as npf
from scipy.stats import norm, lognorm, uniform, expon, binom, poisson, geom, bernoulli, chi2, gamma, weibull_min, hypergeom,multinomial, beta, f
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.arima.model import ARIMA
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import logging
from typing import Dict, List, Any
import os
import math
from pathlib import Path
import shutil
from fastapi import HTTPException
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import wraps
import signal
import threading
import plotly.graph_objects as go
import plotly.io as pio
from xlsxwriter.exceptions import DuplicateWorksheetName

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_FILENAME = "financial_assumptions.xlsx"
DATA_DIR.mkdir(exist_ok=True)
EXCEL_FILE = DATA_DIR / DEFAULT_FILENAME

root_excel = BASE_DIR / DEFAULT_FILENAME
if root_excel.exists() and not EXCEL_FILE.exists():
    shutil.copy(root_excel, EXCEL_FILE)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Timeout decorator
def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(target)
                try:
                    future.result(timeout=seconds)
                except TimeoutError:
                    raise HTTPException(status_code=500, detail=f"Function {func.__name__} timed out after {seconds} seconds")
            if exception[0] is not None:
                raise exception[0]
            return result[0]
        return wrapper
    return decorator





def clean_float(value: float) -> float:
    """Replace non-finite float values with 0.0."""
    return float(value) if np.isfinite(value) else 0.0

# Helper function to validate DataFrame columns
def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        # raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_columns}")
        print(f"Missing required columns: {missing_columns}")

class EcommerceModel:
    def __init__(self):
        self.session_state = {'years_data': {}}
        self.scenarios = {'Base': None, 'Best': None, 'Worst': None}
        self.charts = {}
        self.current_file = 'data/financial_assumptions.xlsx'
        self.inflation_rate = 0.02
        self.labor_rate = 0.03
        self.risk_free_rate = 0.03
        self.terminal_growth_rate = 0.02
        self.num_simulations = 1000
    
    def _to_float(self, value: Any, field: str, year: int, default: float = 0.0) -> float:
        """Convert value to float, preserving zeros and handling errors."""
        try:
            if any(x in field.lower() for x in ['name', 'categories', 'types']):
                return default
            if isinstance(value, (list, tuple, dict)):
                logger.error(f"Expected a number for {field} in year {year}, got a sequence: {value}")
                return default
            if value is None or pd.isna(value):
                return 0.0
            if isinstance(value, str):
                try:
                    parsed = float(value)
                    return parsed
                except ValueError:
                    logger.error(f"Invalid numeric value for {field} in year {year}: {value}, using default {default}")
                    return default
            return float(value)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid numeric value for {field} in year {year}: {value}, using default {default}")
            return default

    def _to_int(self, value: Any, field: str, year: Any, default: int = 0) -> int:
        """Convert value to int, preserving zeros and handling errors."""
        try:
            if isinstance(value, (list, tuple)):
                raise ValueError(f"Expected an integer for {field} in year {year}, got a sequence: {value}")
            if value is None or pd.isna(value):
                return 0
            if isinstance(value, (np.floating, float)):
                return int(round(float(value)))
            return int(value)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid integer value for {field} in year {year}: {value}, using default {default}")
            return default

    def _parse_comma_separated(self, value: Any) -> List[str]:
        """Parse comma-separated string or return empty list."""
        if isinstance(value, str) and value.strip():
            return [item.strip() for item in value.split(',')]
        return []
    
    # Helper function to process Excel data
    def process_excel_data(self, df: pd.DataFrame) -> Dict[int, Dict]:
        # 1) Normalize/strip all column names once
        df = df.copy()
        df.columns = [col.strip() for col in df.columns]

        years_data: Dict[int, Dict] = {}
        for _, row in df.iterrows():
            # Turn the row into a plain dict, but first make sure 'Year' is an integer
            year = int(row['Year'])
            data: Dict[str, Any] = row.to_dict()

            # 2) Clean out NaNs and force numeric types
            for k, v in data.items():
                if isinstance(v, (int, float)) and not pd.isna(v):
                    data[k] = float(v)
                else:
                    # pandas.NA or NaN or None → set to 0.0
                    data[k] = 0.0 if pd.isna(v) else v

            # 3) Pull off Rent Categories (the same logic you already had)
            rent_categories_raw = data.get('Rent Categories', '')
            rent_categories = (
                rent_categories_raw.split(',')
                if isinstance(rent_categories_raw, str) and rent_categories_raw.strip()
                else []
            )
            for category in rent_categories:
                data[f"{category} Square Meters"] = float(data.get(f"{category} Square Meters", 0.0))
                data[f"{category} Cost per SQM"] = float(data.get(f"{category} Cost per SQM", 0.0))
                data[f"{category}"] = float(data.get(f"{category}", 0.0))

            # 4) Pull off Professional Fees (same as before)
            fee_types_raw = data.get('Professional Fee Types', '')
            fee_types = (
                fee_types_raw.split(',')
                if isinstance(fee_types_raw, str) and fee_types_raw.strip()
                else []
            )
            for ft in fee_types:
                data[f"{ft} Cost"] = float(data.get(f"{ft} Cost", 0.0))

            # 5) Pull off Assets
            assets = []
            i = 1
            while True:
                key_name = f"Asset_{i}_Name"
                if key_name not in data:
                    break
                name = data.get(f"Asset_{i}_Name", f"Asset {i}")
                amount = float(data.get(f"Asset_{i}_Amount", 0.0))
                rate = float(data.get(f"Asset_{i}_Rate", 0.0))
                if amount > 0 or rate > 0:
                    assets.append({
                        "name": name,
                        "amount": amount,
                        "rate": rate
                    })
                i += 1
            data['Assets'] = assets

            # 6) Pull off Debts
            debts: List[Dict[str, Any]] = []
            i = 1
            while True:
                key_debt_name = f"Debt_{i}_Name"
                if key_debt_name not in data:
                    break
                name = data.get(key_debt_name, f"Debt {i}")
                amount = float(data.get(f"Debt_{i}_Amount", 0.0))
                interest_rate = float(data.get(f"Debt_{i}_Interest_Rate", 0.0))
                duration = int(data.get(f"Debt_{i}_Duration", 1))

                # Only append if there is actually a non‐zero principal
                if amount > 0:
                    debts.append({
                        "name": name,
                        "amount": amount,
                        "interest_rate": interest_rate,
                        "duration": duration
                    })
                i += 1
            data['Debts'] = debts
            print(f"debts : {data['Debts']} (year {year})")

            years_data[year] = data
            
        return years_data

    def load_existing_data(self) -> Dict:
        """Load financial_assumptions.xlsx if present, preserving zero values."""
        file_path = self.current_file
        logger.info(f"Attempting to load file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return {"status": "error", "data": {}, "detail": f"File not found: {file_path}"}

        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            logger.info(f"Successfully read Excel file with {len(df)} rows")
            
            years_data = {}
            debts_data={}
            expected_columns = [
                'Year', 'Email Traffic', 'Organic Search Traffic', 'Paid Search Traffic', 'Affiliates Traffic',
                'Email Conversion Rate', 'Organic Search Conversion Rate', 'Paid Search Conversion Rate',
                'Affiliates Conversion Rate', 'Average Item Value', 'Number of Items per Order',
                'Average Markdown', 'Average Promotion/Discount', 'COGS Percentage', 'Churn Rate',
                'Email Cost per Click', 'Organic Search Cost per Click', 'Paid Search Cost per Click',
                'Affiliates Cost per Click', 'Freight/Shipping per Order', 'Labor/Handling per Order',
                'General Warehouse Rent', 'Other', 'Interest', 'Tax Rate', 'Direct Staff Hours per Year',
                'Direct Staff Number', 'Direct Staff Hourly Rate', 'Direct Staff Total Cost',
                'Indirect Staff Hours per Year', 'Indirect Staff Number', 'Indirect Staff Hourly Rate',
                'Indirect Staff Total Cost', 'Part-Time Staff Hours per Year', 'Part-Time Staff Number',
                'Part-Time Staff Hourly Rate', 'Part-Time Staff Total Cost', 'CEO Salary', 'COO Salary',
                'CFO Salary', 'Director of HR Salary', 'CIO Salary', 'Pension Cost per Staff',
                'Pension Total Cost', 'Medical Insurance Cost per Staff', 'Medical Insurance Total Cost',
                'Child Benefit Cost per Staff', 'Child Benefit Total Cost', 'Car Benefit Cost per Staff',
                'Car Benefit Total Cost', 'Total Benefits', 'Salaries, Wages & Benefits', 'Office Rent',
                'Rent Categories', 'Professional Fees', 'Professional Fee Types', 'Depreciation',
                'Accounts Receivable Days', 'Inventory Days', 'Accounts Payable Days',
                'Technology Development', 'Office Equipment', 'Technology Depreciation Years',
                'Office Equipment Depreciation Years', 'Interest Rate', 'Equity Raised', 'Dividends Paid',
                'Debt Issued'
            ]
            
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing columns in Excel file: {missing_columns}")
            
            for _, row in df.iterrows():
                year = self._to_int(row.get('Year'), 'Year', row.get('Year'))
                if not year:
                    logger.warning(f"Skipping row with invalid year: {row.get('Year')}")
                    continue
                
                year_data = {}
                for col in expected_columns:
                    if col in ['Rent Categories', 'Professional Fee Types']:
                        year_data[col] = str(row.get(col, '')) if pd.notna(row.get(col)) else ''
                    elif col in ['Assets', 'Debts']:
                        continue
                    else:
                        value = row.get(col, None)
                        year_data[col] = self._to_float(value, col, year) if col != 'Year' else year
                
                rent_categories = self._parse_comma_separated(row.get('Rent Categories', ''))
                for cat in rent_categories:
                    if cat:
                        year_data[f"{cat} Square Meters"] = self._to_float(row.get(f"{cat} Square Meters"), f"{cat} Square Meters", year)
                        year_data[f"{cat} Cost per SQM"] = self._to_float(row.get(f"{cat} Cost per SQM"), f"{cat} Cost per SQM", year)
                        year_data[cat] = self._to_float(row.get(cat), cat, year)
                
                fee_types = self._parse_comma_separated(row.get('Professional Fee Types', ''))
                for fee in fee_types:
                    if fee:
                        year_data[f"{fee} Cost"] = self._to_float(row.get(f"{fee} Cost"), f"{fee} Cost", year)
                
                assets = []
                i = 1
                while f"Asset_{i}_Name" in row:
                    if pd.notna(row[f"Asset_{i}_Name"]):
                        assets.append({
                            'name': str(row[f"Asset_{i}_Name"]),
                            'amount': self._to_float(row.get(f"Asset_{i}_Amount"), f"Asset_{i}_Amount", year),
                            'rate': self._to_float(row.get(f"Asset_{i}_Rate"), f"Asset_{i}_Rate", year)
                        })
                        year_data[f"Asset_{i}_Name"] = str(row[f"Asset_{i}_Name"])
                        year_data[f"Asset_{i}_Amount"] = self._to_float(row.get(f"Asset_{i}_Amount"), f"Asset_{i}_Amount", year)
                        year_data[f"Asset_{i}_Rate"] = self._to_float(row.get(f"Asset_{i}_Rate"), f"Asset_{i}_Rate", year)
                        year_data[f"Asset_{i}_Depreciation"] = self._to_float(row.get(f"Asset_{i}_Depreciation"), f"Asset_{i}_Depreciation", year)
                        year_data[f"Asset_{i}_NBV"] = self._to_float(row.get(f"Asset_{i}_NBV"), f"Asset_{i}_NBV", year)
                    i += 1
                year_data['Assets'] = assets
                
                debts = []
                i = 1
                while f"Debt_{i}_Name" in row:
                    if pd.notna(row[f"Debt_{i}_Name"]):
                        debts.append({
                            'name': str(row[f"Debt_{i}_Name"]),
                            'amount': self._to_float(row.get(f"Debt_{i}_Amount"), f"Debt_{i}_Amount", year),
                            'interest_rate': self._to_float(row.get(f"Debt_{i}_Interest_Rate"), f"Debt_{i}_Interest_Rate", year),
                            'duration': self._to_int(row.get(f"Debt_{i}_Duration"), f"Debt_{i}_Duration", year, default=1)
                        })
                        year_data[f"Debt_{i}_Name"] = str(row[f"Debt_{i}_Name"])
                        year_data[f"Debt_{i}_Amount"] = self._to_float(row.get(f"Debt_{i}_Amount"), f"Debt_{i}_Amount", year)
                        year_data[f"Debt_{i}_Interest_Rate"] = self._to_float(row.get(f"Debt_{i}_Interest_Rate"), f"Debt_{i}_Interest_Rate", year)
                        year_data[f"Debt_{i}_Duration"] = self._to_int(row.get(f"Debt_{i}_Duration"), f"Debt_{i}_Duration", year, default=1)
                    i += 1
                debts_data[f'Debts_{year}']= debts                
                
                years_data[year] = year_data
                logger.info(f"Loaded data for year {year}")

            self.session_state['years_data'] = years_data
            logger.info(f"Loaded {len(years_data)} years of data")
            print("debts_data1 :", debts_data)
            return {"status": "success", "data": years_data,"debts_data":debts_data}
        
        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}")
            return {"status": "error", "data": {}, "detail": f"Failed to load Excel file: {str(e)}"}
        
    def display_customer_metrics(self,filtered_df):
        """Display customer metrics with Year, Burn Rate, and single IRR as a metric"""
        df = filtered_df.copy()        

        # Ensure Year column exists
        if 'Year' not in df.columns:
            df = df.reset_index().rename(columns={'index': 'Year'})   
        
        irr_value =float( df['IRR'].iloc[0] * 100) if 'IRR' in df.columns and pd.notna(df['IRR'].iloc[0]) and np.isfinite(df['IRR'].iloc[0]) else None

        # Select columns
        columns = ['Year', 'NPV', 'CAC', 'Contribution Margin Per Order', 'LTV', 'LTV/CAC Ratio', 'Payback Orders', 'Burn Rate']
        available_columns = [col for col in columns if col in df.columns]
        display_df = df[available_columns].copy()

        # Fill missing columns with NaN
        for col in columns:
            if col not in display_df.columns:
                display_df[col] = np.nan
        
        display_df = display_df.replace([np.inf, -np.inf, np.nan], None)
        
        return display_df,irr_value

    def save_data(self):
        """Save years_data to financial_assumptions.xlsx."""
        logger.info("Saving data to financial_assumptions.xlsx")
        if not self.session_state['years_data']:
            logger.warning("No data to save")
            return        
        try:
            data = []
            for year, values in self.session_state['years_data'].items():
                row = {'Year': year}
                row.update({k: v for k, v in values.items() if k not in ['Assets', 'Debts']})
                for asset in values.get('Assets', []):
                    idx = values['Assets'].index(asset) + 1
                    row[f"Asset_{idx}_Name"] = asset['name']
                    row[f"Asset_{idx}_Amount"] = asset['amount']
                    row[f"Asset_{idx}_Rate"] = asset['rate']
                    row[f"Asset_{idx}_Depreciation"] = values.get(f"Asset_{idx}_Depreciation", 0.0)
                    row[f"Asset_{idx}_NBV"] = values.get(f"Asset_{idx}_NBV", 0.0)
                for debt in values.get('Debts', []):
                    idx = values['Debts'].index(debt) + 1
                    row[f"Debt_{idx}_Name"] = debt['name']
                    row[f"Debt_{idx}_Amount"] = debt['amount']
                    row[f"Debt_{idx}_Interest_Rate"] = debt['interest_rate']
                    row[f"Debt_{idx}_Duration"] = debt['duration']
                data.append(row)
            df = pd.DataFrame(data)
            df.to_excel(self.current_file, index=False, engine='openpyxl')
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error saving Excel file: {str(e)}")
            raise

    def create_dataframe(self) -> pd.DataFrame:
        logger.info("Creating DataFrame from years_data")
        if not self.session_state['years_data']:
            logger.warning("No years_data available")
            return pd.DataFrame()
        
        data = []
        for year, values in self.session_state['years_data'].items():
            row = {'Year': year}
            for key, value in values.items():
                if key in ['Assets', 'Debts']:
                    continue
                elif key in ['Rent Categories', 'Professional Fee Types']:
                    row[key] = str(value)
                else:
                    row[key] = self._to_float(value, key, year)
            for asset in values.get('Assets', []):
                idx = values['Assets'].index(asset) + 1
                row[f"Asset_{idx}_Name"] = asset['name']
                row[f"Asset_{idx}_Amount"] = self._to_float(asset['amount'], f"Asset_{idx}_Amount", year)
                row[f"Asset_{idx}_Rate"] = self._to_float(asset['rate'], f"Asset_{idx}_Rate", year)
                row[f"Asset_{idx}_Depreciation"] = self._to_float(values.get(f"Asset_{idx}_Depreciation", 0.0), f"Asset_{idx}_Depreciation", year)
                row[f"Asset_{idx}_NBV"] = self._to_float(values.get(f"Asset_{idx}_NBV", 0.0), f"Asset_{idx}_NBV", year)
            for debt in values.get('Debts', []):
                idx = values['Debts'].index(debt) + 1
                row[f"Debt_{idx}_Name"] = debt['name']
                row[f"Debt_{idx}_Amount"] = self._to_float(debt['amount'], f"Debt_{idx}_Amount", year)
                row[f"Debt_{idx}_Interest_Rate"] = self._to_float(debt['interest_rate'], f"Debt_{idx}_Interest_Rate", year)
                row[f"Debt_{idx}_Duration"] = self._to_int(debt['duration'], f"Debt_{idx}_Duration", year, default=1)
            data.append(row)
        
        df = pd.DataFrame(data)
        df['Year'] = df['Year'].astype(int)
        logger.info(f"Created DataFrame with {len(df)} rows, columns: {df.columns.tolist()}")
        return df

    def calculate_debt_payment_schedule(self, df,debts_years_data):
        """Calculate debt payment schedules and update cash flow statement"""
        debt_schedules = {}
        total_principal_payments = []
        total_interest_payments = []

        for year in df['Year']:
            year_str = str(year)
            debts = debts_years_data.get(f"Debts_{year_str}", [])
            year_schedule = []
            principal_sum = 0.0
            interest_sum = 0.0

            for i, debt in enumerate(debts):
                amount = debt['amount']
                rate = debt['interest_rate'] / 100.0  # Convert to decimal
                duration = debt['duration']
                name = debt['name']

                # Calculate annual payment using the annuity formula (PMT)
                if rate > 0 and duration > 0 and amount > 0:
                    annual_payment = npf.pmt(rate, duration, -amount, 0)  # Negative amount for loan
                else:
                    annual_payment = amount / duration if duration > 0 else amount

                # Generate payment schedule for this debt
                remaining_balance = amount
                debt_schedule = {'Year': [], 'Principal': [], 'Interest': [], 'Payment': []}

                for t in range(1, duration + 1):
                    year_t = year + t - 1
                    interest_payment = remaining_balance * rate if remaining_balance > 0 else 0
                    principal_payment = annual_payment - interest_payment if remaining_balance > 0 else 0
                    remaining_balance = max(0, remaining_balance - principal_payment)

                    debt_schedule['Year'].append(year_t)
                    debt_schedule['Principal'].append(principal_payment)
                    debt_schedule['Interest'].append(interest_payment)
                    debt_schedule['Payment'].append(annual_payment)

                    if year_t == year:
                        principal_sum += principal_payment
                        interest_sum += interest_payment

                year_schedule.append({
                    'Debt Name': name,
                    'Amount': amount,
                    'Interest Rate': rate * 100,  # Back to percentage
                    'Duration': duration,
                    'Schedule': pd.DataFrame(debt_schedule)
                })

            debt_schedules[year] = year_schedule
            total_principal_payments.append(principal_sum)
            total_interest_payments.append(interest_sum)

        # Update DataFrame with aggregated debt payments for the current year
        df['Debt Principal Payments'] = total_principal_payments
        df['Debt Interest Payments'] = total_interest_payments

        # Adjust Cash from Financing to include principal payments as an outflow
        df['Cash from Financing'] = (df['Equity Raised'] + 
                                    df['Debt Issued'] - 
                                    df['Dividends Paid'] - 
                                    df['Debt Principal Payments'])

        # Update Interest in Income Statement to reflect debt interest
        if 'Interest' in df.columns:
            df['Interest'] = df['Debt Interest Payments']

        return debt_schedules
    
    def calculate_additional_metrics(self, df):
        """Calculate additional financial metrics"""
        new_metrics = {
            'Asset Turnover': df['Net Revenue'] / df['Total Assets'],
            'Inventory Turnover': df['COGS'] / df['Inventory'],
            'Days Sales Outstanding': df['Accounts Receivable'] / df['Net Revenue'] * 365,
            'Return on Assets': df['Net Income'] / df['Total Assets'],
            'Return on Equity': df['Net Income'] / df['Total Equity'],
            'Operating Leverage': (df['EBITDA'].pct_change() / df['Net Revenue'].pct_change()).rolling(2).mean(),
            'Current Ratio': df['Total Current Assets'] / df['Total Current Liabilities'],
            'Quick Ratio': (df['Total Current Assets'] - df['Inventory']) / df['Total Current Liabilities']
        }
        return pd.concat([df, pd.DataFrame(new_metrics)], axis=1)
    
    def calculate_margin_of_safety(self, df):
        """Calculate margin of safety metrics"""
        df['Actual Sales'] = df['Net Revenue']
        df['Margin of Safety Dollars'] = df['Actual Sales'] - df['Break Even Dollars']
        df['Margin of Safety Percentage'] = df['Margin of Safety Dollars'] / df['Actual Sales'] * 100
        return df
    
    def calculate_consideration(self, df):
        """Calculate consideration metrics"""
        total_monthly_traffic = (df['Email Traffic'] + 
                                df['Organic Search Traffic'] +
                                df['Paid Search Traffic'] + 
                                df['Affiliates Traffic'])
        
        df['Weighted Consideration Rate'] = (
            (df['Email Traffic'] * df['Email Conversion Rate'] +
            df['Organic Search Traffic'] * df['Organic Search Conversion Rate'] +
            df['Paid Search Traffic'] * df['Paid Search Conversion Rate'] +
            df['Affiliates Traffic'] * df['Affiliates Conversion Rate']) 
            / total_monthly_traffic
        )
        
        df['Consideration Actions'] = (
            (df['Email Traffic'] * df['Email Conversion Rate'] +
            df['Organic Search Traffic'] * df['Organic Search Conversion Rate'] +
            df['Paid Search Traffic'] * df['Paid Search Conversion Rate'] +
            df['Affiliates Traffic'] * df['Affiliates Conversion Rate']) * 12
        )
        
        epsilon = 1e-6
        df['Consideration to Conversion'] = (
            df['Total Orders'] / (df['Consideration Actions'] + epsilon) * 100
        )
        
        return df

    def calculate_income_statement(self, df, tax_rate=None, inflation_rate=None, direct_labor_rate_increase=None):
        """Calculate income statement metrics with dynamic tax, inflation, and labor rate adjustments"""
        # Default to DataFrame's Tax Rate if slider value not provided; otherwise, use 25% as fallback
        tax_rate = tax_rate if tax_rate is not None else (df['Tax Rate'].iloc[0] if 'Tax Rate' in df.columns else 0.25)
        inflation_rate = inflation_rate if inflation_rate is not None else 0.0
        direct_labor_rate_increase = direct_labor_rate_increase if direct_labor_rate_increase is not None else 0.0

        # Traffic and order calculations
        df['Total Monthly Visits'] = (df['Email Traffic'] + df['Organic Search Traffic'] + 
                                    df['Paid Search Traffic'] + df['Affiliates Traffic'])
        df['Annual Site Traffic'] = df['Total Monthly Visits'] * 12
        
        # Calculate new orders for the current year
        df['Email Orders'] = df['Email Traffic'] * 12 * df['Email Conversion Rate']
        df['Organic Search Orders'] = df['Organic Search Traffic'] * 12 * df['Organic Search Conversion Rate']
        df['Paid Search Orders'] = df['Paid Search Traffic'] * 12 * df['Paid Search Conversion Rate']
        df['Affiliates Orders'] = df['Affiliates Traffic'] * 12 * df['Affiliates Conversion Rate']
        df['New Orders'] = (df['Email Orders'] + df['Organic Search Orders'] + 
                            df['Paid Search Orders'] + df['Affiliates Orders'])

        # Initialize Total Orders column
        df['Total Orders'] = df['New Orders'].copy()

        # Calculate repeat orders from previous year, adjusted for churn
        for i in range(1, len(df)):
            prev_year_orders = df['Total Orders'].iloc[i-1]
            churn_rate = df['Churn Rate'].iloc[i]
            repeat_orders = prev_year_orders * (1 - churn_rate)
            df.loc[df.index[i], 'Total Orders'] = df['New Orders'].iloc[i] + repeat_orders

        # Ensure Total Orders is non-negative and finite
        df['Total Orders'] = df['Total Orders'].clip(lower=0).replace([np.inf, -np.inf], 0)

        # Apply inflation to Average Item Value (revenue side)
        df['Average Item Value'] = df['Average Item Value'] * (1 + inflation_rate) ** (df['Year'] - df['Year'].min())
        df['Average Gross Order Value'] = df['Average Item Value'] * df['Number of Items per Order']
        df['Average Net Order Value'] = (df['Average Gross Order Value'] * 
                                        (1 - df['Average Markdown']) * 
                                        (1 - df['Average Promotion/Discount']))
        df['Gross Revenue'] = df['Total Orders'] * df['Average Gross Order Value']
        df['Discounts, Promotions, Markdowns'] = (df['Gross Revenue'] * 
                                                (df['Average Markdown'] + df['Average Promotion/Discount']))
        df['Net Revenue'] = df['Gross Revenue'] - df['Discounts, Promotions, Markdowns']
        
        # Apply risk adjustment (unchanged)
        if 'Risk Adjustment Factor' in df.columns and hasattr(self, 'df'):
            max_historical_year = self.df['Year'].max()
            df.loc[df['Year'] > max_historical_year, 'Net Revenue'] *= (
                1 - df.loc[df['Year'] > max_historical_year, 'Risk Adjustment Factor'] * 0.1
            )
        
        # Apply inflation to COGS
        df['COGS'] = df['Gross Revenue'] * df['COGS Percentage'] * (1 + inflation_rate) ** (df['Year'] - df['Year'].min())
        df['Gross Profit'] = df['Net Revenue'] - df['COGS']
        df['Gross Margin'] = df['Gross Profit'] / df['Net Revenue']
        
        # Apply inflation to Marketing Expenses
        df['Marketing Expenses'] = ((df['Email Traffic'] * df['Email Cost per Click'] +
                                    df['Organic Search Traffic'] * df['Organic Search Cost per Click'] +
                                    df['Paid Search Traffic'] * df['Paid Search Cost per Click'] +
                                    df['Affiliates Traffic'] * df['Affiliates Cost per Click']) * 12 *
                                (1 + inflation_rate) ** (df['Year'] - df['Year'].min()))
        
        # Apply inflation and direct labor rate increase to Fulfilment Expenses
        df['Freight/Shipping per Order'] = df['Freight/Shipping per Order'] * (1 + inflation_rate) ** (df['Year'] - df['Year'].min())
        df['Labor/Handling per Order'] = (df['Labor/Handling per Order'] * 
                                        (1 + inflation_rate + direct_labor_rate_increase) ** (df['Year'] - df['Year'].min()))
        df['Fulfilment Expenses'] = (df['Freight/Shipping per Order'] + df['Labor/Handling per Order']) * df['Total Orders']
        
        df['Total Variable Costs'] = df['Marketing Expenses'] + df['Fulfilment Expenses']
        df['Contribution Margin'] = df['Gross Profit'] - df['Total Variable Costs']
        df['Contribution Margin Percentage'] = df['Contribution Margin'] / df['Net Revenue']

        # Apply inflation to Fixed Costs
        df['Fixed Fulfilment Expenses'] = df['General Warehouse Rent'] * (1 + inflation_rate) ** (df['Year'] - df['Year'].min())
        df['Fixed General & Administrative Expenses'] = (df['Office Rent'] + 
                                                        df['Salaries, Wages & Benefits'] + 
                                                        df['Professional Fees'] + 
                                                        df['Other']) * (1 + inflation_rate) ** (df['Year'] - df['Year'].min())
        df['Total Fixed Costs'] = df['Fixed Fulfilment Expenses'] + df['Fixed General & Administrative Expenses']
        
        df['EBITDA'] = df['Contribution Margin'] - df['Total Fixed Costs']
        df['EBITDA Margin'] = df['EBITDA'] / df['Net Revenue']
        
        df['Earnings Before Tax'] = df['EBITDA'] - df['Depreciation'] - df['Interest']
        
        # Apply political risk adjustment to Tax Rate (unchanged)
        if 'Political Risk Factor' in df.columns and hasattr(self, 'df'):
            max_historical_year = self.df['Year'].max()
            df.loc[df['Year'] > max_historical_year, 'Tax Rate'] *= (
                1 + df.loc[df['Year'] > max_historical_year, 'Political Risk Factor'] * 0.05
            )
        
        # Use the dynamic tax_rate parameter instead of df['Tax Rate']
        df['Adjusted EBT for Tax'] = df['Earnings Before Tax'].clip(lower=0)
        df['Taxes'] = df['Adjusted EBT for Tax'] * tax_rate
        df['Net Income'] = df['Earnings Before Tax'] - df['Taxes']
        df.drop('Adjusted EBT for Tax', axis=1, inplace=True)
        return df
    
    def calculate_cash_flow_statement_with_debt_schedules(self, df,debts_years_data):
        """Calculate cash flow statement metrics with debt payment schedule"""
        # Ensure required columns exist, default to 0 if missing
        for col in ['Net Income', 'Depreciation', 'Accounts Receivable Days', 'COGS', 
                    'Inventory Days', 'Accounts Payable Days', 'Total Variable Costs',
                    'Technology Development', 'Office Equipment', 'Equity Raised', 
                    'Debt Issued', 'Dividends Paid']:
            if col not in df.columns:
                df[col] = 0.0
        
        # Calculate working capital components
        df['Accounts Receivable'] = df['Net Revenue'] * (df['Accounts Receivable Days'] / 365)
        df['Inventory'] = df['COGS'] * (df['Inventory Days'] / 365)
        df['Accounts Payable'] = ((df['COGS'] + df['Total Variable Costs']) * 
                                df['Accounts Payable Days'] / 365)
        
        # Environmental impact adjustment for inventory days (if applicable)
        if 'Environmental Impact Factor' in df.columns and hasattr(self, 'df'):
            max_historical_year = self.df['Year'].max()
            df.loc[df['Year'] > max_historical_year, 'Inventory Days'] *= (
                1 + df.loc[df['Year'] > max_historical_year, 'Environmental Impact Factor'] * 0.1
            )
            df['Inventory'] = df['COGS'] * (df['Inventory Days'] / 365)
        
        # Calculate changes in working capital
        df['Change in Accounts Receivable'] = df['Accounts Receivable'].diff().fillna(df['Accounts Receivable'])
        df['Change in Inventory'] = df['Inventory'].diff().fillna(df['Inventory'])
        if not df.empty:
            df['Change in Accounts Payable'] = df['Accounts Payable'].diff()
            df.loc[df.index[0], 'Change in Accounts Payable'] = df.loc[df.index[0], 'Accounts Payable'] - 0
        
        # Calculate cash flow components
        df['Cash from Operations'] = (df['Net Income'] + df['Depreciation'] - 
                                    df['Change in Accounts Receivable'] - 
                                    df['Change in Inventory'] + 
                                    df['Change in Accounts Payable'])
        
        df['Capital Expenditures'] = df['Technology Development'] + df['Office Equipment']
        df['Cash from Investing'] = df['Capital Expenditures']  # Corrected to negative for outflows
        
        # Calculate debt payment schedule and update Cash from Financing
        debt_schedules = self.calculate_debt_payment_schedule(df,debts_years_data)
        # Cash from Financing now includes debt principal payments as an outflow
        df['Cash from Financing'] = (df['Equity Raised'] + 
                                    df['Debt Issued'] - 
                                    df['Dividends Paid'] - 
                                    df['Debt Principal Payments'])
        
        df['Net Cash Flow'] = (df['Cash from Operations'] - 
                            df['Cash from Investing'] + 
                            df['Cash from Financing'])  # Corrected to subtract investing
        
        # Calculate cash balances
        df['Opening Cash Balance'] = 0.0
        df['Increase (Decrease)'] = df['Net Cash Flow']
        df['Closing Cash Balance'] = 0.0
        for i in range(len(df)):
            if i > 0:
                df.loc[df.index[i], 'Opening Cash Balance'] = df.loc[df.index[i-1], 'Closing Cash Balance']
            df.loc[df.index[i], 'Closing Cash Balance'] = (df.loc[df.index[i], 'Opening Cash Balance'] + 
                                                        df.loc[df.index[i], 'Increase (Decrease)'])
        
        return debt_schedules    
    
    def calculate_cash_flow_statement(self, income_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculating cash flow statement")
        
        if income_df is None or income_df.empty:
            logger.error("Input income_df is None or empty")
            raise ValueError("Input income_df is None or empty")
        
        result_df = income_df.copy()        
        
        try:
            required_columns = [
                'Net Income', 'Depreciation', 'Accounts Receivable Days', 'COGS',
                'Inventory Days', 'Accounts Payable Days', 'Total Variable Costs',
                'Technology Development', 'Office Equipment', 'Equity Raised',
                'Debt Issued', 'Dividends Paid', 'Debt Principal Payments'
            ]
            for col in required_columns:
                if col not in result_df.columns:
                    result_df[col] = 0.0
                    logger.debug(f"Column {col} missing, set to 0.0")
            
            result_df['Accounts Receivable'] = result_df['Net Revenue'] * (result_df['Accounts Receivable Days'] / 365)
            result_df['Inventory'] = result_df['COGS'] * (result_df['Inventory Days'] / 365)
            result_df['Accounts Payable'] = (
                (result_df['COGS'] + result_df['Total Variable Costs']) *
                result_df['Accounts Payable Days'] / 365
            )           

            if 'Environmental Impact Factor' in result_df.columns and self.session_state['years_data']:
                max_historical_year = max(self.session_state['years_data'].keys())
                result_df.loc[result_df['Year'] > max_historical_year, 'Inventory Days'] *= (
                    1 + result_df.loc[result_df['Year'] > max_historical_year, 'Environmental Impact Factor'] * 0.1
                )
                result_df['Inventory'] = result_df['COGS'] * (result_df['Inventory Days'] / 365)

            result_df['Change in Accounts Receivable'] = result_df['Accounts Receivable'].diff().fillna(result_df['Accounts Receivable'])
            result_df['Change in Inventory'] = result_df['Inventory'].diff().fillna(result_df['Inventory'])
            if not result_df.empty:
                result_df['Change in Accounts Payable'] = result_df['Accounts Payable'].diff()
                result_df.loc[result_df.index[0], 'Change in Accounts Payable'] = result_df.loc[result_df.index[0], 'Accounts Payable']

            result_df['Cash from Operations'] = (
                result_df['Net Income'] +
                result_df['Depreciation'] -
                result_df['Change in Accounts Receivable'] -
                result_df['Change in Inventory'] +
                result_df['Change in Accounts Payable']
            )

            result_df['Capital Expenditures'] = result_df['Technology Development'] + result_df['Office Equipment']
            result_df['Cash from Investing'] = -result_df['Capital Expenditures']

            result_df['Cash from Financing'] = (
                result_df['Equity Raised'] +
                result_df['Debt Issued'] -
                result_df['Dividends Paid'] -
                result_df['Debt Principal Payments']
            )

            result_df['Net Cash Flow'] = (
                result_df['Cash from Operations'] -
                result_df['Cash from Investing'] +
                result_df['Cash from Financing']
            )

            result_df['Opening Cash Balance'] = 0.0
            result_df['Increase (Decrease)'] = result_df['Net Cash Flow']
            result_df['Closing Cash Balance'] = 0.0
            for i in range(len(result_df)):
                if i > 0:
                    result_df.loc[result_df.index[i], 'Opening Cash Balance'] = result_df.loc[result_df.index[i-1], 'Closing Cash Balance']
                result_df.loc[result_df.index[i], 'Closing Cash Balance'] = (
                    result_df.loc[result_df.index[i], 'Opening Cash Balance'] +
                    result_df.loc[result_df.index[i], 'Increase (Decrease)']
                )

            logger.info(f"Cash flow statement calculated with columns: {result_df.columns.tolist()}")
            return result_df
        
        except Exception as e:
            logger.error(f"Error calculating cash flow statement: {str(e)}")
            raise

    def calculate_supporting_schedules(self, cash_flow_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculating supporting schedules")
        
        if cash_flow_df is None or cash_flow_df.empty:
            logger.error("Input cash_flow_df is None or empty")
            raise ValueError("Input cash_flow_df is None or empty")
        
        result_df = cash_flow_df.copy()
        
        try:
            required_columns = [
                'Year', 'Technology Development', 'Office Equipment',
                'Technology Depreciation Years', 'Office Equipment Depreciation Years',
                'Debt Issued', 'Interest Rate'
            ]
            for col in required_columns:
                if col not in result_df.columns:
                    default = 1.0 if 'Depreciation Years' in col else 0.0
                    result_df[col] = default
                    logger.debug(f"Column {col} missing, set to {default}")
            
            missing_metrics = [col for col in ['Year'] if col not in result_df.columns]
            if missing_metrics:
                logger.error(f"Missing required metrics for supporting schedules: {missing_metrics}")
                raise ValueError(f"Missing required metrics: {missing_metrics}")
            
            result_df['Opening Balance Technology'] = 0.0
            result_df['Opening Balance Office Equipment'] = 0.0
            result_df['Additions Technology'] = result_df['Technology Development']
            result_df['Additions Office Equipment'] = result_df['Office Equipment']

            result_df['Depreciation Technology'] = result_df['Technology Development'] / result_df['Technology Depreciation Years'].replace(0, 1)
            result_df['Depreciation Office Equipment'] = result_df['Office Equipment'] / result_df['Office Equipment Depreciation Years'].replace(0, 1)

            result_df['Cumulative Depreciation Technology'] = (
                result_df['Depreciation Technology']
                .rolling(window=5, min_periods=1)
                .sum()
                .fillna(0)
            )
            result_df['Cumulative Depreciation Office Equipment'] = (
                result_df['Depreciation Office Equipment']
                .rolling(window=5, min_periods=1)
                .sum()
                .fillna(0)
            )

            first_year = result_df['Year'].iloc[0] if not result_df.empty else 0
            result_df.loc[result_df['Year'] == first_year, 'Subtotal Technology'] = (
                result_df.loc[result_df['Year'] == first_year, 'Opening Balance Technology'] +
                result_df.loc[result_df['Year'] == first_year, 'Additions Technology']
            )
            result_df.loc[result_df['Year'] == first_year, 'Subtotal Office Equipment'] = (
                result_df.loc[result_df['Year'] == first_year, 'Opening Balance Office Equipment'] +
                result_df.loc[result_df['Year'] == first_year, 'Additions Office Equipment']
            )
            result_df.loc[result_df['Year'] == first_year, 'Closing Balance Technology'] = (
                result_df.loc[result_df['Year'] == first_year, 'Subtotal Technology'] -
                result_df.loc[result_df['Year'] == first_year, 'Cumulative Depreciation Technology']
            )
            result_df.loc[result_df['Year'] == first_year, 'Closing Balance Office Equipment'] = (
                result_df.loc[result_df['Year'] == first_year, 'Subtotal Office Equipment'] -
                result_df.loc[result_df['Year'] == first_year, 'Cumulative Depreciation Office Equipment']
            )

            for i in range(1, len(result_df)):
                current_year = result_df['Year'].iloc[i]
                previous_year = result_df['Year'].iloc[i-1]
                result_df.loc[result_df['Year'] == current_year, 'Opening Balance Technology'] = (
                    result_df.loc[result_df['Year'] == previous_year, 'Closing Balance Technology'].iloc[0]
                )
                result_df.loc[result_df['Year'] == current_year, 'Opening Balance Office Equipment'] = (
                    result_df.loc[result_df['Year'] == previous_year, 'Closing Balance Office Equipment'].iloc[0]
                )
                result_df.loc[result_df['Year'] == current_year, 'Subtotal Technology'] = (
                    result_df.loc[result_df['Year'] == current_year, 'Opening Balance Technology'] +
                    result_df.loc[result_df['Year'] == current_year, 'Additions Technology']
                )
                result_df.loc[result_df['Year'] == current_year, 'Subtotal Office Equipment'] = (
                    result_df.loc[result_df['Year'] == current_year, 'Opening Balance Office Equipment'] +
                    result_df.loc[result_df['Year'] == current_year, 'Additions Office Equipment']
                )
                result_df.loc[result_df['Year'] == current_year, 'Closing Balance Technology'] = (
                    result_df.loc[result_df['Year'] == current_year, 'Subtotal Technology'] -
                    result_df.loc[result_df['Year'] == current_year, 'Cumulative Depreciation Technology']
                )
                result_df.loc[result_df['Year'] == current_year, 'Closing Balance Office Equipment'] = (
                    result_df.loc[result_df['Year'] == current_year, 'Subtotal Office Equipment'] -
                    result_df.loc[result_df['Year'] == current_year, 'Cumulative Depreciation Office Equipment']
                )

            result_df['Total Opening Balance'] = (
                result_df['Opening Balance Technology'] + result_df['Opening Balance Office Equipment']
            )
            result_df['Total Additions'] = (
                result_df['Additions Technology'] + result_df['Additions Office Equipment']
            )
            result_df['Total'] = (
                result_df['Subtotal Technology'] + result_df['Subtotal Office Equipment']
            )
            result_df['Total Depreciation'] = (
                result_df['Cumulative Depreciation Technology'] + result_df['Cumulative Depreciation Office Equipment']
            )
            result_df['Total Closing Balance'] = (
                result_df['Closing Balance Technology'] + result_df['Closing Balance Office Equipment']
            )

            result_df['Opening Balance Debt'] = result_df['Debt Issued'].shift(1).cumsum().fillna(0)
            result_df['Closing Balance Debt'] = result_df['Opening Balance Debt'] + result_df['Debt Issued']
            result_df['Interest on Debt'] = result_df['Closing Balance Debt'] * result_df['Interest Rate']

            logger.info(f"Supporting schedules calculated with columns: {result_df.columns.tolist()}")
            return result_df
        
        except Exception as e:
            logger.error(f"Error calculating supporting schedules: {str(e)}")
            raise

    def calculate_balance_sheet(self, cash_flow_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculating balance sheet")
        
        if cash_flow_df is None or cash_flow_df.empty:
            logger.error("Input cash_flow_df is None or empty")
            raise ValueError("Input cash_flow_df is None or empty")
        
        result_df = cash_flow_df.copy()
        
        try:
            required_cols = [
                'Closing Cash Balance', 'Accounts Receivable', 'Inventory', 'Debt Issued',
                'Equity Raised', 'Net Income', 'Dividends Paid', 'Closing Balance Technology',
                'Closing Balance Office Equipment', 'Accounts Payable'
            ]
            for col in required_cols:
                if col not in result_df.columns:
                    result_df[col] = 0.0
                    logger.debug(f"Column {col} missing, set to 0.0")
            
            if self.session_state['years_data']:
                max_historical_year = max(self.session_state['years_data'].keys())
            else:
                max_historical_year = result_df['Year'].min()
            
            historical_mask = result_df['Year'] <= max_historical_year
            forecast_mask = result_df['Year'] > max_historical_year

            result_df['Cash'] = result_df['Closing Cash Balance']
            result_df['Technology Assets'] = result_df['Closing Balance Technology']
            result_df['Office Equipment Assets'] = result_df['Closing Balance Office Equipment']
            result_df['Total Current Assets'] = (
                result_df['Cash'] + result_df['Accounts Receivable'] + result_df['Inventory']
            )
            result_df['Total Fixed Assets'] = (
                result_df['Technology Assets'] + result_df['Office Equipment Assets']
            )
            result_df['Total Assets'] = result_df['Total Current Assets'] + result_df['Total Fixed Assets']

            result_df['Total Current Liabilities'] = result_df['Accounts Payable']
            result_df['Long Term Debt'] = result_df['Debt Issued'].cumsum()
            result_df['Total Liabilities'] = (
                result_df['Total Current Liabilities'] + result_df['Long Term Debt']
            )

            result_df['Share Capital'] = result_df['Equity Raised'].cumsum()

            result_df['Retained Earnings'] = 0.0
            if historical_mask.any():
                result_df.loc[historical_mask, 'Retained Earnings'] = (
                    (result_df['Net Income'] - result_df['Dividends Paid'])
                    .where(historical_mask)
                    .cumsum()
                    .fillna(0)
                )
                result_df.loc[historical_mask, 'Total Equity'] = (
                    result_df['Share Capital'] + result_df['Retained Earnings']
                ).where(historical_mask)
                result_df.loc[historical_mask, 'Total Liabilities & Shareholder Equity'] = (
                    result_df['Total Equity'] + result_df['Total Liabilities']
                ).where(historical_mask)
            
            for i in result_df.index[historical_mask]:
                shortfall = result_df.loc[i, 'Total Assets'] - result_df.loc[i, 'Total Liabilities & Shareholder Equity']
                result_df.loc[i, 'Retained Earnings'] += shortfall
                result_df.loc[i, 'Total Equity'] = (
                    result_df.loc[i, 'Share Capital'] + result_df.loc[i, 'Retained Earnings']
                )
                result_df.loc[i, 'Total Liabilities & Shareholder Equity'] = (
                    result_df.loc[i, 'Total Equity'] + result_df.loc[i, 'Total Liabilities']
                )
            if forecast_mask.any():
                last_historical_idx = result_df[historical_mask].index[-1] if historical_mask.any() else None
                last_historical_re = (result_df.loc[last_historical_idx, 'Retained Earnings']
                                    if last_historical_idx is not None else 0)
                
                for i in result_df.index[forecast_mask]:
                    prev_idx = result_df.index[result_df.index.get_loc(i) - 1] if result_df.index.get_loc(i) > 0 else i
                    re_increase = result_df.loc[i, 'Net Income'] - result_df.loc[i, 'Dividends Paid']
                    result_df.loc[i, 'Retained Earnings'] = result_df.loc[prev_idx, 'Retained Earnings'] + re_increase
                    
                    result_df.loc[i, 'Total Equity'] = (
                        result_df.loc[i, 'Share Capital'] + result_df.loc[i, 'Retained Earnings']
                    )
                    result_df.loc[i, 'Total Liabilities & Shareholder Equity'] = (
                        result_df.loc[i, 'Total Equity'] + result_df.loc[i, 'Total Liabilities']
                    )
                    
                    shortfall = (result_df.loc[i, 'Total Assets'] -
                                result_df.loc[i, 'Total Liabilities & Shareholder Equity'])
                    result_df.loc[i, 'Retained Earnings'] += shortfall
                    result_df.loc[i, 'Total Equity'] = (
                        result_df.loc[i, 'Share Capital'] + result_df.loc[i, 'Retained Earnings']
                    )
                    result_df.loc[i, 'Total Liabilities & Shareholder Equity'] = (
                        result_df.loc[i, 'Total Equity'] + result_df.loc[i, 'Total Liabilities']
                    )

            result_df['Balance Sheet Check'] = (
                result_df['Total Liabilities & Shareholder Equity'] - result_df['Total Assets']
            )

            logger.info(f"Balance sheet calculated with columns: {result_df.columns.tolist()}")
            return result_df
        
        except Exception as e:
            logger.error(f"Error calculating balance sheet: {str(e)}")
            raise

    def calculate_project_irr(self, df):
        """
        Calculate the Internal Rate of Return (IRR) for the project based on net cash flows.
        
        Parameters:
        - df: DataFrame containing 'Net Cash Flow' and 'Year' columns
        
        Returns:
        - float: IRR value, or np.nan if calculation fails
        """
        try:
            # Ensure DataFrame has necessary columns
            if 'Net Cash Flow' not in df.columns or 'Year' not in df.columns:
                raise ValueError("DataFrame must contain 'Net Cash Flow' and 'Year' columns")

            # Sort by year to ensure chronological order
            df = df.sort_values('Year').reset_index(drop=True)

            # Extract cash flows
            cash_flows = df['Net Cash Flow'].fillna(0).values

            # Check if cash flows are valid
            if len(cash_flows) < 2 or np.all(cash_flows == 0):
                return np.nan

            # Add initial investment if applicable (assuming Year 0 investment if not in data)
            # Modify this based on your actual initial investment logic
            if df['Year'].iloc[0] > df['Year'].min() - 1:
                initial_investment = -df['Equity Raised'].iloc[0] if 'Equity Raised' in df.columns else 0
                cash_flows = np.insert(cash_flows, 0, initial_investment)

            # Clean cash flows: replace inf/nan with 0
            cash_flows = np.where(np.isfinite(cash_flows), cash_flows, 0)

            # Check for sign changes (IRR requires at least one positive and one negative cash flow)
            if not (np.any(cash_flows > 0) and np.any(cash_flows < 0)):
                return np.nan

            # Calculate IRR
            irr = npf.irr(cash_flows)

            # Validate IRR (sometimes IRR can be unrealistic due to numerical issues)
            if not np.isfinite(irr) or irr < -1 or irr > 10:  # Arbitrary bounds; adjust as needed
                return np.nan         
            
            return irr

        except Exception as e:
            print(f"Error calculating IRR: {str(e)}")
            return np.nan           

    def display_metrics_scenario_analysis(self, scenario_df):
        """Display key metrics over the entire (unsliced) scenario DataFrame."""

        # 1) Sort by Year to guarantee chronological order
        # scenario_df = scenario_df.sort_values("Year").reset_index(drop=True)

        # 2) Drop any rows where Net Revenue is 0 or NaN before computing growth
        #    (This matches your Excel, which starts pct_change at the first positive‐revenue year.)
     
    # —————————————————————————————————————————————————————————————————————————
        filtered_df=scenario_df.copy()
    # Take only non-zero “Net Revenue” rows (in case you have a 0 placeholder)
        revenue = filtered_df['Net Revenue'].replace(0, np.nan).dropna()

        # If there are at least two years, compute CAGR = (End/Start)^(1/(n-1)) − 1
        if len(revenue) > 1:
            start_revenue = revenue.iloc[0]
            end_revenue   = revenue.iloc[-1]
            n_periods     = len(revenue) - 1

            if start_revenue > 0:
                cagr = (end_revenue / start_revenue)**(1.0 / n_periods) - 1.0
            else:
                cagr = 0.0

            # “Current growth” = last-year’s percentage change = (this_year/prev_year − 1)
            prev_year_revenue = revenue.iloc[-2]
            if prev_year_revenue > 0:
                last_year_change = (end_revenue / prev_year_revenue) - 1.0
            else:
                last_year_change = 0.0

        else:
            cagr             = 0.0
            last_year_change = 0.0

        # Convert to percentages
        revenue_growth_pct       = cagr * 100.0
        current_revenue_growth_pct = last_year_change * 100.0


        # —————————————————————————————————————————————————————————————————————————
        # 2) EBITDA MARGIN (initial vs. current + delta)
        # —————————————————————————————————————————————————————————————————————————

        if 'EBITDA' in filtered_df.columns and 'Net Revenue' in filtered_df.columns:
            # Compute margin = EBITDA / Net Revenue row-by-row
            margins = (filtered_df['EBITDA'] / filtered_df['Net Revenue'].replace(0, np.nan)).fillna(0.0)

            margin_initial = margins.iloc[0] if len(margins) > 0 else 0.0
            margin_current = margins.iloc[-1] if len(margins) > 0 else 0.0
            delta_margin   = margin_current - margin_initial
        else:
            margin_initial = 0.0
            margin_current = 0.0
            delta_margin   = 0.0

        ebitda_margin_pct       = margin_current * 100.0
        ebitda_margin_delta_pct = delta_margin * 100.0


        # —————————————————————————————————————————————————————————————————————————
        # 3) ENTERPRISE VALUE (last row “Total Enterprise Value” / 1e6; fallback to EBITDA×8)
        # —————————————————————————————————————————————————————————————————————————

        if 'Total Enterprise Value' in filtered_df.columns:
            enterprise_value_m = filtered_df['Total Enterprise Value'].iloc[-1] / 1e6
        elif 'EBITDA' in filtered_df.columns:
            enterprise_value_m = (filtered_df['EBITDA'].iloc[-1] * 8.0) / 1e6
        else:
            enterprise_value_m = 0.0


        # —————————————————————————————————————————————————————————————————————————
        # 4) BUILD RESPONSE
        # —————————————————————————————————————————————————————————————————————————

        response = {
            "revenue_growth":           round(revenue_growth_pct, 1),
            "current_revenue_growth":   round(current_revenue_growth_pct, 1),
            "initial_ebitda_margin":    round(margin_initial * 100.0, 1),
            "ebitda_margin":            round(ebitda_margin_pct, 1),
            "ebitda_margin_delta":      round(ebitda_margin_delta_pct, 1),
            "enterprise_value":         round(enterprise_value_m, 1)
        }
        print("response :", response)
        return response
        
    def display_summary_of_analysis(self,scenarios,df_length):
        summary_data = {
                    'Scenario': ['Base Case', 'Best Case', 'Worst Case'],
                    'Net Income ($M)': [
                        scenarios['Base Case']['Net Income'].iloc[-1] / 1e6 if not pd.isna(scenarios['Base Case']['Net Income'].iloc[-1]) else 0.0,
                        scenarios.get('Best Case', pd.DataFrame({'Net Income': [0] * df_length}))['Net Income'].iloc[-1] / 1e6 if 'Best Case' in scenarios else 0.0,
                        scenarios.get('Worst Case', pd.DataFrame({'Net Income': [0] * df_length}))['Net Income'].iloc[-1] / 1e6 if 'Worst Case' in scenarios else 0.0
                    ],
                    'EBITDA ($M)': [
                        scenarios['Base Case']['EBITDA'].iloc[-1] / 1e6 if not pd.isna(scenarios['Base Case']['EBITDA'].iloc[-1]) else 0.0,
                        scenarios.get('Best Case', pd.DataFrame({'EBITDA': [0] * df_length}))['EBITDA'].iloc[-1] / 1e6 if 'Best Case' in scenarios else 0.0,
                        scenarios.get('Worst Case', pd.DataFrame({'EBITDA': [0] * df_length}))['EBITDA'].iloc[-1] / 1e6 if 'Worst Case' in scenarios else 0.0
                    ],
                    
                    'IRR (%)': [
                        scenarios['Base Case']['IRR'].iloc[0] * 100 if 'IRR' in scenarios['Base Case'].columns and not pd.isna(scenarios['Base Case']['IRR'].iloc[0]) else 0.0,
                        scenarios.get('Best Case', pd.DataFrame({'IRR': [0] * df_length}))['IRR'].iloc[0] * 100 if 'Best Case' in scenarios and 'IRR' in scenarios['Best Case'].columns else 0.0,
                        scenarios.get('Worst Case', pd.DataFrame({'IRR': [0] * df_length}))['IRR'].iloc[0] * 100 if 'Worst Case' in scenarios and 'IRR' in scenarios['Worst Case'].columns else 0.0
                    ],
                    'NPV ($M)': [
                        scenarios['Base Case']['NPV'].iloc[-1] / 1e6 if 'NPV' in scenarios['Base Case'].columns and not pd.isna(scenarios['Base Case']['NPV'].iloc[-1]) else 0.0,
                        scenarios.get('Best Case', pd.DataFrame({'NPV': [0] * df_length}))['NPV'].iloc[-1] / 1e6 if 'Best Case' in scenarios and 'NPV' in scenarios['Best Case'].columns else 0.0,
                        scenarios.get('Worst Case', pd.DataFrame({'NPV': [0] * df_length}))['NPV'].iloc[-1] / 1e6 if 'Worst Case' in scenarios and 'NPV' in scenarios['Worst Case'].columns else 0.0
                    ],
                    'Payback Period (Orders)': [
                        scenarios['Base Case']['Payback Orders'].iloc[-1] if 'Payback Orders' in scenarios['Base Case'].columns and not pd.isna(scenarios['Base Case']['Payback Orders'].iloc[-1]) else 0.0,
                        scenarios.get('Best Case', pd.DataFrame({'Payback Orders': [0] * df_length}))['Payback Orders'].iloc[-1] if 'Best Case' in scenarios and 'Payback Orders' in scenarios['Best Case'].columns else 0.0,
                        scenarios.get('Worst Case', pd.DataFrame({'Payback Orders': [0] * df_length}))['Payback Orders'].iloc[-1] if 'Worst Case' in scenarios and 'Payback Orders' in scenarios['Worst Case'].columns else 0.0
                    ],
                    'Gross Profit Margin (%)': [
                        scenarios['Base Case']['Gross Margin'].iloc[-1] * 100 if not pd.isna(scenarios['Base Case']['Gross Margin'].iloc[-1]) else 0.0,
                        scenarios.get('Best Case', pd.DataFrame({'Gross Margin': [0] * df_length}))['Gross Margin'].iloc[-1] * 100 if 'Best Case' in scenarios else 0.0,
                        scenarios.get('Worst Case', pd.DataFrame({'Gross Margin': [0] * df_length}))['Gross Margin'].iloc[-1] * 100 if 'Worst Case' in scenarios else 0.0
                    ],
                    'Net Profit Margin (%)': [
                        (scenarios['Base Case']['Net Income'].iloc[-1] / scenarios['Base Case']['Net Revenue'].iloc[-1] * 100) if not pd.isna(scenarios['Base Case']['Net Income'].iloc[-1]) and scenarios['Base Case']['Net Revenue'].iloc[-1] != 0 else 0.0,
                        (scenarios.get('Best Case', pd.DataFrame({'Net Income': [0] * df_length, 'Net Revenue': [1] * df_length}))['Net Income'].iloc[-1] / scenarios.get('Best Case', pd.DataFrame({'Net Revenue': [1] * df_length}))['Net Revenue'].iloc[-1] * 100) if 'Best Case' in scenarios else 0.0,
                        (scenarios.get('Worst Case', pd.DataFrame({'Net Income': [0] * df_length, 'Net Revenue': [1] * df_length}))['Net Income'].iloc[-1] / scenarios.get('Worst Case', pd.DataFrame({'Net Revenue': [1] * df_length}))['Net Revenue'].iloc[-1] * 100) if 'Worst Case' in scenarios else 0.0
                    ],
                    'Net Cash Flow ($M)': [
                        scenarios['Base Case']['Net Cash Flow'].iloc[-1] / 1e6 if not pd.isna(scenarios['Base Case']['Net Cash Flow'].iloc[-1]) else 0.0,
                        scenarios.get('Best Case', pd.DataFrame({'Net Cash Flow': [0] * df_length}))['Net Cash Flow'].iloc[-1] / 1e6 if 'Best Case' in scenarios else 0.0,
                        scenarios.get('Worst Case', pd.DataFrame({'Net Cash Flow': [0] * df_length}))['Net Cash Flow'].iloc[-1] / 1e6 if 'Worst Case' in scenarios else 0.0
                    ]
                }
        # Convert to list of dictionaries for response
        summary_df = pd.DataFrame(summary_data)
        return summary_df

    def calculate_customer_metrics(self, income_df: pd.DataFrame, discount_rate) -> pd.DataFrame:
        logger.info("Calculating customer metrics")
        result_df = income_df.copy()

        required_metrics = [
            'Net Revenue', 'Total Orders', 'Marketing Expenses', 'Churn Rate',
            'Paid Search Orders', 'Affiliates Orders', 'Contribution Margin',
            'Cash from Operations', 'Cash from Investing'
        ]

        for col in required_metrics:
            if col not in result_df.columns:
                result_df[col] = 0.0
                logger.debug(f"Column {col} missing, set to 0.0")

        for col in required_metrics:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)

        try:
            paid_aff_orders = result_df['Paid Search Orders'] + result_df['Affiliates Orders']
            result_df['CAC'] = np.where(
                paid_aff_orders > 0,
                result_df['Marketing Expenses'] / paid_aff_orders,
                0
            )

            result_df['Contribution Margin Per Order'] = np.where(
                result_df['Total Orders'] > 0,
                (result_df['Contribution Margin'] + result_df['Marketing Expenses']) / result_df['Total Orders'],
                0
            )

            result_df['LTV'] = np.where(
                result_df['Churn Rate'] > 0,
                result_df['Contribution Margin Per Order'] * (1 / result_df['Churn Rate']),
                0
            )

            result_df['LTV/CAC Ratio'] = np.where(
                result_df['CAC'] > 0,
                result_df['LTV'] / result_df['CAC'],
                0
            )

            result_df['Payback Orders'] = np.where(
                (result_df['CAC'] > 0) & (result_df['Contribution Margin Per Order'] > 0),
                result_df['CAC'] / result_df['Contribution Margin Per Order'],
                0
            )

            # Prepare Free Cash Flow series
            if 'Free Cash Flow' in result_df.columns:
                cash_flow_series = result_df['Free Cash Flow']
            elif 'Net Cash Flow' in result_df.columns:
                cash_flow_series = result_df['Net Cash Flow']
            else:
                cash_flow_series = pd.Series([0] * len(result_df), index=result_df.index)
            cash_flow_series = pd.to_numeric(cash_flow_series, errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)

            # IRR & NPV Calculation per scenario using the entire cash flow stream
            initial_investment = (
                -(result_df['Capital Expenditures'].iloc[0] + result_df['Marketing Expenses'].iloc[0])
                if 'Capital Expenditures' in result_df.columns
                else -result_df['Marketing Expenses'].iloc[0]
            )
            initial_investment = float(initial_investment) if pd.notna(initial_investment) else 0.0
            cash_flows = [initial_investment] + cash_flow_series.tolist()

            # Compute NPV for the scenario as a whole
            npv_value = npf.npv(discount_rate, cash_flows)
            result_df['NPV'] = round(npv_value, 2)

            # Compute IRR for the scenario as a whole
            try:
                irr = npf.irr(cash_flows)
                result_df['IRR'] = round(irr * 100 if irr is not None else 0, 2)
            except:
                result_df['IRR'] = 0.0

            result_df['Burn Rate'] = result_df['Cash from Operations'] - result_df['Cash from Investing']

            for col in ['CAC', 'Contribution Margin Per Order', 'LTV', 'LTV/CAC Ratio', 'Payback Orders', 'Burn Rate']:
                result_df[col] = result_df[col].round(2)

            logger.info(f"Customer metrics calculated with columns: {result_df.columns.tolist()}")
            return result_df

        except Exception as e:
            logger.error(f"Error calculating customer metrics: {str(e)}")
            raise

    def calculate_dcf_valuation(self, cash_flow_df: pd.DataFrame, wacc: float = 0.1, perpetual_growth: float = 0.02) -> Dict:
        logger.info("Calculating DCF valuation")
        
        if cash_flow_df is None or cash_flow_df.empty:
            logger.error("Input cash_flow_df is None or empty")
            raise ValueError("Input cash_flow_df is None or empty")
        
        result_df = cash_flow_df.copy()
        
        try:
            # Ensure required columns exist and are numeric
            required_cols = [
                'Year', 'EBIT', 'Tax Rate', 'Depreciation', 'Change in Working Capital',
                'Capital Expenditures', 'Long Term Debt', 'Cash', 'EBITDA', 'Net Revenue',
                'Cash from Operations', 'Cash from Investing', 'Accounts Receivable',
                'Inventory', 'Accounts Payable', 'Technology Development', 'Office Equipment'
            ]
            for col in required_cols:
                if col not in result_df.columns:
                    result_df[col] = 0.0
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0.0).replace([np.inf, -np.inf], 0)

            # Calculate Unlevered Free Cash Flow (FCF)
            result_df['Change in Working Capital'] = (
                (result_df['Accounts Receivable'] + result_df['Inventory'] - result_df['Accounts Payable']) -
                result_df[['Accounts Receivable', 'Inventory', 'Accounts Payable']].shift(1).sum(axis=1).fillna(0)
            )
            result_df['Capital Expenditures'] = -(result_df['Technology Development'] + result_df['Office Equipment'])
            result_df['Unlevered FCF'] = (
                result_df['EBIT'] * (1 - result_df['Tax Rate']) +
                result_df['Depreciation'] -
                result_df['Change in Working Capital'] -
                result_df['Capital Expenditures']
            ).clip(lower=0).replace([np.inf, -np.inf], 0)

            # Reset index to ensure consecutive integer indices starting from 0
            result_df = result_df.reset_index(drop=True)

            # Determine the forecast start index based on historical data
            if self.session_state and 'years_data' in self.session_state and self.session_state['years_data']:
                years_data = self.session_state['years_data']
                historical_max_year = max(years_data.keys())
                forecast_rows = result_df['Year'] > historical_max_year
                forecast_start_idx = forecast_rows.idxmax() if forecast_rows.any() else 0
            else:
                forecast_start_idx = 0

            # Calculate Present Value of Free Cash Flows
            present_values = []
            
            for i in range(forecast_start_idx, len(result_df)):
                years_from_start = i - forecast_start_idx + 1
                fcf = result_df['Unlevered FCF'].iloc[i]
                if not np.isfinite(fcf):
                    fcf = 0.0
                discount_factor = 1 / ((1 + wacc) ** years_from_start)
                pv_fcf = fcf * discount_factor
                present_values.append(pv_fcf)
                result_df.loc[i, 'PV of FCF'] = pv_fcf

            # Calculate Terminal Value and its Present Value
            final_fcf = result_df['Unlevered FCF'].iloc[-1]
            if not np.isfinite(final_fcf) or final_fcf <= 0:
                final_fcf = max(result_df['Unlevered FCF'].iloc[-2] if len(result_df) > 1 else 0, 0)
            if wacc <= perpetual_growth or wacc <= 0:
                raise ValueError(f"WACC ({wacc:.2%}) must be greater than perpetual growth ({perpetual_growth:.2%}) and positive.")
            terminal_value = final_fcf * (1 + perpetual_growth) / (wacc - perpetual_growth)
            forecast_periods = len(result_df) - forecast_start_idx
            if forecast_periods <= 0:
                forecast_periods = 1  # Ensure at least one period for discounting
            terminal_value_pv = terminal_value / ((1 + wacc) ** forecast_periods)
            result_df.loc[len(result_df) - 1, 'Terminal Value PV'] = terminal_value_pv

            pv_of_forecast_fcf = sum(present_values) if present_values else 0.0
            enterprise_value = pv_of_forecast_fcf + terminal_value_pv
            if not np.isfinite(enterprise_value):
                enterprise_value = 0.0

            net_debt = result_df['Long Term Debt'].iloc[-1] - result_df['Cash'].iloc[-1]
            if not np.isfinite(net_debt):
                net_debt = 0.0
            equity_value = enterprise_value - net_debt

           

            # Calculate valuation multiples
            last_ebitda = result_df['EBITDA'].iloc[-1]
            last_revenue = result_df['Net Revenue'].iloc[-1]
            ebitda_multiple = enterprise_value / last_ebitda if last_ebitda > 0 else np.nan
            revenue_multiple = enterprise_value / last_revenue if last_revenue > 0 else np.nan

            result_df['DCF Enterprise Value'] = enterprise_value
            result_df['DCF Equity Value'] = equity_value
            result_df['Implied EBITDA Multiple'] = ebitda_multiple
            result_df['Implied Revenue Multiple'] = revenue_multiple
            result_df['Net Debt'] = net_debt

            
          
            # Return valuation results as a dictionary
            valuation_results = {
                'enterprise_value': float(enterprise_value),
                'equity_value': float(equity_value),
                'terminal_value': float(terminal_value),
                'terminal_value_pv': float(terminal_value_pv),
                'present_value_fcf': float(pv_of_forecast_fcf),
                'implied_ebitda_multiple': float(ebitda_multiple) if pd.notna(ebitda_multiple) else 0.0,
                'implied_revenue_multiple': float(revenue_multiple) if pd.notna(revenue_multiple) else 0.0,
                'wacc': float(wacc),
                'perpetual_growth': float(perpetual_growth),
                'net_debt': float(net_debt),
                'forecast_start_year': int(result_df['Year'].iloc[forecast_start_idx]),
                'forecast_periods': int(forecast_periods)
            }

           
            logger.info(f"DCF valuation results: {valuation_results}")
            return valuation_results
        
        except Exception as e:
            logger.error(f"Error calculating DCF valuation: {str(e)}")
            raise   
    
    def calculate_valuation(self, df, discount_rate):
        """Calculate valuation metrics"""
        valuation_df = df.copy()
        
        if 'EBIT' not in valuation_df.columns:
            if 'EBITDA' in valuation_df.columns and 'Depreciation' in valuation_df.columns:
                valuation_df['EBIT'] = valuation_df['EBITDA'] - valuation_df['Depreciation']
            else:
                valuation_df['EBIT'] = (valuation_df['Net Income'] + 
                                      valuation_df.get('Interest', 0) + 
                                      valuation_df.get('Taxes', 0))
        
        required_cols = ['EBIT', 'Tax Rate', 'Depreciation', 'Capital Expenditures']
        if all(col in valuation_df.columns for col in required_cols):
            valuation_df['Unlevered FCF'] = (valuation_df['EBIT'] * (1 - valuation_df['Tax Rate']) +
                                           valuation_df['Depreciation'] -
                                           valuation_df['Capital Expenditures'] -
                                           valuation_df.get('Change in Working Capital', 0))
            forecast_years = len(valuation_df.index)
            discount_factors = [(1 + discount_rate) ** -i for i in range(1, forecast_years + 1)]
            valuation_df['PV of FCF'] = valuation_df['Unlevered FCF'] * discount_factors
            
            if 'EBITDA' in valuation_df.columns:
                terminal_value = valuation_df['EBITDA'].iloc[-1] * 8.0
                pv_terminal = terminal_value / ((1 + discount_rate) ** forecast_years)
                valuation_df['Total Enterprise Value'] = valuation_df['PV of FCF'].sum() + pv_terminal
            else:
                valuation_df['Total Enterprise Value'] = valuation_df['EBIT'].iloc[-1] * 8.0
        else:
            valuation_df['Total Enterprise Value'] = (valuation_df['EBITDA'] * 8.0 if 'EBITDA' in valuation_df.columns 
                                                    else valuation_df['Net Revenue'] * 2.0)        
        return valuation_df
    
    def calculate_break_even(self, df):
        """Calculate break-even metrics"""
        df['Fixed Costs'] = df['Total Fixed Costs']
        df['Variable Cost per Order'] = df['Total Variable Costs'] / df['Total Orders']
        df['Contribution Margin per Order'] = df['Average Net Order Value'] - df['Variable Cost per Order']
        df['Break Even Units'] = df['Fixed Costs'] / df['Contribution Margin per Order']
        df['Break Even Dollars'] = df['Break Even Units'] * df['Average Net Order Value']
        return df   

    def forecast_metrics(self, historical_df: pd.DataFrame, forecast_years: int = 10) -> pd.DataFrame:
        logger.info("Starting forecast of input metrics")
        
        if historical_df.empty:
            logger.warning("Historical DataFrame is empty. Returning empty DataFrame.")
            return pd.DataFrame(columns=historical_df.columns)
        
        if 'Year' not in historical_df.columns:
            logger.error("Year column missing in historical DataFrame.")
            raise ValueError("Year column is required in historical DataFrame.")
        
        last_year = int(historical_df['Year'].iloc[-1])
        forecast_range = range(last_year + 1, last_year + forecast_years + 1)
        forecasted_values = {}
        
        def create_forecast(values, n_years):
            values = pd.to_numeric(values, errors='coerce').dropna()
            if values.empty:
                return np.zeros(n_years)
            elif len(values) < 2:
                last_val = values.iloc[-1]
                growth = values.pct_change().mean() if len(values) > 1 else 0.01
                return [last_val * (1 + growth) ** i for i in range(n_years)]
            else:
                X = np.array(range(len(values))).reshape(-1, 1)
                model = LinearRegression().fit(X, values.values.reshape(-1, 1))
                future_X = np.array(range(len(values), len(values) + n_years)).reshape(-1, 1)
                forecast = model.predict(future_X).flatten()
                return np.where(np.isnan(forecast), values.iloc[-1], forecast)

        def get_dynamic_bounds(metric, values):
            values = pd.to_numeric(values, errors='coerce').dropna()
            if values.empty:
                return 0, 1
            mean_val = values.mean()
            std_val = values.std() if len(values) > 1 else mean_val * 0.1
            lower = max(0, values.min() * 0.5)
            upper = values.max() * 1.5
            return lower, upper

        input_metrics = [
            'Email Traffic', 'Organic Search Traffic', 'Paid Search Traffic', 'Affiliates Traffic',
            'Email Conversion Rate', 'Organic Search Conversion Rate',
            'Paid Search Conversion Rate', 'Affiliates Conversion Rate',
            'Average Item Value', 'Number of Items per Order', 'Average Markdown',
            'Average Promotion/Discount', 'COGS Percentage', 'Churn Rate',
            'Email Cost per Click', 'Organic Search Cost per Click',
            'Paid Search Cost per Click', 'Affiliates Cost per Click',
            'Freight/Shipping per Order', 'Labor/Handling per Order',
            'General Warehouse Rent', 'Office Rent', 'Salaries, Wages & Benefits',
            'Professional Fees', 'Other', 'Depreciation', 'Interest', 'Tax Rate',
            'Accounts Receivable Days', 'Inventory Days', 'Accounts Payable Days',
            'Technology Development', 'Office Equipment',
            'Technology Depreciation Years', 'Office Equipment Depreciation Years',
            'Interest Rate', 'Debt Issued', 'Equity Raised', 'Dividends Paid'
        ]

        for metric in input_metrics:
            if metric in historical_df.columns:
                forecast = create_forecast(historical_df[metric], len(forecast_range))
                lower, upper = get_dynamic_bounds(metric, historical_df[metric])
                if 'Rate' in metric or 'Percentage' in metric:
                    forecast = np.clip(forecast, max(0, lower), min(1, upper))
                elif 'Days' in metric or 'Years' in metric:
                    forecast = np.maximum(forecast, max(0, lower))
                else:
                    forecast = np.maximum(forecast, max(0, lower))
                forecasted_values[metric] = forecast
            else:
                if 'Rate' in metric or 'Percentage' in metric:
                    similar_cols = [c for c in historical_df.columns if 'Rate' in c or 'Percentage' in c]
                    default = historical_df[similar_cols].mean().mean() if similar_cols else 0.05
                elif 'Days' in metric:
                    similar_cols = [c for c in historical_df.columns if 'Days' in c]
                    default = historical_df[similar_cols].mean().mean() if similar_cols else 30
                elif 'Years' in metric:
                    similar_cols = [c for c in historical_df.columns if 'Years' in c]
                    default = historical_df[similar_cols].mean().mean() if similar_cols else 5
                else:
                    default = historical_df.mean(numeric_only=True).mean() or 0
                forecasted_values[metric] = np.full(len(forecast_range), default)

        for col in historical_df.columns:
            if col not in input_metrics and col != 'Year':
                forecasted_values[col] = np.full(len(forecast_range), historical_df[col].iloc[-1])

        forecast_df = pd.DataFrame(forecasted_values, index=forecast_range)
        forecast_df['Year'] = list(forecast_range)
        forecast_df = forecast_df[['Year'] + [col for col in forecast_df.columns if col != 'Year']]
        
        # combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
        # combined_df = (
        #     combined_df
        #     .drop_duplicates(subset="Year", keep="first")
        #     .reset_index(drop=True)
        # )
        forecast_df = forecast_df.reset_index(drop=True)
        # logger.info(f"Forecasted DataFrame shape: {combined_df.shape}, Years: {combined_df['Year'].tolist()}")
        return forecast_df

    def _scenario_parameters(self, scenario_type: str) -> Dict:
        """Define default scenario parameters."""
        params = {}
        print("scenario_type ", scenario_type)
        if scenario_type == "Best Case":
            params = {
                'conversion_rate_mult': 1.2,
                'aov_mult': 1.1,
                'cogs_mult': 0.95,
                'interest_mult': 0.9,
                'labor_mult': 0.9,
                'material_mult': 0.9,
                'markdown_mult': 0.9,
                'political_risk': 2,
                'env_impact': 2
            }
        elif scenario_type == "Worst Case":
            params = {
                'conversion_rate_mult': 0.8,
                'aov_mult': 0.9,
                'cogs_mult': 1.05,
                'interest_mult': 1.2,
                'labor_mult': 1.2,
                'material_mult': 1.2,
                'markdown_mult': 1.1,
                'political_risk': 4,
                'env_impact': 4
            }
        else:  # Base Case
            params = {
                'conversion_rate_mult': 1.0,
                'aov_mult': 1.0,
                'cogs_mult': 1.0,
                'interest_mult': 1.0,
                'labor_mult': 1.0,
                'material_mult': 1.0,
                'markdown_mult': 1.0,
                'political_risk': 0,
                'env_impact': 0
            }
        return params

    def _create_scenarios(
        self,
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        params: Dict
    ) -> pd.DataFrame:
        """
        Apply Best/Worst multipliers to forecast_df only,
        then concat to historical_df (no overlapping years).
        """
        # 1) Copy so we don’t clobber the original
        scenario_forecast = forecast_df.copy()

        # 2) Apply all your multipliers
        for rate_col in [
            'Email Conversion Rate',
            'Organic Search Conversion Rate',
            'Paid Search Conversion Rate',
            'Affiliates Conversion Rate',
        ]:
            scenario_forecast[rate_col] *= params['conversion_rate_mult']

        scenario_forecast['Average Item Value']                *= params['aov_mult']
        scenario_forecast['COGS Percentage']                   *= params['cogs_mult']
        scenario_forecast['Average Markdown']                  *= params['markdown_mult']
        scenario_forecast['Interest']                          *= params['interest_mult']
        scenario_forecast['Labor/Handling per Order']          *= params['labor_mult']
        scenario_forecast['Political Risk Factor']             = params['political_risk']
        scenario_forecast['Environmental Impact Factor']       = params['env_impact']
        scenario_forecast['Risk Adjustment Factor']            = (
            params['political_risk'] + params['env_impact']
        ) / 10

        # 3) Stitch them back together (no duplicate years)
        scenario_df = pd.concat([historical_df, scenario_forecast], ignore_index=True)
        # drop any duplicate years (keep the historical row)
        scenario_df = (
            scenario_df
            .drop_duplicates(subset="Year", keep="first")
            .reset_index(drop=True)
        )
        return scenario_df

    def create_scenarios(
        self,
        df: pd.DataFrame,
        scenario_type: str,
        params: Dict
    ) -> Dict[str, pd.DataFrame]:
        """
        Build Base, Best, and Worst scenario DataFrames.
        """
        # 0) Always keep Base unchanged
        self.scenarios['Base'] = df.copy()

        # 1) Determine the last _user-loaded_ year
        years_data = self.session_state.get('years_data', {}) or {}
        if years_data:
            # sort keys to find the max
            hist_years = sorted(years_data.keys())
            max_hist   = hist_years[-1]
            historical_df = df[df['Year'] <= max_hist].reset_index(drop=True)
            forecast_df   = df[df['Year'] >  max_hist].reset_index(drop=True)
        else:
            # nothing loaded → treat all as historical, no forecast
            historical_df = df.copy().reset_index(drop=True)
            forecast_df   = df.iloc[0:0].copy()  # empty

        # 2) If we’re in Best/Worst, re-compute that slice
        if scenario_type in ("Best Case", "Worst Case"):
            scen_key = scenario_type.split()[0]  # "Best" or "Worst"
            self.scenarios[scen_key] = self._create_scenarios(
                historical_df,
                forecast_df,
                params
            )
        else:
            # clear any prior scenarios
            self.scenarios['Best']  = None
            self.scenarios['Worst'] = None

        # 3) Re-run your P&L, CF, BS chain on each non-None scenario
        for key in ("Base", "Best", "Worst"):
            df_scn = self.scenarios.get(key)
            if df_scn is not None:
                df_scn = self.calculate_income_statement(df_scn)
                df_scn = self.calculate_cash_flow_statement(df_scn)
                df_scn = self.calculate_balance_sheet(df_scn)
                self.scenarios[key] = df_scn

        return self.scenarios
    
    def calculate_cagr(self, df: pd.DataFrame, metric: str) -> float:
        logger.info(f"Calculating CAGR for metric: {metric}")
        
        try:
            if df.empty or metric not in df.columns:
                logger.warning(f"DataFrame is empty or metric '{metric}' not found")
                return 0.0
            
            values = pd.to_numeric(df[metric], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) < 2:
                logger.warning(f"Insufficient data for CAGR calculation: {len(values)} periods")
                return 0.0
            
            start_value = values.iloc[0]
            end_value = values.iloc[-1]
            num_periods = len(values) - 1
            
            if start_value <= 0 or end_value <= 0 or num_periods <= 0:
                logger.warning(f"Invalid values for CAGR: start={start_value}, end={end_value}, periods={num_periods}")
                return 0.0
            
            cagr = (end_value / start_value) ** (1.0 / num_periods) - 1
            if not np.isfinite(cagr):
                logger.warning(f"Non-finite CAGR result: {cagr}")
                return 0.0
                
            logger.debug(f"CAGR for {metric}: {cagr:.4f}")
            return float(cagr)
        
        except Exception as e:
            logger.error(f"Error calculating CAGR for {metric}: {str(e)}")
            return 0.0  
        
    def run_precision_tree(self, df: pd.DataFrame) -> Dict:
        """Simulate PrecisionTree: Probabilistic Decision Trees with fully dynamic parameters."""
        logger.info("Running PrecisionTree: Decision Tree Analysis")        
         
        try:
            if df is None or df.empty:
                logger.error("DataFrame is empty or unavailable")
                raise HTTPException(status_code=400, detail="DataFrame is empty or unavailable")

            # Required columns for the analysis
            required_columns = [
                'Email Traffic', 'Paid Search Traffic', 'Email Conversion Rate',
                'Paid Search Conversion Rate', 'Email Cost per Click',
                'Paid Search Cost per Click', 'Average Net Order Value'
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_columns}")

            # Ensure at least 2 rows for pct_change
            if len(df) < 2:
                logger.error("At least 2 rows of data required for growth calculations")
                raise HTTPException(status_code=400, detail="At least 2 rows of data required for growth calculations")

            # Clean DataFrame: replace NaN with 0 and ensure numeric types
            df = df.copy()
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)

            # Calculate growth rates and their volatility
            email_growth = clean_float(df['Email Traffic'].pct_change().mean())
            email_growth_vol = clean_float(df['Email Traffic'].pct_change().std())
            paid_growth = clean_float(df['Paid Search Traffic'].pct_change().mean())
            paid_growth_vol = clean_float(df['Paid Search Traffic'].pct_change().std())

            # Dynamically calculate probabilities based on historical growth trends
            total_growth = email_growth + paid_growth
            if total_growth == 0:
                # Use historical conversion rates as a fallback for probability
                email_conv_rate = clean_float(df['Email Conversion Rate'].mean())
                paid_conv_rate = clean_float(df['Paid Search Conversion Rate'].mean())
                total_conv = email_conv_rate + paid_conv_rate
                email_high_prob = email_conv_rate / total_conv if total_conv != 0 else 0.5
                paid_high_prob = paid_conv_rate / total_conv if total_conv != 0 else 0.5
            else:
                email_high_prob = email_growth / total_growth
                paid_high_prob = paid_growth / total_growth

            # Use volatility to set dynamic bounds (e.g., mean ± 1 std deviation)
            email_prob_bound_lower = max(0.05, email_high_prob - email_growth_vol) if email_growth_vol != 0 else 0.1
            email_prob_bound_upper = min(0.95, email_high_prob + email_growth_vol) if email_growth_vol != 0 else 0.9
            paid_prob_bound_lower = max(0.05, paid_high_prob - paid_growth_vol) if paid_growth_vol != 0 else 0.1
            paid_prob_bound_upper = min(0.95, paid_high_prob + paid_growth_vol) if paid_growth_vol != 0 else 0.9

            email_high_prob = min(max(email_high_prob, email_prob_bound_lower), email_prob_bound_upper)
            paid_high_prob = min(max(paid_high_prob, paid_prob_bound_lower), paid_prob_bound_upper)

            # Define decision tree
            G = nx.DiGraph()
            G.add_node("D1", label="Marketing Budget", type="decision")
            G.add_node("C1", label="Email Traffic Outcome", type="chance")
            G.add_node("C2", label="Paid Traffic Outcome", type="chance")
            G.add_node("T1", label="High Email Success", type="terminal", value=0)
            G.add_node("T2", label="Low Email Success", type="terminal", value=0)
            G.add_node("T3", label="High Paid Success", type="terminal", value=0)
            G.add_node("T4", label="Low Paid Success", type="terminal", value=0)

            # Dynamic costs based on historical marketing expenses
            email_cost = clean_float(df['Email Cost per Click'].mean() * df['Email Traffic'].mean() * 12)
            paid_cost = clean_float(df['Paid Search Cost per Click'].mean() * df['Paid Search Traffic'].mean() * 12)

            # Dynamic decision probabilities based on historical budget allocation
            total_cost = email_cost + paid_cost
            email_decision_prob = email_cost / total_cost if total_cost != 0 else 0.5
            paid_decision_prob = paid_cost / total_cost if total_cost != 0 else 0.5

            G.add_edge("D1", "C1", label="Increase Email Budget", cost=-email_cost, probability=email_decision_prob)
            G.add_edge("D1", "C2", label="Increase Paid Budget", cost=-paid_cost, probability=paid_decision_prob)
            G.add_edge("C1", "T1", label="High", probability=email_high_prob)
            G.add_edge("C1", "T2", label="Low", probability=1 - email_high_prob)
            G.add_edge("C2", "T3", label="High", probability=paid_high_prob)
            G.add_edge("C2", "T4", label="Low", probability=1 - paid_high_prob)

            # Calculate terminal values based on historical data
            avg_email_traffic = clean_float(df['Email Traffic'].mean())
            avg_paid_traffic = clean_float(df['Paid Search Traffic'].mean())
            avg_email_conv = clean_float(df['Email Conversion Rate'].mean())
            avg_paid_conv = clean_float(df['Paid Search Conversion Rate'].mean())
            avg_order_value = clean_float(df['Average Net Order Value'].mean())

            # Dynamic high/low success scenarios based on growth and volatility
            email_high_traffic = avg_email_traffic * (1 + email_growth + email_growth_vol) if email_growth_vol != 0 else avg_email_traffic * (1 + email_growth * 2)
            email_low_traffic = avg_email_traffic * (1 + email_growth - email_growth_vol) if email_growth_vol != 0 else avg_email_traffic * (1 + email_growth * 0.5)
            paid_high_traffic = avg_paid_traffic * (1 + paid_growth + paid_growth_vol) if paid_growth_vol != 0 else avg_paid_traffic * (1 + paid_growth * 2)
            paid_low_traffic = avg_paid_traffic * (1 + paid_growth - paid_growth_vol) if paid_growth_vol != 0 else avg_paid_traffic * (1 + paid_growth * 0.5)

            G.nodes["T1"]['value'] = clean_float(email_high_traffic * avg_email_conv * avg_order_value * 12)
            G.nodes["T2"]['value'] = clean_float(email_low_traffic * avg_email_conv * avg_order_value * 12)
            G.nodes["T3"]['value'] = clean_float(paid_high_traffic * avg_paid_conv * avg_order_value * 12)
            G.nodes["T4"]['value'] = clean_float(paid_low_traffic * avg_paid_conv * avg_order_value * 12)

            # Calculate expected values
            expected_values = {}
            for chance_node in [n for n, d in G.nodes(data=True) if d['type'] == 'chance']:
                successors = list(G.successors(chance_node))
                ev = sum(G.edges[chance_node, succ]['probability'] * G.nodes[succ]['value'] for succ in successors)
                expected_values[chance_node] = clean_float(ev)

            # Decision node evaluation
            decision_outcomes = {}
            for decision in [n for n, d in G.nodes(data=True) if d['type'] == 'decision']:
                for succ in G.successors(decision):
                    cost = G.edges[decision, succ]['cost']
                    ev = expected_values[succ] + cost
                    decision_outcomes[G.edges[decision, succ]['label']] = clean_float(ev)

            
            outcomes_df = pd.DataFrame.from_dict(decision_outcomes, orient='index', columns=['Expected Value ($)'])
      

            # Visualize Decision Tree
            plt.figure(figsize=(10, 6))
            pos = nx.spring_layout(G)
            node_labels = {node: G.nodes[node]['label'] for node in G.nodes}
            nx.draw(G, pos, with_labels=True, labels=node_labels, node_color='lightblue', node_size=2000, font_size=10)
            edge_labels = {(u, v): f"{d.get('label', '')}\nProb: {d.get('probability', ''):.2f}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            plt.title("Decision Tree (PrecisionTree Simulation)")
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close()

            # Prepare response
            response = {
                "decision_outcomes": decision_outcomes,
                "decision_tree_image": f"data:image/png;base64,{img_str}",
                "message": "PrecisionTree analysis completed successfully"
            }

            logger.info("PrecisionTree analysis completed successfully")
         
            return response

        except Exception as e:
            logger.error(f"Error in PrecisionTree analysis: {str(e)}")           
            raise HTTPException(status_code=500, detail=f"Error in PrecisionTree analysis: {str(e)}")
     
  
    def run_top_rank_sensitivity(self, df: pd.DataFrame, variables_to_test: List[str], change_percentage: float, discount_rate: float) -> Dict:
        """Simulate TopRank: What-If Sensitivity Analysis with dynamic variables and full financial impact."""
        logger.info("Running TopRank: Sensitivity Analysis")
        
        if df.empty or not variables_to_test:
            logger.error("DataFrame is empty or no variables selected")
            raise HTTPException(status_code=400, detail="DataFrame is empty or no variables selected")
        if change_percentage <= 0:
            logger.error("Change percentage must be positive")
            raise HTTPException(status_code=400, detail="Change percentage must be positive")
        
        # Create a deep copy to preserve original data
        base_df = df.copy(deep=True)
        self.calculate_income_statement(base_df)
        self.calculate_cash_flow_statement(base_df)
        self.calculate_balance_sheet(base_df)
        self.calculate_valuation(base_df, discount_rate)
        
        # Baseline metrics
        base_metrics = {
            'Net Income': base_df['Net Income'].iloc[-1],
            'EBITDA': base_df['EBITDA'].iloc[-1],
            'Net Cash Flow': base_df['Net Cash Flow'].iloc[-1],
            'Equity Value': base_df['DCF Equity Value'].iloc[-1] if 'DCF Equity Value' in base_df.columns else 0.0
        }
        for metric, value in base_metrics.items():
            if not np.isfinite(value) or value == 0:
                base_metrics[metric] = 1e-6
                logger.warning(f"Baseline {metric} is zero or non-finite, using {1e-6} as default")
        
        # Initialize sensitivity results
        sensitivity_results = []
        sample_sensitivity_results= {var: {'Increase': {}, 'Decrease': {}} for var in variables_to_test}

         # Perform sensitivity analysis
        for var in variables_to_test:
            if var not in base_df.columns:
                print(f"Variable {var} not found in dataset. Skipping.")
                # continue
                return {
                    "status":"error",
            "message": f"Invalid Variable - {var}" ,
            "sensitivity_results":[],
            "sensitivity_insights":[] 
        }

            for direction, factor in [('Increase', 1 + change_percentage / 100), ('Decrease', 1 - change_percentage / 100)]:
                temp_df = base_df.copy(deep=True)
                
                # Apply percentage change to the variable
                temp_df[var] *= factor
                self.calculate_income_statement(temp_df)  # Recalculate income statement
                
                # Recalculate downstream financials
                self.calculate_cash_flow_statement(temp_df)
                self.calculate_balance_sheet(temp_df)
                self.calculate_valuation(temp_df, self.session_state.get('discount_rate', 0.20))

                # Update metrics
                new_metrics = {
                    'Net Income': temp_df['Net Income'].iloc[-1],
                    'EBITDA': temp_df['EBITDA'].iloc[-1],
                    'Net Cash Flow': temp_df['Net Cash Flow'].iloc[-1],
                    'Equity Value': temp_df['DCF Equity Value'].iloc[-1] if 'DCF Equity Value' in temp_df.columns else 0.0
                }
                for metric in base_metrics:
                    if np.isfinite(new_metrics[metric]):
                        sample_sensitivity_results[var][direction][metric] = ((new_metrics[metric] - base_metrics[metric]) / base_metrics[metric] * 100)
                    else:
                        sample_sensitivity_results[var][direction][metric] = 0.0
                        print(f"{metric} is non-finite for {var} {direction}. Set to 0% change.")
        
        # Perform sensitivity analysis
        for var in variables_to_test:
            if var not in base_df.columns:
                logger.warning(f"Variable {var} not found in dataset. Skipping")
                continue
            
            for direction, factor in [('Increase', 1 + change_percentage / 100), ('Decrease', 1 - change_percentage / 100)]:
                temp_df = base_df.copy(deep=True)
                
                # Apply percentage change
                temp_df[var] *= factor
                self.calculate_income_statement(temp_df)
                self.calculate_cash_flow_statement(temp_df)
                self.calculate_balance_sheet(temp_df)
                self.calculate_valuation(temp_df, discount_rate)
                
                # Calculate percentage changes
                new_metrics = {
                    'Net Income': temp_df['Net Income'].iloc[-1],
                    'EBITDA': temp_df['EBITDA'].iloc[-1],
                    'Net Cash Flow': temp_df['Net Cash Flow'].iloc[-1],
                    'Equity Value': temp_df['DCF Equity Value'].iloc[-1] if 'DCF Equity Value' in temp_df.columns else 0.0
                }
                result = {
                    "variable": var,
                    "direction": direction,
                    "net_income_change": 0.0,
                    "ebitda_change": 0.0,
                    "net_cash_flow_change": 0.0,
                    "equity_value_change": 0.0
                }
                for metric in base_metrics:
                    if np.isfinite(new_metrics[metric]):
                        change = ((new_metrics[metric] - base_metrics[metric]) / base_metrics[metric] * 100)
                        result[f"{metric.lower().replace(' ', '_')}_change"] = change
                    else:
                        logger.warning(f"{metric} is non-finite for {var} {direction}. Set to 0% change")                
                sensitivity_results.append(result)
        sensitivity_insights=[]

        # Convert results to DataFrame with unique column names
        metrics_display = ['Net Income', 'EBITDA', 'Net Cash Flow', 'Equity Value']
        results_df = pd.DataFrame({
            f'{var} (+{change_percentage}%)': [sample_sensitivity_results[var]['Increase'].get(metric, 0) for metric in metrics_display]
            for var in variables_to_test
        })
        results_df_decrease = pd.DataFrame({
            f'{var} (-{change_percentage}%)': [sample_sensitivity_results[var]['Decrease'].get(metric, 0) for metric in metrics_display]
            for var in variables_to_test
        })

        # Combine increase and decrease results
        display_df = pd.concat([results_df, results_df_decrease], axis=1)
        display_df.index = metrics_display

        for var in variables_to_test:
            for metric in metrics_display:
                increase_col = f'{var} (+{change_percentage}%)'
                decrease_col = f'{var} (-{change_percentage}%)'
                if increase_col in display_df.columns and decrease_col in display_df.columns:
                    increase_impact = display_df.loc[metric, increase_col]
                    decrease_impact = display_df.loc[metric, decrease_col]
                    sensitivity_insights.append(f"- **{var}**: +{change_percentage}% changes {metric} by {increase_impact:.2f}%, "
                            f"-{change_percentage}% changes it by {decrease_impact:.2f}%.")
                    print(f"- **{var}**: +{change_percentage}% changes {metric} by {increase_impact:.2f}%, "
                            f"-{change_percentage}% changes it by {decrease_impact:.2f}%.")
        
        response = {
            "status":"success",
            "sensitivity_results": sensitivity_results,
            "message": f"Sensitivity analysis completed for {change_percentage}% change",
            "sensitivity_insights":sensitivity_insights
        }
        
        logger.info("TopRank sensitivity analysis completed successfully")
        return response
    
    def run_stat_tools_forecasting(self, filtered_df: pd.DataFrame, forecast_years: int, confidence_level: float) -> Dict:
        """Simulate StatTools: Statistical forecasting with ARIMA starting from last historical year."""
        logger.info("Running StatTools: Statistical Forecasting Analysis")
        
        try:
            historical_data = pd.DataFrame()
            if os.path.exists(EXCEL_FILE):
                try:
                    historical_data = pd.read_excel(EXCEL_FILE)
                    
                    print(f"Loaded historical data from {EXCEL_FILE} with {len(historical_data)} years.")
                except Exception as e:
                    
                    print(f"Error loading historical data: {e}")
                    return
            else:
                print(f"No historical data found in {EXCEL_FILE}. Please ensure the file exists.")
          
                return
            
            if historical_data.empty or 'Year' not in historical_data.columns:
                logger.error("Historical data is empty or missing 'Year' column")
                raise HTTPException(status_code=400, detail="Historical data is empty or missing 'Year' column")
            
            if len(historical_data) < 3:
                logger.error("At least 3 years of historical data required")
                raise HTTPException(status_code=400, detail="At least 3 years of historical data required")
            
            # Ensure numeric data
            for col in historical_data.columns:
                historical_data[col] = pd.to_numeric(historical_data[col], errors='coerce').fillna(0).clip(lower=0)
            
            self.calculate_income_statement(historical_data)
            self.calculate_cash_flow_statement(historical_data)
            self.calculate_supporting_schedules(historical_data)
            self.calculate_balance_sheet(historical_data)
            
            # Define time range
            last_historical_year = int(historical_data['Year'].max())
            forecast_range = range(last_historical_year + 1, last_historical_year + forecast_years + 1)
            all_years = list(historical_data['Year']) + list(forecast_range)
            
            # Metrics to forecast
            metrics_to_forecast = ['Net Revenue', 'EBITDA', 'Net Income', 'Total Orders']
            available_metrics = [m for m in metrics_to_forecast if m in historical_data.columns]
            
            if not available_metrics:
                logger.error("No valid metrics available for forecasting")
                raise HTTPException(status_code=400, detail="No valid metrics available for forecasting")
            
            # Forecasting with ARIMA
            forecast_results = {}
            alpha = 1 - (confidence_level / 100)
            
            for metric in available_metrics:
                historical_values = historical_data[metric].values
                
                try:
                    model = ARIMA(historical_values, order=(1, 1, 1))
                    model_fit = model.fit()
                    forecast_result = model_fit.get_forecast(steps=forecast_years)
                    forecast_mean = forecast_result.predicted_mean
                    conf_int = forecast_result.conf_int(alpha=alpha)
                    
                    full_series = np.concatenate([historical_values, forecast_mean])
                    lower_ci = np.concatenate([historical_values, conf_int[:, 0]])
                    upper_ci = np.concatenate([historical_values, conf_int[:, 1]])
                    
                    forecast_results[metric] = {
                        'full_series': full_series.tolist(),
                        'historical': historical_values.tolist(),
                        'forecast': forecast_mean.tolist(),
                        'lower_ci': lower_ci.tolist(),
                        'upper_ci': upper_ci.tolist()
                    }
                except Exception as e:
                    logger.warning(f"Failed to forecast {metric}: {str(e)}")
                    continue
            
            if not forecast_results:
                logger.error("Failed to generate forecasts for any metrics")
                raise HTTPException(status_code=500, detail="Failed to generate forecasts for any metrics")
            
            # Forecast Data Table
            forecast_df = pd.DataFrame({'Year': all_years})
            for metric in forecast_results:
                forecast_df[f"{metric} Historical"] = np.concatenate([forecast_results[metric]['historical'], [np.nan] * forecast_years])
                forecast_df[f"{metric} Forecast"] = forecast_results[metric]['full_series']
                forecast_df[f"{metric} Lower {confidence_level}% CI"] = forecast_results[metric]['lower_ci']
                forecast_df[f"{metric} Upper {confidence_level}% CI"] = forecast_results[metric]['upper_ci']
            
            format_dict = {'Year': '{:.0f}'}
            for metric in available_metrics:
                format_dict.update({
                    f"{metric} Historical": '${:,.0f}' if metric != 'Total Orders' else '{:,.0f}',
                    f"{metric} Forecast": '${:,.0f}' if metric != 'Total Orders' else '{:,.0f}',
                    f"{metric} Lower {confidence_level}% CI": '${:,.0f}' if metric != 'Total Orders' else '{:,.0f}',
                    f"{metric} Upper {confidence_level}% CI": '${:,.0f}' if metric != 'Total Orders' else '{:,.0f}'
                })
            
            # Summary Statistics
            stats_df = pd.DataFrame(index=available_metrics)
            for metric in forecast_results:
                forecast_values = forecast_results[metric]['forecast']
                stats_df.loc[metric, 'Mean Forecast'] = np.mean(forecast_values)
                stats_df.loc[metric, 'Last Forecast'] = forecast_values[-1]
                stats_df.loc[metric, f'{confidence_level}% CI Width'] = (
                    forecast_results[metric]['upper_ci'][-1] - forecast_results[metric]['lower_ci'][-1]
                )
            
            # Chart Data
            chart_data = []
            for metric in forecast_results:
                traces = [
                    {
                        "x": historical_data['Year'].tolist(),
                        "y": forecast_results[metric]['historical'],
                        "name": "Historical",
                        "type": "scatter",
                        "mode": "lines+markers",
                        "line": {"color": "#2563eb", "width": 2.5},
                        "marker": {"size": 8},
                        "showlegend": True
                    },
                    {
                        "x": all_years,
                        "y": forecast_results[metric]['full_series'],
                        "name": "Forecast",
                        "type": "scatter",
                        "mode": "lines+markers",
                        "line": {"color": "#22c55e", "width": 2.5, "dash": "dash"},
                        "marker": {"size": 8},
                        "showlegend": True
                    },
                    {
                        "x": list(forecast_range),
                        "y": forecast_results[metric]['lower_ci'][len(historical_data):],
                        "name": f"Lower {confidence_level}% CI",
                        "type": "scatter",
                        "mode": "lines",
                        "line": {"color": "#f59e0b", "width": 1, "dash": "dot"},
                        "showlegend": True
                    },
                    {
                        "x": list(forecast_range),
                        "y": forecast_results[metric]['upper_ci'][len(historical_data):],
                        "name": f"Upper {confidence_level}% CI",
                        "type": "scatter",
                        "mode": "lines",
                        "line": {"color": "#f59e0b", "width": 1, "dash": "dot"},
                        "fill": "tonexty",
                        "fillcolor": "rgba(245, 158, 11, 0.25)",
                        "showlegend": True
                    }
                ]
                layout = {
                    "title": {"text": f"{metric} Forecast (Historical + {forecast_years} Years)"},
                    "xaxis": {"title": "Year"},
                    "yaxis": {
                        "title": metric,
                        "tickformat": "$,.0f" if metric != "Total Orders" else ",.0f"
                    },
                    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}
                }
                chart_data.append({"metric": metric, "traces": traces, "layout": layout})
            
            # Prepare response
            response = {
                "forecast_data": forecast_df.to_dict(orient='records'),
                "summary_statistics": stats_df.to_dict(orient='index'),
                "format_dict": format_dict,
                "chart_data": chart_data,
                "message": f"Successfully generated forecasts for {forecast_years} years at {confidence_level}% confidence level"
            }
            
            logger.info("StatTools forecasting completed successfully")
            return response
        
        except Exception as e:
            logger.error(f"Error in StatTools forecasting: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in StatTools forecasting: {str(e)}")

    def run_neural_tools_prediction(self, df: pd.DataFrame, traffic_increase_percentage: float) -> dict:
        """Simulate NeuralTools: Predictive Neural Networks with dynamic inputs."""
        logger.info("Running NeuralTools: Neural Network Prediction")
        
        # Prepare data for neural network
        features = ['Email Traffic', 'Organic Search Traffic', 'Paid Search Traffic', 'Affiliates Traffic']
        target = 'Net Revenue'
        
        # Check if required columns exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing features for neural network: {missing_features}")
            raise HTTPException(status_code=400, detail=f"Missing features for neural network: {missing_features}")
        if target not in df.columns:
            logger.error(f"Target variable {target} not found in dataset")
            raise HTTPException(status_code=400, detail=f"Target variable {target} not found in dataset")
        
        X = df[features].values
        y = df[target].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train neural network
        nn = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
        nn.fit(X_scaled, y)
        
        # Predict future revenue with dynamic traffic increase
        future_X = X[-1:] * (1 + traffic_increase_percentage / 100)
        future_X_scaled = scaler.transform(future_X)
        predicted_revenue = nn.predict(future_X_scaled)[0]
        
        # Feature importance
        importance = np.abs(nn.coefs_[0]).sum(axis=1)
        importance_normalized = importance / importance.sum()
        
        # Prepare response
        response = {
            "traffic_increase_percentage": traffic_increase_percentage,
            "predicted_revenue": float(predicted_revenue),
            "feature_importance": [
                {"feature": feature, "importance": float(imp)}
                for feature, imp in zip(features, importance_normalized)
            ],
            "message": f"Predicted Net Revenue with {traffic_increase_percentage}% Traffic Increase: ${predicted_revenue:,.0f}"
        }
        
        logger.info(f"NeuralTools prediction completed: predicted_revenue=${predicted_revenue:,.0f}")
        return response       

    @timeout(30) 
    def run_evolver_optimization(self, df, budget_dict, forecast_years=10):
        logger.info(f"Running Evolver Optimization with budget: {budget_dict}, years: {forecast_years}")
        if df.empty or 'Year' not in df.columns:
            logger.error("Invalid DataFrame")
            raise HTTPException(status_code=400, detail="DataFrame is empty or missing 'Year' column")
        budget_line = list(budget_dict.keys())[0]
        budget_amount = budget_dict[budget_line]
        last_year = df['Year'].max()
        working_df = df[df['Year'] == last_year].copy()

        # Scale traffic to match historical orders
        historical_orders = working_df['Total Orders'].iloc[0]
        calculated_orders = (
            working_df['Email Traffic'] * working_df['Email Conversion Rate'] +
            working_df['Organic Search Traffic'] * working_df['Organic Search Conversion Rate'] +
            working_df['Paid Search Traffic'] * working_df['Paid Search Conversion Rate'] +
            working_df['Affiliates Traffic'] * working_df['Affiliates Conversion Rate']
        ).iloc[0]
        if calculated_orders > 0:
            scale = historical_orders / calculated_orders
            for traffic in ['Email Traffic', 'Organic Search Traffic', 'Paid Search Traffic', 'Affiliates Traffic']:
                working_df[traffic] *= scale
            logger.info(f"Scaled traffic by {scale:.2f} to match {historical_orders} orders")

        # Extend DataFrame
        base_row = working_df.iloc[-1].copy()
        growth_rate = 1.05
        for year in range(int(last_year) + 1, int(last_year) + forecast_years):
            new_row = base_row.copy()
            new_row['Year'] = year
            for col in ['Average Gross Order Value', 'COGS Percentage', 'Freight/Shipping per Order', 'Labor/Handling per Order']:
                if col in new_row:
                    new_row[col] *= growth_rate
            for traffic in ['Email Traffic', 'Organic Search Traffic', 'Paid Search Traffic', 'Affiliates Traffic']:
                if traffic in new_row:
                    new_row[traffic] = max(1.0, new_row[traffic])
            working_df = pd.concat([working_df, pd.DataFrame([new_row])], ignore_index=True)
        working_df = working_df.head(forecast_years)

        # Reset order/revenue columns
        for col in ['Email Orders', 'Organic Search Orders', 'Paid Search Orders', 'Affiliates Orders', 'Total Orders', 'Gross Revenue', 'Net Revenue', 'COGS', 'Marketing Expenses', 'Fulfilment Expenses', 'Total Variable Costs']:
            if col in working_df.columns:
                working_df[col] = 0

        variables = ['Email Traffic', 'Organic Search Traffic', 'Paid Search Traffic', 'Affiliates Traffic']
        cost_per_click_map = {
            'Email Traffic': 'Email Cost per Click',
            'Organic Search Traffic': 'Organic Search Cost per Click',
            'Paid Search Traffic': 'Paid Search Cost per Click',
            'Affiliates Traffic': 'Affiliates Cost per Click'
        }
        active_variables = [var for var in variables if var in working_df.columns and working_df[cost_per_click_map[var]].iloc[0] > 0]
        if not active_variables:
            raise HTTPException(status_code=400, detail="No traffic channels have positive CPC")

        self.calculate_supporting_schedules(working_df)
        initial_guess = []
        for var in active_variables:
            initial_values = working_df[var].values
            initial_guess.extend([max(1.0, val) for val in initial_values])

        bounds = []
        for var in active_variables:
            for j in range(forecast_years):
                initial_val = working_df[var].iloc[j]
                if var in cost_per_click_map and working_df[cost_per_click_map[var]].iloc[j] > 0:
                    max_traffic = budget_amount / working_df[cost_per_click_map[var]].iloc[j] * 1.5
                else:
                    max_traffic = initial_val * 10
                bounds.append((1.0, max_traffic))

        def objective(x):
            temp_df = working_df.copy()
            for i, var in enumerate(active_variables):
                start_idx = i * forecast_years
                end_idx = start_idx + forecast_years
                temp_df[var] = [max(1.0, val) for val in x[start_idx:end_idx]]
            try:
                self.calculate_supporting_schedules(temp_df)
                self.calculate_income_statement(temp_df)
                total_ebitda = temp_df['EBITDA'].sum()
                logger.debug(f"Objective EBITDA: {total_ebitda}")
                return -total_ebitda / 1e6 if np.isfinite(total_ebitda) else 1e12
            except Exception as e:
                logger.error(f"Objective function error: {str(e)}")
                return 1e12

        def budget_constraint(x):
            temp_df = working_df.copy()
            for i, var in enumerate(active_variables):
                start_idx = i * forecast_years
                end_idx = start_idx + forecast_years
                temp_df[var] = [max(1.0, val) for val in x[start_idx:end_idx]]
            try:
                self.calculate_supporting_schedules(temp_df)
                total_cost = sum(
                    temp_df[f'{traffic} Cost'].sum()
                    for traffic in active_variables
                    if f'{traffic} Cost' in temp_df.columns
                )
                return budget_amount * forecast_years - total_cost
            except Exception as e:
                logger.error(f"Constraint error: {str(e)}")
                return -1e12

        initial_df = working_df.copy()
        self.calculate_supporting_schedules(initial_df)
        total_cost = sum(
            initial_df[f'{traffic} Cost'].sum()
            for traffic in active_variables
            if f'{traffic} Cost' in initial_df.columns
        )
        if total_cost > budget_amount * forecast_years:
            raise HTTPException(status_code=400, detail=f"Initial cost {total_cost} exceeds budget {budget_amount * forecast_years}")

        constraints = [{'type': 'ineq', 'fun': budget_constraint}]
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': True, 'maxiter': 1000, 'ftol': 1e-6}
        )
        if result.success:
            optimized_values = result.x
            optimized_df = working_df.copy()
            for i, var in enumerate(active_variables):
                start_idx = i * forecast_years
                end_idx = start_idx + forecast_years
                optimized_df[var] = optimized_values[start_idx:end_idx]
            self.calculate_supporting_schedules(optimized_df)
            self.calculate_income_statement(optimized_df)
            optimized_data = []
            for idx, row in optimized_df.iterrows():
                variables_dict = {var: row[var] for var in variables}
                changes_dict = {
                    f"{var} Change (%)": ((row[var] - working_df[var].iloc[idx]) / working_df[var].iloc[idx] * 100)
                    if working_df[var].iloc[idx] != 0 else 0
                    for var in variables
                }
                optimized_data.append({
                    "year": row['Year'],
                    "variables": variables_dict,
                    "changes": changes_dict
                })
            return {
                "optimized_data": optimized_data,
                "original_ebitda": working_df['EBITDA'].sum(),
                "optimized_ebitda": optimized_df['EBITDA'].sum(),
                "ebitda_change_percent": (optimized_df['EBITDA'].sum() - working_df['EBITDA'].sum()) / working_df['EBITDA'].sum() * 100,
                "message": f"Optimization successful for {budget_line} budget: ${budget_amount * forecast_years:,.0f}"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Optimization failed: {result.message}")

    def run_schedule_risk_analysis(self, df, num_simulations, confidence_level):
        """Simulate ScheduleRiskAnalysis: Project Timeline Risk Assessment"""
      
       

        tasks = [
            {"name": "Market Research", "base_duration": 30, "std_dev": 5},
            {"name": "Campaign Design", "base_duration": 45, "std_dev": 10},
            {"name": "Implementation", "base_duration": 60, "std_dev": 15}
        ]
        
        sim_durations = np.zeros(num_simulations)
        for _ in range(num_simulations):
            total_duration = 0
            for task in tasks:
                duration = np.random.normal(task["base_duration"], task["std_dev"])
                total_duration += max(duration, 0)  # Ensure non-negative
            sim_durations[_] = total_duration
        
        # Calculate statistics
        mean_duration = np.mean(sim_durations)
        ci_bounds = np.percentile(sim_durations, [(100 - confidence_level) / 2, 100 - (100 - confidence_level) / 2])
        ci_lower = clean_float(ci_bounds[0])
        ci_upper = clean_float(ci_bounds[1])
        
        response = {
            "simulation_durations": [clean_float(val) for val in sim_durations.tolist()],
            "mean_duration": mean_duration,
            "confidence_interval": {
                "lower": ci_lower,
                "upper": ci_upper,
                "confidence_level": confidence_level
            },
            "tasks": tasks
        }

        
        return response                    

    def create_revenue_chart_data(self,df):
        """Get data for Revenue and Profitability Trends chart."""
        try:
            if df is None or df.empty:
                # raise HTTPException(status_code=400, detail="Dataframe is empty or unavailable")
                print("Dataframe is empty or unavailable")

            required_columns = ['Year', 'Net Revenue', 'Gross Margin', 'EBITDA Margin']
            validate_columns(df, required_columns)

            response = {
                "years": df['Year'].tolist(),
                "net_revenue": [clean_float(val) for val in df['Net Revenue'].tolist()],
                "gross_margin": [clean_float(val * 100) for val in df['Gross Margin'].tolist()],  # Convert to percentage
                "ebitda_margin": [clean_float(val * 100) for val in df['EBITDA Margin'].tolist()]  # Convert to percentage
            }
            return response
        except Exception as e:
            logging.error(f"Error in revenue chart data: {str(e)}")
            # raise HTTPException(status_code=500, detail=f"Error in revenue chart data: {str(e)}")
            print(f"Error in revenue chart data: {str(e)}")
    
    def create_traffic_chart_data(self, df, max_existing_year=None):
        """Get data for Customer Economics: LTV vs CAC chart."""
        try:
            if df is None or df.empty:
                # raise HTTPException(status_code=400, detail="Dataframe is empty or unavailable")
                print("Dataframe is empty or unavailable")

            required_columns = ['Year', 'LTV', 'CAC', 'LTV/CAC Ratio']
            validate_columns(df, required_columns)

            # Filter by max_existing_year if provided
            if max_existing_year:
                df = df[df['Year'] <= max_existing_year].copy()

            response = {
                "years": df['Year'].tolist(),
                "ltv": [clean_float(val) for val in df['LTV'].tolist()],
                "cac": [clean_float(val) for val in df['CAC'].tolist()],
                "ltv_cac_ratio": [clean_float(val) for val in df['LTV/CAC Ratio'].tolist()]
            }
            return response
        except Exception as e:
            logging.error(f"Error in traffic chart data: {str(e)}")
            # raise HTTPException(status_code=500, detail=f"Error in traffic chart data: {str(e)}")
            print(f"Error in traffic chart data: {str(e)}")

    def create_profitability_chart_data(self, df):
        """Get data for Cash Balance Over Time chart."""
        try:
            if df is None or df.empty:
                # raise HTTPException(status_code=400, detail="Dataframe is empty or unavailable")
                print("Dataframe is empty or unavailable")

            required_columns = ['Year', 'Closing Cash Balance']
            validate_columns(df, required_columns)

            response = {
                "years": df['Year'].tolist(),
                "closing_cash_balance": [clean_float(val) for val in df['Closing Cash Balance'].tolist()]
            }
            return response
        except Exception as e:
            logging.error(f"Error in profitability chart data: {str(e)}")
            # raise HTTPException(status_code=500, detail=f"Error in profitability chart data: {str(e)}")
            print(f"Error in profitability chart data: {str(e)}")

    def create_waterfall_chart_data(self, df):

        """Get data for Profit Bridge Analysis chart."""
        try:
            if df is None or df.empty:
                # raise HTTPException(status_code=400, detail="Dataframe is empty or unavailable")
                print("Dataframe is empty or unavailable")

            required_columns = ['Gross Revenue', 'Discounts, Promotions, Markdowns', 'COGS', 'Total Variable Costs', 'Total Fixed Costs', 'Net Income']
            validate_columns(df, required_columns)

            # Use the last row for the waterfall chart
            last_row = df.iloc[-1]
            response = {
                "categories": ["Gross Revenue", "Discounts", "COGS", "Operating Expenses", "Net Income"],
                "values": [
                    clean_float(last_row['Gross Revenue']),
                    clean_float(-last_row['Discounts, Promotions, Markdowns']),
                    clean_float(-last_row['COGS']),
                    clean_float(-(last_row['Total Variable Costs'] + last_row['Total Fixed Costs'])),
                    clean_float(last_row['Net Income'])
                ],
                "measures": ["relative", "relative", "relative", "relative", "total"]
            }
            return response
        except Exception as e:
            logging.error(f"Error in waterfall chart data: {str(e)}")
            # raise HTTPException(status_code=500, detail=f"Error in waterfall chart data: {str(e)}")
            print(f"Error in waterfall chart data: {str(e)}")

    def create_break_even_chart_data(self, df):
        """Get data for Break-Even Analysis chart."""
        try:
            if df is None or df.empty:
                # raise HTTPException(status_code=400, detail="Dataframe is empty or unavailable")
                print("Dataframe is empty or unavailable")

            required_columns = ['Year', 'Break Even Dollars', 'Actual Sales']
            validate_columns(df, required_columns)
            response = {
                "years": df['Year'].tolist(),
                "break_even_dollars": [clean_float(val) for val in df['Break Even Dollars'].tolist()],
                "actual_sales": [clean_float(val) for val in df['Actual Sales'].tolist()]
            }
            return response
        except Exception as e:
            logging.error(f"Error in break-even chart data: {str(e)}")
            # raise HTTPException(status_code=500, detail=f"Error in break-even chart data: {str(e)}")
            print(f"Error in break-even chart data: {str(e)}")

    def create_consideration_chart_data(self, df):
        """Get data for Customer Journey Funnel Metrics chart."""
        try:
            if df is None or df.empty:
                # raise HTTPException(status_code=400, detail="Dataframe is empty or unavailable")
                print("Dataframe is empty or unavailable")

            required_columns = ['Year', 'Weighted Consideration Rate', 'Consideration to Conversion']
            validate_columns(df, required_columns)

            response = {
                "years": df['Year'].tolist(),
                "weighted_consideration_rate": [clean_float(val * 100) for val in df['Weighted Consideration Rate'].tolist()],  # Convert to percentage
                "consideration_to_conversion": [clean_float(val) for val in df['Consideration to Conversion'].tolist()]
            }
            return response
        except Exception as e:
            logging.error(f"Error in consideration chart data: {str(e)}")
            # raise HTTPException(status_code=500, detail=f"Error in consideration chart data: {str(e)}")
            print(f"Error in consideration chart data: {str(e)}")
    
    def create_margin_safety_chart_data(self, df):

        """Get data for Margin of Safety Analysis chart."""
        try:
            if df is None or df.empty:
                # raise HTTPException(status_code=400, detail="Dataframe is empty or unavailable")
                print("Dataframe is empty or unavailable")
            required_columns = ['Year', 'Margin of Safety Dollars', 'Margin of Safety Percentage']
            validate_columns(df, required_columns)
            response = {
                "years": df['Year'].tolist(),
                "margin_safety_dollars": [clean_float(val) for val in df['Margin of Safety Dollars'].tolist()],
                "margin_safety_percentage": [clean_float(val) for val in df['Margin of Safety Percentage'].tolist()]
            }
            return response
        
        except Exception as e:
            logging.error(f"Error in margin safety chart data: {str(e)}")
            # raise HTTPException(status_code=500, detail=f"Error in margin safety chart data: {str(e)}")
            print(f"Error in margin safety chart data: {str(e)}")


    def create_dcf_summary_chart_data(self, valuation_details):
        """Get data for DCF Valuation Bridge chart."""
        try:
            if not valuation_details:
                # raise HTTPException(status_code=400, detail="Valuation details are unavailable")
                print("Valuation details are unavailable")
            required_keys = ['present_value_fcf', 'terminal_value_pv', 'net_debt', 'equity_value']
            missing_keys = [key for key in required_keys if key not in valuation_details]
            if missing_keys:
                # raise HTTPException(status_code=400, detail=f"Missing required valuation keys: {missing_keys}")
                print(f"Missing required valuation keys: {missing_keys}")
            response = {
                "categories": ["Present Value of FCF", "Terminal Value (PV)", "Net Debt", "Equity Value"],
                "values": [
                    clean_float(valuation_details['present_value_fcf']),
                    clean_float(valuation_details['terminal_value_pv']),
                    clean_float(-valuation_details['net_debt']),
                    clean_float(valuation_details['equity_value'])
                ],
                "measures": ["relative", "relative", "relative", "total"]
            }
            return response
        except Exception as e:
            logging.error(f"Error in DCF summary chart data: {str(e)}")
            # raise HTTPException(status_code=500, detail=f"Error in DCF summary chart data: {str(e)}")
            print(f"Error in DCF summary chart data: {str(e)}")

    def create_cash_flow_forecast_chart(self, df):
        """Get data for Cash Flow Forecast chart."""
        try:
            if df is None or df.empty:
                # raise HTTPException(status_code=400, detail="Dataframe is empty or unavailable")
                print("Dataframe is empty or unavailable")
            required_columns = ['Year', 'Cash from Operations', 'Cash from Investing', 'Net Cash Flow']
            validate_columns(df, required_columns)
            response = {
                "years": df['Year'].tolist(),
                "cash_from_operations": [clean_float(val) for val in df['Cash from Operations'].tolist()],
                "cash_from_investing": [clean_float(val) for val in df['Cash from Investing'].tolist()],
                "net_cash_flow": [clean_float(val) for val in df['Net Cash Flow'].tolist()]
            }
            return response
        except Exception as e:
            logging.error(f"Error in cash flow forecast chart data: {str(e)}")
            # raise HTTPException(status_code=500, detail=f"Error in cash flow forecast chart data: {str(e)}")
            print(f"Error in cash flow forecast chart data: {str(e)}")

    def create_profitability_margin_trends_chart(self, df):
        """Get data for Profitability Margin Trends chart."""
        try:
            if df is None or df.empty:
                # raise HTTPException(status_code=400, detail="Dataframe is empty or unavailable")
                print("Dataframe is empty or unavailable")

            required_columns = ['Year', 'Gross Margin', 'EBITDA Margin', 'Net Income', 'Net Revenue']
            validate_columns(df, required_columns)
            # Calculate Net Profit Margin
            net_profit_margin = (df['Net Income'] / df['Net Revenue']).fillna(0) * 100

            response = {
                "years": df['Year'].tolist(),
                "gross_margin": [clean_float(val * 100) for val in df['Gross Margin'].tolist()],  # Convert to percentage
                "ebitda_margin": [clean_float(val * 100) for val in df['EBITDA Margin'].tolist()],  # Convert to percentage
                "net_profit_margin": [clean_float(val) for val in net_profit_margin.tolist()]
            }
            return response
        except Exception as e:
            logging.error(f"Error in profitability margin trends chart data: {str(e)}")
            # raise HTTPException(status_code=500, detail=f"Error in profitability margin trends chart data: {str(e)}")
            print(f"Error in profitability margin trends chart data: {str(e)}")

    def combine_exports_to_excel(self, df, scenarios,valuation): #(self, df, scenarios, charts):
        """Combine detailed financial statements and full analysis with charts into one Excel file"""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter', engine_kwargs={'options': {'nan_inf_to_errors': True}}) as writer:
            workbook = writer.book
            # 1. Export Detailed Financial Statements (unchanged structure, added NaN/INF handling)
            header_format = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1})
            subheader_format = workbook.add_format({'bold': True})
            number_format = workbook.add_format({'num_format': '$#,##0'})
            percent_format = workbook.add_format({'num_format': '0.0%'})
            row_position = 0            
            # Clean Contribution Margin to avoid NaN/INF downstream
            df['Contribution Margin'] = df['Contribution Margin'].replace([np.inf, -np.inf], np.nan).fillna(0) / 1000
            
            statements = {
                'Income Statement': [
                    'Gross Revenue', 'Discounts, Promotions, Markdowns', 'Net Revenue',
                    'COGS', 'Gross Profit', 'Gross Margin', 'Marketing Expenses',
                    'Fulfilment Expenses', 'Total Variable Costs', 'Contribution Margin',
                    'Contribution Margin Percentage', 'Fixed Fulfilment Expenses',
                    'Fixed General & Administrative Expenses', 'Total Fixed Costs',
                    'EBITDA', 'EBITDA Margin', 'Depreciation', 'Interest',
                    'Earnings Before Tax', 'Taxes', 'Net Income'
                ],
                'Cash Flow': [
                    'Net Income', 'Deprecation', 'Change in Accounts Receivable',
                    'Change in Inventory', 'Change in Accounts Payable',
                    'Cash from Operations', 'Cash from Investing',
                    'Equity Raised', 'Debt Issued', 'Dividends Paid', 'Cash from Financing',
                    'Net Cash Flow', 'Opening Cash Balance', 'Closing Cash Balance'
                ],
                'Balance Sheet': [
                    'Cash', 'Accounts Receivable', 'Inventory', 'Total Current Assets',
                    'Technology Assets', 'Office Equipment Assets', 'Total Fixed Assets',
                    'Total Assets', 'Accounts Payable', 'Total Current Liabilities',
                    'Long Term Debt', 'Total Liabilities', 'Share Capital',
                    'Retained Earnings', 'Total Equity', 'Total Liabilities & Shareholder Equity',
                    'Balance Sheet Check'
                ],
                'Capital Assets': [
                    'Opening Balance Technology', 'Opening Balance Office Equipment',
                    'Total Opening Balance', 'Additions Technology', 'Additions Office Equipment',
                    'Total Additions', 'Subtotal Technology', 'Subtotal Office Equipment',
                    'Total', 'Depreciation Technology', 'Depreciation Office Equipment',
                    'Total Depreciation', 'Closing Balance Technology',
                    'Closing Balance Office Equipment', 'Total Closing Balance'
                ],
                'Valuation': ['EBIT', 'Unlevered FCF', 'PV of FCF', 'Total Enterprise Value'],
                'Customer Metrics': [
                    'CAC', 'Contribution Margin Per Order', 'LTV', 'LTV/CAC Ratio',
                    'Payback Orders', 'NPV'
                ]
            }            
            for sheet_name, cols in statements.items():
                available_cols = [col for col in cols if col in df.columns]
                if available_cols:
                    # Clean data to prevent NaN/INF issues
                    cleaned_df = df[available_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
                    transposed_df = cleaned_df.set_index(df['Year']).T
                    transposed_df.to_excel(writer, sheet_name=sheet_name)
                    worksheet = writer.sheets[sheet_name]
                    for col_num, value in enumerate(transposed_df.columns):
                        worksheet.write(0, col_num + 1, value, header_format)
                    for row_num, value in enumerate(transposed_df.index):
                        worksheet.write(row_num + 1, 0, value, header_format)
                    
                    for row in range(1, len(transposed_df) + 1):
                        for col in range(len(transposed_df.columns)):
                            value = transposed_df.iloc[row - 1, col]
                            if pd.isna(value) or np.isinf(value):  # This check is now redundant with nan_inf_to_errors, but kept for clarity
                                worksheet.write(row, col + 1, "N/A", number_format)
                            elif any(x in transposed_df.index[row - 1] for x in ['Margin', 'Percentage']):
                                worksheet.write(row, col + 1, value, percent_format)
                            else:
                                worksheet.write(row, col + 1, value, number_format)
            # 2. Export Scenarios with Structured Layout
            for name, data in scenarios.items():
                worksheet = workbook.add_worksheet(f"Scenario_{name}")
                row_position = 0                
                # Define formatting
                section_header_format = workbook.add_format({'bold': True, 'bg_color': '#4B5EAA', 'font_color': 'white', 'border': 1})
                subheader_format = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1})                
                # Clean scenario data to avoid NaN/INF
                cleaned_data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
                                
                # Section 1: Revenue Schedule
                worksheet.write(row_position, 0, "Section 1:", section_header_format)
                row_position += 1
                revenue_cols = [
                    'Year', 'Email Traffic', 'Organic Search Traffic', 'Paid Search Traffic', 
                    'Affiliates Traffic', 'Email Conversion Rate', 'Organic Search Conversion Rate', 
                    'Paid Search Conversion Rate', 'Affiliates Conversion Rate', 'Average Item Value', 
                    'Number of Items per Order', 'Average Markdown', 'Average Promotion/Discount', 
                    'COGS Percentage', 'Churn Rate', 'Email Cost per Click', 'Organic Search Cost per Click', 
                    'Paid Search Cost per Click', 'Affiliates Cost per Click'
                ]
                available_revenue_cols = [col for col in revenue_cols if col in cleaned_data.columns]
                if not available_revenue_cols:
                    worksheet.write(row_position, 0, "No revenue data available", subheader_format)
                    row_position += 2
                else:
                    revenue_df = cleaned_data[available_revenue_cols].set_index('Year').T
                    for col_num, year in enumerate(revenue_df.columns):
                        worksheet.write(row_position, col_num + 1, year, subheader_format)
                    for row_num, metric in enumerate(revenue_df.index):
                        worksheet.write(row_position + row_num + 1, 0, metric, subheader_format)
                        for col_num, value in enumerate(revenue_df.loc[metric]):
                            worksheet.write(row_position + row_num + 1, col_num + 1, value if not pd.isna(value) else "N/A", number_format)
                    row_position += len(revenue_df) + 2

                # Section 2: Cost Schedule
                worksheet.write(row_position, 0, "Section 2:", section_header_format)
                row_position += 1
                cost_cols = ['Year', 'Freight/Shipping per Order', 'Labor/Handling per Order', 'General Warehouse Rent', 'Office Rent']
                cost_df = cleaned_data[cost_cols].set_index('Year').T
                for col_num, year in enumerate(cost_df.columns):
                    worksheet.write(row_position, col_num + 1, year, subheader_format)
                for row_num, metric in enumerate(cost_df.index):
                    worksheet.write(row_position + row_num + 1, 0, metric, subheader_format)
                    for col_num, value in enumerate(cost_df.loc[metric]):
                        worksheet.write(row_position + row_num + 1, col_num + 1, value if not pd.isna(value) else "N/A", number_format)
                row_position += len(cost_df) + 2

                # Section 3: Depreciation Schedule
                worksheet.write(row_position, 0, "Section 3:", section_header_format)
                row_position += 1
                dep_cols = ['Year', 'Salaries, Wages & Benefits']
                dep_df = cleaned_data[dep_cols].set_index('Year').T
                for col_num, year in enumerate(dep_df.columns):
                    worksheet.write(row_position, col_num + 1, year, subheader_format)
                for row_num, metric in enumerate(dep_df.index):
                    worksheet.write(row_position + row_num + 1, 0, metric, subheader_format)
                    for col_num, value in enumerate(dep_df.loc[metric]):
                        worksheet.write(row_position + row_num + 1, col_num + 1, value if not pd.isna(value) else "N/A", number_format)
                row_position += len(dep_df) + 2

                # Section 4: Working Capital Schedule
                worksheet.write(row_position, 0, "Section 4:", section_header_format)
                row_position += 1
                wc_cols = ['Year', 'Professional Fees', 'Other']
                wc_df = cleaned_data[wc_cols].set_index('Year').T
                for col_num, year in enumerate(wc_df.columns):
                    worksheet.write(row_position, col_num + 1, year, subheader_format)
                for row_num, metric in enumerate(wc_df.index):
                    worksheet.write(row_position + row_num + 1, 0, metric, subheader_format)
                    for col_num, value in enumerate(wc_df.loc[metric]):
                        worksheet.write(row_position + row_num + 1, col_num + 1, value if not pd.isna(value) else "N/A", number_format)
                row_position += len(wc_df) + 2

                # Section 5: Tax Schedule
                worksheet.write(row_position, 0, "Section 5:", section_header_format)
                row_position += 1
                tax_cols = ['Year', 'Depreciation']
                tax_df = cleaned_data[tax_cols].set_index('Year').T
                for col_num, year in enumerate(tax_df.columns):
                    worksheet.write(row_position, col_num + 1, year, subheader_format)
                for row_num, metric in enumerate(tax_df.index):
                    worksheet.write(row_position + row_num + 1, 0, metric, subheader_format)
                    for col_num, value in enumerate(tax_df.loc[metric]):
                        worksheet.write(row_position + row_num + 1, col_num + 1, value if not pd.isna(value) else "N/A", number_format)
                row_position += len(tax_df) + 2

                # Section 6: Debt Schedule
                worksheet.write(row_position, 0, "Section 6:", section_header_format)
                row_position += 1
                debt_cols = [
                    'Year', 'Interest on Debt', 'Cash', 'Technology Assets', 'Office Equipment Assets', 
                    'Total Current Assets', 'Total Fixed Assets', 'Total Assets', 'Total Current Liabilities', 
                    'Long Term Debt', 'Total Liabilities', 'Share Capital', 'Retained Earnings', 
                    'Total Equity', 'Total Liabilities & Shareholder Equity', 'Balance Sheet Check'
                ]
                debt_df = cleaned_data[debt_cols].set_index('Year').T
                for col_num, year in enumerate(debt_df.columns):
                    worksheet.write(row_position, col_num + 1, year, subheader_format)
                for row_num, metric in enumerate(debt_df.index):
                    worksheet.write(row_position + row_num + 1, 0, metric, subheader_format)
                    for col_num, value in enumerate(debt_df.loc[metric]):
                        worksheet.write(row_position + row_num + 1, col_num + 1, value if not pd.isna(value) else "N/A", number_format)
                row_position += len(debt_df) + 2

                # Section 7: Customer Metrics and Valuation
                worksheet.write(row_position, 0, "Section 7:", section_header_format)
                row_position += 1
                customer_cols = [
                    'Year', 'CAC', 'Contribution Margin Per Order', 'LTV', 'LTV/CAC Ratio', 
                    'Payback Orders', 'NPV', 'IRR', 'Burn Rate', 'EBIT', 'Unlevered FCF', 
                    'PV of FCF', 'Total Enterprise Value'
                ]
                customer_df = cleaned_data[customer_cols].set_index('Year').T
                for col_num, year in enumerate(customer_df.columns):
                    worksheet.write(row_position, col_num + 1, year, subheader_format)
                for row_num, metric in enumerate(customer_df.index):
                    worksheet.write(row_position + row_num + 1, 0, metric, subheader_format)
                    for col_num, value in enumerate(customer_df.loc[metric]):
                        format_to_use = percent_format if metric in ['IRR'] else number_format
                        worksheet.write(row_position + row_num + 1, col_num + 1, value if not pd.isna(value) else "N/A", format_to_use)
                row_position += len(customer_df) + 2

                # Section 8: Compounded Annual Growth Rate (CAGR)
                worksheet.write(row_position, 0, "Section 8: Compound Annual Growth Rate", section_header_format)
                row_position += 1
                if len(cleaned_data) >= 2:
                    cagr_net_revenue = self.calculate_cagr(cleaned_data, 'Net Revenue')
                    cagr_net_income = self.calculate_cagr(cleaned_data, 'Net Income')
                    cagr_df = pd.DataFrame({
                        'Metric': ['CAGR Revenue', 'CAGR Net Income'],
                        'Value': [cagr_net_revenue, cagr_net_income]
                    })
                    # Handle NaN/INF explicitly
                    cagr_df['Value'] = cagr_df['Value'].replace([np.inf, -np.inf, np.nan], 0)
                    for row_num, row in cagr_df.iterrows():
                        worksheet.write(row_position + row_num, 0, row['Metric'], subheader_format)
                        worksheet.write(row_position + row_num, 1, row['Value'], percent_format)
                else:
                    worksheet.write(row_position, 0, "Insufficient data for CAGR", subheader_format)
                row_position += 3

                # Section 9: Adjusted EBITDA
                worksheet.write(row_position, 0, "Section 9: Adjusted EBITDA", section_header_format)
                row_position += 1
                adjusted_ebitda_cols = ['Year', 'EBITDA']
                adjusted_ebitda_df = cleaned_data[adjusted_ebitda_cols].set_index('Year').T
                for col_num, year in enumerate(adjusted_ebitda_df.columns):
                    worksheet.write(row_position, col_num + 1, year, subheader_format)
                for row_num, metric in enumerate(adjusted_ebitda_df.index):
                    worksheet.write(row_position + row_num + 1, 0, metric, subheader_format)
                    for col_num, value in enumerate(adjusted_ebitda_df.loc[metric]):
                        worksheet.write(row_position + row_num + 1, col_num + 1, value if not pd.isna(value) else "N/A", number_format)
                row_position += len(adjusted_ebitda_df) + 2

                # Section 10: Capital Intensity Ratio
                worksheet.write(row_position, 0, "Section 10: Capital Intensity Ratio", section_header_format)
                row_position += 1
                cleaned_data['Capital Intensity Ratio'] = cleaned_data['Total Assets'] / cleaned_data['Net Revenue'].replace(0, np.nan)  # Avoid division by zero
                cleaned_data['Capital Intensity Ratio'] = cleaned_data['Capital Intensity Ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
                cap_intensity_cols = ['Year', 'Capital Intensity Ratio']
                cap_intensity_df = cleaned_data[cap_intensity_cols].set_index('Year').T
                for col_num, year in enumerate(cap_intensity_df.columns):
                    worksheet.write(row_position, col_num + 1, year, subheader_format)
                for row_num, metric in enumerate(cap_intensity_df.index):
                    worksheet.write(row_position + row_num + 1, 0, metric, subheader_format)
                    for col_num, value in enumerate(cap_intensity_df.loc[metric]):
                        worksheet.write(row_position + row_num + 1, col_num + 1, value if not pd.isna(value) else "N/A", number_format)

                chart_specs = [
                        ("Revenue & Profitability",       self.create_revenue_chart_data,       df),
                        ("LTV vs CAC",                    self.create_traffic_chart_data,       df),
                        ("Cash Balance Over Time",        self.create_profitability_chart_data, df),
                        ("Profit Bridge",                 self.create_waterfall_chart_data,     df),
                        ("Break-Even Analysis",           self.create_break_even_chart_data,    df),
                        ("Customer Funnel",               self.create_consideration_chart_data, df),
                        ("Margin of Safety",              self.create_margin_safety_chart_data, df),
                        ("DCF Valuation Bridge",          self.create_dcf_summary_chart_data,valuation),
                        ("Cash Flow Forecast",            self.create_cash_flow_forecast_chart, df),
                        ("Profitability Margin Trends",   self.create_profitability_margin_trends_chart, df),
                    ]
                try:
                    charts_ws = workbook.add_worksheet("Charts")
                    # register in writer.sheets so future lookups work
                    writer.sheets["Charts"] = charts_ws
                except DuplicateWorksheetName:
                    # already there, just reuse it
                    charts_ws = writer.sheets["Charts"]
                img_row = 0
                for title, data_fn, source in chart_specs:
                    # 1) pull the raw chart data
                    raw = data_fn(source)
                    if "years" in raw:
                        x = raw["years"]
                    elif "categories" in raw:
                        x = raw["categories"]
                    else:
                        # nothing to plot
                        continue

                    # 2) build a Plotly figure based on its keys
                    fig = go.Figure()
                    
                    if "net_revenue" in raw:
                        fig.add_trace(go.Bar(x=x, y=raw["net_revenue"],    name="Net Revenue"))
                        fig.add_trace(go.Scatter(x=x, y=np.array(raw["gross_margin"])/100,
                                                name="Gross Margin", yaxis="y2", mode="lines+markers"))
                        fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%"))
                    elif "cac" in raw:
                        fig.add_trace(go.Bar(x=x, y=raw["cac"], name="CAC"))
                        fig.add_trace(go.Scatter(x=x, y=raw["ltv"], name="LTV"))
                        fig.add_trace(go.Scatter(x=x, y=raw["ltv_cac_ratio"], name="LTV/CAC"))
                    elif "closing_cash_balance" in raw:
                        fig.add_trace(go.Scatter(x=x, y=raw["closing_cash_balance"], name="Closing Cash", fill="tozeroy"))
                    elif "categories" in raw and "values" in raw and raw.get("measures"):
                        fig = go.Figure(go.Waterfall(
                            name=title,
                            x=raw["categories"],
                            y=raw["values"],
                            measure=raw["measures"],
                            textposition="outside",
                            texttemplate="%{y:$,}"
                        ))
                    elif "break_even_dollars" in raw:
                        fig.add_trace(go.Scatter(x=x, y=raw["break_even_dollars"], name="Break-Even"))
                        fig.add_trace(go.Scatter(x=x, y=raw["actual_sales"],     name="Actual Sales"))
                    elif "weighted_consideration_rate" in raw:
                        fig.add_trace(go.Scatter(x=x, y=np.array(raw["weighted_consideration_rate"])/100,
                                                name="Consideration", yaxis="y"))
                        fig.add_trace(go.Scatter(x=x, y=raw["consideration_to_conversion"],
                                                name="Conversion", yaxis="y2"))
                        fig.update_layout(yaxis2=dict(overlaying="y", side="right"))
                    
                    elif "cash_from_operations" in raw:
                        fig.add_trace(go.Bar(x=x, y=raw["cash_from_operations"], name="Operating CF"))
                        fig.add_trace(go.Bar(x=x, y=raw["cash_from_investing"],   name="Investing CF"))
                        fig.add_trace(go.Scatter(x=x, y=raw["net_cash_flow"],    name="Net CF", mode="lines+markers"))
                        fig.update_layout(barmode="stack")
                    elif "gross_margin" in raw:
                        fig.add_trace(go.Scatter(x=x, y=np.array(raw["gross_margin"])/100,    name="Gross Margin"))
                        fig.add_trace(go.Scatter(x=x, y=np.array(raw["ebitda_margin"])/100,   name="EBITDA Margin"))
                        fig.add_trace(go.Scatter(x=x, y=np.array(raw["net_profit_margin"])/100, name="Net Profit Margin"))

                    fig.update_layout(title=title, margin=dict(l=30, r=30, t=30, b=30), 
                                    xaxis_title="Year", legend=dict(orientation="h", y=-0.2))

                    # 3) render to PNG
                    img_data = pio.to_image(fig, format="png", width=700, height=400)
                    img_bytes = BytesIO(img_data)

                    # 4) embed at the next free row
                    charts_ws.insert_image(img_row, 0, f"{title}.png", {'image_data': img_bytes})
                    img_row += 22  # adjust spacing between charts

        output.seek(0)
        return output.getvalue()

    def run_monte_carlo_simulation(self, df, forecast_years, num_simulations, confidence_level, 
                              discount_rate, wacc, perpetual_growth, distribution_type="Normal"):
        """Run Monte Carlo simulation with selectable distribution types, including EBITDA"""
        try:
            # Key variables to simulate, including valuation parameters
            variables = ['Net Revenue', 'COGS', 'Total Variable Costs', 'Total Fixed Costs', 'Interest',
                        'discount_rate', 'wacc', 'perpetual_growth']
            sim_results = {var: np.zeros((num_simulations, len(df) + forecast_years)) for var in variables[:5]}
            sim_results['Net Income'] = np.zeros((num_simulations, len(df) + forecast_years))
            sim_results['Unlevered FCF'] = np.zeros((num_simulations, len(df) + forecast_years))
            sim_results['EBITDA'] = np.zeros((num_simulations, len(df) + forecast_years))  # Add EBITDA
            sim_results['NPV'] = np.zeros(num_simulations)
            sim_valuation = {
                'discount_rate': np.zeros((num_simulations, len(df) + forecast_years)),
                'wacc': np.zeros((num_simulations, len(df) + forecast_years)),
                'perpetual_growth': np.zeros((num_simulations, len(df) + forecast_years))
            }
            
            # Historical data
            historical_years = len(df)
            forecast_start = historical_years
            
            # Base values from historical data or input parameters
            base_values = {var: df[var].values for var in variables[:5] if var in df.columns}
            base_values['discount_rate'] = np.full(historical_years, discount_rate)
            base_values['wacc'] = np.full(historical_years, wacc)
            base_values['perpetual_growth'] = np.full(historical_years, perpetual_growth)
            
            # Extend the time frame
            total_years = historical_years + forecast_years
            years = np.arange(df['Year'].min(), df['Year'].max() + forecast_years + 1)
            
            # Volatility estimation
            volatility = {var: np.std(df[var].pct_change().dropna()) if len(df[var]) > 1 else 0.2 
                        for var in variables[:5] if var in df.columns}
            volatility['discount_rate'] = 0.1
            volatility['wacc'] = 0.1
            volatility['perpetual_growth'] = 0.05
            
            # Distribution sampling functions
            dist_map = {
                "Normal": lambda mean, std: norm.rvs(loc=mean, scale=max(std, 1e-6)),
                "Lognormal": lambda mean, std: lognorm.rvs(s=max(std, 1e-6), scale=max(np.exp(mean), 1e-6)),
                "Uniform": lambda mean, std: uniform.rvs(loc=mean - max(std, 1e-6) * np.sqrt(3), 
                                                    scale=2 * max(std, 1e-6) * np.sqrt(3)),
                "Exponential": lambda mean, _: expon.rvs(scale=max(mean, 1e-6)),
                "Binomial": lambda mean, std: binom.rvs(n=max(int(mean / max(std**2, 1e-6)), 1), 
                                                    p=min(max(std**2 / mean, 0.01), 0.99)),
                "Poisson": lambda mean, _: poisson.rvs(mu=max(mean, 1e-6)),
                "Geometric": lambda mean, _: geom.rvs(p=min(max(1 / mean, 0.01), 0.99)),
                "Bernoulli": lambda mean, _: bernoulli.rvs(p=min(max(mean, 0.01), 0.99)),
                "Chi-square": lambda mean, _: chi2.rvs(df=max(mean, 1e-6)),
                "Gamma": lambda mean, std: gamma.rvs(a=max(mean**2 / max(std**2, 1e-6), 1e-6), 
                                                scale=max(std**2 / mean, 1e-6)),
                "Weibull": lambda mean, std: weibull_min.rvs(c=max(mean / max(std, 1e-6), 1e-6), 
                                                        scale=max(mean / gamma(1 + 1 / max(mean / max(std, 1e-6), 1e-6)), 1e-6)),
                "Hypergeometric": lambda mean, std: hypergeom.rvs(M=max(int(mean * 10), 1), 
                                                                n=max(int(mean), 1), 
                                                                N=max(int(mean * 2), 1)),
                "Multinomial": lambda mean, _: multinomial.rvs(n=max(int(mean), 1), p=[0.25, 0.25, 0.25, 0.25])[:, 0],
                # "T-distribution": lambda mean, std: t.rvs(df=max(int(mean / max(std, 1e-6)), 1), 
                #                                         loc=mean, scale=max(std, 1e-6)),
                "Beta": lambda mean, std: beta.rvs(a=max(mean * (mean * (1 - mean) / max(std**2, 1e-6) - 1), 1e-6), 
                                                b=max((1 - mean) * (mean * (1 - mean) / max(std**2, 1e-6) - 1), 1e-6)) * max(mean / std, 1e-6),
                "F-distribution": lambda mean, _: f.rvs(dfn=max(int(mean), 1), dfd=max(int(mean), 1), 
                                                    loc=mean),
                "Discrete": lambda mean, std: np.random.choice([max(mean - std, 0), mean, mean + std], 
                                                            p=[0.25, 0.5, 0.25]),
                "Continuous": lambda mean, std: norm.rvs(loc=mean, scale=max(std, 1e-6)),
                "Cumulative": lambda mean, std: np.cumsum(norm.rvs(loc=mean / total_years, 
                                                                scale=max(std, 1e-6), 
                                                                size=total_years))[-1]
            }
            
            dist_func = dist_map.get(distribution_type, dist_map["Normal"])
            
            for sim in range(num_simulations):
                # Historical period
                for var in variables:
                    if var in base_values:
                        if var in sim_results:
                            sim_results[var][sim, :historical_years] = base_values[var]
                        else:
                            sim_valuation[var][sim, :historical_years] = base_values[var]
                
                # Historical EBITDA (if data available)
                if all(var in df.columns for var in ['Net Revenue', 'COGS', 'Total Variable Costs', 'Total Fixed Costs']):
                    sim_results['EBITDA'][sim, :historical_years] = (
                        df['Net Revenue'] - df['COGS'] - df['Total Variable Costs'] - df['Total Fixed Costs']
                    )
                
                # Forecast period
                for t in range(forecast_start, total_years):
                    for var in variables:
                        prev_value = (sim_results[var][sim, t-1] if var in sim_results else 
                                    sim_valuation[var][sim, t-1])
                        mean = prev_value * (1 + (0.05 if var not in ['discount_rate', 'wacc', 'perpetual_growth'] else 0.01))
                        std = volatility.get(var, 0.2) * mean
                        if std <= 0:
                            std = mean * 0.1
                        if mean <= 0:
                            mean = 1e-6
                        
                        try:
                            sampled_value = dist_func(mean, std)
                            if isinstance(sampled_value, (list, np.ndarray)):
                                sampled_value = sampled_value[0]
                            if var == 'discount_rate' or var == 'wacc':
                                sampled_value = max(0.01, sampled_value)
                            elif var == 'perpetual_growth':
                                sampled_value = np.clip(sampled_value, -0.05, 0.05)
                            else:
                                sampled_value = max(0, sampled_value)
                            
                            if var in sim_results:
                                sim_results[var][sim, t] = sampled_value
                            else:
                                sim_valuation[var][sim, t] = sampled_value
                        except ValueError as e:
                          
                            if var in sim_results:
                                sim_results[var][sim, t] = prev_value
                            else:
                                sim_valuation[var][sim, t] = prev_value
                    
                    # Calculate EBITDA
                    sim_results['EBITDA'][sim, t] = (
                        sim_results['Net Revenue'][sim, t] -
                        sim_results['COGS'][sim, t] -
                        sim_results['Total Variable Costs'][sim, t] -
                        sim_results['Total Fixed Costs'][sim, t]
                    )
                    
                    # Calculate Net Income
                    sim_results['Net Income'][sim, t] = (
                        sim_results['EBITDA'][sim, t] - sim_results['Interest'][sim, t]
                    ) * (1 - df['Tax Rate'].mean())
                    
                    # Simplified Unlevered FCF
                    sim_results['Unlevered FCF'][sim, t] = (
                        sim_results['Net Income'][sim, t] +
                        df['Depreciation'].mean() - df['Capital Expenditures'].mean()
                    )
                
                # Calculate NPV
                cash_flows = sim_results['Unlevered FCF'][sim, forecast_start:]
                discount_rates = sim_valuation['discount_rate'][sim, forecast_start:]
                terminal_value = (sim_results['Unlevered FCF'][sim, -1] * (1 + sim_valuation['perpetual_growth'][sim, -1]) /
                                (sim_valuation['wacc'][sim, -1] - sim_valuation['perpetual_growth'][sim, -1])
                                if sim_valuation['wacc'][sim, -1] > sim_valuation['perpetual_growth'][sim, -1] else 0)
                discounted_cf = [cf / (1 + dr)**(i+1) for i, (cf, dr) in enumerate(zip(cash_flows, discount_rates))]
                sim_results['NPV'][sim] = sum(discounted_cf) + (terminal_value / (1 + sim_valuation['wacc'][sim, -1])**forecast_years)
            
            # Confidence bounds
            confidence_bounds = [(100 - confidence_level) / 2, 100 - (100 - confidence_level) / 2]
            
            sim_results.update(sim_valuation)
            return sim_results, confidence_bounds
        
        except Exception as e:
            print(f"Monte Carlo Simulation Error: {str(e)}")
            return None, None

    def apply_what_if_adjustments(self, forecast_df, adjustments):
        """Apply What-If adjustments with dynamic e-commerce-specific logic."""
        
        adjusted_df = forecast_df.copy()
        
        
        # Calculate historical correlations to determine secondary effect multipliers
        if 'Total Orders' in adjusted_df.columns and 'Freight/Shipping per Order' in adjusted_df.columns:
            freight_correlation = adjusted_df['Total Orders'].corr(adjusted_df['Freight/Shipping per Order'])
            labor_correlation = adjusted_df['Total Orders'].corr(adjusted_df['Labor/Handling per Order'])
            marketing_correlation = adjusted_df['Total Orders'].corr(adjusted_df['Marketing Expenses'])
        else:
            freight_correlation = labor_correlation = marketing_correlation = 0.8  # Fallback
        
        if 'Average Item Value' in adjusted_df.columns and 'COGS Percentage' in adjusted_df.columns:
            cogs_correlation = adjusted_df['Average Item Value'].corr(adjusted_df['COGS Percentage'])
        else:
            cogs_correlation = -0.5  # Fallback (negative correlation)
        
        if 'Paid Search Traffic' in adjusted_df.columns:
            conv_correlation = adjusted_df['Paid Search Traffic'].corr(adjusted_df['Paid Search Conversion Rate']) if 'Paid Search Conversion Rate' in adjusted_df.columns else 0.2
            cost_correlation = adjusted_df['Paid Search Traffic'].corr(adjusted_df['Paid Search Cost per Click']) if 'Paid Search Cost per Click' in adjusted_df.columns else 0.7
        else:
            conv_correlation = 0.2
            cost_correlation = 0.7
        
        if 'Marketing Expenses' in adjusted_df.columns and 'Paid Search Traffic' in adjusted_df.columns:
            traffic_correlation = adjusted_df['Marketing Expenses'].corr(adjusted_df['Paid Search Traffic'])
        else:
            traffic_correlation = 0.3
        
        for adj in adjustments:
            year, variable, multiplier = adj['year'], adj['variable'], adj['multiplier']
            if year in adjusted_df['Year'].values:
                # Apply primary adjustment
                adjusted_df.loc[adjusted_df['Year'] == year, variable] *= multiplier
                
                # Apply secondary adjustments with dynamic correlations
                if variable == 'Total Orders':
                    adjusted_df.loc[adjusted_df['Year'] == year, 'Freight/Shipping per Order'] *= (multiplier ** freight_correlation)
                    adjusted_df.loc[adjusted_df['Year'] == year, 'Labor/Handling per Order'] *= (multiplier ** labor_correlation)
                    adjusted_df.loc[adjusted_df['Year'] == year, 'Marketing Expenses'] *= (multiplier ** marketing_correlation)
                elif variable == 'Average Item Value':
                    adjusted_df.loc[adjusted_df['Year'] == year, 'COGS Percentage'] *= (multiplier ** cogs_correlation)
                elif variable == 'Paid Search Traffic':
                    adjusted_df.loc[adjusted_df['Year'] == year, 'Paid Search Conversion Rate'] *= (multiplier ** conv_correlation)
                    adjusted_df.loc[adjusted_df['Year'] == year, 'Paid Search Cost per Click'] *= (multiplier ** cost_correlation)
                elif variable == 'Email Conversion Rate':
                    adjusted_df.loc[adjusted_df['Year'] == year, 'Email Orders'] *= multiplier
                    # Validate conversion rate
                    adjusted_df.loc[adjusted_df['Year'] == year, 'Email Conversion Rate'] = adjusted_df.loc[adjusted_df['Year'] == year, 'Email Conversion Rate'].clip(upper=1.0)
                elif variable == 'COGS Percentage':
                    pass
                elif variable == 'Labor/Handling per Order' or variable == 'Freight/Shipping per Order':
                    adjusted_df.loc[adjusted_df['Year'] == year, 'Fulfilment Expenses'] *= multiplier
                elif variable == 'Marketing Expenses':
                    adjusted_df.loc[adjusted_df['Year'] == year, 'Paid Search Traffic'] *= (multiplier ** traffic_correlation)
                elif variable == 'Interest Rate':
                    adjusted_df.loc[adjusted_df['Year'] == year, 'Interest'] *= multiplier

        # # Ensure all values are non-negative and realistic
        # try:
        #     for col in adjusted_df.columns:
        #         print("col ",col)
        #         adjusted_df[col] = adjusted_df[col].clip(lower=0)
        #         if 'Conversion Rate' in col:
        #             print("conversion rate",col)
        #             adjusted_df[col] = adjusted_df[col].clip(upper=1.0)  # Conversion rates cannot exceed 100%
        # except Exception as e:
        #     print("Exception in what if ",e)

        # Ensure all values are non-negative and realistic, but only for numeric columns
        try:
            for col in adjusted_df.columns:               
                if pd.api.types.is_numeric_dtype(adjusted_df[col]):  # Only apply clip to numeric columns
                    adjusted_df[col] = adjusted_df[col].clip(lower=0)
                    if 'Conversion Rate' in col:                        
                        adjusted_df[col] = adjusted_df[col].clip(upper=1.0)  # Conversion rates cannot exceed 100%
        except Exception as e:
            print("Exception in what if ", e)
            raise ValueError(f"Error during clipping operation of what if : {str(e)}")
      

        return adjusted_df


    def run_goal_seek(
        self,
        df: pd.DataFrame,
        session_years_data: Dict,
        target_profit_margin: float,
        variable_to_adjust: str,
        year_to_adjust: int,
        max_iterations: int = 100,
        tolerance: float = 0.001
    ) -> (float, pd.DataFrame):
        """
        Perform goal seek to find the required value of a variable to achieve a target profit margin.

        Returns:
        - multiplier: the factor we apply to the chosen variable in the goal year
        - adjusted_df: the full “what‐if” DataFrame (historical + forecast) after re‐calculation
        """

        # 1) Work with a fresh copy so we don’t overwrite session_state
        working_df = df.copy()

        # 2) Build a DataFrame of historical years from session_years_data
        historical_data_df = pd.DataFrame.from_dict(session_years_data, orient='index')
        if 'Year' not in historical_data_df.columns:
            print("Error: 'Year' column not found in historical data from session_state.")
            return None, None

        max_historical_year = historical_data_df['Year'].max()
        # 3) Slice working_df into historical_df and full forecast_df (for all years ≥ max_historical_year)
        historical_df = working_df[working_df['Year'] <= max_historical_year].copy()
        full_forecast_df = working_df[working_df['Year'] > max_historical_year].copy()

        # 4) RESET INDEX on both pieces immediately, so there are no duplicate labels
        historical_df = historical_df.reset_index(drop=True)
        full_forecast_df = full_forecast_df.reset_index(drop=True)

        # 5) Ensure the “goal” year exists in the historical+forecast combined set?
        combined = pd.concat([historical_df, full_forecast_df], ignore_index=True)
        if year_to_adjust not in combined['Year'].values:
            print(f"No data found for year {year_to_adjust}")
            return None, None

        # 6) Compute the current profit margin in that goal year
        year_data = combined[combined['Year'] == year_to_adjust]
        current_net_income = year_data['Net Income'].iloc[0]
        current_net_revenue = year_data['Net Revenue'].iloc[0]
        current_profit_margin = (current_net_income / current_net_revenue) if current_net_revenue != 0 else 0

        # 7) If already within tolerance, return 1.0 (no change) and the original
        if abs(current_profit_margin - target_profit_margin) <= tolerance:
            return 1.0, combined

        # 8) Set up binary‐search bounds
        lower_bound = 0.5
        upper_bound = 2.0
        multiplier = 1.0

        # 9) Find the “original” value of the chosen variable in that year
        original_value = year_data[variable_to_adjust].iloc[0]

        # 10) Begin binary search
        for iteration in range(max_iterations):
            # 10a) Make a fresh copy of the forecast slice and apply the multiplier in the goal year
            forecast_df = full_forecast_df.copy()
            # 10b) Only adjust the row where Year == year_to_adjust
            forecast_df.loc[forecast_df['Year'] == year_to_adjust, variable_to_adjust] = original_value * multiplier
            # 10c) RESET INDEX on the adjusted forecast so no duplicates
            forecast_df = forecast_df.reset_index(drop=True)

            # 10d) Concatenate historical_df + adjusted forecast_df, ignoring indices
            what_if_df = pd.concat([historical_df, forecast_df], ignore_index=True)

            # 10e) Run the full chain of calculations on this what_if_df
            what_if_df = self.calculate_income_statement(
                what_if_df,
                tax_rate=0,               # or pass in whatever rates are needed
                inflation_rate=0,
                direct_labor_rate_increase=0
            )
            what_if_df = self.calculate_cash_flow_statement(what_if_df)
            what_if_df = self.calculate_supporting_schedules(what_if_df)
            what_if_df = self.calculate_balance_sheet(what_if_df)
            what_if_df = self.calculate_valuation(what_if_df, discount_rate=0.1)  # example discount

            # 10f) Extract the new profit margin in the goal year
            new_year_data = what_if_df[what_if_df['Year'] == year_to_adjust]
            new_net_income = new_year_data['Net Income'].iloc[0]
            new_net_revenue = new_year_data['Net Revenue'].iloc[0]
            new_profit_margin = (new_net_income / new_net_revenue) if new_net_revenue != 0 else 0

            # 10g) Check for convergence
            if abs(new_profit_margin - target_profit_margin) <= tolerance:
                return multiplier, what_if_df

            # 10h) Adjust binary‐search bounds according to overshoot/undershoot
            revenue_vars = ['Net Revenue', 'Total Orders', 'Average Item Value', 'Paid Search Traffic']
            if new_profit_margin < target_profit_margin:
                # If margin too low, increase “revenue” multipliers, or decrease “cost” multipliers
                if variable_to_adjust in revenue_vars:
                    lower_bound = multiplier
                else:
                    upper_bound = multiplier
            else:
                # If margin too high, decrease revenue multiplier or increase cost multiplier
                if variable_to_adjust in revenue_vars:
                    upper_bound = multiplier
                else:
                    lower_bound = multiplier

            multiplier = (lower_bound + upper_bound) / 2.0

        # 11) If we exit loop without converging
        print(f"Goal Seek did not converge within {max_iterations} iterations. Closest PM: {new_profit_margin:.2%}")
        return multiplier, what_if_df

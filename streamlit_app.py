"""Streamlit dashboard for the ecommerce financial model API."""
from __future__ import annotations

import base64
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Streamlit configuration & constants
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Ecommerce Financial Model", layout="wide", page_icon="💼")

DEFAULT_API_BASE = "http://localhost:8000"
API_BASE_ENV_VARS = ("ECOMMERCE_API_BASE", "ECOM_API_BASE", "API_BASE_URL")
SCENARIO_TYPES = ["Base Case", "Best Case", "Worst Case"]
SCENARIO_DEFAULTS = {
    "Base Case": {
        "conversion_rate_mult": 1.0,
        "aov_mult": 1.0,
        "cogs_mult": 1.0,
        "interest_mult": 1.0,
        "labor_mult": 1.0,
        "material_mult": 1.0,
        "markdown_mult": 1.0,
        "political_risk": 0.0,
        "env_impact": 0.0,
    },
    "Best Case": {
        "conversion_rate_mult": 1.2,
        "aov_mult": 1.1,
        "cogs_mult": 0.95,
        "interest_mult": 0.9,
        "labor_mult": 0.9,
        "material_mult": 0.9,
        "markdown_mult": 0.9,
        "political_risk": 2.0,
        "env_impact": 2.0,
    },
    "Worst Case": {
        "conversion_rate_mult": 0.8,
        "aov_mult": 0.9,
        "cogs_mult": 1.05,
        "interest_mult": 1.2,
        "labor_mult": 1.2,
        "material_mult": 1.2,
        "markdown_mult": 1.1,
        "political_risk": 4.0,
        "env_impact": 4.0,
    },
}
SCENARIO_PARAM_LABELS = {
    "conversion_rate_mult": "Conversion rate multiplier",
    "aov_mult": "Average order value multiplier",
    "cogs_mult": "COGS multiplier",
    "interest_mult": "Interest multiplier",
    "labor_mult": "Labor multiplier",
    "material_mult": "Material multiplier",
    "markdown_mult": "Markdown multiplier",
    "political_risk": "Political risk score",
    "env_impact": "Environmental impact score",
}
WHAT_IF_VARIABLES = [
    "Total Orders",
    "Average Item Value",
    "Email Conversion Rate",
    "Paid Search Traffic",
    "COGS Percentage",
    "Labor/Handling per Order",
    "Freight/Shipping per Order",
    "Marketing Expenses",
    "Interest Rate",
]
GOAL_SEEK_VARIABLES = [
    "Net Revenue",
    "Total Orders",
    "Average Item Value",
    "Paid Search Traffic",
    "COGS Percentage",
    "Labor/Handling per Order",
    "Freight/Shipping per Order",
    "Marketing Expenses",
]
SENSITIVITY_VARIABLES = [
    "Average Item Value",
    "COGS Percentage",
    "Email Traffic",
    "Paid Search Traffic",
    "Email Conversion Rate",
    "Organic Search Conversion Rate",
    "Paid Search Conversion Rate",
    "Affiliates Conversion Rate",
]
BUDGET_LINES = [
    "Total Marketing Budget",
    "Freight/Shipping per Order",
    "Labor/Handling per Order",
    "General Warehouse Rent",
    "Office Rent",
    "Salaries, Wages & Benefits",
    "Professional Fees",
]
SCHEDULE_OPTIONS = {
    "Income Statement": ["Income Statement"],
    "Financial Position": ["Balance Sheet", "Capital Assets", "Debt Payment Schedule"],
    "Cash Flow": ["Cash Flow Statement", "Valuation"],
}

DEFAULT_ASSUMPTION_YEARS = [2023, 2024, 2025, 2026, 2027]
ASSUMPTION_SCHEDULES: List[Dict[str, Any]] = [
    {
        "name": "Demand & Conversion",
        "description": "Traffic, conversion, and churn drivers for each forecast year.",
        "columns": [
            "Year",
            "Email Traffic",
            "Organic Search Traffic",
            "Paid Search Traffic",
            "Affiliates Traffic",
            "Email Conversion Rate",
            "Organic Search Conversion Rate",
            "Paid Search Conversion Rate",
            "Affiliates Conversion Rate",
            "Churn Rate",
        ],
        "editable_year": True,
    },
    {
        "name": "Pricing & Order Economics",
        "description": "Average order value, markdowns, and unit economics assumptions.",
        "columns": [
            "Year",
            "Average Item Value",
            "Number of Items per Order",
            "Average Markdown",
            "Average Promotion/Discount",
            "COGS Percentage",
        ],
        "editable_year": False,
    },
    {
        "name": "Acquisition Costs",
        "description": "Cost per click inputs for each acquisition channel.",
        "columns": [
            "Year",
            "Email Cost per Click",
            "Organic Search Cost per Click",
            "Paid Search Cost per Click",
            "Affiliates Cost per Click",
        ],
        "editable_year": False,
    },
    {
        "name": "Fulfillment & Operating Costs",
        "description": "Per-order fulfillment costs plus warehouse overhead and tax rate assumptions.",
        "columns": [
            "Year",
            "Freight/Shipping per Order",
            "Labor/Handling per Order",
            "General Warehouse Rent",
            "Other",
            "Interest",
            "Tax Rate",
        ],
        "editable_year": False,
    },
    {
        "name": "Staffing Levels",
        "description": "Direct, indirect, and part-time staffing assumptions and costs.",
        "columns": [
            "Year",
            "Direct Staff Hours per Year",
            "Direct Staff Number",
            "Direct Staff Hourly Rate",
            "Direct Staff Total Cost",
            "Indirect Staff Hours per Year",
            "Indirect Staff Number",
            "Indirect Staff Hourly Rate",
            "Indirect Staff Total Cost",
            "Part-Time Staff Hours per Year",
            "Part-Time Staff Number",
            "Part-Time Staff Hourly Rate",
            "Part-Time Staff Total Cost",
        ],
        "editable_year": False,
    },
    {
        "name": "Executive Compensation",
        "description": "Leadership salaries captured separately from staffing totals.",
        "columns": [
            "Year",
            "CEO Salary",
            "COO Salary",
            "CFO Salary",
            "Director of HR Salary",
            "CIO Salary",
        ],
        "editable_year": False,
    },
    {
        "name": "Employee Benefits",
        "description": "Benefits allocations and employer-paid allowances.",
        "columns": [
            "Year",
            "Pension Cost per Staff",
            "Pension Total Cost",
            "Medical Insurance Cost per Staff",
            "Medical Insurance Total Cost",
            "Child Benefit Cost per Staff",
            "Child Benefit Total Cost",
            "Car Benefit Cost per Staff",
            "Car Benefit Total Cost",
            "Total Benefits",
        ],
        "editable_year": False,
    },
    {
        "name": "Overheads & Fees",
        "description": "General overhead expenses and professional fees.",
        "columns": [
            "Year",
            "Salaries, Wages & Benefits",
            "Office Rent",
            "Rent Categories",
            "Professional Fees",
            "Professional Fee Types",
            "Depreciation",
        ],
        "editable_year": False,
    },
    {
        "name": "Working Capital",
        "description": "Receivable, inventory, and payable day assumptions.",
        "columns": [
            "Year",
            "Accounts Receivable Days",
            "Inventory Days",
            "Accounts Payable Days",
        ],
        "editable_year": False,
    },
    {
        "name": "Capital Investments",
        "description": "Capital expenditure and depreciation schedules.",
        "columns": [
            "Year",
            "Technology Development",
            "Office Equipment",
            "Technology Depreciation Years",
            "Office Equipment Depreciation Years",
        ],
        "editable_year": False,
    },
    {
        "name": "Financing Activities",
        "description": "Equity, debt, and dividend financing assumptions.",
        "columns": [
            "Year",
            "Interest Rate",
            "Equity Raised",
            "Dividends Paid",
            "Debt Issued",
        ],
        "editable_year": False,
    },
    {
        "name": "Property Portfolio",
        "description": "Warehouse footprint and rent assumptions by site.",
        "columns": [
            "Year",
            "Warehouse2 Square Meters",
            "Warehouse2 Cost per SQM",
            "Warehouse2",
            "sun warehouse Square Meters",
            "sun warehouse Cost per SQM",
            "sun warehouse",
            "new warehouse Square Meters",
            "new warehouse Cost per SQM",
            "new warehouse",
        ],
        "editable_year": False,
    },
    {
        "name": "Legal & Compliance",
        "description": "Legal obligations, recurring fees, and future legal spend planning.",
        "columns": [
            "Year",
            "Legal Cost",
            "Legal",
            "legal_2024 Cost",
            "legal_2024",
            "legal_2025 Cost",
            "legal_2025",
        ],
        "editable_year": False,
    },
    {
        "name": "Asset Register",
        "description": "Fixed asset balances and depreciation assumptions.",
        "columns": [
            "Year",
            "Asset_1_Name",
            "Asset_1_Amount",
            "Asset_1_Rate",
            "Asset_1_Depreciation",
            "Asset_1_NBV",
        ],
        "editable_year": False,
    },
    {
        "name": "Debt Schedule",
        "description": "Outstanding debt facilities and repayment terms.",
        "columns": [
            "Year",
            "Debt_1_Name",
            "Debt_1_Amount",
            "Debt_1_Interest_Rate",
            "Debt_1_Duration",
        ],
        "editable_year": False,
    },
]


ASSUMPTION_SCHEDULE_TIPS: Dict[str, List[str]] = {
    "Demand & Conversion": [
        "Update channel traffic volumes to match your marketing plan; lower numbers reflect tighter budgets while higher values signal expansion.",
        "Tune each conversion rate to represent expected landing-page or campaign performance—use the Add line control when you need duplicate years for A/B assumptions.",
        "Adjust the churn rate when retention programs change; removing a line lets you discard outdated historical scenarios.",
    ],
    "Pricing & Order Economics": [
        "Set Average Item Value and Items per Order from your merchandising plan; edit values directly in the grid to reflect updated pricing tests.",
        "Use Average Markdown and Promotion/Discount to capture seasonal campaigns; insert rows for new year-specific promo calendars.",
        "COGS Percentage should reflect anticipated supplier terms—remove a row if that year no longer applies to the forecast horizon.",
    ],
    "Acquisition Costs": [
        "Enter the per-click spend for each channel as negotiated with marketing vendors.",
        "Add a row to stage alternative cost outlooks for the same year (e.g., optimistic vs. conservative).",
        "Drop a row when the channel plan is retired so calculations no longer use stale assumptions.",
    ],
    "Fulfillment & Operating Costs": [
        "Update freight, labor, and rent line items to mirror logistics contracts or staffing plans.",
        "Adjust the Interest and Tax Rate fields when capital structure or jurisdiction changes occur.",
        "Use Add/Remove line buttons to separate special projects (like temporary warehouses) from base operations.",
    ],
    "Staffing Levels": [
        "Edit the direct, indirect, and part-time staffing counts and hours to align with workforce planning.",
        "Set hourly rates or total costs when compensation benchmarks change; ensure totals reconcile with payroll guidance.",
        "Insert extra rows to capture alternate hiring scenarios for the same year, then prune outdated entries with Remove line.",
    ],
    "Executive Compensation": [
        "Revise each executive salary when contract renewals occur; the schedule supports multiple years of adjustments.",
        "Add a line to reflect overlapping leadership transitions where two executives share a year.",
        "Remove rows for vacant positions so the forecast excludes unneeded salary expense.",
    ],
    "Employee Benefits": [
        "Populate per-staff and total cost fields based on benefits provider quotes.",
        "Add rows to reflect plan changes that apply mid-year or to separate cohorts (e.g., hourly vs. salaried).",
        "Delete rows when benefit programs are sunset to keep totals clean.",
    ],
    "Overheads & Fees": [
        "Update salaries and rent categories to reflect general and administrative plans.",
        "Modify Professional Fees and Types when advisory engagements start or end.",
        "Use Add line to introduce new fee categories tied to future projects, and Remove line once they conclude.",
    ],
    "Working Capital": [
        "Adjust Accounts Receivable Days to match credit terms—lowering the figure speeds up cash collection.",
        "Inventory Days should reflect turnover targets; add a line to represent alternative sourcing assumptions for the same year.",
        "Set Accounts Payable Days to negotiated supplier terms and remove rows once legacy agreements expire.",
    ],
    "Capital Investments": [
        "Enter capex for technology and office equipment directly in the table when budgeting new projects.",
        "Update depreciation years to model policy changes; separate overlapping initiatives by inserting additional rows.",
        "Remove obsolete investment lines to avoid carrying forward cancelled projects.",
    ],
    "Financing Activities": [
        "Set interest rates, equity raises, and dividend payouts according to your capital plan.",
        "Add rows for bridge rounds or debt tranches that occur within the same fiscal year.",
        "Remove financing rows when a transaction is deferred to keep projections realistic.",
    ],
    "Property Portfolio": [
        "Update square meters and cost per SQM for each warehouse as leases change or expansions occur.",
        "Insert lines to capture overlapping facilities or temporary space requirements.",
        "Remove warehouse rows after consolidation to keep rent forecasts current.",
    ],
    "Legal & Compliance": [
        "Edit legal cost buckets to reflect regulatory filings or casework expected in that year.",
        "Add lines when multiple legal initiatives share a single fiscal period for more granular tracking.",
        "Remove stale line items once matters close to avoid double-counting spend.",
    ],
    "Asset Register": [
        "Update asset names, amounts, and depreciation rates as fixed assets are acquired or revalued.",
        "Add rows for each new asset class introduced during the forecast period.",
        "Remove assets once they are disposed or fully depreciated to keep the register accurate.",
    ],
    "Debt Schedule": [
        "Edit debt names, balances, and interest rates to align with current facility agreements.",
        "Insert lines for additional borrowings or refinancing that occur mid-year.",
        "Remove loans once they are repaid so amortization schedules no longer include them.",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_api_base() -> str:
    if "api_base_url" not in st.session_state:
        for env_var in API_BASE_ENV_VARS:
            candidate = os.environ.get(env_var)
            if candidate:
                set_api_base(candidate)
                break
        else:
            st.session_state["api_base_url"] = DEFAULT_API_BASE
    return st.session_state["api_base_url"]


def set_api_base(url: str) -> None:
    cleaned = (url or "").strip()
    if cleaned:
        cleaned = cleaned.rstrip("/")
    st.session_state["api_base_url"] = cleaned or DEFAULT_API_BASE


def _normalize_year_list(years: Iterable[Any]) -> List[int]:
    normalized: List[int] = []
    for item in years:
        try:
            value = int(float(item))
        except (TypeError, ValueError):
            continue
        if value not in normalized:
            normalized.append(value)
    return sorted(normalized)


def create_blank_schedule(columns: Iterable[str], years: Iterable[int]) -> pd.DataFrame:
    year_list = _normalize_year_list(years)
    if year_list:
        frame = pd.DataFrame({"Year": year_list})
    else:
        frame = pd.DataFrame(columns=["Year"])
    for column in columns:
        if column == "Year":
            continue
        frame[column] = None
    # Ensure column order matches the schedule definition
    return frame[[col for col in columns if col in frame.columns]]


def _coerce_schedule_frame(data: Any, columns: Sequence[str]) -> pd.DataFrame:
    column_list = list(columns)
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif data is None:
        frame = pd.DataFrame(columns=column_list)
    else:
        frame = pd.DataFrame(data)
    for column in column_list:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[column_list]
    return frame.reset_index(drop=True)


def _dataframes_equal(left: Any, right: Any) -> bool:
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        try:
            return left.equals(right)
        except Exception:  # pragma: no cover - defensive
            return False
    return left is right


def _format_edit_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _parse_edit_value(column: str, text_value: str) -> Any:
    cleaned = (text_value or "").strip()
    if not cleaned:
        return None
    if column == "Year":
        try:
            year_value = int(float(cleaned))
        except (TypeError, ValueError):
            return cleaned
        return year_value
    numeric_candidate = pd.to_numeric(pd.Series([cleaned]), errors="coerce").iloc[0]
    if pd.isna(numeric_candidate):
        return cleaned
    float_value = float(numeric_candidate)
    if float_value.is_integer():
        return int(float_value)
    return float_value


def _apply_incremental_fill(
    frame: pd.DataFrame, percent_changes: Dict[str, float]
) -> pd.DataFrame:
    """Fill subsequent years using percentage changes from prior year values."""

    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame

    working = frame.copy()
    if "Year" not in working.columns:
        return working

    def _sort_key(idx: int) -> Any:
        year_value = working.iloc[idx].get("Year")
        try:
            return (float(year_value), idx)
        except (TypeError, ValueError):
            return (float("inf"), idx)

    order = sorted(range(len(working)), key=_sort_key)
    if not order:
        return working

    for pos in range(1, len(order)):
        prev_idx = order[pos - 1]
        idx = order[pos]
        for column, percent in percent_changes.items():
            if column not in working.columns:
                continue
            prev_value = working.at[prev_idx, column]
            if prev_value is None or pd.isna(prev_value):
                continue
            try:
                prev_numeric = float(prev_value)
            except (TypeError, ValueError):
                continue
            try:
                percent_value = float(percent or 0.0)
            except (TypeError, ValueError):
                percent_value = 0.0
            new_value = prev_numeric * (1.0 + percent_value / 100.0)
            if float(prev_numeric).is_integer() and float(new_value).is_integer():
                working.at[idx, column] = int(round(new_value))
            else:
                working.at[idx, column] = round(new_value, 6)

    return working


def collect_all_years(tables: Dict[str, pd.DataFrame]) -> List[int]:
    observed: Set[int] = set()
    for frame in tables.values():
        if frame is None or not isinstance(frame, pd.DataFrame):
            continue
        if "Year" not in frame.columns:
            continue
        observed.update(_normalize_year_list(frame["Year"].tolist()))
    return sorted(observed)


def ensure_assumption_tables() -> None:
    if "assumption_tables" not in st.session_state:
        st.session_state["assumption_years"] = DEFAULT_ASSUMPTION_YEARS.copy()
        initial_tables = {
            schedule["name"]: create_blank_schedule(
                schedule["columns"], st.session_state["assumption_years"]
            )
            for schedule in ASSUMPTION_SCHEDULES
        }
        st.session_state["assumption_tables"] = sync_schedule_years(
            st.session_state["assumption_years"], initial_tables
        )


def get_assumption_years() -> List[int]:
    if years := st.session_state.get("assumption_years"):
        return years
    tables: Dict[str, pd.DataFrame] = st.session_state.get("assumption_tables", {})
    collected: List[int] = []
    for frame in tables.values():
        if frame is None or frame.empty or "Year" not in frame.columns:
            continue
        collected.extend(_normalize_year_list(frame["Year"].tolist()))
    if not collected:
        collected = DEFAULT_ASSUMPTION_YEARS.copy()
    st.session_state["assumption_years"] = collected
    return collected


def add_assumption_year(year: int) -> None:
    current_years = _normalize_year_list(get_assumption_years())
    if year not in current_years:
        current_years.append(year)
    tables: Dict[str, pd.DataFrame] = st.session_state.get("assumption_tables", {})
    updated_tables: Dict[str, pd.DataFrame] = {}
    for schedule in ASSUMPTION_SCHEDULES:
        frame = tables.get(schedule["name"])
        columns = schedule["columns"]
        new_row = {column: None for column in columns}
        if "Year" in new_row:
            new_row["Year"] = year
        if frame is None or frame.empty:
            updated_tables[schedule["name"]] = pd.DataFrame([new_row], columns=columns)
            continue
        working = frame.copy()
        if "Year" not in working.columns:
            working.insert(0, "Year", [None] * len(working))
        working["Year"] = pd.to_numeric(working["Year"], errors="coerce")
        existing_years = _normalize_year_list(working["Year"].tolist())
        if year not in existing_years:
            working = pd.concat([working, pd.DataFrame([new_row])], ignore_index=True)
        updated_tables[schedule["name"]] = working
    st.session_state["assumption_tables"] = sync_schedule_years(
        sorted(set(current_years)), updated_tables
    )


def remove_assumption_year(year: int) -> None:
    tables: Dict[str, pd.DataFrame] = st.session_state.get("assumption_tables", {})
    cleaned: Dict[str, pd.DataFrame] = {}
    for schedule in ASSUMPTION_SCHEDULES:
        frame = tables.get(schedule["name"])
        if frame is None or frame.empty:
            cleaned[schedule["name"]] = frame
            continue
        frame = frame.copy()
        if "Year" in frame.columns:
            frame["Year"] = pd.to_numeric(frame["Year"], errors="coerce")
            frame = frame[frame["Year"] != year]
        cleaned[schedule["name"]] = frame
    remaining_years = [value for value in collect_all_years(cleaned) if value != year]
    st.session_state["assumption_tables"] = sync_schedule_years(remaining_years, cleaned)


def reset_assumption_tables(years: Optional[Iterable[int]] = None) -> None:
    base_years = _normalize_year_list(years or DEFAULT_ASSUMPTION_YEARS)
    st.session_state["assumption_years"] = base_years
    fresh_tables = {
        schedule["name"]: create_blank_schedule(schedule["columns"], base_years)
        for schedule in ASSUMPTION_SCHEDULES
    }
    st.session_state["assumption_tables"] = sync_schedule_years(base_years, fresh_tables)
    st.session_state["assumptions_raw"] = []


def split_assumptions_into_tables(df: pd.DataFrame, years: Iterable[int]) -> Dict[str, pd.DataFrame]:
    year_list = _normalize_year_list(years)
    tables: Dict[str, pd.DataFrame] = {}
    for schedule in ASSUMPTION_SCHEDULES:
        available_cols = [col for col in schedule["columns"] if col in df.columns]
        if available_cols:
            frame = df[available_cols].copy()
            if "Year" in frame.columns:
                frame["Year"] = pd.to_numeric(frame["Year"], errors="coerce")
            for column in schedule["columns"]:
                if column not in frame.columns:
                    frame[column] = None
            frame = frame[schedule["columns"]]
        else:
            frame = create_blank_schedule(schedule["columns"], year_list)
        tables[schedule["name"]] = frame.reset_index(drop=True)
    return tables


def combine_assumption_tables(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for schedule in ASSUMPTION_SCHEDULES:
        frame = tables.get(schedule["name"])
        if frame is None or frame.empty or "Year" not in frame.columns:
            continue
        normalized = frame.copy()
        normalized["Year"] = pd.to_numeric(normalized["Year"], errors="coerce")
        frames.append(normalized)
    if not frames:
        return pd.DataFrame()
    merged = frames[0]
    for frame in frames[1:]:
        merged = pd.merge(merged, frame, on="Year", how="outer")
    merged = merged.sort_values("Year").reset_index(drop=True)
    ordered_columns: List[str] = []
    for schedule in ASSUMPTION_SCHEDULES:
        for column in schedule["columns"]:
            if column not in ordered_columns:
                ordered_columns.append(column)
    merged = merged[[column for column in ordered_columns if column in merged.columns]]
    merged["Year"] = pd.to_numeric(merged["Year"], errors="coerce").astype("Int64")
    return merged


def set_assumptions_data(rows: List[Dict[str, Any]]) -> None:
    df = to_dataframe(rows)
    if not df.empty and "Year" in df.columns:
        years = _normalize_year_list(df["Year"].tolist())
    else:
        years = DEFAULT_ASSUMPTION_YEARS.copy()
    st.session_state["assumptions_raw"] = rows
    st.session_state["assumption_years"] = years
    tables = split_assumptions_into_tables(df, years)
    st.session_state["assumption_tables"] = sync_schedule_years(years, tables)


def sync_schedule_years(
    base_years: List[int], tables: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    normalized_years = _normalize_year_list(base_years)
    observed: Set[int] = set(normalized_years)
    synced: Dict[str, pd.DataFrame] = {}
    for schedule in ASSUMPTION_SCHEDULES:
        columns = schedule["columns"]
        frame = tables.get(schedule["name"])
        if frame is None:
            frame = create_blank_schedule(columns, normalized_years)
        else:
            frame = frame.copy()
            if "Year" in frame.columns:
                frame["Year"] = pd.to_numeric(frame["Year"], errors="coerce")
                observed.update(_normalize_year_list(frame["Year"].tolist()))
            else:
                frame.insert(0, "Year", [None] * len(frame))
            for column in columns:
                if column not in frame.columns:
                    frame[column] = None
            frame = frame[columns]
        synced[schedule["name"]] = frame.reset_index(drop=True)
    final_years = sorted(year for year in observed if year is not None)
    if not final_years:
        final_years = normalized_years or DEFAULT_ASSUMPTION_YEARS.copy()
    st.session_state["assumption_years"] = final_years
    return synced


def api_request(
    method: str,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Any] = None,
    files: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> Any:
    base_url = get_api_base()
    if not base_url:
        raise RuntimeError("Configure the API base URL before making requests.")
    url = f"{base_url}{path if path.startswith('/') else '/' + path}"
    try:
        response = requests.request(
            method,
            url,
            params=params,
            json=json,
            files=files,
            data=data,
            timeout=timeout,
        )
    except requests.RequestException as exc:  # pragma: no cover - network failure
        raise RuntimeError(f"Request to {url} failed: {exc}") from exc

    if response.status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise RuntimeError(f"{response.status_code} {response.reason}: {detail}")

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        return response.json()
    if "application/vnd.openxmlformats" in content_type or path.endswith("export_excel"):
        return response.content
    return response.text


def api_get(path: str, **kwargs: Any) -> Any:
    return api_request("GET", path, **kwargs)


def api_post(path: str, **kwargs: Any) -> Any:
    return api_request("POST", path, **kwargs)


@lru_cache(maxsize=None)
def load_scenario_defaults_from_api(scenario_type: str) -> Dict[str, float]:
    fallback = dict(SCENARIO_DEFAULTS.get(scenario_type, SCENARIO_DEFAULTS["Base Case"]))
    try:
        response = api_get(f"/get_scenario_parameters/{requests.utils.quote(scenario_type)}")
    except RuntimeError:
        return fallback
    parameters = response.get("parameters") if isinstance(response, dict) else response
    if isinstance(parameters, dict):
        merged: Dict[str, float] = {**fallback}
        for key, value in parameters.items():
            try:
                merged[key] = float(value)
            except (TypeError, ValueError):
                continue
        return merged
    return fallback


def to_dataframe(records: Any) -> pd.DataFrame:
    if records is None:
        return pd.DataFrame()
    if isinstance(records, dict):
        records = list(records.values())
    if isinstance(records, Iterable) and not isinstance(records, (str, bytes)):
        try:
            return pd.DataFrame(list(records))
        except ValueError:
            return pd.DataFrame()
    return pd.DataFrame()


def render_table(title: str, data: Any) -> None:
    df = to_dataframe(data)
    if df.empty:
        st.info(f"No {title.lower()} available.")
    else:
        st.subheader(title)
        st.dataframe(df, use_container_width=True)


def render_metric_cards(metrics: List[Dict[str, Any]]) -> None:
    if not metrics:
        st.info("No operational metrics available yet.")
        return
    columns = st.columns(min(4, len(metrics)))
    for idx, metric in enumerate(metrics):
        col = columns[idx % len(columns)]
        col.metric(metric.get("metric", "Metric"), f"{metric.get('current', 0):,.2f}")


def build_revenue_figure(payload: Dict[str, Any]) -> go.Figure:
    years = payload.get("years") or []
    net_revenue = payload.get("net_revenue") or []
    gross_margin = payload.get("gross_margin") or []
    ebitda_margin = payload.get("ebitda_margin") or []
    fig = go.Figure()
    if years and any(pd.notna(val) for val in net_revenue):
        fig.add_bar(name="Net Revenue", x=years, y=net_revenue, marker_color="#2563eb")
    if years and any(pd.notna(val) for val in gross_margin):
        fig.add_trace(
            go.Scatter(
                name="Gross Margin",
                x=years,
                y=gross_margin,
                mode="lines+markers",
                marker=dict(color="#0ea5e9"),
                yaxis="y2",
            )
        )
    if years and any(pd.notna(val) for val in ebitda_margin):
        fig.add_trace(
            go.Scatter(
                name="EBITDA Margin",
                x=years,
                y=ebitda_margin,
                mode="lines+markers",
                marker=dict(color="#a855f7"),
                yaxis="y2",
                line=dict(dash="dot"),
            )
        )
    fig.update_layout(
        title="Revenue & margin profile",
        barmode="group",
        yaxis=dict(title="Net Revenue", tickprefix="$", separatethousands=True),
        yaxis2=dict(title="Margin %", overlaying="y", side="right", tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, r=40, l=40, b=40),
    )
    return fig


def build_traffic_figure(payload: Dict[str, Any]) -> go.Figure:
    years = payload.get("years") or []
    ltv = payload.get("ltv") or []
    cac = payload.get("cac") or []
    ratio = payload.get("ltv_cac_ratio") or []
    fig = go.Figure()
    if years and any(pd.notna(val) for val in ltv):
        fig.add_bar(name="LTV", x=years, y=ltv, marker_color="#22c55e")
    if years and any(pd.notna(val) for val in cac):
        fig.add_bar(name="CAC", x=years, y=cac, marker_color="#ef4444")
    if years and any(pd.notna(val) for val in ratio):
        fig.add_trace(
            go.Scatter(
                name="LTV/CAC",
                x=years,
                y=ratio,
                mode="lines+markers",
                marker=dict(color="#6366f1"),
                yaxis="y2",
            )
        )
    fig.update_layout(
        title="Customer economics",
        barmode="group",
        yaxis=dict(title="Value", tickprefix="$", separatethousands=True),
        yaxis2=dict(title="Ratio", overlaying="y", side="right", tickformat=".2f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, r=40, l=40, b=40),
    )
    return fig


def build_profitability_figure(payload: Dict[str, Any]) -> go.Figure:
    years = payload.get("years") or []
    closing_cash = payload.get("closing_cash_balance") or []
    fig = go.Figure()
    if years and any(pd.notna(val) for val in closing_cash):
        fig.add_trace(
            go.Scatter(
                name="Closing Cash Balance",
                x=years,
                y=closing_cash,
                mode="lines+markers",
                line=dict(color="#f97316", width=3),
            )
        )
    fig.update_layout(
        title="Closing cash balance",
        yaxis=dict(title="Cash", tickprefix="$", separatethousands=True),
        margin=dict(t=60, r=40, l=40, b=40),
    )
    return fig


def build_cashflow_figure(payload: Dict[str, Any]) -> go.Figure:
    years = payload.get("years") or []
    ops = payload.get("cash_from_operations") or []
    investing = payload.get("cash_from_investing") or []
    net = payload.get("net_cash_flow") or []
    fig = go.Figure()
    if years and any(pd.notna(val) for val in ops):
        fig.add_bar(name="Operations", x=years, y=ops, marker_color="#22c55e")
    if years and any(pd.notna(val) for val in investing):
        fig.add_bar(name="Investing", x=years, y=investing, marker_color="#f59e0b")
    if years and any(pd.notna(val) for val in net):
        fig.add_trace(
            go.Scatter(
                name="Net Cash Flow",
                x=years,
                y=net,
                mode="lines+markers",
                marker=dict(color="#0ea5e9"),
            )
        )
    fig.update_layout(
        title="Cash flow forecast",
        barmode="group",
        yaxis=dict(title="Cash Flow", tickprefix="$", separatethousands=True),
        margin=dict(t=60, r=40, l=40, b=40),
    )
    return fig


def build_waterfall(payload: Dict[str, Any], title: str) -> go.Figure:
    categories = payload.get("categories") or []
    values = payload.get("values") or []
    measures = payload.get("measures") or []
    fig = go.Figure()
    if categories and any(pd.notna(val) for val in values):
        fig.add_trace(
            go.Waterfall(
                name=payload.get("title", title),
                orientation="v",
                measure=measures,
                x=categories,
                y=values,
            )
        )
    fig.update_layout(title=title, margin=dict(t=60, r=40, l=40, b=40))
    return fig


def render_schedule_section(title: str, schedules: Iterable[Dict[str, Any]]) -> None:
    for schedule in schedules:
        name = schedule.get("schedule", "Schedule")
        data = schedule.get("data")
        df = to_dataframe(data)
        st.subheader(name)
        if df.empty:
            st.info("No data available.")
        else:
            st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------


def configure_sidebar() -> None:
    with st.sidebar:
        st.empty()


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


def render_input_tab(tab: st.delta_generator.DeltaGenerator) -> None:
    with tab:
        st.header("Workbook setup & assumptions")
        st.write(
            "Upload or refresh the working Excel file, edit the grouped assumption tables,"
            " and control base analysis inputs before running downstream workflows."
        )

        ensure_assumption_tables()

        st.subheader("Edit assumptions")
        st.caption(
            "Populate each schedule below to rebuild the model inputs manually. "
            "The Demand & Conversion table controls the forecast years—add or remove years "
            "there and the remaining schedules will stay in sync."
        )

        current_years = get_assumption_years()
        add_col, remove_col, reset_col = st.columns([2, 2, 1])
        with add_col:
            default_year = (current_years[-1] + 1) if current_years else 2023
            new_year_value = st.number_input(
                "New forecast year",
                value=default_year,
                step=1,
                key="assumptions_add_year",
            )
            if st.button("Add year", use_container_width=True, key="assumptions_add_year_btn"):
                add_assumption_year(int(new_year_value))
        with remove_col:
            if current_years:
                remove_choice = st.selectbox(
                    "Remove forecast year",
                    current_years,
                    key="assumptions_remove_year",
                )
                if st.button(
                    "Remove year", use_container_width=True, key="assumptions_remove_year_btn"
                ):
                    remove_assumption_year(int(remove_choice))
            else:
                st.selectbox(
                    "Remove forecast year",
                    ["No years available"],
                    disabled=True,
                    key="assumptions_remove_year_disabled",
                )
                st.button(
                    "Remove year",
                    use_container_width=True,
                    disabled=True,
                    key="assumptions_remove_year_btn_disabled",
                )
        with reset_col:
            st.write("")
            st.write("")
            if st.button("Reset", use_container_width=True, key="assumptions_reset_btn"):
                reset_assumption_tables()

        tables: Dict[str, pd.DataFrame] = st.session_state.get("assumption_tables", {})
        schedule_tabs = st.tabs([schedule["name"] for schedule in ASSUMPTION_SCHEDULES])
        updated_tables: Dict[str, pd.DataFrame] = {}
        for schedule, schedule_tab in zip(ASSUMPTION_SCHEDULES, schedule_tabs):
            with schedule_tab:
                st.caption(schedule["description"])
                tips = ASSUMPTION_SCHEDULE_TIPS.get(schedule["name"])
                if tips:
                    tip_lines = "\n".join(f"- {tip}" for tip in tips)
                    st.info(f"**How to edit line items**\n{tip_lines}")
                frame = tables.get(schedule["name"])
                if frame is None:
                    frame = create_blank_schedule(schedule["columns"], current_years)
                frame = _coerce_schedule_frame(frame, schedule["columns"])
                column_config: Dict[str, Any] = {}
                column_config["Year"] = st.column_config.NumberColumn(
                    "Year",
                    step=1,
                    format="%d",
                    disabled=False,
                )

            editor_key = "assumptions_" + "".join(
                char.lower() if char.isalnum() else "_" for char in schedule["name"]
            )
            data_state_key = f"{editor_key}_table"
            edit_state_key = f"{editor_key}_active_edit"
            if edit_state_key not in st.session_state:
                st.session_state[edit_state_key] = None

            stored_table = st.session_state.get(data_state_key)
            if stored_table is None:
                st.session_state[data_state_key] = frame.copy()
            else:
                coerced = _coerce_schedule_frame(stored_table, schedule["columns"])
                if not _dataframes_equal(coerced, frame):
                    coerced = frame.copy()
                st.session_state[data_state_key] = coerced.copy()

            working_table = _coerce_schedule_frame(
                st.session_state[data_state_key], schedule["columns"]
            )

            table_col, actions_col = st.columns([4, 1.7], gap="large")

            with table_col:
                updated = st.data_editor(
                    working_table,
                    num_rows="dynamic",
                    use_container_width=True,
                    key=editor_key,
                    column_config=column_config,
                )
                updated = _coerce_schedule_frame(updated, schedule["columns"])

                with st.expander("Line item controls", expanded=False):
                    insert_col, remove_col = st.columns(2)
                    with insert_col:
                        insert_position = st.number_input(
                            "Insert at row",
                            min_value=0,
                            max_value=len(updated),
                            value=len(updated),
                            step=1,
                            key=f"{editor_key}_insert_position",
                        )
                        if "Year" in updated.columns:
                            existing_years = _normalize_year_list(
                                pd.to_numeric(updated["Year"], errors="coerce").tolist()
                            )
                            if existing_years:
                                suggested_year = existing_years[-1] + 1
                            elif current_years:
                                suggested_year = current_years[0]
                            else:
                                suggested_year = 2023
                            new_row_year = st.number_input(
                                "Year for new row",
                                value=int(suggested_year),
                                step=1,
                                key=f"{editor_key}_new_row_year",
                            )
                        else:
                            new_row_year = None
                        if st.button(
                            "Add line",
                            key=f"{editor_key}_add_line",
                            use_container_width=True,
                        ):
                            new_row = {column: None for column in schedule["columns"]}
                            if "Year" in new_row and new_row_year is not None:
                                new_row["Year"] = int(new_row_year)
                            top = updated.iloc[: insert_position]
                            bottom = updated.iloc[insert_position:]
                            updated = pd.concat(
                                [top, pd.DataFrame([new_row]), bottom], ignore_index=True
                            )
                            st.session_state[data_state_key] = updated.copy()
                            st.success("Line inserted.")
                    with remove_col:
                        if not updated.empty:
                            remove_options = list(range(len(updated)))
                            remove_labels: List[str] = []
                            for idx, (_, row) in enumerate(updated.iterrows()):
                                year_value = row.get("Year")
                                if pd.notna(year_value):
                                    remove_labels.append(f"Row {idx + 1} – Year {int(year_value)}")
                                else:
                                    remove_labels.append(f"Row {idx + 1} – (no year)")
                            remove_choice = st.selectbox(
                                "Row to remove",
                                remove_options,
                                format_func=lambda idx: remove_labels[idx],
                                key=f"{editor_key}_remove_choice",
                            )
                            if st.button(
                                "Remove line",
                                key=f"{editor_key}_remove_line",
                                use_container_width=True,
                            ):
                                updated = (
                                    updated.drop(index=remove_choice)
                                    .reset_index(drop=True)
                                )
                                st.session_state[data_state_key] = updated.copy()
                                st.success("Line removed.")
                        else:
                            st.selectbox(
                                "Row to remove",
                                ["No rows available"],
                                disabled=True,
                                key=f"{editor_key}_remove_disabled",
                            )
                            st.button(
                                "Remove line",
                                key=f"{editor_key}_remove_btn_disabled",
                                disabled=True,
                                use_container_width=True,
                            )

                numeric_columns = [
                    column for column in schedule["columns"] if column != "Year"
                ]
                if len(updated) >= 2 and numeric_columns:
                    with st.expander("Yearly increment helper", expanded=False):
                        st.caption(
                            "Fill the first year's values, then set annual percentage changes "
                            "to populate the remaining years automatically."
                        )
                        rate_inputs: Dict[str, float] = {}
                        for column in numeric_columns:
                            rate_inputs[column] = st.number_input(
                                f"{column} annual change (%)",
                                value=0.0,
                                step=0.5,
                                format="%.2f",
                                key=f"{editor_key}_increment_{column}",
                            )
                        helper_col1, helper_col2, helper_col3 = st.columns(3)
                        with helper_col1:
                            if st.button(
                                "Copy forward",
                                key=f"{editor_key}_copy_forward",
                                use_container_width=True,
                            ):
                                updated = _apply_incremental_fill(
                                    updated,
                                    {column: 0.0 for column in numeric_columns},
                                )
                                updated = _coerce_schedule_frame(
                                    updated, schedule["columns"]
                                )
                                st.session_state[data_state_key] = updated.copy()
                                st.success("Copied the first year across later years.")
                        with helper_col2:
                            if st.button(
                                "Apply increases",
                                key=f"{editor_key}_apply_increase",
                                use_container_width=True,
                            ):
                                increments = {
                                    column: abs(rate_inputs.get(column) or 0.0)
                                    for column in numeric_columns
                                }
                                updated = _apply_incremental_fill(updated, increments)
                                updated = _coerce_schedule_frame(
                                    updated, schedule["columns"]
                                )
                                st.session_state[data_state_key] = updated.copy()
                                st.success("Applied annual increases to future years.")
                        with helper_col3:
                            if st.button(
                                "Apply decreases",
                                key=f"{editor_key}_apply_decrease",
                                use_container_width=True,
                            ):
                                decrements = {
                                    column: -abs(rate_inputs.get(column) or 0.0)
                                    for column in numeric_columns
                                }
                                updated = _apply_incremental_fill(updated, decrements)
                                updated = _coerce_schedule_frame(
                                    updated, schedule["columns"]
                                )
                                st.session_state[data_state_key] = updated.copy()
                                st.success("Applied annual decreases to future years.")

            active_edit = st.session_state.get(edit_state_key)
            if active_edit is not None and active_edit >= len(updated):
                st.session_state[edit_state_key] = None
                active_edit = None

            with actions_col:
                if updated.empty:
                    st.info("Add a line to begin editing this schedule.")
                else:
                    st.markdown("**Row actions**")
                    for idx, (_, row) in enumerate(updated.iterrows()):
                        summary_bits = []
                        year_value = row.get("Year") if isinstance(row, pd.Series) else None
                        if pd.notna(year_value):
                            try:
                                summary_bits.append(f"Year {int(float(year_value))}")
                            except (TypeError, ValueError):
                                summary_bits.append(f"Year {year_value}")
                        preview_cols = [col for col in schedule["columns"] if col != "Year"][:2]
                        for col in preview_cols:
                            value = row.get(col)
                            if pd.notna(value):
                                summary_bits.append(f"{col}: {value}")
                        summary_text = " | ".join(summary_bits) or f"Row {idx + 1}"
                        row_label_col, row_button_col = st.columns([1, 0.6], gap="small")
                        with row_label_col:
                            st.write(summary_text)
                        with row_button_col:
                            if st.button(
                                "Edit",
                                key=f"{editor_key}_edit_button_{idx}",
                                use_container_width=True,
                            ):
                                st.session_state[edit_state_key] = idx
                                active_edit = idx

                if active_edit is not None:
                    st.divider()
                    st.markdown(
                        f"**Editing {schedule['name']} – Row {active_edit + 1}**"
                    )
                    row_data = updated.iloc[active_edit].copy()
                    with st.form(f"{editor_key}_edit_form"):
                        new_values: Dict[str, Any] = {}
                        for column in schedule["columns"]:
                            current_value = row_data.get(column)
                            new_values[column] = st.text_input(
                                column,
                                value=_format_edit_value(current_value),
                                key=f"{editor_key}_edit_field_{active_edit}_{column}",
                            )
                        submit_col, cancel_col = st.columns(2)
                        submitted = submit_col.form_submit_button(
                            "Apply changes", use_container_width=True
                        )
                        cancelled = cancel_col.form_submit_button(
                            "Cancel", use_container_width=True, type="secondary"
                        )
                        if submitted:
                            for column, raw_value in new_values.items():
                                parsed_value = _parse_edit_value(column, raw_value)
                                updated.at[active_edit, column] = parsed_value
                            st.session_state[edit_state_key] = None
                            st.session_state[data_state_key] = updated.copy()
                            st.success("Row updated.")
                        elif cancelled:
                            st.session_state[edit_state_key] = None

            st.session_state[data_state_key] = updated.copy()
            updated_tables[schedule["name"]] = updated

        combined_years = collect_all_years(updated_tables)
        base_years = (
            combined_years
            if combined_years
            else (st.session_state.get("assumption_years") or current_years)
        )
        st.session_state["assumption_tables"] = sync_schedule_years(base_years, updated_tables)

        if st.button("Save assumptions", type="primary"):
            combined_df = combine_assumption_tables(st.session_state["assumption_tables"])
            if combined_df.empty:
                st.warning("Add at least one forecast year before saving assumptions.")
            else:
                payload = [
                    {k: (None if pd.isna(v) else v) for k, v in row.items()}
                    for row in combined_df.to_dict(orient="records")
                ]
                try:
                    with st.spinner("Saving assumptions..."):
                        response = api_post("/save_assumptions", json=payload)
                    st.success(response.get("message", "Assumptions saved."))
                    st.session_state["assumptions_raw"] = payload
                except RuntimeError as exc:
                    st.error(str(exc))

        with st.form("filter_form"):
            st.subheader("Filter time period & rebuild caches")
            scenario_type = st.selectbox("Scenario", SCENARIO_TYPES, key="filter_scenario")
            col1, col2, col3, col4 = st.columns(4)
            start_year = col1.number_input("Start year", value=2023, step=1)
            end_year = col2.number_input("End year", value=2028, step=1)
            discount_rate = col3.number_input("Discount rate", value=0.2, step=0.01)
            wacc = col4.number_input("WACC", value=0.1, step=0.01)
            col5, col6, col7 = st.columns(3)
            perpetual_growth = col5.number_input("Perpetual growth", value=0.02, step=0.01)
            tax_rate = col6.number_input("Tax rate", value=0.25, step=0.01)
            inflation_rate = col7.number_input("Inflation rate", value=0.02, step=0.01)
            direct_labor_rate = st.number_input(
                "Direct labor rate increase",
                value=0.03,
                step=0.01,
            )
            submitted = st.form_submit_button("Apply filter & refresh")
            if submitted:
                payload = {
                    "scenario_type": scenario_type,
                    "start_year": int(start_year),
                    "end_year": int(end_year),
                    "discount_rate": float(discount_rate),
                    "wacc": float(wacc),
                    "perpetual_growth": float(perpetual_growth),
                    "tax_rate": float(tax_rate),
                    "inflation_rate": float(inflation_rate),
                    "direct_labor_rate_increase": float(direct_labor_rate),
                }
                try:
                    with st.spinner("Applying filters..."):
                        response = api_post("/filter_time_period", json=payload)
                    st.success(response.get("message", "Time period updated."))
                except RuntimeError as exc:
                    st.error(str(exc))

        with st.form("base_analysis_form"):
            st.subheader("Run base analysis")
            col1, col2, col3 = st.columns(3)
            discount_rate = col1.number_input("Discount rate", value=20.0, step=1.0)
            wacc = col2.number_input("WACC", value=10.0, step=1.0)
            perpetual_growth = col3.number_input("Perpetual growth", value=2.0, step=0.5)
            col4, col5, col6 = st.columns(3)
            tax_rate = col4.number_input("Tax rate", value=0.0, step=0.5)
            inflation_rate = col5.number_input("Inflation rate", value=0.0, step=0.5)
            direct_labor_rate = col6.number_input(
                "Direct labor rate increase", value=0.0, step=0.5
            )
            forecast_years = st.number_input(
                "Forecast years", value=10, min_value=1, max_value=30, step=1
            )
            run_base = st.form_submit_button("Run analysis", type="primary")
            if run_base:
                payload = {
                    "discount_rate": float(discount_rate),
                    "wacc": float(wacc),
                    "perpetual_growth": float(perpetual_growth),
                    "tax_rate": float(tax_rate),
                    "inflation_rate": float(inflation_rate),
                    "direct_labor_rate_increase": float(direct_labor_rate),
                    "normal_forecast_years": int(forecast_years),
                }
                try:
                    with st.spinner("Running base analysis..."):
                        response = api_post("/run_base_analysis", json=payload)
                    st.session_state["base_analysis_result"] = response
                    st.success("Base analysis completed.")
                except RuntimeError as exc:
                    st.error(str(exc))

        if st.session_state.get("base_analysis_result"):
            st.subheader("Latest base analysis snapshot")
            st.json(st.session_state["base_analysis_result"])


def render_metrics_tab(tab: st.delta_generator.DeltaGenerator) -> None:
    with tab:
        st.header("Key financial metrics")
        st.write(
            "Review summary KPIs, operational metrics, valuation outputs, and scenario"
            " comparisons. Adjust scenario parameters to recalculate downstream tables."
        )

        if st.button("Refresh metrics", key="refresh_metrics") or "summary_metrics" not in st.session_state:
            try:
                with st.spinner("Fetching metrics..."):
                    st.session_state["summary_metrics"] = api_get("/display_metrics_summary_of_analysis")
                    st.session_state["operational_metrics"] = api_get("/operational_metrics").get("metrics", [])
                    st.session_state["valuation"] = api_get("/dcf_valuation")
                    st.session_state["scenario_metrics"] = api_get("/display_metrics_scenario_analysis").get("data")
                    st.session_state["implications"] = api_get("/key_implications")
                    charts_payload = api_get("/revenue_chart_data")
                    traffic_payload = api_get("/traffic_chart_data")
                    profitability_payload = api_get("/profitability_chart_data")
                st.session_state["revenue_chart"] = charts_payload.get("revenue_chart_data", charts_payload)
                st.session_state["traffic_chart"] = traffic_payload.get("traffic_chart_data", traffic_payload)
                st.session_state["profitability_chart"] = profitability_payload.get(
                    "profitability_chart_data", profitability_payload
                )
            except RuntimeError as exc:
                st.error(str(exc))

        render_table("Summary metrics", st.session_state.get("summary_metrics"))
        render_metric_cards(st.session_state.get("operational_metrics", []))

        valuation = st.session_state.get("valuation")
        if valuation:
            col1, col2 = st.columns(2)
            col1.metric("Enterprise Value ($M)", f"{valuation.get('enterprise_value_m', 0):,.1f}")
            col2.metric("Equity Value ($M)", f"{valuation.get('equity_value_m', 0):,.1f}")

        render_table("Scenario metrics", st.session_state.get("scenario_metrics"))

        implications = st.session_state.get("implications")
        if implications:
            st.subheader("Narrative implications")
            items = implications.get("implications") if isinstance(implications, dict) else implications
            if isinstance(items, list) and items:
                for item in items:
                    st.write(f"• {item}")
            elif isinstance(items, str):
                st.info(items)

        st.subheader("Scenario management")
        scenario_type = st.selectbox("Scenario type", SCENARIO_TYPES, key="scenario_type")
        if st.button("Load scenario defaults", key="load_scenario_defaults"):
            try:
                params = load_scenario_defaults_from_api(scenario_type)
                for key, value in params.items():
                    st.session_state[f"scenario_param_{key}"] = value
                st.success("Scenario defaults loaded.")
            except RuntimeError as exc:
                st.warning(str(exc))

        with st.form("scenario_form"):
            col1, col2, col3, col4 = st.columns(4)
            discount_rate = col1.number_input(
                "Discount rate", value=st.session_state.get("scenario_discount_rate", 0.2), step=0.01
            )
            tax_rate = col2.number_input(
                "Tax rate", value=st.session_state.get("scenario_tax_rate", 0.25), step=0.01
            )
            inflation_rate = col3.number_input(
                "Inflation rate", value=st.session_state.get("scenario_inflation", 0.02), step=0.01
            )
            direct_labor_rate = col4.number_input(
                "Direct labor increase", value=st.session_state.get("scenario_labor", 0.03), step=0.01
            )
            params_container: Dict[str, float] = {}
            st.markdown("**Scenario parameters**")
            for key, label in SCENARIO_PARAM_LABELS.items():
                default_value = st.session_state.get(
                    f"scenario_param_{key}",
                    SCENARIO_DEFAULTS.get(scenario_type, SCENARIO_DEFAULTS["Base Case"]).get(key, 1.0),
                )
                params_container[key] = st.number_input(
                    label,
                    value=float(default_value),
                    step=0.05,
                    key=f"scenario_param_input_{key}",
                )
            scenario_submit = st.form_submit_button("Recalculate scenario", type="primary")
            if scenario_submit:
                payload = {
                    "scenario_type": scenario_type,
                    "scenario_params": params_container,
                    "discount_rate": float(discount_rate),
                    "tax_rate": float(tax_rate),
                    "inflation_rate": float(inflation_rate),
                    "direct_labor_rate_increase": float(direct_labor_rate),
                }
                try:
                    with st.spinner("Updating scenario..."):
                        response = api_post("/select_scenario", json=payload)
                    st.success(response.get("message", "Scenario updated."))
                    st.session_state["scenario_discount_rate"] = float(discount_rate)
                    st.session_state["scenario_tax_rate"] = float(tax_rate)
                    st.session_state["scenario_inflation"] = float(inflation_rate)
                    st.session_state["scenario_labor"] = float(direct_labor_rate)
                    for key, value in params_container.items():
                        st.session_state[f"scenario_param_{key}"] = value
                except RuntimeError as exc:
                    st.error(str(exc))

        if st.session_state.get("revenue_chart"):
            col1, col2 = st.columns(2)
            col1.plotly_chart(
                build_revenue_figure(st.session_state["revenue_chart"]), use_container_width=True
            )
            col2.plotly_chart(
                build_traffic_figure(st.session_state["traffic_chart"]), use_container_width=True
            )
        if st.session_state.get("profitability_chart"):
            st.plotly_chart(
                build_profitability_figure(st.session_state["profitability_chart"]),
                use_container_width=True,
            )


def render_performance_tab(tab: st.delta_generator.DeltaGenerator) -> None:
    with tab:
        st.header("Financial performance dashboards")
        st.write("Visualize revenue drivers, breakeven analysis, waterfall bridges, and margin trends.")

        if st.button("Refresh performance visuals", key="refresh_performance") or "waterfall_chart" not in st.session_state:
            try:
                with st.spinner("Loading performance visuals..."):
                    st.session_state["waterfall_chart"] = api_get("/waterfall_chart_data")
                    st.session_state["breakeven_chart"] = api_get("/breakeven_chart_data")
                    st.session_state["consideration_chart"] = api_get("/consideration_chart_data")
                    st.session_state["margin_safety_chart"] = api_get("/margin_safety_chart")
                    st.session_state["margin_trends_chart"] = api_get("/profitability_margin_trends_chart_data")
            except RuntimeError as exc:
                st.error(str(exc))

        col1, col2 = st.columns(2)
        if st.session_state.get("waterfall_chart"):
            col1.plotly_chart(
                build_waterfall(st.session_state["waterfall_chart"], "Waterfall"),
                use_container_width=True,
            )
        if st.session_state.get("breakeven_chart"):
            breakeven = st.session_state["breakeven_chart"]
            traces = breakeven.get("traces") if isinstance(breakeven, dict) else None
            if traces:
                fig = go.Figure(traces)
                fig.update_layout(
                    title="Breakeven analysis",
                    yaxis=dict(title="Revenue", tickprefix="$", separatethousands=True),
                    legend=dict(orientation="h", y=-0.2),
                )
                col2.plotly_chart(fig, use_container_width=True)
        if st.session_state.get("consideration_chart"):
            consideration = st.session_state["consideration_chart"]
            traces = consideration.get("traces") if isinstance(consideration, dict) else None
            if traces:
                fig = go.Figure(traces)
                fig.update_layout(
                    title="Customer consideration funnel",
                    yaxis=dict(title="Rate", tickformat=".0%"),
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig, use_container_width=True)
        if st.session_state.get("margin_safety_chart"):
            margin_safety = st.session_state["margin_safety_chart"]
            traces = margin_safety.get("traces") if isinstance(margin_safety, dict) else None
            if traces:
                fig = go.Figure(traces)
                fig.update_layout(
                    title="Margin of safety",
                    yaxis=dict(title="Margin ($)", tickprefix="$", separatethousands=True),
                    yaxis2=dict(title="Margin %", overlaying="y", side="right", tickformat=".0%"),
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig, use_container_width=True)
        if st.session_state.get("margin_trends_chart"):
            margin_trends = st.session_state["margin_trends_chart"]
            traces = margin_trends.get("traces") if isinstance(margin_trends, dict) else None
            if traces:
                fig = go.Figure(traces)
                fig.update_layout(title="Margin trends", yaxis=dict(tickformat=".0%"))
                st.plotly_chart(fig, use_container_width=True)


def render_position_tab(tab: st.delta_generator.DeltaGenerator) -> None:
    with tab:
        st.header("Financial position & supporting schedules")
        st.write("Inspect balance sheet, capital assets, and debt amortization outputs.")

        if st.button("Load financial position", key="refresh_position") or "financial_position" not in st.session_state:
            try:
                with st.spinner("Loading financial position..."):
                    response = api_get(
                        "/financial_schedules",
                        params=[("schedules", item) for item in SCHEDULE_OPTIONS["Financial Position"]],
                    )
                st.session_state["financial_position"] = response.get("schedules", [])
            except RuntimeError as exc:
                st.error(str(exc))

        schedules = st.session_state.get("financial_position", [])
        if schedules:
            render_schedule_section("Financial Position", schedules)


def render_cashflow_tab(tab: st.delta_generator.DeltaGenerator) -> None:
    with tab:
        st.header("Cash flow and valuation")
        st.write("Review detailed cash flow statements and valuation bridges.")

        if st.button("Load cash flow", key="refresh_cashflow") or "cashflow_schedules" not in st.session_state:
            try:
                with st.spinner("Loading cash flow schedules..."):
                    response = api_get(
                        "/financial_schedules",
                        params=[("schedules", item) for item in SCHEDULE_OPTIONS["Cash Flow"]],
                    )
                    cashflow_payload = api_get("/cashflow_forecast_chart_data")
                    dcf_payload = api_get("/dcf_summary_chart_data")
                st.session_state["cashflow_schedules"] = response.get("schedules", [])
                st.session_state["cashflow_chart"] = cashflow_payload.get(
                    "cashflow_forecast_chart_data", cashflow_payload
                )
                st.session_state["dcf_chart"] = dcf_payload.get("data", dcf_payload)
            except RuntimeError as exc:
                st.error(str(exc))

        schedules = st.session_state.get("cashflow_schedules", [])
        if schedules:
            render_schedule_section("Cash Flow", schedules)
        if st.session_state.get("cashflow_chart"):
            st.plotly_chart(build_cashflow_figure(st.session_state["cashflow_chart"]), use_container_width=True)
        if st.session_state.get("dcf_chart"):
            st.plotly_chart(build_waterfall(st.session_state["dcf_chart"], "DCF summary"), use_container_width=True)


def render_sensitivity_tab(tab: st.delta_generator.DeltaGenerator) -> None:
    with tab:
        st.header("Sensitivity analysis & what-if tooling")
        st.write("Test key drivers, apply what-if adjustments, and run goal seek routines.")

        with st.form("top_rank_form"):
            st.subheader("Top-rank sensitivity")
            variables = st.multiselect(
                "Variables to test",
                options=SENSITIVITY_VARIABLES,
                default=[SENSITIVITY_VARIABLES[0]],
            )
            change_pct = st.slider("Change percentage", min_value=5.0, max_value=20.0, value=10.0, step=0.5)
            discount_rate = st.number_input("Discount rate", value=0.2, step=0.01)
            run_top_rank = st.form_submit_button("Run sensitivity", type="primary")
            if run_top_rank:
                payload = {
                    "variables_to_test": variables,
                    "change_percentage": float(change_pct),
                    "discount_rate": float(discount_rate),
                }
                try:
                    with st.spinner("Running sensitivity..."):
                        response = api_post("/top_rank_sensitivity", json=payload)
                    st.session_state["top_rank_results"] = response
                    st.success(response.get("message", "Sensitivity completed."))
                except RuntimeError as exc:
                    st.error(str(exc))

        results = st.session_state.get("top_rank_results")
        if results:
            render_table("Sensitivity results", results.get("sensitivity_results"))
            insights = results.get("sensitivity_insights")
            if insights:
                st.subheader("Insights")
                for item in insights:
                    st.write(f"• {item}")

        with st.form("what_if_form"):
            st.subheader("What-if adjustments")
            num_adjustments = st.number_input("Number of adjustments", min_value=1, max_value=5, value=1, step=1)
            adjustments: List[Dict[str, Any]] = []
            for idx in range(int(num_adjustments)):
                col1, col2, col3 = st.columns(3)
                year = col1.number_input(f"Year #{idx + 1}", value=2025, step=1, key=f"what_if_year_{idx}")
                variable = col2.selectbox(
                    f"Variable #{idx + 1}",
                    options=WHAT_IF_VARIABLES,
                    key=f"what_if_var_{idx}",
                )
                multiplier = col3.number_input(
                    f"Multiplier #{idx + 1}",
                    value=1.1,
                    step=0.05,
                    key=f"what_if_multiplier_{idx}",
                )
                adjustments.append({"year": int(year), "variable": variable, "multiplier": float(multiplier)})
            discount_rate = st.number_input("Discount rate", value=0.2, step=0.01)
            run_what_if = st.form_submit_button("Apply what-if")
            if run_what_if:
                payload = {
                    "num_adjustments": len(adjustments),
                    "adjustments": adjustments,
                    "discount_rate": float(discount_rate),
                }
                try:
                    with st.spinner("Applying what-if adjustments..."):
                        response = api_post("/what_if", json=payload)
                    st.session_state["what_if_results"] = response
                    st.success(response.get("message", "What-if complete."))
                except RuntimeError as exc:
                    st.error(str(exc))

        what_if_results = st.session_state.get("what_if_results")
        if what_if_results:
            render_table("What-if results", what_if_results.get("results"))
            warnings = what_if_results.get("warnings")
            if warnings:
                st.warning("\n".join(warnings))

        with st.form("goal_seek_form"):
            st.subheader("Goal seek")
            target_profit_margin = st.number_input("Target profit margin (decimal)", value=0.1, step=0.01)
            variable_to_adjust = st.selectbox("Driver to adjust", GOAL_SEEK_VARIABLES)
            year_to_adjust = st.number_input("Year", value=2025, step=1)
            max_iterations = st.number_input("Max iterations", value=100, min_value=1, step=1)
            tolerance = st.number_input("Tolerance", value=0.001, min_value=0.0001, step=0.0001, format="%.4f")
            discount_rate = st.number_input("Discount rate", value=0.1, step=0.01)
            run_goal_seek = st.form_submit_button("Run goal seek")
            if run_goal_seek:
                payload = {
                    "target_profit_margin": float(target_profit_margin),
                    "variable_to_adjust": variable_to_adjust,
                    "year_to_adjust": int(year_to_adjust),
                    "max_iterations": int(max_iterations),
                    "tolerance": float(tolerance),
                    "discount_rate": float(discount_rate),
                }
                try:
                    with st.spinner("Running goal seek..."):
                        response = api_post("/goal_seek", json=payload)
                    st.session_state["goal_seek_results"] = response
                    st.success(response.get("message", "Goal seek completed."))
                except RuntimeError as exc:
                    st.error(str(exc))

        goal_seek_results = st.session_state.get("goal_seek_results")
        if goal_seek_results:
            render_table("Goal seek results", goal_seek_results.get("results"))
            st.json(goal_seek_results)


def render_advanced_tab(tab: st.delta_generator.DeltaGenerator) -> None:
    with tab:
        st.header("Advanced analytics & exports")
        st.write(
            "Run Monte Carlo simulations, schedule risk, neural predictions, statistical"
            " forecasts, and budget optimizations. Download the consolidated Excel report"
            " for offline review."
        )

        if st.button("Download Excel report"):
            try:
                with st.spinner("Preparing Excel report..."):
                    content = api_get("/export_excel")
                st.download_button(
                    label="Save Excel report",
                    data=content,
                    file_name="ecommerce_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except RuntimeError as exc:
                st.error(str(exc))

        with st.form("monte_carlo_form"):
            st.subheader("Monte Carlo simulation")
            forecast_years = st.number_input("Forecast years", value=5, min_value=1, max_value=20, step=1)
            num_simulations = st.number_input("Number of simulations", value=1000, min_value=100, step=100)
            confidence_level = st.slider("Confidence level", min_value=80, max_value=99, value=90, step=1)
            distribution_type = st.selectbox(
                "Distribution type",
                [
                    "Normal",
                    "Lognormal",
                    "Uniform",
                    "Exponential",
                    "Binomial",
                    "Poisson",
                    "Geometric",
                    "Bernoulli",
                    "Chi-square",
                    "Gamma",
                    "Weibull",
                    "Hypergeometric",
                    "Multinomial",
                    "Beta",
                    "F-distribution",
                    "Discrete",
                    "Continuous",
                    "Cumulative",
                ],
            )
            discount_rate = st.number_input("Discount rate (%)", value=10.0, step=0.5)
            wacc = st.number_input("WACC (%)", value=10.0, step=0.5)
            perpetual_growth = st.number_input("Perpetual growth (%)", value=2.0, step=0.5)
            run_monte_carlo = st.form_submit_button("Run simulation")
            if run_monte_carlo:
                payload = {
                    "forecast_years": int(forecast_years),
                    "num_simulations": int(num_simulations),
                    "confidence_level": float(confidence_level),
                    "distribution_type": distribution_type,
                    "discount_rate": float(discount_rate),
                    "wacc": float(wacc),
                    "perpetual_growth": float(perpetual_growth),
                }
                try:
                    with st.spinner("Running Monte Carlo simulation..."):
                        response = api_post("/monte_carlo", json=payload)
                    st.session_state["monte_carlo"] = response
                    st.success("Monte Carlo simulation complete.")
                except RuntimeError as exc:
                    st.error(str(exc))

        monte_carlo = st.session_state.get("monte_carlo")
        if monte_carlo:
            st.subheader("Monte Carlo metrics")
            st.json(monte_carlo)

        with st.form("schedule_risk_form"):
            st.subheader("Schedule risk analysis")
            num_simulations = st.number_input("Number of simulations", value=1000, min_value=100, step=100)
            confidence_level = st.slider("Confidence level (%)", min_value=80, max_value=99, value=90, step=1)
            run_schedule_risk = st.form_submit_button("Run schedule risk")
            if run_schedule_risk:
                payload = {
                    "num_simulations": int(num_simulations),
                    "confidence_level": float(confidence_level),
                }
                try:
                    with st.spinner("Running schedule risk analysis..."):
                        response = api_post("/schedule_risk_analysis", json=payload)
                    st.session_state["schedule_risk"] = response
                    st.success("Schedule risk analysis complete.")
                except RuntimeError as exc:
                    st.error(str(exc))

        schedule_risk = st.session_state.get("schedule_risk")
        if schedule_risk:
            st.subheader("Schedule risk results")
            st.json(schedule_risk)

        with st.form("neural_tools_form"):
            st.subheader("Neural tools forecast")
            traffic_increase = st.number_input("Traffic increase (%)", value=5.0, step=0.5)
            run_neural = st.form_submit_button("Run neural prediction")
            if run_neural:
                payload = {"traffic_increase_percentage": float(traffic_increase)}
                try:
                    with st.spinner("Running neural prediction..."):
                        response = api_post("/neural_tools", json=payload)
                    st.session_state["neural_tools"] = response
                    st.success("Neural tools prediction ready.")
                except RuntimeError as exc:
                    st.error(str(exc))

        neural_tools = st.session_state.get("neural_tools")
        if neural_tools:
            st.subheader("Neural tools output")
            st.json(neural_tools)

        with st.form("stat_tools_form"):
            st.subheader("Statistical forecasting")
            forecast_years = st.number_input("Forecast years", value=5, min_value=1, max_value=20, step=1)
            confidence_level = st.slider("Confidence level", min_value=80, max_value=99, value=90, step=1, key="stat_confidence")
            run_stat = st.form_submit_button("Run forecasting")
            if run_stat:
                payload = {
                    "forecast_years": int(forecast_years),
                    "confidence_level": float(confidence_level),
                }
                try:
                    with st.spinner("Running statistical forecast..."):
                        response = api_post("/stat_tools_forecasting", json=payload)
                    st.session_state["stat_tools"] = response
                    st.success("Statistical forecasting ready.")
                except RuntimeError as exc:
                    st.error(str(exc))

        stat_tools = st.session_state.get("stat_tools")
        if stat_tools:
            st.subheader("Forecasting output")
            st.json(stat_tools)

        with st.form("evolver_form"):
            st.subheader("Budget optimization (Evolver)")
            budget_line = st.selectbox("Budget line", BUDGET_LINES)
            budget_value = st.number_input("Budget amount", value=100000.0, min_value=0.0, step=1000.0)
            forecast_years = st.number_input("Forecast years", value=5, min_value=1, max_value=20, step=1)
            run_evolver = st.form_submit_button("Run optimization")
            if run_evolver:
                payload = {
                    "budget_dict": {budget_line: float(budget_value)},
                    "forecast_years": int(forecast_years),
                }
                try:
                    with st.spinner("Running optimization..."):
                        response = api_post("/evolver_optimization", json=payload)
                    st.session_state["evolver"] = response
                    st.success("Optimization completed.")
                except RuntimeError as exc:
                    st.error(str(exc))

        evolver = st.session_state.get("evolver")
        if evolver:
            st.subheader("Optimization results")
            st.json(evolver)

        if st.button("Run precision tree analysis"):
            try:
                with st.spinner("Generating precision tree..."):
                    response = api_get("/precision_tree")
                st.session_state["precision_tree"] = response
                st.success("Precision tree generated.")
            except RuntimeError as exc:
                st.error(str(exc))

        precision_tree = st.session_state.get("precision_tree")
        if precision_tree:
            st.subheader("Precision tree output")
            st.json({k: v for k, v in precision_tree.items() if k != "decision_tree_image"})
            image_data = precision_tree.get("decision_tree_image")
            if image_data:
                try:
                    st.image(base64.b64decode(image_data), caption="Decision tree")
                except (ValueError, TypeError):
                    st.warning("Unable to decode decision tree image.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    configure_sidebar()
    (
        input_tab,
        metrics_tab,
        performance_tab,
        position_tab,
        cashflow_tab,
        sensitivity_tab,
        advanced_tab,
    ) = st.tabs(
        [
            "Input & Assumptions",
            "Key Financial Metrics",
            "Financial Performance",
            "Financial Position",
            "Cash Flow Statement",
            "Sensitivity Analysis",
            "Advanced Analysis",
        ]
    )

    render_input_tab(input_tab)
    render_metrics_tab(metrics_tab)
    render_performance_tab(performance_tab)
    render_position_tab(position_tab)
    render_cashflow_tab(cashflow_tab)
    render_sensitivity_tab(sensitivity_tab)
    render_advanced_tab(advanced_tab)


AUTORUN_ENV_VAR = "STREAMLIT_AUTORUN"


if __name__ == "__main__":
    if os.environ.get("STREAMLIT_SERVER_SCRIPT_PATH") or os.environ.get(AUTORUN_ENV_VAR) == "1":
        main()
    else:
        os.environ[AUTORUN_ENV_VAR] = "1"
        from streamlit.web import bootstrap

        script_path = Path(__file__).resolve()
        try:
            bootstrap.run(str(script_path), "", [], flag_options={})
        except RuntimeError as exc:  # pragma: no cover - defensive guard for hosted runners
            if "Runtime instance already exists" in str(exc):
                main()
            else:
                raise

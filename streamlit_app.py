"""Streamlit dashboard for manual ecommerce financial modeling."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import math

try:  # numpy removed np.irr in v2.0; prefer numpy-financial when available
    import numpy_financial as npf
except Exception:  # pragma: no cover - fallback when package is unavailable
    npf = None

# ---------------------------------------------------------------------------
# Streamlit configuration & constants
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ecommerce Financial Model",
    layout="wide",
    page_icon="💼",
    initial_sidebar_state="collapsed",
)


def inject_global_styles() -> None:
    """Remove the default Streamlit sidebar chrome so the layout stays centered."""

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none !important;}
        [data-testid="collapsedControl"] {display: none !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_global_styles()

DEFAULT_TAX_RATE = 0.25
DEFAULT_DISCOUNT_RATE = 0.10
CHANNEL_DEFINITIONS = [
    {
        "label": "Email",
        "traffic": "Email Traffic",
        "conversion": "Email Conversion Rate",
        "cpc": "Email Cost per Click",
    },
    {
        "label": "Organic Search",
        "traffic": "Organic Search Traffic",
        "conversion": "Organic Search Conversion Rate",
        "cpc": "Organic Search Cost per Click",
    },
    {
        "label": "Paid Search",
        "traffic": "Paid Search Traffic",
        "conversion": "Paid Search Conversion Rate",
        "cpc": "Paid Search Cost per Click",
    },
    {
        "label": "Affiliates",
        "traffic": "Affiliates Traffic",
        "conversion": "Affiliates Conversion Rate",
        "cpc": "Affiliates Cost per Click",
    },
]
SENSITIVITY_STEPS = [-0.1, -0.05, 0.05, 0.1]

DEFAULT_PRODUCTION_START_YEAR = 2023
DEFAULT_PRODUCTION_END_YEAR = 2027
PRODUCTION_YEAR_CHOICES = list(range(2000, 2101))


def default_production_years() -> List[int]:
    return list(range(DEFAULT_PRODUCTION_START_YEAR, DEFAULT_PRODUCTION_END_YEAR + 1))


def get_production_horizon() -> Tuple[int, int]:
    start_raw = st.session_state.get("production_start_year", DEFAULT_PRODUCTION_START_YEAR)
    end_raw = st.session_state.get("production_end_year", DEFAULT_PRODUCTION_END_YEAR)
    try:
        start_year = int(start_raw)
    except (TypeError, ValueError):
        start_year = DEFAULT_PRODUCTION_START_YEAR
    try:
        end_year = int(end_raw)
    except (TypeError, ValueError):
        end_year = DEFAULT_PRODUCTION_END_YEAR
    if end_year < start_year:
        end_year = start_year
    return start_year, end_year


def get_production_years() -> List[int]:
    start_year, end_year = get_production_horizon()
    return list(range(start_year, end_year + 1))


def set_production_horizon(start_year: int, end_year: int) -> None:
    try:
        start = int(start_year)
    except (TypeError, ValueError):
        start = DEFAULT_PRODUCTION_START_YEAR
    try:
        end = int(end_year)
    except (TypeError, ValueError):
        end = DEFAULT_PRODUCTION_END_YEAR
    if end < start:
        end = start
    years = list(range(start, end + 1))
    if not years:
        years = default_production_years()
    st.session_state["production_start_year"] = years[0]
    st.session_state["production_end_year"] = years[-1]
    st.session_state["assumption_years"] = years
    tables: Dict[str, pd.DataFrame] = st.session_state.get("assumption_tables", {})
    synced = sync_schedule_years(years, tables)
    st.session_state["assumption_tables"] = synced
    refresh_model_from_assumptions(synced)
EXECUTIVE_ROLES = [
    "CEO Salary",
    "COO Salary",
    "CFO Salary",
    "Director of HR Salary",
    "CIO Salary",
]
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

SCHEDULE_COLUMN_MAP: Dict[str, Sequence[str]] = {
    schedule["name"]: schedule["columns"] for schedule in ASSUMPTION_SCHEDULES
}


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


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    return value


def _table_signature_payload(name: str, frame: Any) -> List[Dict[str, Any]]:
    columns = list(SCHEDULE_COLUMN_MAP.get(name, []))
    if columns:
        coerced = _coerce_schedule_frame(frame, columns)
    elif isinstance(frame, pd.DataFrame):
        coerced = _coerce_schedule_frame(frame, list(frame.columns))
    else:
        coerced = pd.DataFrame(columns=columns)
    if coerced.empty:
        return []
    sanitized = coerced.applymap(_jsonable)
    return sanitized.to_dict(orient="records")


def _build_tables_signature(tables: Dict[str, pd.DataFrame]) -> str:
    all_names = sorted(set(tables.keys()) | set(SCHEDULE_COLUMN_MAP.keys()))
    payload: Dict[str, Any] = {}
    for name in all_names:
        payload[name] = _table_signature_payload(name, tables.get(name))
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


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
        start_year, end_year = get_production_horizon()
        if year_value < start_year:
            year_value = start_year
        if year_value > end_year:
            year_value = end_year
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


def ensure_assumption_tables() -> None:
    if "production_start_year" not in st.session_state:
        st.session_state["production_start_year"] = DEFAULT_PRODUCTION_START_YEAR
    if "production_end_year" not in st.session_state:
        st.session_state["production_end_year"] = DEFAULT_PRODUCTION_END_YEAR
    if "assumption_tables" not in st.session_state:
        years = get_production_years() or default_production_years()
        st.session_state["assumption_years"] = years
        initial_tables = {
            schedule["name"]: create_blank_schedule(schedule["columns"], years)
            for schedule in ASSUMPTION_SCHEDULES
        }
        st.session_state["assumption_tables"] = sync_schedule_years(years, initial_tables)
        refresh_model_from_assumptions(st.session_state["assumption_tables"])


def get_assumption_years() -> List[int]:
    years = st.session_state.get("assumption_years")
    normalized = _normalize_year_list(years or [])
    if normalized:
        return normalized
    production_years = get_production_years()
    st.session_state["assumption_years"] = production_years
    return production_years


def reset_assumption_tables(years: Optional[Iterable[int]] = None) -> None:
    if years is not None:
        base_years = _normalize_year_list(years)
    else:
        base_years = get_production_years()
    if not base_years:
        base_years = default_production_years()
    st.session_state["production_start_year"] = base_years[0]
    st.session_state["production_end_year"] = base_years[-1]
    st.session_state["assumption_years"] = base_years
    fresh_tables = {
        schedule["name"]: create_blank_schedule(schedule["columns"], base_years)
        for schedule in ASSUMPTION_SCHEDULES
    }
    st.session_state["assumption_tables"] = sync_schedule_years(base_years, fresh_tables)
    st.session_state["assumptions_raw"] = []
    st.session_state.pop("model_tables_signature", None)
    refresh_model_from_assumptions(st.session_state["assumption_tables"])


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


def refresh_model_from_assumptions(
    tables: Dict[str, pd.DataFrame], combined_df: Optional[pd.DataFrame] = None
) -> bool:
    if not isinstance(tables, dict):
        return False
    signature = _build_tables_signature(tables)
    existing_signature = st.session_state.get("model_tables_signature")
    if combined_df is None:
        combined_df = combine_assumption_tables(tables)
    if (
        signature == existing_signature
        and "model_results" in st.session_state
        and "assumptions_raw" in st.session_state
    ):
        return not combined_df.empty
    if combined_df.empty:
        st.session_state["assumptions_raw"] = []
        st.session_state["model_results"] = {}
        st.session_state["model_tables_signature"] = signature
        return False
    sanitized = combined_df.applymap(_jsonable)
    st.session_state["assumptions_raw"] = sanitized.to_dict(orient="records")
    st.session_state["model_results"] = compute_model_outputs(tables)
    st.session_state["model_tables_signature"] = signature
    return True


def set_assumptions_data(rows: List[Dict[str, Any]]) -> None:
    df = to_dataframe(rows)
    if not df.empty and "Year" in df.columns:
        years = _normalize_year_list(df["Year"].tolist())
    else:
        years = get_production_years()
    if not years:
        years = default_production_years()
    st.session_state["production_start_year"] = years[0]
    st.session_state["production_end_year"] = years[-1]
    st.session_state["assumptions_raw"] = rows
    st.session_state["assumption_years"] = years
    tables = split_assumptions_into_tables(df, years)
    st.session_state["assumption_tables"] = sync_schedule_years(years, tables)
    refresh_model_from_assumptions(st.session_state["assumption_tables"])


def _build_editor_key(name: str) -> str:
    return "assumptions_" + "".join(
        char.lower() if char.isalnum() else "_" for char in name
    )


def sync_schedule_years(
    base_years: List[int], tables: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    normalized_years = _normalize_year_list(base_years)
    if not normalized_years:
        normalized_years = default_production_years()
    min_year, max_year = normalized_years[0], normalized_years[-1]
    synced: Dict[str, pd.DataFrame] = {}
    for schedule in ASSUMPTION_SCHEDULES:
        columns = schedule["columns"]
        frame = tables.get(schedule["name"])
        working = _coerce_schedule_frame(frame, columns)
        if "Year" in working.columns:
            working["Year"] = pd.to_numeric(working["Year"], errors="coerce")
            working = working.dropna(subset=["Year"]).copy()
            working = working[(working["Year"] >= min_year) & (working["Year"] <= max_year)]
            working["Year"] = working["Year"].astype(int)
            existing_years = _normalize_year_list(working["Year"].tolist())
        else:
            existing_years = []
        missing_years = [year for year in normalized_years if year not in existing_years]
        if missing_years and "Year" in working.columns:
            blanks = pd.DataFrame(
                [{column: None for column in columns} for _ in missing_years],
                columns=columns,
            )
            blanks["Year"] = missing_years
            working = pd.concat([working, blanks], ignore_index=True)
        if "Year" in working.columns:
            working = working.sort_values("Year", kind="mergesort").reset_index(drop=True)
        synced[schedule["name"]] = working.reset_index(drop=True)
    st.session_state["assumption_years"] = normalized_years
    st.session_state["production_start_year"] = normalized_years[0]
    st.session_state["production_end_year"] = normalized_years[-1]
    return apply_derived_assumption_values(synced)


def to_number(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_decimal(value: Any, default: float = 0.0) -> float:
    number = to_number(value, None)
    if number is None:
        return default
    if abs(number) > 1:
        return number / 100.0
    return number


def sum_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    series = pd.to_numeric(frame[column], errors="coerce")
    return float(series.fillna(0.0).sum())


def avg_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return 0.0
    return float(series.mean())


def rows_for_year(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    if frame.empty or "Year" not in frame.columns:
        return pd.DataFrame(columns=frame.columns)
    return frame[frame["Year"] == year].copy()


def sanitize_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    sanitized: Dict[str, pd.DataFrame] = {}
    start_year, end_year = get_production_horizon()
    for schedule in ASSUMPTION_SCHEDULES:
        frame = tables.get(schedule["name"])
        if frame is None:
            frame = pd.DataFrame(columns=schedule["columns"])
        frame = _coerce_schedule_frame(frame, schedule["columns"])
        if "Year" in frame.columns:
            frame["Year"] = pd.to_numeric(frame["Year"], errors="coerce")
            frame = frame.dropna(subset=["Year"]).copy()
            frame = frame[(frame["Year"] >= start_year) & (frame["Year"] <= end_year)]
            frame["Year"] = frame["Year"].astype(int)
        sanitized[schedule["name"]] = frame
    return sanitized


def gather_years(tables: Dict[str, pd.DataFrame]) -> List[int]:
    years: Set[int] = set()
    for frame in tables.values():
        if "Year" in frame.columns:
            years.update(int(year) for year in frame["Year"] if pd.notna(year))
    return sorted(years)


def _sum_for_year(frame: Optional[pd.DataFrame], year: int, column: str) -> float:
    if frame is None or column not in frame.columns:
        return 0.0
    working = frame.copy()
    if "Year" in working.columns:
        years = pd.to_numeric(working["Year"], errors="coerce")
        working = working.loc[years == year]
    if working.empty:
        return 0.0
    values = pd.to_numeric(working[column], errors="coerce")
    return float(values.fillna(0.0).sum())


def _apply_staffing_totals(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    for prefix in ["Direct Staff", "Indirect Staff", "Part-Time Staff"]:
        hours_col = f"{prefix} Hours per Year"
        number_col = f"{prefix} Number"
        rate_col = f"{prefix} Hourly Rate"
        total_col = f"{prefix} Total Cost"
        required = [hours_col, number_col, rate_col, total_col]
        if not all(col in working.columns for col in required):
            continue
        hours = pd.to_numeric(working[hours_col], errors="coerce")
        headcount = pd.to_numeric(working[number_col], errors="coerce")
        rate = pd.to_numeric(working[rate_col], errors="coerce")
        total = hours * headcount * rate
        working[total_col] = total.round(2)
    return working


BENEFIT_TOTAL_COLUMNS = [
    ("Pension Cost per Staff", "Pension Total Cost"),
    ("Medical Insurance Cost per Staff", "Medical Insurance Total Cost"),
    ("Child Benefit Cost per Staff", "Child Benefit Total Cost"),
    ("Car Benefit Cost per Staff", "Car Benefit Total Cost"),
]


def _count_executives_for_year(executives: Optional[pd.DataFrame], year: int) -> float:
    if executives is None or executives.empty:
        return 0.0
    if "Year" in executives.columns:
        years = pd.to_numeric(executives["Year"], errors="coerce")
        filtered = executives.loc[years == year]
    else:
        filtered = executives.copy()
    if filtered.empty:
        return 0.0
    count = 0.0
    for column in EXECUTIVE_ROLES:
        if column not in filtered.columns:
            continue
        series = filtered[column]
        if series.notna().any():
            count += 1.0
    return count


def _apply_benefit_totals(
    benefits: pd.DataFrame,
    staffing: Optional[pd.DataFrame],
    executives: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if benefits.empty:
        return benefits
    working = benefits.copy()
    years = pd.to_numeric(working.get("Year"), errors="coerce") if "Year" in working else None
    for idx in working.index:
        year_value = None if years is None else years.iloc[idx]
        if year_value is None or pd.isna(year_value):
            total_staff = None
        else:
            year_int = int(year_value)
            direct = _sum_for_year(staffing, year_int, "Direct Staff Number")
            indirect = _sum_for_year(staffing, year_int, "Indirect Staff Number")
            exec_count = _count_executives_for_year(executives, year_int)
            total_staff = direct + indirect + exec_count
        for per_cost_col, total_col in BENEFIT_TOTAL_COLUMNS:
            if per_cost_col not in working.columns or total_col not in working.columns:
                continue
            per_cost = to_number(working.at[idx, per_cost_col], None)
            if per_cost is None or total_staff is None:
                working.at[idx, total_col] = None
            else:
                working.at[idx, total_col] = round(per_cost * total_staff, 2)
        totals: List[float] = []
        for _, total_col in BENEFIT_TOTAL_COLUMNS:
            if total_col not in working.columns:
                continue
            value = to_number(working.at[idx, total_col], None)
            if value is not None:
                totals.append(value)
        if "Total Benefits" in working.columns:
            working.at[idx, "Total Benefits"] = (
                round(sum(totals), 2) if totals else None
            )
    return working


def _normalize_rate(value: Any) -> Optional[float]:
    rate = to_number(value, None)
    if rate is None:
        return None
    if abs(rate) > 1:
        rate = rate / 100.0
    return rate


def _apply_asset_depreciation(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    required = [
        "Asset_1_Amount",
        "Asset_1_Rate",
        "Asset_1_Depreciation",
        "Asset_1_NBV",
    ]
    if not all(column in frame.columns for column in required):
        return frame
    working = frame.copy()

    def _label(idx: Any) -> str:
        if "Asset_1_Name" not in working.columns:
            return f"asset_{idx}"
        raw = working.at[idx, "Asset_1_Name"]
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return f"asset_{idx}"
        label = str(raw).strip()
        return label if label else f"asset_{idx}"

    def _year_key(idx: Any) -> float:
        if "Year" not in working.columns:
            return float(idx) if isinstance(idx, (int, float)) else 0.0
        raw_year = working.at[idx, "Year"]
        try:
            return float(int(float(raw_year)))
        except (TypeError, ValueError):
            return float("inf")

    ordered_indices = sorted(
        working.index,
        key=lambda idx: (_label(idx).lower(), _year_key(idx), idx),
    )

    cumulative_by_asset: Dict[str, float] = {}
    depreciation_results: Dict[Any, Optional[float]] = {}
    nbv_results: Dict[Any, Optional[float]] = {}

    for idx in ordered_indices:
        asset_label = _label(idx)
        previous_cumulative = cumulative_by_asset.get(asset_label, 0.0)

        amount = to_number(working.at[idx, "Asset_1_Amount"], None)
        rate = _normalize_rate(working.at[idx, "Asset_1_Rate"])

        depreciation: Optional[float]
        nbv: Optional[float]

        if amount is None:
            depreciation = None
            nbv = None
        else:
            if rate is None:
                depreciation = None
                nbv_value = amount - previous_cumulative
            else:
                depreciation = round(amount * rate, 2)
                nbv_value = amount - depreciation - previous_cumulative
            if nbv_value < 0:
                nbv_value = 0.0
            nbv = round(nbv_value, 2)

        depreciation_results[idx] = depreciation
        nbv_results[idx] = nbv

        if depreciation is not None:
            cumulative_by_asset[asset_label] = previous_cumulative + depreciation
        else:
            cumulative_by_asset[asset_label] = previous_cumulative

    for idx, value in depreciation_results.items():
        working.at[idx, "Asset_1_Depreciation"] = value
    for idx, value in nbv_results.items():
        working.at[idx, "Asset_1_NBV"] = value

    return working


def build_asset_schedule(
    assets: Optional[pd.DataFrame], years: Sequence[int]
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, float]]]:
    columns = [
        "Year",
        "Asset",
        "Beginning Balance",
        "Additions",
        "Depreciation",
        "Cumulative Depreciation",
        "Ending Balance",
        "Rate %",
    ]
    if assets is None or assets.empty:
        return pd.DataFrame(columns=columns), {}

    entries: List[Dict[str, Any]] = []
    for idx, row in assets.iterrows():
        year_raw = row.get("Year")
        try:
            year = int(float(year_raw)) if year_raw is not None else None
        except (TypeError, ValueError):
            year = None
        amount = to_number(row.get("Asset_1_Amount"), None)
        rate = _normalize_rate(row.get("Asset_1_Rate")) or 0.0
        if year is None or amount in (None, 0):
            continue
        name = str(row.get("Asset_1_Name") or "").strip()
        if not name:
            name = f"Asset {idx + 1}"
        entries.append(
            {
                "name": name,
                "start_year": year,
                "amount": float(amount),
                "rate": rate,
            }
        )

    if not entries:
        return pd.DataFrame(columns=columns), {}

    label_counts: Dict[str, int] = {}
    for entry in entries:
        base = entry["name"]
        label_counts[base] = label_counts.get(base, 0) + 1
        if label_counts[base] > 1:
            entry["label"] = f"{base} #{label_counts[base]}"
        else:
            entry["label"] = base

    observed_years: Set[int] = {
        int(year)
        for year in years
        if year is not None and not (isinstance(year, float) and pd.isna(year))
    }
    observed_years.update(entry["start_year"] for entry in entries)
    sorted_years = sorted(observed_years)

    asset_states: List[Dict[str, Any]] = [
        {
            "name": entry["name"],
            "label": entry.get("label", entry["name"]),
            "start_year": entry["start_year"],
            "amount": entry["amount"],
            "rate": entry["rate"],
            "balance": 0.0,
            "cumulative": 0.0,
        }
        for entry in entries
    ]

    schedule_rows: List[Dict[str, Any]] = []
    totals: Dict[int, Dict[str, float]] = {}

    for year in sorted_years:
        year_totals = {
            "beginning": 0.0,
            "additions": 0.0,
            "depreciation": 0.0,
            "cumulative": 0.0,
            "ending": 0.0,
        }
        active = False
        for state in asset_states:
            start_year = state["start_year"]
            if year < start_year and state["balance"] <= 0:
                continue
            beginning = state["balance"]
            addition = 0.0
            if year == start_year:
                addition = state["amount"]
            balance_before_depr = beginning + addition
            if balance_before_depr <= 0 and addition <= 0:
                continue
            rate = state["rate"]
            depreciation = balance_before_depr * rate
            if depreciation > balance_before_depr:
                depreciation = balance_before_depr
            ending = balance_before_depr - depreciation
            state["balance"] = ending
            state["cumulative"] = state.get("cumulative", 0.0) + depreciation
            schedule_rows.append(
                {
                    "Year": year,
                    "Asset": state["label"],
                    "Beginning Balance": round(beginning, 2),
                    "Additions": round(addition, 2),
                    "Depreciation": round(depreciation, 2),
                    "Cumulative Depreciation": round(state["cumulative"], 2),
                    "Ending Balance": round(ending, 2),
                    "Rate %": round(rate * 100.0, 4),
                }
            )
            year_totals["beginning"] += beginning
            year_totals["additions"] += addition
            year_totals["depreciation"] += depreciation
            year_totals["ending"] += ending
            active = True
        if active:
            year_totals["cumulative"] = round(
                sum(
                    state.get("cumulative", 0.0)
                    for state in asset_states
                    if year >= state.get("start_year", year)
                ),
                2,
            )
            totals[year] = {
                "beginning": round(year_totals["beginning"], 2),
                "additions": round(year_totals["additions"], 2),
                "depreciation": round(year_totals["depreciation"], 2),
                "cumulative": year_totals["cumulative"],
                "ending": round(year_totals["ending"], 2),
            }

    schedule_df = pd.DataFrame(schedule_rows, columns=columns)
    return schedule_df, totals


def apply_derived_assumption_values(
    tables: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    derived: Dict[str, pd.DataFrame] = {}
    for schedule in ASSUMPTION_SCHEDULES:
        frame = tables.get(schedule["name"])
        derived[schedule["name"]] = frame.copy() if isinstance(frame, pd.DataFrame) else frame
    staffing = derived.get("Staffing Levels")
    if isinstance(staffing, pd.DataFrame):
        derived["Staffing Levels"] = _apply_staffing_totals(staffing)
    executives = derived.get("Executive Compensation")
    benefits = derived.get("Employee Benefits")
    if isinstance(benefits, pd.DataFrame):
        derived["Employee Benefits"] = _apply_benefit_totals(
            benefits,
            derived.get("Staffing Levels"),
            executives if isinstance(executives, pd.DataFrame) else None,
        )
    assets = derived.get("Asset Register")
    if isinstance(assets, pd.DataFrame):
        derived["Asset Register"] = _apply_asset_depreciation(assets)
    return derived


def _calculate_staff_cost(frame: pd.DataFrame, prefix: str) -> float:
    total_column = f"{prefix} Total Cost"
    total_cost = sum_numeric(frame, total_column)
    if total_cost:
        return total_cost
    hours_column = f"{prefix} Hours per Year"
    rate_column = f"{prefix} Hourly Rate"
    hours = sum_numeric(frame, hours_column)
    rate = avg_numeric(frame, rate_column)
    return hours * rate


def _sum_numeric_columns(frame: pd.DataFrame) -> float:
    total = 0.0
    for column in frame.columns:
        if column == "Year":
            continue
        total += sum_numeric(frame, column)
    return total


def _npv(rate: float, cash_flows: Sequence[float]) -> float:
    return sum(cf / ((1 + rate) ** idx) for idx, cf in enumerate(cash_flows))


def _irr_newton(cash_flows: Sequence[float]) -> Optional[float]:
    rate = 0.1
    for _ in range(100):
        npv_value = _npv(rate, cash_flows)
        derivative = sum(
            -idx * cf / ((1 + rate) ** (idx + 1))
            for idx, cf in enumerate(cash_flows)
        )
        if abs(derivative) < 1e-12:
            return None
        next_rate = rate - npv_value / derivative
        if not np.isfinite(next_rate) or next_rate <= -0.9999:
            return None
        if abs(next_rate - rate) < 1e-6:
            return max(next_rate, -0.9999)
        rate = next_rate
    return None


def compute_irr_value(cash_flows: Sequence[float]) -> Optional[float]:
    valid_flows = [float(cf) for cf in cash_flows if pd.notna(cf)]
    if not valid_flows:
        return None
    has_positive = any(cf > 0 for cf in valid_flows)
    has_negative = any(cf < 0 for cf in valid_flows)
    if not (has_positive and has_negative):
        return None
    try:
        if npf is not None:
            irr = npf.irr(valid_flows)
        else:
            irr = _irr_newton(valid_flows)
        if isinstance(irr, np.ndarray):
            irr = irr.item()
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return None
    if irr is None:
        return None
    if isinstance(irr, complex):
        return None
    if np.isnan(irr):
        return None
    return irr


def build_discount_table(cash_flows: Sequence[float], rate: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, cash_flow in enumerate(cash_flows, start=1):
        discount_factor = 1 / ((1 + rate) ** idx) if rate is not None else 1.0
        discounted_value = cash_flow * discount_factor
        rows.append(
            {
                "Year": idx,
                "Net Cash Flow": cash_flow,
                "Discount Factor": discount_factor,
                "Discounted Cash Flow": discounted_value,
            }
        )
    return pd.DataFrame(rows)


def build_debt_amortization_schedule(
    debt_frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, float]]]:
    if debt_frame.empty:
        columns = [
            "Year",
            "Loan",
            "Beginning Balance",
            "Interest",
            "Principal",
            "Ending Balance",
            "Payment",
        ]
        return pd.DataFrame(columns=columns), {}

    schedule_rows: List[Dict[str, Any]] = []
    aggregates: Dict[int, Dict[str, float]] = {}

    for idx, row in debt_frame.iterrows():
        start_year = row.get("Year")
        amount = to_number(row.get("Debt_1_Amount"), None)
        rate = _normalize_rate(row.get("Debt_1_Interest_Rate")) or 0.0
        duration_raw = row.get("Debt_1_Duration")
        try:
            duration = int(float(duration_raw)) if duration_raw is not None else 0
        except (TypeError, ValueError):
            duration = 0

        if start_year is None or amount in (None, 0) or duration <= 0:
            continue

        loan_name = row.get("Debt_1_Name") or f"Debt {idx + 1}"
        balance = float(amount)

        if rate:
            payment = balance * rate / (1 - (1 + rate) ** (-duration))
        else:
            payment = balance / duration

        for period in range(duration):
            year = int(start_year) + period
            beginning_balance = balance
            interest = beginning_balance * rate if rate else 0.0
            principal = payment - interest if rate else payment

            if principal > balance or math.isclose(balance, principal, rel_tol=1e-9, abs_tol=1e-6):
                principal = balance
            ending_balance = balance - principal

            # Guard against floating noise
            if ending_balance < 1e-6:
                ending_balance = 0.0

            interest = round(interest, 2)
            principal = round(principal, 2)
            payment_value = round(interest + principal, 2)
            beginning_balance = round(beginning_balance, 2)
            ending_balance = round(ending_balance, 2)

            schedule_rows.append(
                {
                    "Year": year,
                    "Loan": str(loan_name),
                    "Beginning Balance": beginning_balance,
                    "Interest": interest,
                    "Principal": principal,
                    "Ending Balance": ending_balance,
                    "Payment": payment_value,
                }
            )

            year_totals = aggregates.setdefault(
                year,
                {
                    "beginning": 0.0,
                    "interest": 0.0,
                    "principal": 0.0,
                    "ending": 0.0,
                    "issued": 0.0,
                },
            )
            year_totals["beginning"] += beginning_balance
            year_totals["interest"] += interest
            year_totals["principal"] += principal
            year_totals["ending"] += ending_balance
            if period == 0:
                year_totals["issued"] += beginning_balance

            balance = ending_balance

    schedule_df = pd.DataFrame(schedule_rows)
    if schedule_df.empty:
        return schedule_df, aggregates

    schedule_df = schedule_df.sort_values(["Year", "Loan"]).reset_index(drop=True)

    totals = (
        schedule_df.groupby("Year")[
            ["Beginning Balance", "Interest", "Principal", "Ending Balance", "Payment"]
        ]
        .sum()
        .reset_index()
    )
    if not totals.empty:
        totals.insert(1, "Loan", "Total")
        schedule_df = pd.concat([schedule_df, totals], ignore_index=True)
        schedule_df = schedule_df.sort_values(["Year", "Loan"]).reset_index(drop=True)

    return schedule_df, aggregates


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator in (0, None) or (isinstance(denominator, float) and math.isclose(denominator, 0.0)):
        return 0.0
    return numerator / denominator


def monte_carlo_analysis(
    base_cashflows: Sequence[float],
    iterations: int,
    revenue_sigma: float,
    cost_sigma: float,
    discount_rate: float,
) -> pd.DataFrame:
    if not base_cashflows or iterations <= 0:
        return pd.DataFrame()
    base_array = np.array(list(base_cashflows), dtype=float)
    results = []
    for _ in range(iterations):
        revenue_factor = np.random.normal(1.0, revenue_sigma)
        cost_factor = np.random.normal(1.0, cost_sigma)
        simulated = base_array * revenue_factor - (base_array.clip(min=0) * (cost_factor - 1))
        discount = 1 / ((1 + discount_rate) ** np.arange(1, len(simulated) + 1))
        npv = float(np.sum(simulated * discount))
        cumulative = simulated.cumsum()
        results.append(
            {
                "NPV": npv,
                "Ending Cash": float(cumulative[-1]),
                "Min Cash": float(np.min(cumulative)),
            }
        )
    return pd.DataFrame(results)


def simple_forecast(series: Sequence[float], periods: int) -> List[float]:
    values = [float(val) for val in series if pd.notna(val)]
    if not values:
        return []
    if len(values) == 1:
        growth = 0.0
    else:
        growth_rates = [
            (values[idx] - values[idx - 1]) / values[idx - 1]
            for idx in range(1, len(values))
            if values[idx - 1]
        ]
        growth = np.mean(growth_rates) if growth_rates else 0.0
    last = values[-1]
    forecast = []
    for _ in range(periods):
        last = last * (1 + growth)
        forecast.append(last)
    return forecast


def goal_seek_margin(target_margin: float, base_revenue: float, base_margin: float) -> Tuple[float, float]:
    if base_revenue == 0:
        return 1.0, base_margin
    required_margin = target_margin
    if required_margin <= 0:
        return 1.0, base_margin
    multiplier = required_margin / base_margin if base_margin else 1.0
    return multiplier, base_margin * multiplier


def compute_model_outputs(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    sanitized = sanitize_tables(tables)
    debt_schedule = sanitized.get("Debt Schedule", pd.DataFrame())
    debt_amortization_df, debt_totals = build_debt_amortization_schedule(debt_schedule)

    years = gather_years(sanitized)
    if not years:
        years = get_assumption_years()
    if not debt_amortization_df.empty:
        amort_years = [
            int(year)
            for year in debt_amortization_df["Year"].tolist()
            if pd.notna(year)
        ]
        if amort_years:
            years = sorted(set(years) | set(amort_years))

    asset_schedule_df, asset_totals = build_asset_schedule(
        sanitized.get("Asset Register"), years
    )
    asset_rollforward_df = (
        pd.DataFrame(
            [
                {
                    "Year": year,
                    "Beginning Balance": values.get("beginning", 0.0),
                    "Additions": values.get("additions", 0.0),
                    "Depreciation": values.get("depreciation", 0.0),
                    "Cumulative Depreciation": values.get("cumulative", 0.0),
                    "Ending Balance": values.get("ending", 0.0),
                }
                for year, values in sorted(asset_totals.items())
            ]
        )
        if asset_totals
        else pd.DataFrame(
            columns=[
                "Year",
                "Beginning Balance",
                "Additions",
                "Depreciation",
                "Cumulative Depreciation",
                "Ending Balance",
            ]
        )
    )

    summary_rows: List[Dict[str, Any]] = []
    performance_rows: List[Dict[str, Any]] = []
    position_rows: List[Dict[str, Any]] = []
    cashflow_rows: List[Dict[str, Any]] = []

    revenue_matrix_rows: List[Dict[str, Any]] = []
    depreciation_rows: List[Dict[str, Any]] = []
    equity_debt_rows: List[Dict[str, Any]] = []
    customer_metrics_rows: List[Dict[str, Any]] = []
    operational_kpis_rows: List[Dict[str, Any]] = []

    traffic_rows: List[Dict[str, Any]] = []
    profitability_rows: List[Dict[str, Any]] = []

    contribution_rows: List[Dict[str, Any]] = []
    consideration_rows: List[Dict[str, Any]] = []

    cumulative_cash = 0.0
    fixed_assets = 0.0
    other_fixed_assets = 0.0
    prior_working_capital = 0.0
    retained_equity = 0.0

    for year in years:
        demand = rows_for_year(sanitized["Demand & Conversion"], year)
        pricing = rows_for_year(sanitized["Pricing & Order Economics"], year)
        acquisition = rows_for_year(sanitized["Acquisition Costs"], year)
        fulfillment = rows_for_year(sanitized["Fulfillment & Operating Costs"], year)
        staffing = rows_for_year(sanitized["Staffing Levels"], year)
        executives = rows_for_year(sanitized["Executive Compensation"], year)
        benefits = rows_for_year(sanitized["Employee Benefits"], year)
        overheads = rows_for_year(sanitized["Overheads & Fees"], year)
        working_capital = rows_for_year(sanitized["Working Capital"], year)
        capital = rows_for_year(sanitized["Capital Investments"], year)
        financing = rows_for_year(sanitized["Financing Activities"], year)
        property_df = rows_for_year(sanitized["Property Portfolio"], year)
        legal = rows_for_year(sanitized["Legal & Compliance"], year)
        assets = rows_for_year(sanitized["Asset Register"], year)
        debt_activity = debt_totals.get(year, {})
        asset_activity = asset_totals.get(
            year,
            {
                "beginning": 0.0,
                "additions": 0.0,
                "depreciation": 0.0,
                "cumulative": 0.0,
                "ending": 0.0,
            },
        )
        asset_additions = asset_activity.get("additions", 0.0) or 0.0
        asset_depreciation = asset_activity.get("depreciation", 0.0) or 0.0
        asset_cumulative_depr = asset_activity.get("cumulative", 0.0) or 0.0
        asset_ending_balance = asset_activity.get("ending", 0.0) or 0.0

        total_traffic = 0.0
        total_orders = 0.0
        marketing_spend = 0.0
        for channel in CHANNEL_DEFINITIONS:
            traffic = sum_numeric(demand, channel["traffic"])
            conversion = to_decimal(avg_numeric(demand, channel["conversion"]))
            total_traffic += traffic
            total_orders += traffic * conversion
            cpc = avg_numeric(acquisition, channel["cpc"])
            marketing_spend += traffic * cpc
            consideration_rows.append(
                {
                    "Year": year,
                    "Channel": channel["label"],
                    "Traffic": traffic,
                    "Conversion Rate": conversion,
                    "Cost per Click": cpc,
                }
            )

        churn_rate = to_decimal(avg_numeric(demand, "Churn Rate"))
        avg_item_value = avg_numeric(pricing, "Average Item Value")
        items_per_order = max(avg_numeric(pricing, "Number of Items per Order"), 1.0)
        markdown = to_decimal(avg_numeric(pricing, "Average Markdown"))
        promotions = to_decimal(avg_numeric(pricing, "Average Promotion/Discount"))
        cogs_pct = to_decimal(avg_numeric(pricing, "COGS Percentage"))

        gross_order_value = total_orders * items_per_order * avg_item_value
        net_revenue = gross_order_value * (1 - markdown) * (1 - promotions)
        cogs = net_revenue * cogs_pct
        gross_profit = net_revenue - cogs

        freight = avg_numeric(fulfillment, "Freight/Shipping per Order")
        labor_per_order = avg_numeric(fulfillment, "Labor/Handling per Order")
        per_order_cost = freight + labor_per_order
        fulfillment_cost = per_order_cost * total_orders
        warehouse_rent = sum_numeric(fulfillment, "General Warehouse Rent")
        other_operating = sum_numeric(fulfillment, "Other")
        interest_expense = sum_numeric(fulfillment, "Interest")
        tax_rate_override = to_decimal(avg_numeric(fulfillment, "Tax Rate"), DEFAULT_TAX_RATE)

        staffing_cost = (
            _calculate_staff_cost(staffing, "Direct Staff")
            + _calculate_staff_cost(staffing, "Indirect Staff")
            + _calculate_staff_cost(staffing, "Part-Time Staff")
        )
        executive_comp = sum(
            sum_numeric(executives, column)
            for column in executives.columns
            if column != "Year"
        )
        benefits_total = sum_numeric(benefits, "Total Benefits")
        if not benefits_total:
            benefit_columns = [
                "Pension Total Cost",
                "Medical Insurance Total Cost",
                "Child Benefit Total Cost",
                "Car Benefit Total Cost",
            ]
            benefits_total = sum(sum_numeric(benefits, col) for col in benefit_columns)

        overhead_salaries = sum_numeric(overheads, "Salaries, Wages & Benefits")
        office_rent = sum_numeric(overheads, "Office Rent")
        professional_fees = sum_numeric(overheads, "Professional Fees")
        overhead_depreciation = sum_numeric(overheads, "Depreciation")
        depreciation = overhead_depreciation + asset_depreciation

        property_cost = _sum_numeric_columns(property_df)
        legal_cost = _sum_numeric_columns(legal)

        operating_expenses = (
            marketing_spend
            + fulfillment_cost
            + warehouse_rent
            + other_operating
            + staffing_cost
            + executive_comp
            + benefits_total
            + overhead_salaries
            + office_rent
            + professional_fees
            + property_cost
            + legal_cost
        )

        ebitda = gross_profit - operating_expenses
        ebit = ebitda - depreciation

        interest_rate_input = to_decimal(avg_numeric(financing, "Interest Rate"))
        schedule_interest = debt_activity.get("interest", 0.0)
        schedule_beginning = debt_activity.get("beginning", 0.0)
        schedule_principal = debt_activity.get("principal", 0.0)
        debt_balance = debt_activity.get("ending", 0.0)
        schedule_issued = debt_activity.get("issued", 0.0)

        if schedule_beginning:
            effective_interest_rate = _safe_ratio(schedule_interest, schedule_beginning)
        else:
            effective_interest_rate = interest_rate_input

        interest_total = interest_expense + schedule_interest
        if not schedule_interest and schedule_beginning and interest_rate_input:
            interest_total += schedule_beginning * interest_rate_input

        pre_tax_income = ebit - interest_total
        tax_rate = tax_rate_override if tax_rate_override else DEFAULT_TAX_RATE
        taxes = max(pre_tax_income, 0.0) * tax_rate
        net_income = pre_tax_income - taxes

        receivable_days = avg_numeric(working_capital, "Accounts Receivable Days")
        inventory_days = avg_numeric(working_capital, "Inventory Days")
        payable_days = avg_numeric(working_capital, "Accounts Payable Days")
        receivables = net_revenue / 365.0 * receivable_days
        inventory_value = cogs / 365.0 * inventory_days
        payables = cogs / 365.0 * payable_days
        current_working_capital = receivables + inventory_value - payables
        delta_working_capital = current_working_capital - prior_working_capital
        prior_working_capital = current_working_capital

        tech_capex = sum_numeric(capital, "Technology Development")
        equipment_capex = sum_numeric(capital, "Office Equipment")
        other_capex = tech_capex + equipment_capex
        capex = other_capex + asset_additions
        other_fixed_assets = max(
            other_fixed_assets + other_capex - overhead_depreciation, 0.0
        )
        equity_raised = sum_numeric(financing, "Equity Raised")
        dividends = sum_numeric(financing, "Dividends Paid")
        debt_issued = sum_numeric(financing, "Debt Issued") + schedule_issued

        cfo = net_income + depreciation - delta_working_capital
        cfi = -capex
        cff = equity_raised + debt_issued - dividends - schedule_principal
        net_cash_flow = cfo + cfi + cff
        cumulative_cash += net_cash_flow
        fixed_assets = max(asset_ending_balance, 0.0) + other_fixed_assets
        retained_equity += net_income + equity_raised - dividends

        total_assets = cumulative_cash + receivables + inventory_value + fixed_assets
        total_liabilities = payables + debt_balance
        equity_value = total_assets - total_liabilities

        gross_margin_pct = (gross_profit / net_revenue * 100.0) if net_revenue else 0.0
        ebitda_margin_pct = (ebitda / net_revenue * 100.0) if net_revenue else 0.0
        net_margin_pct = (net_income / net_revenue * 100.0) if net_revenue else 0.0
        operating_costs = (
            marketing_spend
            + fulfillment_cost
            + warehouse_rent
            + other_operating
            + staffing_cost
            + executive_comp
            + benefits_total
            + overhead_salaries
            + office_rent
            + professional_fees
            + property_cost
            + legal_cost
        )
        contribution_margin = net_revenue - cogs - marketing_spend - fulfillment_cost
        contribution_margin_pct = (
            (contribution_margin / net_revenue * 100.0) if net_revenue else 0.0
        )
        customer_count = total_orders if total_orders else 0.0
        cac = (marketing_spend / customer_count) if customer_count else 0.0
        contribution_per_order = (
            (contribution_margin / customer_count) if customer_count else 0.0
        )
        payback_months = (
            (cac / contribution_per_order * 12.0)
            if contribution_per_order
            else None
        )
        burn_rate = max(-(cfo) / 12.0, 0.0)
        ltv = (
            (gross_profit / customer_count) if customer_count else 0.0
        )

        summary_rows.append(
            {
                "Year": year,
                "Net Revenue": net_revenue,
                "Gross Profit": gross_profit,
                "EBITDA": ebitda,
                "Net Income": net_income,
                "Gross Margin %": gross_margin_pct,
                "EBITDA Margin %": ebitda_margin_pct,
                "Net Margin %": net_margin_pct,
                "Total Orders": total_orders,
            }
        )

        revenue_matrix_rows.append(
            {
                "Year": year,
                "Total Traffic": total_traffic,
                "Total Orders": total_orders,
                "Gross Order Value": gross_order_value,
                "Net Revenue": net_revenue,
                "COGS": cogs,
                "Gross Profit": gross_profit,
                "Contribution Margin %": contribution_margin_pct,
            }
        )

        performance_rows.append(
            {
                "Year": year,
                "Total Traffic": total_traffic,
                "Total Orders": total_orders,
                "Churn Rate": churn_rate,
                "Average Item Value": avg_item_value,
                "Items per Order": items_per_order,
                "Marketing Spend": marketing_spend,
                "Fulfillment Cost": fulfillment_cost + warehouse_rent + other_operating,
                "Staffing Cost": staffing_cost + executive_comp,
                "Benefits": benefits_total,
                "Overheads": overhead_salaries + office_rent + professional_fees,
                "Property & Legal": property_cost + legal_cost,
            }
        )

        depreciation_rows.append(
            {
                "Year": year,
                "Technology Development": tech_capex,
                "Office Equipment": equipment_capex,
                "Asset Additions": asset_additions,
                "Asset Depreciation": asset_depreciation,
                "Cumulative Asset Depreciation": asset_cumulative_depr,
                "Depreciation Expense": depreciation,
                "Net New Capex": capex,
                "Net Book Value": fixed_assets,
            }
        )

        position_rows.append(
            {
                "Year": year,
                "Cash": cumulative_cash,
                "Accounts Receivable": receivables,
                "Inventory": inventory_value,
                "Fixed Assets": fixed_assets,
                "Accounts Payable": payables,
                "Debt": debt_balance,
                "Equity": equity_value,
            }
        )

        equity_debt_rows.append(
            {
                "Year": year,
                "Equity Raised": equity_raised,
                "Retained Earnings": retained_equity,
                "Dividends Paid": dividends,
                "Debt Issued": debt_issued,
                "Debt Balance": debt_balance,
                "Principal Repaid": schedule_principal,
                "Interest Rate %": (effective_interest_rate or 0.0) * 100,
                "Equity Value": equity_value,
            }
        )

        cashflow_rows.append(
            {
                "Year": year,
                "Cash Flow from Operations": cfo,
                "Cash Flow from Investing": cfi,
                "Cash Flow from Financing": cff,
                "Net Cash Flow": net_cash_flow,
                "Ending Cash": cumulative_cash,
            }
        )

        customer_metrics_rows.append(
            {
                "Year": year,
                "Customers": customer_count,
                "CAC": cac,
                "LTV": ltv,
                "LTV/CAC": _safe_ratio(ltv, cac) if cac else 0.0,
                "Payback Months": payback_months,
                "Monthly Burn": burn_rate,
                "Customer IRR": (
                    (contribution_margin - marketing_spend) / marketing_spend
                    if marketing_spend
                    else 0.0
                ),
            }
        )

        contribution_rows.append(
            {
                "Year": year,
                "Revenue": net_revenue,
                "COGS": -cogs,
                "Marketing": -marketing_spend,
                "Fulfillment": -fulfillment_cost,
                "Operating Expenses": -(
                    staffing_cost
                    + executive_comp
                    + benefits_total
                    + overhead_salaries
                    + office_rent
                    + professional_fees
                    + property_cost
                    + legal_cost
                ),
                "EBITDA": ebitda,
                "Depreciation": -depreciation,
                "Interest": -interest_total,
                "Taxes": -taxes,
                "Net Income": net_income,
            }
        )

        previous_summary = summary_rows[-2] if len(summary_rows) > 1 else None
        revenue_growth = 0.0
        if previous_summary:
            revenue_growth = (
                _safe_ratio(
                    net_revenue - previous_summary["Net Revenue"],
                    previous_summary["Net Revenue"],
                )
                * 100.0
            )
        total_assets = cumulative_cash + receivables + inventory_value + fixed_assets
        operational_kpis_rows.append(
            {
                "Year": year,
                "Revenue Growth %": revenue_growth,
                "Gross Margin %": gross_margin_pct,
                "EBITDA Margin %": ebitda_margin_pct,
                "Net Margin %": net_margin_pct,
                "ROE %": (_safe_ratio(net_income, equity_value) * 100.0) if equity_value else 0.0,
                "Asset Turnover": _safe_ratio(net_revenue, total_assets),
            }
        )

        traffic_rows.append(
            {
                "Year": year,
                "Traffic": total_traffic,
                "Orders": total_orders,
            }
        )

        profitability_rows.append(
            {
                "Year": year,
                "EBITDA": ebitda,
                "Net Income": net_income,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    performance_df = pd.DataFrame(performance_rows)
    position_df = pd.DataFrame(position_rows)
    cashflow_df = pd.DataFrame(cashflow_rows)
    revenue_matrix = pd.DataFrame(revenue_matrix_rows)
    depreciation_df = pd.DataFrame(depreciation_rows)
    equity_debt_df = pd.DataFrame(equity_debt_rows)
    customer_metrics_df = pd.DataFrame(customer_metrics_rows)
    operational_kpis_df = pd.DataFrame(operational_kpis_rows)
    contribution_df = pd.DataFrame(contribution_rows)

    metrics_cards: List[Dict[str, Any]] = []
    if not summary_df.empty:
        latest = summary_df.iloc[-1]
        metrics_cards = [
            {"metric": "Net Revenue", "current": latest["Net Revenue"]},
            {"metric": "Gross Profit", "current": latest["Gross Profit"]},
            {"metric": "EBITDA", "current": latest["EBITDA"]},
            {"metric": "Net Income", "current": latest["Net Income"]},
        ]

    traffic_df = pd.DataFrame(traffic_rows)
    profitability_df = pd.DataFrame(profitability_rows)
    consideration_df = pd.DataFrame(consideration_rows)

    cash_flows = cashflow_df["Net Cash Flow"].to_list() if not cashflow_df.empty else []
    discount_rate = to_decimal(
        avg_numeric(sanitized["Financing Activities"], "Interest Rate"), DEFAULT_DISCOUNT_RATE
    )
    discounted = [
        cf / ((1 + discount_rate) ** idx) for idx, cf in enumerate(cash_flows, start=1)
    ] if discount_rate is not None and cash_flows else []
    npv = sum(discounted) if discounted else 0.0
    irr_value = compute_irr_value(cash_flows)
    discount_table = build_discount_table(cash_flows, discount_rate if discount_rate else 0.0)
    cumulative = cashflow_df["Ending Cash"].tolist() if not cashflow_df.empty else []
    payback_period = None
    if cumulative:
        for idx, balance in enumerate(cumulative, start=1):
            if balance >= 0:
                payback_period = years[idx - 1]
                break

    sensitivity_data: List[Dict[str, Any]] = []
    if not summary_df.empty:
        base_net_income = summary_df.iloc[-1]["Net Income"]
        base_revenue = summary_df.iloc[-1]["Net Revenue"]
        for step in SENSITIVITY_STEPS:
            sensitivity_data.append(
                {
                    "Adjustment": f"Revenue {int(step * 100)}%",
                    "Projected Net Income": base_net_income * (1 + step),
                }
            )
            sensitivity_data.append(
                {
                    "Adjustment": f"Orders {int(step * 100)}%",
                    "Projected Net Revenue": base_revenue * (1 + step),
                }
            )

    scenario_summary = pd.DataFrame()
    if not summary_df.empty:
        base_row = summary_df.iloc[-1]
        scenario_rows: List[Dict[str, Any]] = []
        base_cashflows = cash_flows or []

        def scenario_entry(name: str, revenue_factor: float, margin_shift: float) -> Dict[str, Any]:
            revenue = base_row["Net Revenue"] * revenue_factor
            gross_margin = max((base_row["Gross Margin %"] / 100.0) + margin_shift, 0.0)
            ebitda_margin = max((base_row["EBITDA Margin %"] / 100.0) + margin_shift / 2, 0.0)
            net_margin = max((base_row["Net Margin %"] / 100.0) + margin_shift / 2, 0.0)
            gross_profit_val = revenue * gross_margin
            ebitda_val = revenue * ebitda_margin
            net_income_val = revenue * net_margin
            scaled_cashflows = [cf * revenue_factor for cf in base_cashflows]
            irr = compute_irr_value(scaled_cashflows)
            dcf = build_discount_table(
                scaled_cashflows,
                discount_rate if discount_rate else DEFAULT_DISCOUNT_RATE,
            )
            scenario_npv = dcf["Discounted Cash Flow"].sum() if not dcf.empty else 0.0
            return {
                "Scenario": name,
                "Net Revenue": revenue,
                "Gross Profit": gross_profit_val,
                "EBITDA": ebitda_val,
                "Net Income": net_income_val,
                "NPV": scenario_npv,
                "IRR": irr,
            }

        scenario_rows.append(scenario_entry("Base Case", 1.0, 0.0))
        scenario_rows.append(scenario_entry("Best Case", 1.1, 0.02))
        scenario_rows.append(scenario_entry("Worst Case", 0.9, -0.02))
        scenario_summary = pd.DataFrame(scenario_rows)

    breakeven_details: Dict[str, Any] = {}
    margin_of_safety: Dict[str, Any] = {}
    if not contribution_df.empty:
        latest_contrib = contribution_df.iloc[-1]
        revenue = latest_contrib["Revenue"]
        variable_costs = -(
            latest_contrib["COGS"]
            + latest_contrib["Marketing"]
            + latest_contrib["Fulfillment"]
        )
        contribution_margin_pct = (
            (revenue - variable_costs) / revenue if revenue else 0.0
        )
        fixed_costs = -latest_contrib["Operating Expenses"]
        breakeven_revenue = (
            fixed_costs / contribution_margin_pct if contribution_margin_pct else 0.0
        )
        breakeven_details = {
            "Contribution Margin %": contribution_margin_pct * 100.0,
            "Fixed Costs": fixed_costs,
            "Break-even Revenue": breakeven_revenue,
        }
        margin_of_safety = {
            "Actual Revenue": revenue,
            "Break-even Revenue": breakeven_revenue,
            "Margin of Safety %": (
                (revenue - breakeven_revenue) / revenue * 100.0 if revenue else 0.0
            ),
        }

    chart_payloads = {
        "revenue": {
            "years": summary_df["Year"].tolist() if not summary_df.empty else [],
            "net_revenue": summary_df["Net Revenue"].tolist() if not summary_df.empty else [],
            "gross_margin": (summary_df["Gross Margin %"].tolist() if not summary_df.empty else []),
            "ebitda_margin": (
                summary_df["EBITDA Margin %"].tolist() if not summary_df.empty else []
            ),
        },
        "traffic": {
            "years": traffic_df["Year"].tolist() if not traffic_df.empty else [],
            "traffic": traffic_df["Traffic"].tolist() if not traffic_df.empty else [],
            "orders": traffic_df["Orders"].tolist() if not traffic_df.empty else [],
        },
        "profitability": {
            "years": profitability_df["Year"].tolist() if not profitability_df.empty else [],
            "ebitda": profitability_df["EBITDA"].tolist() if not profitability_df.empty else [],
            "net_income": profitability_df["Net Income"].tolist() if not profitability_df.empty else [],
            "cash": cashflow_df["Ending Cash"].tolist() if not cashflow_df.empty else [],
        },
        "cashflow": {
            "years": cashflow_df["Year"].tolist() if not cashflow_df.empty else [],
            "operations": cashflow_df["Cash Flow from Operations"].tolist()
            if not cashflow_df.empty
            else [],
            "investing": cashflow_df["Cash Flow from Investing"].tolist()
            if not cashflow_df.empty
            else [],
            "financing": cashflow_df["Cash Flow from Financing"].tolist()
            if not cashflow_df.empty
            else [],
        },
        "margin_trend": operational_kpis_df[["Year", "Gross Margin %", "EBITDA Margin %", "Net Margin %"]]
        if not operational_kpis_df.empty
        else pd.DataFrame(),
        "waterfall": contribution_df.iloc[-1] if not contribution_df.empty else pd.Series(),
        "breakeven": breakeven_details,
        "margin_of_safety": margin_of_safety,
        "valuation": discount_table,
        "customer_consideration": consideration_df,
        "dcf_summary": discount_table,
    }

    income_statement = summary_df.merge(
        performance_df[
            [
                "Year",
                "Marketing Spend",
                "Fulfillment Cost",
                "Staffing Cost",
                "Benefits",
                "Overheads",
                "Property & Legal",
            ]
        ],
        on="Year",
        how="left",
    ) if not summary_df.empty else pd.DataFrame()

    liquidity_rows: List[Dict[str, Any]] = []
    if not position_df.empty:
        for _, row in position_df.iterrows():
            current_assets = row["Cash"] + row["Accounts Receivable"] + row["Inventory"]
            current_liabilities = row["Accounts Payable"] + row["Debt"]
            liquidity_rows.append(
                {
                    "Year": int(row["Year"]),
                    "Current Ratio": _safe_ratio(current_assets, current_liabilities),
                    "Quick Ratio": _safe_ratio(
                        row["Cash"] + row["Accounts Receivable"], current_liabilities
                    ),
                    "Debt to Equity": _safe_ratio(row["Debt"], row["Equity"]),
                }
            )
    liquidity_df = pd.DataFrame(liquidity_rows)

    valuation_table = discount_table.copy()
    if not valuation_table.empty and not summary_df.empty:
        valuation_table.insert(0, "Year Label", summary_df["Year"].tolist()[: len(valuation_table)])

    top_sensitivity_rows: List[Dict[str, Any]] = []
    if not scenario_summary.empty:
        base_metrics = scenario_summary.iloc[0]
        for idx in range(1, len(scenario_summary)):
            scenario = scenario_summary.iloc[idx]
            top_sensitivity_rows.append(
                {
                    "Scenario": scenario["Scenario"],
                    "Net Income Delta": scenario["Net Income"] - base_metrics["Net Income"],
                    "EBITDA Delta": scenario["EBITDA"] - base_metrics["EBITDA"],
                    "Equity Delta": scenario["NPV"] - scenario_summary.iloc[0]["NPV"],
                }
            )
    top_sensitivity_df = pd.DataFrame(top_sensitivity_rows)

    return {
        "summary": summary_df,
        "performance": performance_df,
        "position": position_df,
        "cashflow": cashflow_df,
        "income_statement": income_statement,
        "liquidity": liquidity_df,
        "revenue_matrix": revenue_matrix,
        "depreciation_matrix": depreciation_df,
        "equity_debt_matrix": equity_debt_df,
        "scenario_summary": scenario_summary,
        "operational_kpis": operational_kpis_df,
        "customer_metrics": customer_metrics_df,
        "valuation_table": valuation_table,
        "chart_payloads": chart_payloads,
        "debt_amortization": debt_amortization_df,
        "asset_schedule": asset_schedule_df,
        "asset_rollforward": asset_rollforward_df,
        "metrics_cards": metrics_cards,
        "traffic": traffic_df,
        "profitability": profitability_df,
        "customer_consideration": consideration_df,
        "npv": npv,
        "irr": irr_value,
        "payback_year": payback_period,
        "sensitivity": pd.DataFrame(sensitivity_data),
        "breakeven": breakeven_details,
        "margin_of_safety": margin_of_safety,
        "top_sensitivity": top_sensitivity_df,
    }


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
    traffic = payload.get("traffic") or []
    orders = payload.get("orders") or []
    fig = go.Figure()
    if years and any(pd.notna(val) for val in traffic):
        fig.add_bar(name="Traffic", x=years, y=traffic, marker_color="#2563eb")
    if years and any(pd.notna(val) for val in orders):
        fig.add_trace(
            go.Scatter(
                name="Orders",
                x=years,
                y=orders,
                mode="lines+markers",
                marker=dict(color="#16a34a"),
                yaxis="y2",
            )
        )
    fig.update_layout(
        title="Traffic & orders",
        barmode="group",
        yaxis=dict(title="Traffic", separatethousands=True),
        yaxis2=dict(title="Orders", overlaying="y", side="right", separatethousands=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, r=40, l=40, b=40),
    )
    return fig


def build_profitability_figure(payload: Dict[str, Any]) -> go.Figure:
    years = payload.get("years") or []
    ebitda = payload.get("ebitda") or []
    net_income = payload.get("net_income") or []
    cash = payload.get("cash") or []
    fig = go.Figure()
    if years and any(pd.notna(val) for val in ebitda):
        fig.add_bar(name="EBITDA", x=years, y=ebitda, marker_color="#a855f7")
    if years and any(pd.notna(val) for val in net_income):
        fig.add_bar(name="Net Income", x=years, y=net_income, marker_color="#f97316")
    if years and any(pd.notna(val) for val in cash):
        fig.add_trace(
            go.Scatter(
                name="Ending Cash",
                x=years,
                y=cash,
                mode="lines+markers",
                marker=dict(color="#0ea5e9"),
                yaxis="y2",
            )
        )
    fig.update_layout(
        title="Profitability & cash",
        barmode="group",
        yaxis=dict(title="Profit", tickprefix="$", separatethousands=True),
        yaxis2=dict(title="Ending Cash", overlaying="y", side="right", tickprefix="$", separatethousands=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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


def build_margin_trend_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig
    for column, color in [
        ("Gross Margin %", "#2563eb"),
        ("EBITDA Margin %", "#a855f7"),
        ("Net Margin %", "#22c55e"),
    ]:
        fig.add_trace(
            go.Scatter(
                name=column,
                x=df["Year"],
                y=df[column],
                mode="lines+markers",
                marker=dict(color=color),
            )
        )
    fig.update_layout(
        title="Margin trends",
        yaxis=dict(title="Margin %", tickformat=".1f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_breakeven_indicator(breakeven: Dict[str, Any], margin_safety: Dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    if not breakeven:
        return fig
    actual_revenue = margin_safety.get("Actual Revenue", 0.0) if margin_safety else 0.0
    max_value = max(actual_revenue, breakeven.get("Break-even Revenue", 0.0))
    if max_value <= 0:
        max_value = breakeven.get("Break-even Revenue", 1.0) or 1.0
    fig.add_trace(
        go.Indicator(
            mode="number+gauge+delta",
            value=breakeven.get("Break-even Revenue", 0.0),
            number=dict(prefix="$", valueformat=",.0f"),
            delta=dict(
                reference=actual_revenue if actual_revenue else None,
                valueformat=",.0f",
                increasing_color="#22c55e",
                decreasing_color="#ef4444",
            ),
            gauge={
                "axis": {"range": [0, max_value * 1.2]},
                "bar": {"color": "#2563eb"},
            },
            title={"text": "Break-even revenue"},
        )
    )
    return fig


def build_customer_consideration_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig
    latest_year = df["Year"].max()
    latest = df[df["Year"] == latest_year]
    fig.add_trace(
        go.Bar(
            name="Traffic",
            x=latest["Channel"],
            y=latest["Traffic"],
            marker_color="#2563eb",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Conversion Rate",
            x=latest["Channel"],
            y=latest["Conversion Rate"] * 100.0,
            mode="lines+markers",
            marker=dict(color="#f97316"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=f"Customer consideration – {int(latest_year)}",
        yaxis=dict(title="Traffic", separatethousands=True),
        yaxis2=dict(title="Conversion %", overlaying="y", side="right", tickformat=".1f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_cashflow_forecast_chart(payload: Dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    years = payload.get("years") or []
    if not years:
        return fig
    for key, label, color in [
        ("operations", "Operations", "#22c55e"),
        ("investing", "Investing", "#f97316"),
        ("financing", "Financing", "#3b82f6"),
    ]:
        fig.add_trace(
            go.Scatter(
                name=label,
                x=years,
                y=payload.get(key, []),
                mode="lines+markers",
                marker=dict(color=color),
            )
        )
    fig.update_layout(
        title="Cash flow forecast detail",
        yaxis=dict(title="Cash Flow", tickprefix="$", separatethousands=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_dcf_summary_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig
    fig.add_trace(
        go.Bar(
            name="Discounted CF",
            x=df.get("Year Label", df["Year"]),
            y=df["Discounted Cash Flow"],
            marker_color="#0ea5e9",
        )
    )
    fig.update_layout(
        title="DCF summary",
        yaxis=dict(title="Value", tickprefix="$", separatethousands=True),
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

        st.subheader("Production horizon")
        st.caption(
            "Select the start and end years for the production horizon. All assumption "
            "schedules and downstream dashboards stay within this range."
        )

        current_years = get_assumption_years()
        if not current_years:
            current_years = default_production_years()
        current_start = current_years[0]
        current_end = current_years[-1]
        min_option = min(PRODUCTION_YEAR_CHOICES[0], current_start, current_end)
        max_option = max(PRODUCTION_YEAR_CHOICES[-1], current_start, current_end)
        year_options = list(range(min_option, max_option + 1))

        start_col, end_col, reset_col = st.columns([2, 2, 1])
        with start_col:
            start_index = year_options.index(current_start) if current_start in year_options else 0
            selected_start = st.selectbox(
                "Start year",
                year_options,
                index=start_index,
                key="production_start_select",
            )
        with end_col:
            end_options = [year for year in year_options if year >= selected_start]
            if not end_options:
                end_options = [selected_start]
            adjusted_end = current_end if current_end >= selected_start else selected_start
            end_index = (
                end_options.index(adjusted_end)
                if adjusted_end in end_options
                else len(end_options) - 1
            )
            selected_end = st.selectbox(
                "End year",
                end_options,
                index=end_index,
                key="production_end_select",
            )
        with reset_col:
            st.write("")
            st.write("")
            if st.button("Reset tables", use_container_width=True, key="assumptions_reset_btn"):
                reset_assumption_tables(list(range(selected_start, selected_end + 1)))
                current_years = get_assumption_years()
                selected_start, selected_end = get_production_horizon()

        if selected_start != current_start or selected_end != current_end:
            set_production_horizon(selected_start, selected_end)
            current_years = get_assumption_years()

        st.subheader("Edit assumptions")
        st.caption(
            "Populate each schedule below to rebuild the model inputs manually. The tables "
            "stay aligned with the production horizon above."
        )

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
                if schedule["name"] == "Staffing Levels":
                    for total_col in [
                        "Direct Staff Total Cost",
                        "Indirect Staff Total Cost",
                        "Part-Time Staff Total Cost",
                    ]:
                        if total_col in schedule["columns"]:
                            column_config[total_col] = st.column_config.NumberColumn(
                                total_col,
                                format="$%0.2f",
                                disabled=True,
                            )
                if schedule["name"] == "Employee Benefits":
                    for total_col in [
                        "Pension Total Cost",
                        "Medical Insurance Total Cost",
                        "Child Benefit Total Cost",
                        "Car Benefit Total Cost",
                        "Total Benefits",
                    ]:
                        if total_col in schedule["columns"]:
                            column_config[total_col] = st.column_config.NumberColumn(
                                total_col,
                                format="$%0.2f",
                                disabled=True,
                            )
                if schedule["name"] == "Asset Register":
                    for derived_col in [
                        "Asset_1_Depreciation",
                        "Asset_1_NBV",
                    ]:
                        if derived_col in schedule["columns"]:
                            column_config[derived_col] = st.column_config.NumberColumn(
                                derived_col,
                                format="$%0.2f",
                                disabled=True,
                            )

            editor_key = _build_editor_key(schedule["name"])
            data_state_key = f"{editor_key}_table"
            edit_state_key = f"{editor_key}_active_edit"
            if edit_state_key not in st.session_state:
                st.session_state[edit_state_key] = None

            stored_table = st.session_state.get(data_state_key)
            frame_coerced = _coerce_schedule_frame(frame, schedule["columns"])
            if stored_table is None:
                working_table = frame_coerced.copy()
            else:
                stored_coerced = _coerce_schedule_frame(
                    stored_table, schedule["columns"]
                )
                allowed_years = set(current_years)
                if "Year" in stored_coerced.columns and allowed_years:
                    stored_years = pd.to_numeric(
                        stored_coerced["Year"], errors="coerce"
                    ).dropna()
                    if not set(stored_years.astype(int)).issubset(allowed_years):
                        stored_coerced = frame_coerced.copy()
                columns_match = list(frame_coerced.columns) == list(
                    stored_coerced.columns
                )
                if columns_match:
                    working_table = stored_coerced.copy()
                else:
                    working_table = frame_coerced.copy()
            st.session_state[data_state_key] = working_table.copy()

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
                                suggested_year = default_production_years()[0]
                            if current_years:
                                min_year = current_years[0]
                                max_year = current_years[-1]
                            else:
                                horizon_default = default_production_years()
                                min_year = horizon_default[0]
                                max_year = horizon_default[-1]
                            suggested_year = max(min(int(suggested_year), max_year), min_year)
                            new_row_year = st.number_input(
                                "Year for new row",
                                value=int(suggested_year),
                                step=1,
                                min_value=int(min_year),
                                max_value=int(max_year),
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

        base_years = get_assumption_years()
        computed_tables = sync_schedule_years(base_years, updated_tables)
        st.session_state["assumption_tables"] = computed_tables
        for schedule in ASSUMPTION_SCHEDULES:
            editor_key = _build_editor_key(schedule["name"])
            data_state_key = f"{editor_key}_table"
            if data_state_key in st.session_state:
                st.session_state[data_state_key] = (
                    computed_tables.get(schedule["name"], pd.DataFrame()).copy()
                )
        refresh_model_from_assumptions(st.session_state["assumption_tables"])

        if st.button("Apply assumptions", type="primary"):
            combined_df = combine_assumption_tables(st.session_state["assumption_tables"])
            if combined_df.empty:
                st.warning("Add at least one forecast year before applying assumptions.")
            else:
                with st.spinner("Rebuilding dashboards from manual inputs..."):
                    refresh_model_from_assumptions(
                        st.session_state["assumption_tables"], combined_df
                    )
                st.success("Assumptions applied. All tabs now reflect your manual inputs.")


def render_metrics_tab(tab: st.delta_generator.DeltaGenerator) -> None:
    with tab:
        st.header("Key financial metrics")
        st.write("Metrics update automatically from the manual assumption schedules.")

        results = st.session_state.get("model_results")
        if not results:
            st.info("Apply assumptions on the Input tab to calculate metrics.")
            return

        summary_df: pd.DataFrame = results.get("summary", pd.DataFrame())
        if summary_df.empty:
            st.info("No summary metrics available yet.")
        else:
            st.dataframe(summary_df.set_index("Year"), use_container_width=True)

        render_metric_cards(results.get("metrics_cards", []))

        scenario_df: pd.DataFrame = results.get("scenario_summary", pd.DataFrame())
        if not scenario_df.empty:
            st.subheader("Scenario analysis summary")
            formatted = scenario_df.copy()
            if "IRR" in formatted.columns:
                formatted["IRR"] = formatted["IRR"].apply(
                    lambda val: f"{val*100:,.2f}%" if pd.notna(val) else "n/a"
                )
            st.dataframe(formatted.set_index("Scenario"), use_container_width=True)

        key_metrics = st.columns(3)
        npv_value = results.get("npv", 0.0)
        irr_value = results.get("irr")
        payback_year = results.get("payback_year")
        key_metrics[0].metric("Net Present Value", f"${npv_value:,.0f}")
        irr_display = f"{irr_value*100:,.2f}%" if irr_value is not None else "n/a"
        key_metrics[1].metric("Internal Rate of Return", irr_display)
        payback_text = str(payback_year) if payback_year else "Not achieved"
        key_metrics[2].metric("Payback year", payback_text)

        revenue_matrix = results.get("revenue_matrix", pd.DataFrame())
        if not revenue_matrix.empty:
            st.subheader("Revenue matrix")
            st.dataframe(revenue_matrix.set_index("Year"), use_container_width=True)

        depreciation_matrix = results.get("depreciation_matrix", pd.DataFrame())
        equity_debt_matrix = results.get("equity_debt_matrix", pd.DataFrame())
        asset_schedule = results.get("asset_schedule", pd.DataFrame())
        asset_rollforward = results.get("asset_rollforward", pd.DataFrame())
        matrix_cols = st.columns(2)
        with matrix_cols[0]:
            if depreciation_matrix.empty:
                st.info("No depreciation schedule available yet.")
            else:
                st.subheader("Depreciation matrix")
                st.dataframe(
                    depreciation_matrix.set_index("Year"), use_container_width=True
                )
        with matrix_cols[1]:
            if equity_debt_matrix.empty:
                st.info("No equity & debt data available yet.")
            else:
                st.subheader("Equity & debt matrix")
                st.dataframe(
                    equity_debt_matrix.set_index("Year"), use_container_width=True
                )

        if not asset_schedule.empty:
            st.subheader("Asset schedule")
            display_schedule = asset_schedule.copy()
            if {"Year", "Asset"}.issubset(display_schedule.columns):
                display_schedule = display_schedule.set_index(["Year", "Asset"])
            st.dataframe(display_schedule, use_container_width=True)

        if not asset_rollforward.empty:
            st.subheader("Asset additions roll-forward")
            st.dataframe(
                asset_rollforward.set_index("Year"), use_container_width=True
            )
            st.info(
                "Add new assets on the Input & Assumptions → Asset Register table. "
                "Each row seeds an addition in the asset roll-forward, drives "
                "depreciation for the remaining horizon, and updates the "
                "income statement, cash flow, and balance sheet automatically."
            )

        chart_payloads: Dict[str, Any] = results.get("chart_payloads", {})

        if not summary_df.empty:
            years = summary_df["Year"].tolist()
            revenue_payload = {
                "years": years,
                "net_revenue": summary_df["Net Revenue"].tolist(),
                "gross_margin": (summary_df["Gross Margin %"] / 100.0).tolist(),
                "ebitda_margin": (summary_df["EBITDA Margin %"] / 100.0).tolist(),
            }
            traffic_df: pd.DataFrame = results.get("traffic", pd.DataFrame())
            profitability_df: pd.DataFrame = results.get("profitability", pd.DataFrame())

            charts_row = st.columns(2)
            charts_row[0].plotly_chart(
                build_revenue_figure(revenue_payload), use_container_width=True
            )
            if not traffic_df.empty:
                traffic_payload = {
                    "years": traffic_df["Year"].tolist(),
                    "traffic": traffic_df["Traffic"].tolist(),
                    "orders": traffic_df["Orders"].tolist(),
                }
                charts_row[1].plotly_chart(
                    build_traffic_figure(traffic_payload), use_container_width=True
                )
            if not profitability_df.empty:
                cashflow_df = results.get("cashflow", pd.DataFrame())
                cash_series = (
                    cashflow_df["Ending Cash"].tolist()
                    if not cashflow_df.empty
                    else []
                )
                profitability_payload = {
                    "years": profitability_df["Year"].tolist(),
                    "ebitda": profitability_df["EBITDA"].tolist(),
                    "net_income": profitability_df["Net Income"].tolist(),
                    "cash": cash_series,
                }
                st.plotly_chart(
                    build_profitability_figure(profitability_payload),
                    use_container_width=True,
                )

        kpi_df: pd.DataFrame = results.get("operational_kpis", pd.DataFrame())
        if not kpi_df.empty:
            st.subheader("Operational KPIs")
            st.dataframe(kpi_df.set_index("Year"), use_container_width=True)
            st.plotly_chart(
                build_margin_trend_figure(kpi_df), use_container_width=True
            )

        customer_metrics_df: pd.DataFrame = results.get("customer_metrics", pd.DataFrame())
        if not customer_metrics_df.empty:
            st.subheader("Customer metrics")
            st.dataframe(
                customer_metrics_df.set_index("Year"), use_container_width=True
            )
            consideration_df: pd.DataFrame = results.get(
                "customer_consideration", pd.DataFrame()
            )
            if not consideration_df.empty:
                st.plotly_chart(
                    build_customer_consideration_chart(consideration_df),
                    use_container_width=True,
                )

        if chart_payloads:
            breakeven = results.get("breakeven", {})
            margin_safety = results.get("margin_of_safety", {})
            indicator_cols = st.columns(2)
            indicator_cols[0].plotly_chart(
                build_breakeven_indicator(breakeven, margin_safety),
                use_container_width=True,
            )
            if margin_safety:
                indicator_cols[1].metric(
                    "Margin of safety",
                    f"{margin_safety.get('Margin of Safety %', 0.0):,.2f}%",
                )

            waterfall_series: pd.Series = chart_payloads.get("waterfall", pd.Series())
            if isinstance(waterfall_series, pd.Series) and not waterfall_series.empty:
                categories = [
                    "Revenue",
                    "COGS",
                    "Marketing",
                    "Fulfillment",
                    "Operating Expenses",
                    "EBITDA",
                    "Depreciation",
                    "Interest",
                    "Taxes",
                    "Net Income",
                ]
                values = [waterfall_series.get(cat, 0.0) for cat in categories]
                measures = ["absolute", "relative", "relative", "relative", "relative", "absolute", "relative", "relative", "relative", "absolute"]
                waterfall_payload = {
                    "categories": categories,
                    "values": values,
                    "measures": measures,
                }
                st.plotly_chart(
                    build_waterfall(waterfall_payload, "Income waterfall"),
                    use_container_width=True,
                )

            cashflow_payload = chart_payloads.get("cashflow", {})
            if cashflow_payload:
                st.plotly_chart(
                    build_cashflow_forecast_chart(cashflow_payload),
                    use_container_width=True,
                )

            valuation_df: pd.DataFrame = chart_payloads.get("dcf_summary", pd.DataFrame())
            if isinstance(valuation_df, pd.DataFrame) and not valuation_df.empty:
                st.plotly_chart(
                    build_dcf_summary_chart(valuation_df), use_container_width=True
                )


def render_performance_tab(tab: st.delta_generator.DeltaGenerator) -> None:
    with tab:
        st.header("Financial performance dashboards")
        st.write("Inspect demand, conversion, and operating expense trends derived from your inputs.")

        results = st.session_state.get("model_results")
        if not results:
            st.info("Apply assumptions on the Input tab to calculate performance views.")
            return

        income_statement = results.get("income_statement", pd.DataFrame())
        if income_statement.empty:
            st.info("Add assumption data to see the income statement.")
            return

        st.subheader("Comprehensive income statement")
        st.dataframe(income_statement.set_index("Year"), use_container_width=True)

        performance_df: pd.DataFrame = results.get("performance", pd.DataFrame())
        if not performance_df.empty:
            st.subheader("Operating drivers")
            st.dataframe(performance_df.set_index("Year"), use_container_width=True)

            fig = go.Figure()
            for column, color in [
                ("Marketing Spend", "#3b82f6"),
                ("Fulfillment Cost", "#10b981"),
                ("Staffing Cost", "#f97316"),
                ("Benefits", "#8b5cf6"),
                ("Overheads", "#ef4444"),
            ]:
                fig.add_trace(
                    go.Scatter(
                        name=column,
                        x=performance_df["Year"],
                        y=performance_df[column],
                        mode="lines+markers",
                        marker=dict(color=color),
                    )
                )
            fig.update_layout(
                title="Operating cost profile",
                yaxis=dict(title="Cost", tickprefix="$", separatethousands=True),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)

        ratios_df: pd.DataFrame = results.get("operational_kpis", pd.DataFrame())
        if not ratios_df.empty:
            st.subheader("Profitability ratios")
            ratio_view = ratios_df[[
                "Year",
                "Gross Margin %",
                "EBITDA Margin %",
                "Net Margin %",
                "ROE %",
                "Asset Turnover",
            ]]
            st.dataframe(ratio_view.set_index("Year"), use_container_width=True)

        chart_payloads: Dict[str, Any] = results.get("chart_payloads", {})
        profitability_payload = chart_payloads.get("profitability")
        if profitability_payload:
            st.subheader("Profitability analysis")
            st.plotly_chart(
                build_profitability_figure(profitability_payload),
                use_container_width=True,
            )


def render_position_tab(tab: st.delta_generator.DeltaGenerator) -> None:
    with tab:
        st.header("Financial position")
        st.write("Review balance sheet items calculated from the manual schedules.")

        results = st.session_state.get("model_results")
        if not results:
            st.info("Apply assumptions on the Input tab to calculate the balance sheet.")
            return

        position_df: pd.DataFrame = results.get("position", pd.DataFrame())
        if position_df.empty:
            st.info("No balance sheet data available yet.")
            return

        st.subheader("Balance sheet")
        st.dataframe(position_df.set_index("Year"), use_container_width=True)

        assets_fig = go.Figure()
        for column, color in [
            ("Cash", "#0ea5e9"),
            ("Accounts Receivable", "#22c55e"),
            ("Inventory", "#facc15"),
            ("Fixed Assets", "#6366f1"),
        ]:
            assets_fig.add_trace(
                go.Scatter(
                    name=column,
                    x=position_df["Year"],
                    y=position_df[column],
                    mode="lines+markers",
                    marker=dict(color=color),
                )
            )
        assets_fig.update_layout(
            title="Asset mix",
            yaxis=dict(title="Value", tickprefix="$", separatethousands=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(assets_fig, use_container_width=True)

        liabilities_fig = go.Figure()
        for column, color in [
            ("Accounts Payable", "#f87171"),
            ("Debt", "#334155"),
            ("Equity", "#a855f7"),
        ]:
            liabilities_fig.add_trace(
                go.Scatter(
                    name=column,
                    x=position_df["Year"],
                    y=position_df[column],
                    mode="lines+markers",
                    marker=dict(color=color),
                )
            )
        liabilities_fig.update_layout(
            title="Liabilities & equity",
            yaxis=dict(title="Value", tickprefix="$", separatethousands=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(liabilities_fig, use_container_width=True)

        liquidity_df: pd.DataFrame = results.get("liquidity", pd.DataFrame())
        if not liquidity_df.empty:
            st.subheader("Liquidity ratios")
            st.dataframe(liquidity_df.set_index("Year"), use_container_width=True)


def render_cashflow_tab(tab: st.delta_generator.DeltaGenerator) -> None:
    with tab:
        st.header("Cash flow statement")
        st.write("Track operating, investing, and financing cash flows derived from manual assumptions.")

        results = st.session_state.get("model_results")
        if not results:
            st.info("Apply assumptions on the Input tab to calculate cash flow results.")
            return

        cashflow_df: pd.DataFrame = results.get("cashflow", pd.DataFrame())
        if cashflow_df.empty:
            st.info("No cash flow data available yet.")
            return

        st.dataframe(cashflow_df.set_index("Year"), use_container_width=True)

        payload = {
            "years": cashflow_df["Year"].tolist(),
            "cash_from_operations": cashflow_df["Cash Flow from Operations"].tolist(),
            "cash_from_investing": cashflow_df["Cash Flow from Investing"].tolist(),
            "net_cash_flow": cashflow_df["Net Cash Flow"].tolist(),
        }
        st.plotly_chart(build_cashflow_figure(payload), use_container_width=True)

        chart_payloads: Dict[str, Any] = results.get("chart_payloads", {})
        detailed_payload = chart_payloads.get("cashflow")
        if detailed_payload:
            st.plotly_chart(
                build_cashflow_forecast_chart(detailed_payload),
                use_container_width=True,
            )

        valuation_table = results.get("valuation_table", pd.DataFrame())
        if not valuation_table.empty:
            st.subheader("DCF summary schedule")
            st.dataframe(valuation_table.set_index("Year Label"), use_container_width=True)

        debt_amort = results.get("debt_amortization", pd.DataFrame())
        if not debt_amort.empty:
            st.subheader("Debt amortization")
            st.dataframe(debt_amort.set_index("Year"), use_container_width=True)


def render_sensitivity_tab(tab: st.delta_generator.DeltaGenerator) -> None:
    with tab:
        st.header("Sensitivity analysis")
        st.write("Quickly compare revenue and profit outcomes for common percentage shifts.")

        results = st.session_state.get("model_results")
        if not results:
            st.info("Apply assumptions on the Input tab to generate sensitivity snapshots.")
            return

        sensitivity_df: pd.DataFrame = results.get("sensitivity", pd.DataFrame())
        if sensitivity_df.empty:
            st.info("No sensitivity data available yet.")
            return

        summary_df: pd.DataFrame = results.get("summary", pd.DataFrame())
        if not summary_df.empty:
            latest = summary_df.iloc[-1]
            st.subheader("Current performance snapshot")
            cols = st.columns(4)
            cols[0].metric("Revenue", f"${latest['Net Revenue']:,.0f}")
            cols[1].metric("EBITDA", f"${latest['EBITDA']:,.0f}")
            cols[2].metric("Net Income", f"${latest['Net Income']:,.0f}")
            cols[3].metric("Orders", f"{latest['Total Orders']:,.0f}")

        st.dataframe(sensitivity_df, use_container_width=True)

        bars = go.Figure()
        bars.add_trace(
            go.Bar(
                name="Impact",
                x=sensitivity_df["Adjustment"],
                y=sensitivity_df.iloc[:, 1],
                marker_color="#2563eb",
            )
        )
        bars.update_layout(
            title="Sensitivity comparison",
            yaxis=dict(title="Projected value", tickprefix="$", separatethousands=True),
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(bars, use_container_width=True)

        breakeven = results.get("breakeven", {})
        margin_safety = results.get("margin_of_safety", {})
        metrics_cols = st.columns(3)
        metrics_cols[0].metric(
            "Break-even revenue",
            f"${breakeven.get('Break-even Revenue', 0.0):,.0f}",
        )
        metrics_cols[1].metric(
            "Margin of safety",
            f"{margin_safety.get('Margin of Safety %', 0.0):,.2f}%",
        )
        metrics_cols[2].metric(
            "Contribution margin",
            f"{breakeven.get('Contribution Margin %', 0.0):,.2f}%",
        )

        npv_value = results.get("npv", 0.0)
        irr_value = results.get("irr")
        payback_year = results.get("payback_year")
        decision = "Proceed" if npv_value >= 0 else "Re-evaluate"
        irr_text = f"IRR = {irr_value*100:,.2f}%" if irr_value is not None else "IRR unavailable."
        st.info(f"**Decision guidance:** {decision}. NPV = ${npv_value:,.0f}. {irr_text}")
        st.write(
            f"Payback period: {payback_year if payback_year else 'Not achieved'}"
        )

        chart_payloads: Dict[str, Any] = results.get("chart_payloads", {})
        revenue_payload = chart_payloads.get("revenue")
        if revenue_payload:
            forecast_payload = {
                "years": revenue_payload.get("years", []),
                "net_revenue": revenue_payload.get("net_revenue", []),
                "gross_margin": [val / 100 if val > 1 else val for val in revenue_payload.get("gross_margin", [])],
                "ebitda_margin": [val / 100 if val > 1 else val for val in revenue_payload.get("ebitda_margin", [])],
            }
            st.plotly_chart(
                build_revenue_figure(forecast_payload), use_container_width=True
            )

        top_sensitivity = results.get("top_sensitivity", pd.DataFrame())
        if not top_sensitivity.empty:
            st.subheader("Top-rank scenario deltas")
            st.dataframe(top_sensitivity.set_index("Scenario"), use_container_width=True)


def render_advanced_tab(tab: st.delta_generator.DeltaGenerator) -> None:
    with tab:
        st.header("Advanced analysis")
        st.write("Run valuation, risk, and optimization tooling on the offline model outputs.")

        results = st.session_state.get("model_results")
        if not results:
            st.info("Apply assumptions on the Input tab to generate valuation insights.")
            return

        npv_value = results.get("npv", 0.0)
        irr_value = results.get("irr")
        payback_year = results.get("payback_year")
        cashflow_df: pd.DataFrame = results.get("cashflow", pd.DataFrame())
        summary_df: pd.DataFrame = results.get("summary", pd.DataFrame())

        valuation_cols = st.columns(3)
        valuation_cols[0].metric("Net Present Value", f"${npv_value:,.0f}")
        irr_display = f"{irr_value*100:,.2f}%" if irr_value is not None else "n/a"
        valuation_cols[1].metric("IRR", irr_display)
        valuation_cols[2].metric(
            "Payback year", str(payback_year) if payback_year else "Not achieved"
        )

        if not cashflow_df.empty:
            cumulative_fig = go.Figure()
            cumulative_fig.add_trace(
                go.Scatter(
                    name="Ending Cash",
                    x=cashflow_df["Year"],
                    y=cashflow_df["Ending Cash"],
                    mode="lines+markers",
                    marker=dict(color="#0ea5e9"),
                )
            )
            cumulative_fig.update_layout(
                title="Cumulative cash position",
                yaxis=dict(title="Cash", tickprefix="$", separatethousands=True),
            )
            st.plotly_chart(cumulative_fig, use_container_width=True)

        st.caption(
            "Valuation metrics reference the manual cash flows, discount assumptions, and net income derived from your schedules."
        )

        st.subheader("Monte Carlo simulation")
        mc_iterations = st.slider("Iterations", 500, 5000, 2000, step=500)
        revenue_sigma = st.slider("Revenue volatility", 0.0, 0.3, 0.1, step=0.01)
        cost_sigma = st.slider("Cost volatility", 0.0, 0.3, 0.05, step=0.01)
        discount_rate = st.slider("Discount rate", 0.01, 0.3, DEFAULT_DISCOUNT_RATE, step=0.01)
        cash_flows = cashflow_df["Net Cash Flow"].tolist() if not cashflow_df.empty else []
        if st.button("Run Monte Carlo", use_container_width=True):
            mc_df = monte_carlo_analysis(
                cash_flows, mc_iterations, revenue_sigma, cost_sigma, discount_rate
            )
            if mc_df.empty:
                st.warning("No cash flows available for simulation.")
            else:
                summary = mc_df.describe(percentiles=[0.1, 0.5, 0.9]).T
                st.dataframe(summary, use_container_width=True)
                st.plotly_chart(
                    go.Figure(
                        data=[
                            go.Histogram(
                                x=mc_df["NPV"], nbinsx=40, marker_color="#2563eb"
                            )
                        ],
                        layout=go.Layout(title="NPV distribution"),
                    ),
                    use_container_width=True,
                )

        st.subheader("What-if overlays")
        if not summary_df.empty:
            base_row = summary_df.iloc[-1]
            driver = st.selectbox(
                "Driver to scale",
                ["Net Revenue", "Gross Profit", "EBITDA", "Net Income", "Total Orders"],
            )
            multiplier = st.slider("Multiplier", 0.5, 1.5, 1.1, step=0.05)
            if st.button("Apply what-if", use_container_width=True):
                adjusted = base_row.copy()
                if driver in adjusted:
                    adjusted[driver] = adjusted[driver] * multiplier
                updated_margin = (
                    adjusted.get("Net Income", 0.0) / adjusted.get("Net Revenue", 1.0)
                ) * 100.0 if adjusted.get("Net Revenue", 0.0) else 0.0
                st.success(
                    f"{driver} scaled by {multiplier:.2f}. Updated Net Income: ${adjusted.get('Net Income', 0.0):,.0f}. "
                    f"Net margin: {updated_margin:,.2f}%"
                )

        st.subheader("Decision analysis")
        scenario_df: pd.DataFrame = results.get("scenario_summary", pd.DataFrame())
        if scenario_df.empty:
            st.info("Provide assumptions to generate scenario analysis.")
        else:
            probabilities = []
            prob_cols = st.columns(len(scenario_df))
            for idx, (_, row) in enumerate(scenario_df.iterrows()):
                probabilities.append(
                    prob_cols[idx].number_input(
                        f"{row['Scenario']} probability",
                        min_value=0.0,
                        max_value=1.0,
                        value=round(1.0 / len(scenario_df), 2),
                        key=f"scenario_prob_{idx}",
                    )
                )
            total_prob = sum(probabilities) or 1.0
            normalized = [p / total_prob for p in probabilities]
            scenario_df = scenario_df.copy()
            scenario_df["Probability"] = normalized
            scenario_df["Expected Value"] = scenario_df["NPV"] * scenario_df["Probability"]
            st.dataframe(scenario_df.set_index("Scenario"), use_container_width=True)
            st.metric(
                "Expected NPV", f"${scenario_df['Expected Value'].sum():,.0f}"
            )

        st.subheader("Goal-seek profit margin")
        target_margin = st.slider("Target net margin %", -20.0, 40.0, 15.0, step=1.0)
        if not summary_df.empty:
            base_row = summary_df.iloc[-1]
            base_margin = base_row.get("Net Margin %", 0.0)
            multiplier, achieved = goal_seek_margin(
                target_margin, base_row.get("Net Revenue", 0.0), base_margin
            )
            st.write(
                f"Multiplier required on net margin: {multiplier:,.2f}. Projected margin: {achieved:,.2f}%"
            )

        st.subheader("Schedule risk simulation")
        if not cashflow_df.empty:
            rng = np.random.default_rng()
            base_duration = len(cashflow_df) * 12
            optimistic = st.slider("Optimistic months", 6, base_duration, 12, step=6)
            pessimistic = st.slider(
                "Pessimistic months", base_duration, base_duration * 2, base_duration + 12, step=6
            )
            trials = rng.triangular(optimistic, base_duration, pessimistic, size=2000)
            st.write(
                f"Expected completion: {np.mean(trials):.1f} months (p90 = {np.percentile(trials, 90):.1f} months)"
            )

        st.subheader("Neural-style demand forecast")
        revenue_series = summary_df["Net Revenue"].tolist() if not summary_df.empty else []
        if revenue_series:
            forecast_periods = st.slider("Forecast periods", 1, 5, 3)
            forecast_values = simple_forecast(revenue_series, forecast_periods)
            forecast_years = [
                summary_df.iloc[-1]["Year"] + idx for idx in range(1, len(forecast_values) + 1)
            ]
            forecast_df = pd.DataFrame(
                {"Year": forecast_years, "Forecast Revenue": forecast_values}
            )
            st.dataframe(forecast_df.set_index("Year"), use_container_width=True)

        st.subheader("Budget optimization (Evolver)")
        performance_df: pd.DataFrame = results.get("performance", pd.DataFrame())
        if not performance_df.empty:
            spend_options = [
                "Marketing Spend",
                "Fulfillment Cost",
                "Staffing Cost",
                "Benefits",
                "Overheads",
            ]
            spend_choice = st.selectbox("Spend line to constrain", spend_options)
            target_spend = st.number_input(
                "Target spend (per year)",
                min_value=0.0,
                value=float(
                    performance_df[spend_choice].iloc[-1]
                    if spend_choice in performance_df
                    else 0.0
                ),
            )
            optimized = performance_df.copy()
            if spend_choice in optimized:
                optimized[spend_choice] = np.minimum(
                    optimized[spend_choice], target_spend
                )
            st.dataframe(optimized.set_index("Year"), use_container_width=True)

        st.caption(
            "Advanced tooling approximates Monte Carlo, what-if analysis, decision trees, goal seek, schedule risk, neural-style forecasts, and budget optimization using the offline model outputs."
        )


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

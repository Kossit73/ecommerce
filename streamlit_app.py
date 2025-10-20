"""Streamlit dashboard for the ecommerce financial model API."""
from __future__ import annotations

import base64
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Streamlit configuration & constants
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ecommerce Financial Model",
    layout="wide",
    page_icon="💼",
)

DEFAULT_API_BASE = "http://localhost:8000"
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_api_base() -> str:
    return st.session_state.setdefault("api_base_url", DEFAULT_API_BASE)


def set_api_base(url: str) -> None:
    cleaned = (url or "").strip()
    if cleaned:
        cleaned = cleaned.rstrip("/")
    st.session_state["api_base_url"] = cleaned or DEFAULT_API_BASE


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
        yaxis2=dict(
            title="Margin %",
            overlaying="y",
            side="right",
            tickformat=".0%",
        ),
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
        yaxis2=dict(
            title="Ratio",
            overlaying="y",
            side="right",
            tickformat=".2f",
        ),
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

with st.sidebar:
    st.header("API configuration")
    api_base_input = st.text_input("FastAPI base URL", value=get_api_base())
    if api_base_input != get_api_base():
        set_api_base(api_base_input)
    if st.button("Test connection"):
        try:
            response = api_get("/file_action", params={"action": "Load Existing"})
            st.success(
                f"Connected. Workbook exists: {response.get('exists', False)}"
            )
        except Exception as exc:  # pragma: no cover - interactive feedback
            st.error(str(exc))

    st.markdown("""
    Configure the FastAPI endpoint and use the tabs to manage assumptions,
    recalculate scenarios, review performance, and run analytics.
    """)


# ---------------------------------------------------------------------------
# Input & assumptions tab
# ---------------------------------------------------------------------------

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


with input_tab:
    st.header("Workbook setup & assumptions")
    st.write(
        "Upload or refresh the working Excel file, edit the grouped assumption tables,"
        " and control base analysis inputs before running downstream workflows."
    )

    action_cols = st.columns(3)
    if action_cols[0].button("Load existing workbook", use_container_width=True):
        try:
            response = api_get("/file_action", params={"action": "Load Existing"})
            st.session_state["assumptions_raw"] = response.get("data")
            st.session_state["file_status"] = response
            st.success("Existing workbook loaded.")
        except RuntimeError as exc:
            st.error(str(exc))
    if action_cols[1].button("Start new workbook", use_container_width=True):
        try:
            response = api_get("/file_action", params={"action": "Start New"})
            st.session_state["assumptions_raw"] = response.get("data")
            st.session_state["file_status"] = response
            st.success("Blank workbook initialized.")
        except RuntimeError as exc:
            st.error(str(exc))
    with action_cols[2]:
        uploaded = st.file_uploader(
            "Upload Excel assumptions", type=["xlsx"], label_visibility="collapsed"
        )
        if uploaded is not None and st.button("Send upload"):
            try:
                files = {
                    "file": (
                        uploaded.name,
                        uploaded.getvalue(),
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                }
                response = api_post("/upload_file", files=files)
                st.session_state["assumptions_raw"] = response.get("data")
                st.session_state["file_status"] = response
                st.success("Workbook uploaded and processed.")
            except RuntimeError as exc:
                st.error(str(exc))

    if st.session_state.get("file_status"):
        status = st.session_state["file_status"]
        st.caption(
            f"Current file: {status.get('filename', 'financial_assumptions.xlsx')}"
            f" | Exists: {status.get('exists', False)}"
        )

    assumptions_df = to_dataframe(st.session_state.get("assumptions_raw"))
    if assumptions_df.empty:
        st.info("Load or upload a workbook to edit assumptions.")
    else:
        st.subheader("Edit assumptions")
        edited_df = st.data_editor(
            assumptions_df,
            num_rows="dynamic",
            use_container_width=True,
            key="assumptions_editor",
        )
        if st.button("Save assumptions", type="primary"):
            try:
                payload = [
                    {k: (None if pd.isna(v) else v) for k, v in row.items()}
                    for row in edited_df.to_dict(orient="records")
                ]
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
                response = api_post("/run_base_analysis", json=payload)
                st.session_state["base_analysis_result"] = response
                st.success("Base analysis completed.")
            except RuntimeError as exc:
                st.error(str(exc))

    if st.session_state.get("base_analysis_result"):
        st.subheader("Latest base analysis snapshot")
        st.json(st.session_state["base_analysis_result"])


# ---------------------------------------------------------------------------
# Key financial metrics tab
# ---------------------------------------------------------------------------

with metrics_tab:
    st.header("Key financial metrics")
    st.write(
        "Review summary KPIs, operational metrics, valuation outputs, and scenario"
        " comparisons. Adjust scenario parameters to recalculate downstream tables."
    )

    if st.button("Refresh metrics", key="refresh_metrics") or "summary_metrics" not in st.session_state:
        try:
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
            st.session_state["profitability_chart"] = profitability_payload.get("profitability_chart_data", profitability_payload)
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
        params_container = {}
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
        col1.plotly_chart(build_revenue_figure(st.session_state["revenue_chart"]), use_container_width=True)
        col2.plotly_chart(build_traffic_figure(st.session_state["traffic_chart"]), use_container_width=True)
    if st.session_state.get("profitability_chart"):
        st.plotly_chart(
            build_profitability_figure(st.session_state["profitability_chart"]),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Financial performance tab
# ---------------------------------------------------------------------------

with performance_tab:
    st.header("Financial performance dashboards")
    st.write("Visualize revenue drivers, breakeven analysis, waterfall bridges, and margin trends.")

    if st.button("Refresh performance visuals", key="refresh_performance") or "waterfall_chart" not in st.session_state:
        try:
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


# ---------------------------------------------------------------------------
# Financial position tab
# ---------------------------------------------------------------------------

with position_tab:
    st.header("Financial position & supporting schedules")
    st.write("Inspect balance sheet, capital assets, and debt amortization outputs.")

    if st.button("Load financial position", key="refresh_position") or "financial_position" not in st.session_state:
        try:
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


# ---------------------------------------------------------------------------
# Cash flow tab
# ---------------------------------------------------------------------------

with cashflow_tab:
    st.header("Cash flow and valuation")
    st.write("Review detailed cash flow statements and valuation bridges.")

    if st.button("Load cash flow", key="refresh_cashflow") or "cashflow_schedules" not in st.session_state:
        try:
            response = api_get(
                "/financial_schedules",
                params=[("schedules", item) for item in SCHEDULE_OPTIONS["Cash Flow"]],
            )
            st.session_state["cashflow_schedules"] = response.get("schedules", [])
            cashflow_payload = api_get("/cashflow_forecast_chart_data")
            st.session_state["cashflow_chart"] = cashflow_payload.get("cashflow_forecast_chart_data", cashflow_payload)
            dcf_payload = api_get("/dcf_summary_chart_data")
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


# ---------------------------------------------------------------------------
# Sensitivity tab
# ---------------------------------------------------------------------------

with sensitivity_tab:
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
        adjustments = []
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
                response = api_post("/goal_seek", json=payload)
                st.session_state["goal_seek_results"] = response
                st.success(response.get("message", "Goal seek completed."))
            except RuntimeError as exc:
                st.error(str(exc))

    goal_seek_results = st.session_state.get("goal_seek_results")
    if goal_seek_results:
        render_table("Goal seek results", goal_seek_results.get("results"))
        st.json(goal_seek_results)


# ---------------------------------------------------------------------------
# Advanced analysis tab
# ---------------------------------------------------------------------------

with advanced_tab:
    st.header("Advanced analytics & exports")
    st.write(
        "Run Monte Carlo simulations, schedule risk, neural predictions, statistical"
        " forecasts, and budget optimizations. Download the consolidated Excel report"
        " for offline review."
    )

    if st.button("Download Excel report"):
        try:
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

"""Streamlit front end for interacting with the Ecommerce financial model."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import streamlit as st

from ecommerce import DATA_DIR, EcommerceModel, EXCEL_FILE


def _ensure_directory(path: Path) -> None:
    """Create the given directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def _write_uploaded_file(uploaded_file: Any) -> Path:
    """Persist an uploaded workbook to the data directory and return its path."""
    destination = DATA_DIR / uploaded_file.name
    with destination.open("wb") as buffer:
        buffer.write(uploaded_file.getbuffer())
    return destination


def _list_available_workbooks(directory: Path) -> Iterable[Path]:
    return sorted(directory.glob("*.xlsx"))


@st.cache_data(show_spinner=False)
def _normalize_years_data(years_data: Dict[int, Dict]) -> pd.DataFrame:
    """Convert the nested years dictionary into a DataFrame for presentation."""
    if not years_data:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(years_data, orient="index")
    df.index.name = "Year"
    return df.reset_index()


def _load_model(current_file: Path) -> Dict:
    model = EcommerceModel()
    model.current_file = str(current_file)
    return model.load_existing_data()


def main() -> None:
    st.set_page_config(page_title="Ecommerce Financial Model", layout="wide")
    st.title("Ecommerce Financial Model Dashboard")
    st.caption(
        "Use this dashboard to upload financial assumption workbooks and explore the "
        "parsed scenario data."
    )

    _ensure_directory(DATA_DIR)

    if "current_file" not in st.session_state:
        st.session_state["current_file"] = EXCEL_FILE

    with st.sidebar:
        st.header("Workbook")
        available_files = list(_list_available_workbooks(DATA_DIR))
        if EXCEL_FILE not in available_files and EXCEL_FILE.exists():
            available_files.insert(0, EXCEL_FILE)

        if available_files:
            labels = [path.name for path in available_files]
            default_index = labels.index(Path(st.session_state["current_file"]).name) if Path(
                st.session_state["current_file"]
            ).exists() and Path(st.session_state["current_file"]).name in labels else 0
            selected_label = st.selectbox(
                "Select workbook",
                labels,
                index=default_index,
            )
            selected_path = DATA_DIR / selected_label
        else:
            selected_path = EXCEL_FILE
            st.info("No saved workbooks yet. Upload one to get started.")

        uploaded_file = st.file_uploader("Upload new workbook", type=["xlsx"], accept_multiple_files=False)
        if uploaded_file is not None:
            saved_path = _write_uploaded_file(uploaded_file)
            st.session_state["current_file"] = saved_path
            try:
                display_path = saved_path.relative_to(Path.cwd())
            except ValueError:
                display_path = saved_path
            st.success(f"Uploaded {uploaded_file.name} to {display_path}")
            selected_path = saved_path

        if st.button("Reload data"):
            st.session_state.pop("load_result", None)

    current_file = selected_path if selected_path.exists() else EXCEL_FILE
    st.session_state["current_file"] = current_file

    if "load_result" not in st.session_state:
        with st.spinner(f"Loading data from {current_file.name}..."):
            st.session_state["load_result"] = _load_model(current_file)

    load_result = st.session_state["load_result"]

    if load_result.get("status") != "success":
        st.error(load_result.get("detail", "Unable to load the selected workbook."))
        return

    years_data = load_result.get("data", {})
    debts_data = load_result.get("debts_data", {})

    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Available years", len(years_data))
    total_revenue = 0.0
    total_costs = 0.0
    if years_data:
        for year_info in years_data.values():
            total_revenue += float(year_info.get("Average Item Value", 0.0)) * float(
                year_info.get("Email Traffic", 0.0)
            )
            total_costs += float(year_info.get("Salaries, Wages & Benefits", 0.0)) + float(
                year_info.get("General Warehouse Rent", 0.0)
            )
    col2.metric("Approx. revenue", f"${total_revenue:,.0f}")
    col3.metric("Labor & rent costs", f"${total_costs:,.0f}")

    st.subheader("Yearly data")
    years_df = _normalize_years_data(years_data)
    if years_df.empty:
        st.info("The selected workbook does not contain any yearly data rows.")
    else:
        st.dataframe(years_df, use_container_width=True)

    if debts_data:
        st.subheader("Debts by year")
        for year, debts in debts_data.items():
            st.write(f"**{year}**")
            if debts:
                debt_df = pd.DataFrame(debts)
                st.table(debt_df)
            else:
                st.write("No debt entries recorded.")

    st.subheader("Raw data preview")
    with st.expander("Download processed data"):
        if years_df.empty:
            st.write("Nothing to download yet.")
        else:
            buffer = BytesIO()
            years_df.to_excel(buffer, index=False)
            buffer.seek(0)
            st.download_button(
                label="Download yearly data",
                data=buffer,
                file_name=f"processed_{Path(current_file).stem}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


if __name__ == "__main__":
    main()

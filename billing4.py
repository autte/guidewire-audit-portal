import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Guidewire Audit Portal", layout="wide")

# ======================================================
# TAB 1 — DATA EXPLORER
# ======================================================

def run_data_explorer():

    st.header("📊 Dynamic Aggregation & Visualization")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="explorer_upload")

    if uploaded is None:
        st.info("Upload a CSV to start.")
        return

    df = pd.read_csv(uploaded, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    st.success("File loaded successfully")

    # -------------------------
    # Field Selection
    # -------------------------
    st.subheader("1️⃣ Select Fields")

    picked_fields = st.multiselect(
        "Choose fields to analyze",
        df.columns.tolist(),
        default=df.columns.tolist()
    )

    if not picked_fields:
        st.warning("Pick at least one field.")
        return

    df_work = df[picked_fields].copy()

    # Save dataset for fraud tab
    st.session_state["filtered_data"] = df_work.copy()

    # -------------------------
    # Aggregation Engine
    # -------------------------
    st.subheader("2️⃣ Aggregation")

    num_cols = df_work.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_work.select_dtypes(exclude=np.number).columns.tolist()

    groupby = st.multiselect("Group by (categorical fields)", cat_cols)

    measure_type = st.radio(
        "Measure Type",
        ["Count rows", "Aggregate numeric column"]
    )

    if measure_type == "Aggregate numeric column" and num_cols:
        measure_col = st.selectbox("Numeric column", num_cols)
        agg_func = st.selectbox("Aggregation function", ["sum", "mean", "min", "max", "median"])
    else:
        measure_col = None
        agg_func = "count"

    chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter"])

    if groupby:

        if measure_type == "Count rows":
            result = df_work.groupby(groupby, dropna=False).size().reset_index(name="measure")
        else:
            result = (
                df_work.groupby(groupby, dropna=False)[measure_col]
                .agg(agg_func)
                .reset_index(name="measure")
            )

        # Sorting
        result = result.sort_values("measure", ascending=False)

        st.subheader("Result Table")
        st.dataframe(result, use_container_width=True)

        st.subheader("Visualization")

        x_axis = groupby[0]

        if chart_type == "Bar":
            fig = px.bar(result, x=x_axis, y="measure")
        elif chart_type == "Line":
            fig = px.line(result, x=x_axis, y="measure")
        else:
            fig = px.scatter(result, x=x_axis, y="measure")

        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download Aggregated Result",
            result.to_csv(index=False).encode("utf-8"),
            file_name="aggregation_result.csv",
            mime="text/csv"
        )

    # Preview data
    st.subheader("3️⃣ Data Preview")
    st.dataframe(df_work.head(1000), use_container_width=True)


# ======================================================
# TAB 2 — FRAUD MODEL
# ======================================================

def run_fraud_model():

    st.header("🚨 Fraud Risk Model — Rapid Modification")

    if "filtered_data" not in st.session_state:
        st.warning("No dataset available. Please upload and select data in Tab 1.")
        return

    df_fraud = st.session_state["filtered_data"].copy()

    required_cols = ["CREATETIME", "UPDATETIME", "TRANSACTIONAMOUNT_AMT"]

    for col in required_cols:
        if col not in df_fraud.columns:
            st.error(f"Missing required column: {col}")
            return

    df_fraud["CREATETIME"] = pd.to_datetime(df_fraud["CREATETIME"], errors="coerce")
    df_fraud["UPDATETIME"] = pd.to_datetime(df_fraud["UPDATETIME"], errors="coerce")

    df_fraud["update_delay_minutes"] = (
        (df_fraud["UPDATETIME"] - df_fraud["CREATETIME"])
        .dt.total_seconds() / 60
    )

    threshold = st.slider("Rapid modification threshold (minutes)", 1, 30, 2)

    df_fraud["rapid_flag"] = df_fraud["update_delay_minutes"] < threshold

    # KPIs
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df_fraud))
    col2.metric("Rapid Modifications", int(df_fraud["rapid_flag"].sum()))
    col3.metric(
        "Rapid %",
        f"{(df_fraud['rapid_flag'].mean() * 100):.2f}%"
    )

    st.subheader("Flagged Cases")

    flagged = df_fraud[df_fraud["rapid_flag"] == True]

    st.dataframe(flagged, use_container_width=True)

    st.download_button(
        "Download Flagged Cases",
        flagged.to_csv(index=False).encode("utf-8"),
        file_name="rapid_modification_cases.csv",
        mime="text/csv"
    )

    # Visualization
    st.subheader("Risk Scatter View")

    fig = px.scatter(
        df_fraud,
        x="CREATETIME",
        y="TRANSACTIONAMOUNT_AMT",
        color="rapid_flag",
        title="Rapid Modification Risk Signal"
    )

    st.plotly_chart(fig, use_container_width=True)


# ======================================================
# MAIN APP
# ======================================================

tab1, tab2 = st.tabs(["1️⃣ Data Explorer", "2️⃣ Fraud Risk Model"])

with tab1:
    run_data_explorer()

with tab2:
    run_fraud_model()
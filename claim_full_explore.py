import streamlit as st
import pandas as pd
import numpy as np

# ======================================================
# 1) DEMO DATA - 替换成你自己的 claims DataFrame 即可
# ======================================================
def load_demo_claims():
    np.random.seed(42)
    n = 400
    df = pd.DataFrame({
        "claim_id": [f"CLM-{1000+i}" for i in range(n)],
        "claim_type": np.random.choice(["Marine", "Commercial Property", "Property", "Auto"], n),
        "state": np.random.choice(["CA", "NY", "TX", "FL", "QC"], n),
        "region": np.random.choice(["East", "West", "South", "Canada"], n),
        "adjuster": np.random.choice(["Miller", "Smith", "Chen", "Singh"], n),
        "severity": np.random.choice(["Low", "Medium", "High", "Cat"], n),
        "claim_amount": np.random.gamma(2.0, 5000, n).round(0),
        "days_to_litigation_referral": np.random.randint(0, 60, n),
        "days_to_decision": np.random.randint(0, 45, n),
        "policy_age_years": np.random.randint(0, 20, n),
        "documentation_complete": np.random.choice([True, False], n, p=[0.75, 0.25]),
    })
    return df

# ======================================================
# 2) PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Claims Audit Explorer",
    page_icon="🧭",
    layout="wide",
)

st.title("🧭 Claims Audit Explorer (A's 2nd Prototype)")
st.caption("Not a fixed dashboard – a workspace where auditors can ask any question using the same UI.")

# ======================================================
# 3) DATA SOURCE (你可以改成从 CSV / DB 读)
# ======================================================
with st.sidebar:
    st.header("Data Source")
    uploaded = st.file_uploader("Upload a CSV file (optional)", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df)} rows from uploaded file.")
    else:
        df = load_demo_claims()
        st.info("Using demo claims data. Upload a CSV to use real data.")

# Basic introspection
cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype) == "category" or df[c].dtype == "bool"]
num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]

# ======================================================
# 4) SIDEBAR - 通用过滤器（对所有 column 都适用）
# ======================================================
st.sidebar.markdown("---")
st.sidebar.header("Global Filters")

filtered_df = df.copy()

# Categorical columns: multi-select filter
for col in cat_cols:
    values = sorted(filtered_df[col].dropna().unique().tolist())
    if len(values) > 1 and len(values) <= 50:  # 避免太多
        selected = st.sidebar.multiselect(
            f"{col} (categorical)",
            options=values,
            default=values,
        )
        filtered_df = filtered_df[filtered_df[col].isin(selected)]

# Numeric columns: optional range filter
with st.sidebar.expander("Numeric filters (optional)"):
    for col in num_cols:
        col_min = float(filtered_df[col].min())
        col_max = float(filtered_df[col].max())
        if col_min == col_max:
            continue
        use_filter = st.checkbox(f"Filter {col}", value=False)
        if use_filter:
            min_val, max_val = st.slider(
                f"{col} range",
                min_value=col_min,
                max_value=col_max,
                value=(col_min, col_max),
            )
            filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]

st.sidebar.markdown("---")
st.sidebar.write(f"Rows after filters: **{len(filtered_df)}** / {len(df)}")

# ======================================================
# 5) TOP: DATA PREVIEW
# ======================================================
with st.expander("Preview filtered dataset", expanded=False):
    st.dataframe(filtered_df.head(50), use_container_width=True)

st.markdown("---")

# ======================================================
# 6) MAIN TABS: EXPLORER + SAMPLES + AI
# ======================================================
tab_explore, tab_samples, tab_ai = st.tabs(
    ["📊 Explore / Create Views", "📋 Drilldown & Samples", "🧠 Ask (AI-style) About This Data"]
)

# ======================================================
# TAB 1: 通用探索引擎 - 动态 groupby + aggregation + chart
# ======================================================
with tab_explore:
    st.subheader("📊 Dynamic Aggregation & Visualization")

    if len(filtered_df) == 0:
        st.warning("No rows after filters. Try relaxing filters in the sidebar.")
    else:
        col_controls, col_output = st.columns([1.1, 2])

        with col_controls:
            st.markdown("### 1) Choose grouping")

            group_cols = st.multiselect(
                "Group by (categorical columns)",
                options=cat_cols,
                default=["claim_type"] if "claim_type" in cat_cols else [],
                help="You can group by 1 or more columns, e.g. claim_type + state + adjuster.",
            )

            st.markdown("### 2) Choose measure")

            measure_mode = st.radio(
                "Measure type",
                options=["Count rows", "Aggregate numeric column"],
                horizontal=False,
            )

            agg_col = None
            agg_func = None

            if measure_mode == "Count rows":
                agg_col = None
                agg_func = "size"
            else:
                agg_col = st.selectbox(
                    "Numeric column to aggregate",
                    options=num_cols,
                )
                agg_func = st.selectbox(
                    "Aggregation function",
                    options=["sum", "mean", "median", "max", "min"],
                    index=1,
                )

            st.markdown("### 3) Choose chart type")
            chart_type = st.selectbox(
                "Chart type",
                options=["Bar", "Line", "Table only"],
            )

            st.markdown("### 4) Optional sort")
            sort_desc = st.checkbox("Sort by measure (descending)", value=True)

        with col_output:
            if len(group_cols) == 0:
                st.info("Please select at least one column to group by.")
            else:
                # Do aggregation dynamically
                if measure_mode == "Count rows":
                    grouped = (
                        filtered_df.groupby(group_cols)
                        .size()
                        .reset_index(name="measure")
                    )
                    measure_label = "row_count"
                else:
                    grouped = (
                        filtered_df.groupby(group_cols)[agg_col]
                        .agg(agg_func)
                        .reset_index(name="measure")
                    )
                    measure_label = f"{agg_func}({agg_col})"

                if sort_desc:
                    grouped = grouped.sort_values("measure", ascending=False)

                st.markdown(f"#### Result – {measure_label}")
                st.dataframe(grouped, use_container_width=True)

                # Simple pivot-like chart: if group-by 1 column → 1D
                if chart_type != "Table only":
                    if len(group_cols) == 1:
                        chart_df = grouped.set_index(group_cols[0])["measure"]
                        if chart_type == "Bar":
                            st.bar_chart(chart_df, use_container_width=True)
                        elif chart_type == "Line":
                            st.line_chart(chart_df, use_container_width=True)
                    else:
                        # 多维 groupby，用 table 展示 + 提示
                        st.info(
                            "Multiple group-by columns selected; showing table. "
                            "For clearer charts, try grouping by only 1 column."
                        )

# ======================================================
# TAB 2: Drilldown & Samples – 任意子集抽样
# ======================================================
with tab_samples:
    st.subheader("📋 Drilldown & Sample Extraction")

    st.markdown(
        "Use the global filters + grouping above to narrow down the population, "
        "then use this tab to inspect and export samples."
    )

    # Row limit for view
    max_rows = st.slider("Max rows to display in table", min_value=10, max_value=500, value=100, step=10)
    st.dataframe(filtered_df.head(max_rows), use_container_width=True)

    # Simple random sample extraction
    st.markdown("### Random sample (for audit testing)")
    sample_size = st.number_input(
        "Sample size",
        min_value=1,
        max_value=max(1, len(filtered_df)),
        value=min(20, len(filtered_df)),
        step=1,
    )

    if st.button("Draw sample"):
        sample_df = filtered_df.sample(sample_size, random_state=42)
        st.success(f"Sample of {sample_size} rows drawn from filtered population.")
        st.dataframe(sample_df, use_container_width=True)

        # Optionally allow download
        csv = sample_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download sample as CSV",
            data=csv,
            file_name="claim_audit_sample.csv",
            mime="text/csv",
        )

# ======================================================
# TAB 3: AI-style Q&A（占位，未来接 LLM）
# ======================================================
with tab_ai:
    st.subheader("🧠 Ask (AI-style) About the Current Filtered Data")

    st.markdown(
        "In a real deployment, this tab would send both your question AND a summary of the filtered dataset "
        "to an LLM (e.g., GPT) to generate explanations, risk analysis, and suggested audit findings.\n\n"
        "Here we only simulate the UI."
    )

    question = st.text_area(
        "Example questions an auditor might ask:",
        placeholder=(
            "• Which claim types show the highest severity and payout?\n"
            "• Are BI litigation referrals late for any particular adjuster?\n"
            "• How do Workers Comp decisions in Texas compare to New York?"
        ),
        height=120,
    )

    if st.button("Generate AI-style answer (demo)"):
        st.info(
            "In production, this would call an LLM with a prompt built from your question and the filtered data.\n"
            "The model would return root causes, patterns, and audit findings."
        )
        st.write("💬 _Demo placeholder:_\n\n"
                 "Based on the current filters, there appears to be a concentration of higher claim_amount "
                 "and longer days_to_decision in certain states and adjusters. This may indicate potential "
                 "control weaknesses or capacity issues that warrant further audit testing.")

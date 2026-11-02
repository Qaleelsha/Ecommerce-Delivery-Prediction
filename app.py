
from pathlib import Path
import glob

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Delivery Intelligence | Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. PROFESSIONAL DARK THEME
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0b1120;
    color: #e5e7eb;
}

[data-testid="stHeader"] {
    background: rgba(11, 17, 32, 0.9);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    background: #111a2b;
    border-right: 1px solid #253149;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label {
    color: #e5e7eb;
}

[data-testid="stSelectbox"] > div > div {
    background-color: #1a2538;
    border: 1px solid #334155;
    border-radius: 8px;
    color: #f8fafc;
}

.dashboard-header {
    font-size: 32px;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -1px;
    margin-bottom: 2px;
}

.dashboard-subtitle {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 20px;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #f1f5f9;
    margin-top: 10px;
    margin-bottom: 12px;
}

.section-caption {
    color: #94a3b8;
    font-size: 13px;
    margin-top: -8px;
    margin-bottom: 16px;
}

.kpi-card {
    background: linear-gradient(145deg, #172338, #111b2d);
    border: 1px solid #293950;
    border-radius: 14px;
    padding: 22px 22px 20px 22px;
    min-height: 135px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.12);
}

.kpi-label {
    font-size: 13px;
    font-weight: 600;
    color: #94a3b8;
    margin-bottom: 12px;
}

.kpi-value {
    font-size: 30px;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.8px;
}

.kpi-foot {
    font-size: 12px;
    color: #64748b;
    margin-top: 8px;
}

.kpi-accent-blue {
    border-top: 3px solid #38bdf8;
}

.kpi-accent-red {
    border-top: 3px solid #fb7185;
}

.kpi-accent-purple {
    border-top: 3px solid #a78bfa;
}

.kpi-accent-green {
    border-top: 3px solid #34d399;
}

div[data-testid="stMetric"] {
    background: #172338;
    border: 1px solid #293950;
    padding: 15px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

div[data-testid="stMetricValue"] {
    color: #f8fafc;
}

div[data-testid="stTabs"] button {
    color: #94a3b8;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #38bdf8;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #293950;
    border-radius: 10px;
}

hr {
    border-color: #263449;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 20px 0 10px 0;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. PLOTLY DARK CHART THEME
# =========================================================

CHART_TEMPLATE = "plotly_dark"

CHART_BG = "#111a2b"
PAPER_BG = "#111a2b"
GRID_COLOR = "#263449"
TEXT_COLOR = "#cbd5e1"

BLUE = "#38bdf8"
PURPLE = "#a78bfa"
RED = "#fb7185"
GREEN = "#34d399"
ORANGE = "#fbbf24"


def style_chart(fig, height=380):

    fig.update_layout(
        template=CHART_TEMPLATE,
        height=height,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        font=dict(
            family="Inter",
            color=TEXT_COLOR,
            size=12
        ),
        title_font=dict(
            size=16,
            color="#f1f5f9"
        ),
        margin=dict(
            l=25,
            r=25,
            t=55,
            b=35
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zeroline=False
        )
    )

    return fig


# =========================================================
# 4. LOAD PREDICTIONS
# =========================================================

@st.cache_data
def load_data():

    files = glob.glob("outputs/predictions/*.csv")

    if not files:
        return None

    df = pd.read_csv(
        files[0],
        encoding="utf-8",
        encoding_errors="replace"
    )

    df["Late_delivery_risk"] = (
        df["Late_delivery_risk"].astype(int)
    )

    df["prediction"] = (
        df["prediction"].astype(int)
    )

    df["Actual Status"] = df["Late_delivery_risk"].map({
        0: "On Time",
        1: "Late"
    })

    df["Predicted Status"] = df["prediction"].map({
        0: "On Time",
        1: "Late"
    })

    return df


df = load_data()


if df is None:

    st.error("Prediction data not found.")

    st.info(
        "Please run `python src/delivery_prediction.py` "
        "to generate the predictions."
    )

    st.stop()


# =========================================================
# 5. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 📦 Delivery Analytics")

    st.caption("MACHINE LEARNING • SUPPLY CHAIN")

    st.divider()

    st.markdown("### Filters")

    regions = ["All Regions"] + sorted(
        df["Order Region"].dropna().unique().tolist()
    )

    modes = ["All Modes"] + sorted(
        df["Shipping Mode"].dropna().unique().tolist()
    )

    selected_region = st.selectbox(
        "🌍 Order Region",
        regions
    )

    selected_mode = st.selectbox(
        "🚚 Shipping Mode",
        modes
    )

    st.divider()

    st.markdown("### Model Information")

    st.markdown("""
    **Algorithm:** Decision Tree

    **Framework:** Apache Spark MLlib

    **Training split:** 80%

    **Testing split:** 20%
    """)

    st.divider()

    st.caption("Dataset: DataCo Smart Supply Chain")


# =========================================================
# 6. APPLY FILTERS
# =========================================================

filtered_df = df.copy()

if selected_region != "All Regions":

    filtered_df = filtered_df[
        filtered_df["Order Region"] == selected_region
    ]

if selected_mode != "All Modes":

    filtered_df = filtered_df[
        filtered_df["Shipping Mode"] == selected_mode
    ]


# =========================================================
# 7. HEADER
# =========================================================

st.markdown(
    '<div class="dashboard-header">'
    '📦 E-Commerce Delivery Intelligence'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Predictive analytics for delivery performance, '
    'shipment risk, and supply chain insights.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# 8. KEY PERFORMANCE INDICATORS
# =========================================================

total_orders = len(filtered_df)

actual_late = (
    filtered_df["Late_delivery_risk"] == 1
).sum()

predicted_late = (
    filtered_df["prediction"] == 1
).sum()

correct_predictions = (
    filtered_df["Late_delivery_risk"]
    == filtered_df["prediction"]
).sum()

accuracy = (
    correct_predictions / total_orders * 100
    if total_orders > 0 else 0
)

actual_late_rate = (
    actual_late / total_orders * 100
    if total_orders > 0 else 0
)

predicted_late_rate = (
    predicted_late / total_orders * 100
    if total_orders > 0 else 0
)


def kpi_card(title, value, subtitle, accent):

    st.markdown(
        f"""
        <div class="kpi-card {accent}">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-foot">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


c1, c2, c3, c4 = st.columns(4)

with c1:
    kpi_card(
        "TOTAL TEST ORDERS",
        f"{total_orders:,}",
        "Orders evaluated by the model",
        "kpi-accent-blue"
    )

with c2:
    kpi_card(
        "ACTUAL LATE DELIVERIES",
        f"{actual_late:,}",
        f"{actual_late_rate:.1f}% of test orders",
        "kpi-accent-red"
    )

with c3:
    kpi_card(
        "PREDICTED LATE DELIVERIES",
        f"{predicted_late:,}",
        f"{predicted_late_rate:.1f}% predicted risk",
        "kpi-accent-purple"
    )

with c4:
    kpi_card(
        "MODEL ACCURACY",
        f"{accuracy:.2f}%",
        "Correct predictions in selected data",
        "kpi-accent-green"
    )


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# 9. TABS
# =========================================================

tab_overview, tab_delivery, tab_model, tab_data = st.tabs([
    "📊 Overview",
    "🚚 Delivery Insights",
    "🧠 Model Performance",
    "📋 Data Explorer"
])


# =========================================================
# TAB 1: OVERVIEW
# =========================================================

with tab_overview:

    st.markdown(
        '<div class="section-title">'
        'Actual vs Predicted Delivery Status'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Comparison of actual delivery risk and model predictions.'
        '</div>',
        unsafe_allow_html=True
    )

    actual_counts = (
        filtered_df["Actual Status"]
        .value_counts()
        .reindex(["Late", "On Time"], fill_value=0)
    )

    predicted_counts = (
        filtered_df["Predicted Status"]
        .value_counts()
        .reindex(["Late", "On Time"], fill_value=0)
    )

    status_chart = go.Figure()

    status_chart.add_trace(
        go.Bar(
            name="Actual",
            x=["Late", "On Time"],
            y=actual_counts.values,
            marker_color=BLUE
        )
    )

    status_chart.add_trace(
        go.Bar(
            name="Predicted",
            x=["Late", "On Time"],
            y=predicted_counts.values,
            marker_color=PURPLE
        )
    )

    status_chart.update_layout(
        barmode="group",
        xaxis_title="Delivery Status",
        yaxis_title="Number of Orders"
    )

    style_chart(status_chart, 390)

    st.plotly_chart(
        status_chart,
        use_container_width=True
    )

    left, right = st.columns(2)

    with left:

        st.markdown(
            '<div class="section-title">'
            'Actual Delivery Distribution'
            '</div>',
            unsafe_allow_html=True
        )

        pie_data = filtered_df["Actual Status"].value_counts()

        fig = px.pie(
            values=pie_data.values,
            names=pie_data.index,
            hole=0.65,
            color=pie_data.index,
            color_discrete_map={
                "Late": RED,
                "On Time": GREEN
            }
        )

        fig.update_traces(
            textinfo="percent",
            textfont=dict(size=14)
        )

        style_chart(fig, 340)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.markdown(
            '<div class="section-title">'
            'Prediction Distribution'
            '</div>',
            unsafe_allow_html=True
        )

        prediction_data = (
            filtered_df["Predicted Status"]
            .value_counts()
        )

        fig = px.pie(
            values=prediction_data.values,
            names=prediction_data.index,
            hole=0.65,
            color=prediction_data.index,
            color_discrete_map={
                "Late": RED,
                "On Time": BLUE
            }
        )

        fig.update_traces(
            textinfo="percent",
            textfont=dict(size=14)
        )

        style_chart(fig, 340)

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# TAB 2: DELIVERY INSIGHTS
# =========================================================

with tab_delivery:

    st.markdown(
        '<div class="section-title">'
        'Shipping Mode Performance'
        '</div>',
        unsafe_allow_html=True
    )

    shipping_analysis = (
        filtered_df.groupby("Shipping Mode")
        .agg(
            Total_Orders=("Late_delivery_risk", "count"),
            Late_Orders=("Late_delivery_risk", "sum")
        )
        .reset_index()
    )

    shipping_analysis["Late Delivery Rate (%)"] = (
        shipping_analysis["Late_Orders"]
        / shipping_analysis["Total_Orders"] * 100
    )

    fig = px.bar(
        shipping_analysis,
        x="Shipping Mode",
        y="Late Delivery Rate (%)",
        color="Late Delivery Rate (%)",
        color_continuous_scale=[
            "#38bdf8",
            "#818cf8",
            "#fb7185"
        ],
        text=shipping_analysis["Late Delivery Rate (%)"].round(1)
    )

    fig.update_traces(
        texttemplate="%{text}%",
        textposition="outside"
    )

    fig.update_layout(
        coloraxis_showscale=False,
        yaxis_title="Late Delivery Rate (%)"
    )

    style_chart(fig, 400)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">'
        'Regional Delivery Risk'
        '</div>',
        unsafe_allow_html=True
    )

    region_analysis = (
        filtered_df.groupby("Order Region")
        .agg(
            Total_Orders=("Late_delivery_risk", "count"),
            Late_Orders=("Late_delivery_risk", "sum")
        )
        .reset_index()
    )

    region_analysis["Late Delivery Rate (%)"] = (
        region_analysis["Late_Orders"]
        / region_analysis["Total_Orders"] * 100
    )

    region_analysis = region_analysis.sort_values(
        "Late Delivery Rate (%)",
        ascending=True
    )

    fig = px.bar(
        region_analysis,
        x="Late Delivery Rate (%)",
        y="Order Region",
        orientation="h",
        color="Late Delivery Rate (%)",
        color_continuous_scale=[
            "#38bdf8",
            "#818cf8",
            "#fb7185"
        ],
        hover_data=[
            "Total_Orders",
            "Late_Orders"
        ]
    )

    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Late Delivery Rate (%)",
        yaxis_title=""
    )

    style_chart(
        fig,
        max(400, len(region_analysis) * 30)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">'
        'Scheduled Shipping Days'
        '</div>',
        unsafe_allow_html=True
    )

    schedule_analysis = (
        filtered_df.groupby("Days for shipment (scheduled)")
        .agg(
            Total_Orders=("Late_delivery_risk", "count"),
            Late_Orders=("Late_delivery_risk", "sum")
        )
        .reset_index()
    )

    schedule_analysis["Late Delivery Rate (%)"] = (
        schedule_analysis["Late_Orders"]
        / schedule_analysis["Total_Orders"] * 100
    )

    fig = px.line(
        schedule_analysis,
        x="Days for shipment (scheduled)",
        y="Late Delivery Rate (%)",
        markers=True
    )

    fig.update_traces(
        line=dict(color=BLUE, width=3),
        marker=dict(size=9, color=PURPLE)
    )

    fig.update_layout(
        xaxis_title="Scheduled Shipping Days",
        yaxis_title="Late Delivery Rate (%)"
    )

    style_chart(fig, 350)

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# TAB 3: MODEL PERFORMANCE
# =========================================================

with tab_model:

    st.markdown(
        '<div class="section-title">'
        'Decision Tree Classifier'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Overall evaluation metrics calculated on the test dataset.'
        '</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Accuracy", "69.50%")
    m2.metric("Weighted Precision", "73.66%")
    m3.metric("Weighted Recall", "69.50%")
    m4.metric("F1 Score", "68.99%")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">'
        'Confusion Matrix'
        '</div>',
        unsafe_allow_html=True
    )

    confusion = pd.crosstab(
        filtered_df["Late_delivery_risk"],
        filtered_df["prediction"]
    )

    confusion = confusion.reindex(
        index=[0, 1],
        columns=[0, 1],
        fill_value=0
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=confusion.values,
            x=["Predicted On Time", "Predicted Late"],
            y=["Actual On Time", "Actual Late"],
            colorscale=[
                [0, "#172338"],
                [0.5, "#2563eb"],
                [1, "#38bdf8"]
            ],
            text=confusion.values,
            texttemplate="%{text:,}",
            textfont=dict(
                size=20,
                color="white"
            ),
            showscale=True
        )
    )

    fig.update_layout(
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class"
    )

    style_chart(fig, 400)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "The model correctly identifies some late deliveries, "
        "but also misses late orders. This highlights the "
        "importance of improving recall for late-delivery risk."
    )


# =========================================================
# TAB 4: DATA EXPLORER
# =========================================================

with tab_data:

    st.markdown(
        '<div class="section-title">'
        'Prediction Records'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Explore the test dataset and compare actual and predicted results.'
        '</div>',
        unsafe_allow_html=True
    )

    search = st.text_input(
        "🔍 Search by country, region, or shipping mode"
    )

    display_df = filtered_df.copy()

    if search:

        mask = (
            display_df["Order Country"].astype(str)
            .str.contains(search, case=False, na=False)
            |
            display_df["Order Region"].astype(str)
            .str.contains(search, case=False, na=False)
            |
            display_df["Shipping Mode"].astype(str)
            .str.contains(search, case=False, na=False)
        )

        display_df = display_df[mask]

    st.dataframe(
        display_df[
            [
                "Late_delivery_risk",
                "prediction",
                "Actual Status",
                "Predicted Status",
                "Shipping Mode",
                "Order Region",
                "Order Country",
                "Market",
                "Days for shipment (scheduled)"
            ]
        ],
        use_container_width=True,
        height=400
    )

    csv = display_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Predictions",
        data=csv,
        file_name="delivery_predictions.csv",
        mime="text/csv"
    )


# =========================================================
# 10. FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    E-COMMERCE DELIVERY DELAY PREDICTION
    <br>
    Powered by Apache Spark MLlib • PySpark • Streamlit • Plotly
</div>
""", unsafe_allow_html=True)

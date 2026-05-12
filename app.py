import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Nordic Electricity Market Analytics",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("Nordic Electricity Market Intelligence Platform")

st.markdown("""
AI-driven electricity market analytics and forecasting platform
built using real Finland electricity pricing data from ENTSO-E.
""")

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/cleaned_finland_energy_prices.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Filters")

selected_month = st.sidebar.selectbox(
    "Select Month",
    sorted(df["month"].unique())
)

filtered_df = df[df["month"] == selected_month]

# ============================================================
# METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Price",
    f"{filtered_df['price'].mean():.2f} EUR/MWh"
)

col2.metric(
    "Maximum Price",
    f"{filtered_df['price'].max():.2f} EUR/MWh"
)

col3.metric(
    "Minimum Price",
    f"{filtered_df['price'].min():.2f} EUR/MWh"
)

# ============================================================
# PRICE TREND
# ============================================================

st.subheader("Electricity Prices Over Time")

fig = px.line(
    filtered_df,
    x="timestamp",
    y="price",
    title="Finland Electricity Prices"
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# HOURLY ANALYSIS
# ============================================================

st.subheader("Average Electricity Price by Hour")

hourly_avg = filtered_df.groupby("hour")["price"].mean()

fig2 = px.line(
    x=hourly_avg.index,
    y=hourly_avg.values,
    labels={
        "x": "Hour",
        "y": "Average Price"
    }
)

st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# MONTHLY ANALYSIS
# ============================================================

st.subheader("Average Electricity Price by Month")

monthly_avg = df.groupby("month")["price"].mean()

fig3 = px.bar(
    x=monthly_avg.index,
    y=monthly_avg.values,
    labels={
        "x": "Month",
        "y": "Average Price"
    }
)

st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# VOLATILITY DETECTION
# ============================================================

st.subheader("Market Volatility Detection")

df["price_change"] = df["price"].diff()

threshold = df["price_change"].std() * 2

volatility = df[
    abs(df["price_change"]) > threshold
]

st.write("Detected High Volatility Events")

st.dataframe(
    volatility[[
        "timestamp",
        "price",
        "price_change"
    ]].head(20)
)

# ============================================================
# RAW DATA
# ============================================================

st.subheader("Raw Dataset")

st.dataframe(filtered_df.head(100))

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
Developed using:
- Python
- Streamlit
- Scikit-learn
- ENTSO-E Electricity Market Data
""")

"""
Nordic Electricity Market Intelligence Platform — Streamlit Dashboard

A production-grade analytics dashboard featuring:
- Real-time KPI metrics with delta indicators
- Multi-model comparison table and radar chart
- Forecast visualization with actual vs predicted overlay
- Feature importance comparison (RF vs XGBoost)
- Residual analysis and error heatmaps
- Enhanced volatility detection with rolling regime analysis
- Interactive data explorer with filters
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DATA_DIR, RESULTS_DIR, MODELS_DIR, TARGET_COL, TIMESTAMP_COL,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Nordic Electricity Market Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        opacity: 0.8;
        font-size: 1rem;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(0, 210, 255, 0.2);
        padding: 1rem;
        border-radius: 10px;
    }
    [data-testid="stMetricLabel"] {
        color: #a0a0c0 !important;
    }
    [data-testid="stMetricValue"] {
        color: #00d2ff !important;
    }

    /* Section headers */
    .section-header {
        border-left: 4px solid #3a7bd5;
        padding-left: 12px;
        margin: 1.5rem 0 1rem 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }

    /* Tables */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    div.stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    div.stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_dataset():
    """Load the main dataset."""
    df = pd.read_csv(DATA_DIR / "cleaned_finland_energy_prices.csv")
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    return df


@st.cache_data
def load_results():
    """Load pipeline results if available."""
    results = {}

    # Model comparison
    comp_path = RESULTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        results["comparison"] = pd.read_csv(comp_path)

    # Residuals
    res_path = RESULTS_DIR / "residuals.csv"
    if res_path.exists():
        results["residuals"] = pd.read_csv(res_path)

    # Error breakdowns
    for key in ["by_hour", "by_month", "by_day_of_week"]:
        bp = RESULTS_DIR / f"error_{key}.csv"
        if bp.exists():
            results[f"error_{key}"] = pd.read_csv(bp)

    # Summary
    summary_path = RESULTS_DIR / "evaluation_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            results["summary"] = json.load(f)

    # Feature importances
    for model_name in ["random_forest", "xgboost", "lightgbm"]:
        fi_path = RESULTS_DIR / f"feature_importance_{model_name}.csv"
        if fi_path.exists():
            results[f"fi_{model_name}"] = pd.read_csv(fi_path)

    return results


df = load_dataset()
results = load_results()
has_results = "comparison" in results

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>⚡ Nordic Electricity Market Intelligence Platform</h1>
    <p>AI-driven electricity market analytics and forecasting | Finland Day-Ahead Prices | ENTSO-E Data</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Flag_of_Finland.svg/255px-Flag_of_Finland.svg.png", width=60)
    st.title("🎛️ Controls")

    st.markdown("---")

    # Date range filter
    st.subheader("📅 Date Range")
    min_date = df[TIMESTAMP_COL].min().date()
    max_date = df[TIMESTAMP_COL].max().date()
    date_range = st.date_input(
        "Select range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    st.markdown("---")

    # Month filter
    st.subheader("📆 Month Filter")
    months = sorted(df["month"].unique())
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    selected_months = st.multiselect(
        "Select months",
        options=months,
        default=months,
        format_func=lambda x: month_names.get(x, str(x)),
    )

    st.markdown("---")

    # Hour range
    st.subheader("🕐 Hour Range")
    hour_range = st.slider("Hours", 0, 23, (0, 23))

    st.markdown("---")
    st.markdown("**Pipeline Status**")
    if has_results:
        st.success("✅ Models trained")
        st.info(f"Best: {results['summary']['best_model']}")
    else:
        st.warning("⚠️ Run `python run_pipeline.py` first")

# Apply filters
filtered_df = df.copy()
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df[TIMESTAMP_COL].dt.date >= date_range[0]) &
        (filtered_df[TIMESTAMP_COL].dt.date <= date_range[1])
    ]
filtered_df = filtered_df[filtered_df["month"].isin(selected_months)]
filtered_df = filtered_df[
    (filtered_df["hour"] >= hour_range[0]) &
    (filtered_df["hour"] <= hour_range[1])
]

# ============================================================
# KPI METRICS ROW
# ============================================================

st.markdown('<div class="section-header"><h3>📊 Market Overview</h3></div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

avg_price = filtered_df[TARGET_COL].mean()
max_price = filtered_df[TARGET_COL].max()
min_price = filtered_df[TARGET_COL].min()
std_price = filtered_df[TARGET_COL].std()
total_records = len(filtered_df)

col1.metric("Avg Price", f"€{avg_price:.2f}/MWh")
col2.metric("Max Price", f"€{max_price:.2f}/MWh")
col3.metric("Min Price", f"€{min_price:.2f}/MWh")
col4.metric("Volatility (σ)", f"€{std_price:.2f}")
col5.metric("Records", f"{total_records:,}")

# ============================================================
# TABS
# ============================================================

if has_results:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Price Trends", "🏆 Model Comparison", "🔮 Forecast",
        "📊 Feature Importance", "🔍 Error Analysis", "⚡ Volatility"
    ])
else:
    tab1, tab5_alt, tab6 = st.tabs(["📈 Price Trends", "📊 Hourly/Monthly", "⚡ Volatility"])

# ── TAB 1: Price Trends ───────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header"><h3>Finland Electricity Prices Over Time</h3></div>', unsafe_allow_html=True)

    fig = px.line(
        filtered_df, x=TIMESTAMP_COL, y=TARGET_COL,
        labels={TIMESTAMP_COL: "Date", TARGET_COL: "Price (EUR/MWh)"},
        template="plotly_dark",
    )
    fig.update_traces(line=dict(color="#00d2ff", width=1))
    fig.update_layout(
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig, width='stretch')

    # Hourly and Monthly side by side
    col_h, col_m = st.columns(2)

    with col_h:
        st.markdown("**Average Price by Hour**")
        hourly_avg = filtered_df.groupby("hour")[TARGET_COL].mean().reset_index()
        fig_h = px.bar(
            hourly_avg, x="hour", y=TARGET_COL,
            labels={"hour": "Hour of Day", TARGET_COL: "Avg Price (EUR/MWh)"},
            template="plotly_dark",
            color=TARGET_COL,
            color_continuous_scale="Viridis",
        )
        fig_h.update_layout(height=350, showlegend=False,
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_h, width='stretch')

    with col_m:
        st.markdown("**Average Price by Month**")
        monthly_avg = filtered_df.groupby("month")[TARGET_COL].mean().reset_index()
        fig_m = px.bar(
            monthly_avg, x="month", y=TARGET_COL,
            labels={"month": "Month", TARGET_COL: "Avg Price (EUR/MWh)"},
            template="plotly_dark",
            color=TARGET_COL,
            color_continuous_scale="Plasma",
        )
        fig_m.update_layout(height=350, showlegend=False,
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_m, width='stretch')

    # Weekday analysis
    st.markdown("**Average Price by Day of Week**")
    weekday_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    weekday_avg = filtered_df.groupby("weekday")[TARGET_COL].mean().reset_index()
    weekday_avg["day_name"] = weekday_avg["weekday"].map(weekday_names)
    fig_w = px.bar(
        weekday_avg, x="day_name", y=TARGET_COL,
        labels={"day_name": "Day", TARGET_COL: "Avg Price (EUR/MWh)"},
        template="plotly_dark",
        color=TARGET_COL,
        color_continuous_scale="Cividis",
    )
    fig_w.update_layout(height=300, showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_w, width='stretch')


if has_results:
    # ── TAB 2: Model Comparison ───────────────────────────────
    with tab2:
        st.markdown('<div class="section-header"><h3>🏆 Model Comparison</h3></div>', unsafe_allow_html=True)

        comp_df = results["comparison"]
        best = results["summary"]["best_model"]

        # Highlight best model
        st.success(f"**Best Model: {best.upper()}** (selected by lowest RMSE)")

        # Styled comparison table
        st.dataframe(
            comp_df.style.highlight_min(
                subset=["mae", "rmse", "mape", "train_time_s"],
                color="rgba(0, 210, 255, 0.2)",
            ).highlight_max(
                subset=["r2", "directional_accuracy"],
                color="rgba(0, 210, 255, 0.2)",
            ).format({
                "mae": "{:.4f}", "rmse": "{:.4f}", "mape": "{:.2f}%",
                "r2": "{:.4f}", "directional_accuracy": "{:.1f}%",
                "train_time_s": "{:.2f}s",
            }),
            width='stretch',
            hide_index=True,
        )

        # Radar chart
        st.markdown("**Performance Radar**")
        categories = ["MAE", "RMSE", "R²", "Dir. Acc.", "Speed"]

        fig_radar = go.Figure()
        for _, row in comp_df.iterrows():
            # Normalize metrics to 0-1 scale for radar
            mae_norm = 1 - (row["mae"] / comp_df["mae"].max()) if comp_df["mae"].max() > 0 else 0
            rmse_norm = 1 - (row["rmse"] / comp_df["rmse"].max()) if comp_df["rmse"].max() > 0 else 0
            r2_norm = row["r2"] if row["r2"] > 0 else 0
            da_norm = row["directional_accuracy"] / 100 if not pd.isna(row["directional_accuracy"]) else 0
            speed_norm = 1 - (row["train_time_s"] / comp_df["train_time_s"].max()) if comp_df["train_time_s"].max() > 0 else 0

            fig_radar.add_trace(go.Scatterpolar(
                r=[mae_norm, rmse_norm, r2_norm, da_norm, speed_norm],
                theta=categories,
                fill="toself",
                name=row["model"],
                opacity=0.6,
            ))

        fig_radar.update_layout(
            polar=dict(bgcolor="rgba(0,0,0,0)"),
            template="plotly_dark",
            height=450,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar, width='stretch')

        # Bar chart comparison
        col_bar1, col_bar2 = st.columns(2)
        with col_bar1:
            fig_mae = px.bar(
                comp_df, x="model", y="mae",
                title="MAE by Model", template="plotly_dark",
                color="mae", color_continuous_scale="Reds_r",
            )
            fig_mae.update_layout(height=300, showlegend=False,
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_mae, width='stretch')

        with col_bar2:
            fig_r2 = px.bar(
                comp_df, x="model", y="r2",
                title="R² by Model", template="plotly_dark",
                color="r2", color_continuous_scale="Blues",
            )
            fig_r2.update_layout(height=300, showlegend=False,
                                 plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_r2, width='stretch')


    # ── TAB 3: Forecast Visualization ─────────────────────────
    with tab3:
        st.markdown('<div class="section-header"><h3>🔮 Actual vs Predicted</h3></div>', unsafe_allow_html=True)

        if "residuals" in results:
            res_df = results["residuals"]
            res_df["timestamp"] = pd.to_datetime(res_df["timestamp"])

            # Show a window of predictions
            n_show = st.slider("Number of points to display", 100, len(res_df), min(500, len(res_df)), step=100)
            show_df = res_df.head(n_show)

            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(
                x=show_df["timestamp"], y=show_df["actual"],
                name="Actual", line=dict(color="#00d2ff", width=1.5),
            ))
            fig_forecast.add_trace(go.Scatter(
                x=show_df["timestamp"], y=show_df["predicted"],
                name="Predicted", line=dict(color="#ff6b6b", width=1.5, dash="dot"),
            ))
            fig_forecast.update_layout(
                template="plotly_dark", height=450,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Date", yaxis_title="Price (EUR/MWh)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_forecast, width='stretch')

            # Residual distribution
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown("**Residual Distribution**")
                fig_hist = px.histogram(
                    res_df, x="residual", nbins=80,
                    template="plotly_dark",
                    color_discrete_sequence=["#3a7bd5"],
                    labels={"residual": "Prediction Error (EUR/MWh)"},
                )
                fig_hist.update_layout(height=350, showlegend=False,
                                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_hist, width='stretch')

            with col_res2:
                st.markdown("**Predicted vs Actual Scatter**")
                fig_scatter = px.scatter(
                    res_df.sample(min(2000, len(res_df))),
                    x="actual", y="predicted",
                    template="plotly_dark",
                    color="abs_residual",
                    color_continuous_scale="YlOrRd",
                    labels={"actual": "Actual Price", "predicted": "Predicted Price"},
                    opacity=0.5,
                )
                # Perfect prediction line
                price_range = [res_df["actual"].min(), res_df["actual"].max()]
                fig_scatter.add_trace(go.Scatter(
                    x=price_range, y=price_range,
                    mode="lines", line=dict(color="white", dash="dash", width=1),
                    showlegend=False,
                ))
                fig_scatter.update_layout(height=350,
                                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_scatter, width='stretch')


    # ── TAB 4: Feature Importance ─────────────────────────────
    with tab4:
        st.markdown('<div class="section-header"><h3>📊 Feature Importance Analysis</h3></div>', unsafe_allow_html=True)

        fi_models = [k.replace("fi_", "") for k in results if k.startswith("fi_")]

        if fi_models:
            cols = st.columns(len(fi_models))
            for i, model_name in enumerate(fi_models):
                with cols[i]:
                    st.markdown(f"**{model_name.replace('_', ' ').title()}**")
                    fi_df = results[f"fi_{model_name}"].head(15)
                    fig_fi = px.bar(
                        fi_df, x="importance", y="feature",
                        orientation="h", template="plotly_dark",
                        color="importance",
                        color_continuous_scale="Viridis",
                    )
                    fig_fi.update_layout(
                        height=500, showlegend=False,
                        yaxis=dict(autorange="reversed"),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_fi, width='stretch')
        else:
            st.info("No feature importance data available. Run the pipeline first.")


    # ── TAB 5: Error Analysis ─────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header"><h3>🔍 Error Analysis</h3></div>', unsafe_allow_html=True)

        col_e1, col_e2 = st.columns(2)

        with col_e1:
            if "error_by_hour" in results:
                st.markdown("**MAE by Hour of Day**")
                err_h = results["error_by_hour"]
                fig_eh = px.bar(
                    err_h, x=err_h.columns[0], y="mae",
                    template="plotly_dark",
                    color="mae", color_continuous_scale="OrRd",
                    labels={err_h.columns[0]: "Hour", "mae": "MAE (EUR/MWh)"},
                )
                fig_eh.update_layout(height=350, showlegend=False,
                                     plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_eh, width='stretch')

        with col_e2:
            if "error_by_month" in results:
                st.markdown("**MAE by Month**")
                err_m = results["error_by_month"]
                fig_em = px.bar(
                    err_m, x=err_m.columns[0], y="mae",
                    template="plotly_dark",
                    color="mae", color_continuous_scale="Purples",
                    labels={err_m.columns[0]: "Month", "mae": "MAE (EUR/MWh)"},
                )
                fig_em.update_layout(height=350, showlegend=False,
                                     plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_em, width='stretch')

        if "error_by_day_of_week" in results:
            st.markdown("**MAE by Day of Week**")
            err_dow = results["error_by_day_of_week"]
            weekday_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
            err_dow["day_name"] = err_dow.iloc[:, 0].map(weekday_map)
            fig_edow = px.bar(
                err_dow, x="day_name", y="mae",
                template="plotly_dark",
                color="mae", color_continuous_scale="Tealgrn",
                labels={"day_name": "Day", "mae": "MAE (EUR/MWh)"},
            )
            fig_edow.update_layout(height=300, showlegend=False,
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_edow, width='stretch')


# ── Volatility Tab (always available) ─────────────────────────
with tab6:
    st.markdown('<div class="section-header"><h3>⚡ Market Volatility Detection</h3></div>', unsafe_allow_html=True)

    vol_df = filtered_df.copy()
    vol_df["price_change"] = vol_df[TARGET_COL].diff()
    vol_df["rolling_vol_24h"] = vol_df[TARGET_COL].rolling(96).std()

    # Volatility trend
    st.markdown("**24-Hour Rolling Volatility**")
    fig_vol = px.line(
        vol_df, x=TIMESTAMP_COL, y="rolling_vol_24h",
        template="plotly_dark",
        labels={TIMESTAMP_COL: "Date", "rolling_vol_24h": "Rolling Std Dev (EUR/MWh)"},
    )
    fig_vol.update_traces(line=dict(color="#ff6b6b", width=1))
    fig_vol.update_layout(height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_vol, width='stretch')

    # Spike detection
    threshold = vol_df["price_change"].std() * 2
    spikes = vol_df[abs(vol_df["price_change"]) > threshold].copy()

    col_v1, col_v2 = st.columns([1, 2])
    with col_v1:
        st.metric("Volatility Events", f"{len(spikes):,}")
        st.metric("Threshold", f"±{threshold:.2f} EUR/MWh")
        st.metric("Max Spike", f"{vol_df['price_change'].max():.2f} EUR/MWh")
        st.metric("Max Drop", f"{vol_df['price_change'].min():.2f} EUR/MWh")

    with col_v2:
        st.markdown("**High Volatility Events (Top 20)**")
        if not spikes.empty:
            display_spikes = spikes[[TIMESTAMP_COL, TARGET_COL, "price_change"]].sort_values(
                "price_change", key=abs, ascending=False
            ).head(20)
            st.dataframe(display_spikes, width='stretch', hide_index=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.markdown("**Tech Stack**: Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Streamlit")
with col_f2:
    st.markdown("**Data Source**: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)")
with col_f3:
    st.markdown("**Author**: M. Sabda Pyari | B.Tech EE (CS)")

# ⚡ Nordic Electricity Market Forecasting Platform

A production-grade ML pipeline for **Finland day-ahead electricity price forecasting** using real market data from the ENTSO-E Transparency Platform. Built as a multi-model time-series forecasting system with proper temporal validation, advanced feature engineering, and an interactive analytics dashboard.

---

## 📌 Problem Statement

Electricity markets are highly dynamic — prices fluctuate with demand, renewable generation, seasonal patterns, and consumption cycles. Accurate day-ahead price forecasting is critical for:

- **Energy traders** optimizing bidding strategies
- **Grid operators** managing load balancing
- **Renewable energy producers** planning dispatch
- **Asset managers** hedging market risk

This project demonstrates how ML can be applied to real-world energy market forecasting with proper time-series methodology.

---

## 🧠 Methodology

### Why Temporal Validation Matters

A common pitfall in time-series ML is using random train/test splits, which causes **data leakage** — the model sees future data during training, inflating metrics unrealistically. This project uses **chronological splitting** (the last 20% of timestamps become the test set), ensuring all predictions are truly out-of-sample.

### Feature Engineering

All transforms are **fully vectorized** using NumPy/Pandas (zero Python loops):

| Feature Group | Features | Purpose |
|---|---|---|
| **Lag Features** | `price_lag_1h`, `price_lag_24h`, `price_lag_168h` | Autoregressive signal from past prices |
| **Rolling Statistics** | `rolling_mean_24h`, `rolling_std_24h`, `rolling_mean_168h`, `rolling_std_168h` | Trend and volatility regime capture |
| **Price Derivatives** | `price_diff_1`, `price_diff_4`, `price_pct_change_1` | Momentum and rate of change |
| **Fourier Encoding** | `sin_hour`, `cos_hour`, `sin_day_of_week`, `cos_day_of_week`, `sin_month`, `cos_month` | Cyclical patterns without discontinuities |
| **Calendar** | `hour`, `day_of_month`, `month`, `day_of_week`, `is_weekend`, `is_business_hour` | Temporal context |
| **Interactions** | `hour_x_weekend`, `hour_x_business` | Peak/off-peak regime differences |

### Models Trained

| Model | Type | Purpose |
|---|---|---|
| **Ridge Regression** | Linear | Baseline — establishes minimum bar |
| **Random Forest** | Ensemble (bagging) | Non-linear patterns, feature importance |
| **XGBoost** | Ensemble (boosting) | State-of-the-art gradient boosting |
| **LightGBM** | Ensemble (boosting) | Fast, efficient gradient boosting |

All models use `TimeSeriesSplit(n_splits=5)` for cross-validation and optional `RandomizedSearchCV` for hyperparameter tuning.

### Evaluation Metrics

- **MAE** — Mean Absolute Error (EUR/MWh)
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error
- **R²** — Coefficient of Determination
- **Directional Accuracy** — Did the model predict price direction correctly?

---

## 🏗️ Project Structure

```
nordic-electricity-market-forecasting/
├── src/                           # Core ML pipeline package
│   ├── __init__.py
│   ├── config.py                  # Centralized configuration
│   ├── data_loader.py             # Data loading + temporal split
│   ├── feature_engineering.py     # Vectorized feature transforms
│   ├── model_training.py          # Multi-model training + tuning
│   ├── evaluation.py              # Metrics, comparisons, residuals
│   └── utils.py                   # Logging, timing decorators
├── data/
│   └── cleaned_finland_energy_prices.csv
├── models/                        # Saved model artifacts (.joblib)
├── results/                       # Evaluation outputs
├── notebook/
│   └── electricity_market_analysis.ipynb
├── images/
├── app.py                         # Streamlit analytics dashboard
├── run_pipeline.py                # CLI entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10+
- pip

### Install

```bash
git clone https://github.com/your-username/nordic-electricity-market-forecasting.git
cd nordic-electricity-market-forecasting
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Full Pipeline

```bash
python run_pipeline.py
```

### With Hyperparameter Tuning

```bash
python run_pipeline.py --tune
```

### Train a Single Model

```bash
python run_pipeline.py --model xgboost
```

### Load Saved Models (Skip Training)

```bash
python run_pipeline.py --skip-training
```

### Launch Dashboard

```bash
streamlit run app.py
```

---

## 📊 Dashboard Features

The Streamlit dashboard provides:

- **📈 Market KPIs** — Average, max, min prices with filters
- **🏆 Model Comparison** — Side-by-side metrics table + radar chart
- **🔮 Forecast Visualization** — Actual vs predicted with confidence analysis
- **📊 Feature Importance** — Comparison across tree-based models
- **🔍 Error Analysis** — Residual distribution, MAE by hour/month/weekday
- **⚡ Volatility Detection** — Rolling volatility with spike identification

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | ENTSO-E Transparency Platform |
| **Region** | Finland |
| **Market** | Day-Ahead |
| **Frequency** | 15-minute intervals |
| **Records** | ~35,000 |
| **Period** | 2025 |
| **Target** | Price (EUR/MWh) |

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.12 |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM |
| **Visualization** | Plotly, Matplotlib |
| **Dashboard** | Streamlit |
| **Serialization** | Joblib |

---

## 🔮 Future Improvements

- [ ] LSTM / Transformer-based deep learning models
- [ ] Real-time ENTSO-E API integration for live forecasting
- [ ] Multi-country comparison (Finland, Sweden, Norway, Denmark)
- [ ] Renewable energy generation features (wind, solar)
- [ ] Docker containerization for deployment
- [ ] CI/CD pipeline with automated model retraining

---

## 👨‍💻 Author

**M. Sabda Pyari**
B.Tech Electrical Engineering (Computer Science Specialization)
Dayalbagh Educational Institute

---

## 📚 References

- [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

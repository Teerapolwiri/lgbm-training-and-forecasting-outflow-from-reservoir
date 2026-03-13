# 🌊 LightGBM Training & Forecasting — Reservoir Outflow

A **physics-informed machine learning pipeline** for forecasting daily reservoir outflow using LightGBM with custom physical constraints and Bayesian hyperparameter optimization.

> Applied to major reservoirs in Thailand using data from the Royal Irrigation Department (RID) API.

---

## 📋 Overview

Traditional ML models optimize purely for prediction accuracy, which can produce physically impossible results — such as outflow volumes that violate water balance. This project addresses that limitation by embedding **engineering constraints directly into the LightGBM loss function**, ensuring the model respects both data patterns and physical laws.

### Key Features

- **Automated data fetching** from RID API via Shapefile-based reservoir identification
- **Physics-informed custom loss function** with 3 components (accuracy + water balance + boundary constraints)
- **Bayesian Optimization** with Optuna TPE Sampler (2,000 trials) for hyperparameter tuning
- **Sequential forecasting** — each day's predicted outflow updates the next day's storage state
- **Automated visualization** for both optimization history and model performance

---

## 🗂️ Repository Structure

```
├── train_model_lgbm.py       # Full training pipeline (data fetch → train → evaluate)
├── forecast_model_lgbm.py    # Sequential outflow forecasting using a saved model
├── Model/
│   └── lgb_model_<dam_id>.txt    # Saved LightGBM models (one per reservoir)
├── data/
│   └── input_pmp.txt             # Input data for forecasting (date, Qin cms, Volume mcm, Qout cms)
└── pic/
    ├── bayesian_optimization_results_<dam>.png
    └── <dam>_reservoir_analysis.png
```

---

## ⚙️ Installation

```bash
pip install lightgbm optuna scikit-learn pandas numpy matplotlib geopandas requests
```

---

## 🚀 Usage

### 1. Train the Model

```bash
python train_model_lgbm.py
```

The training script will:
1. Read reservoir IDs from Shapefiles (`Stations.shp`, `Reservoir.shp`)
2. Fetch historical data from the RID API (inflow, outflow, volume, rule curve)
3. Clean data using IQR outlier removal
4. Train LightGBM with a custom physics-informed loss function
5. Run Bayesian Optimization (2,000 trials) to find optimal hyperparameters
6. Save the trained model to `Model/lgb_model_<dam_id>.txt`
7. Generate performance plots in `pic/`

### 2. Forecast Outflow

```bash
python forecast_model_lgbm.py
```

Loads a saved model and runs a sequential day-by-day forecast. Each prediction feeds back into the next step by updating reservoir storage:

```
Volume(t+1) = Volume(t) + (Inflow(t) - PredictedOutflow(t)) × 0.0864
```

**Input file format** (`data/input_pmp.txt`):

| Column | Unit | Description |
|---|---|---|
| `date` | YYYY-MM-DD | Forecast date |
| `Qin cms` | m³/s | Inflow forecast |
| `Volume mcm` | MCM | Initial reservoir storage |
| `Qout cms` | m³/s | Previous day outflow (for lag feature) |

---

## 🧠 Custom Loss Function

Instead of standard MSE, the model uses a 3-component weighted loss:

```
L = μ₁·L1 + μ₂·L2 + μ₃·L3
```

| Term | Weight | Description |
|---|---|---|
| **L1** — Accuracy | μ₁ = 1.0 | MSE between predicted and observed outflow |
| **L2** — Water Balance | μ₂ = 0.5 | Penalizes violations of `(Inflow − Outflow) × Δt ≈ ΔStorage` |
| **L3** — Boundary Constraint | μ₃ = 0.8 | Penalizes outflow exceeding Upper/Lower Rule Curve limits |

LightGBM was chosen over LSTM because it **natively supports custom gradient/Hessian**, making it straightforward to inject physical knowledge directly into the training process.

---

## 📊 Model Features

Each reservoir model uses 3 input features:

| Feature | Unit | Description |
|---|---|---|
| `inflow mcm` | m³/s | Current day inflow |
| `volume mcm` | MCM | Current reservoir storage |
| `outflow mcm` | m³/s | Previous day outflow (lag-1) |

---

## 🏞️ Supported Reservoirs

| Dam ID | Reservoir Name |
|---|---|
| `200101` | Bhumibol Dam |
| `200102` | Sirikit Dam |
| `200103` | Mae Ngat Somboon Chon Dam |
| `100104` | Mae Kuang Udom Thara Dam |
| `100105` | Kio Lom Dam |
| `200203` | Nam Phung Dam |
| `100201` | Huai Luang Dam |
| `rsv21` | Aemsrwy Reservoir |
| `rsv53` | Aemtam Reservoir |
| `rsv54` | Aempuuem Reservoir |
| `rsv502` | Namely Reservoir |

---

## 📈 Output Visualizations

For each reservoir, the pipeline generates 2 sets of plots:

**1. Bayesian Optimization Results** (`bayesian_optimization_results_<dam>.png`)
- RMSE vs Trial number
- Learning Rate vs RMSE scatter
- Num Leaves vs RMSE scatter
- Best hyperparameters summary panel
- Predicted vs Actual scatter plot
- Optimization performance summary

**2. Reservoir Analysis** (`<dam>_reservoir_analysis.png`)
- Residual error distribution
- Absolute error vs actual values
- Time series comparison (predicted vs observed)
- Model statistics summary

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `lightgbm` | Gradient boosting model with custom loss support |
| `optuna` | Bayesian hyperparameter optimization (TPE Sampler) |
| `scikit-learn` | Train/test split, RMSE evaluation |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Visualization |
| `geopandas` | Shapefile reading for reservoir IDs |
| `requests` | RID API data fetching |

---

## 📄 Reference

This implementation is inspired by the methodology in:

> Tan et al. (2025). *Integrated probabilistic forecasting framework for long-term reservoir outflow through dynamic coupling of meteorological–hydrological–engineering processes.* Journal of Hydrology, 663, 134214.

---

## 🔗 Links

- 📦 Repository: [https://github.com/Teerapolwiri/lgbm-training-and-forecasting-outflow-from-reservoir](https://github.com/Teerapolwiri/lgbm-training-and-forecasting-outflow-from-reservoir)
- 🌐 RID Data Portal: [https://app.rid.go.th/reservoir](https://app.rid.go.th/reservoir)

---

## 📝 License

This project is open-source. Feel free to use, modify, and distribute with attribution.

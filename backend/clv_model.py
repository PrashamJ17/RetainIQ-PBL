"""
RetainIQ — Customer Lifetime Value (CLV) Prediction Engine
============================================================
XGBRegressor-based model that predicts continuous monetary value
a customer is expected to generate over the next 12 months.

Unlike the churn classifier (binary Yes/No), this model answers:
"How much money is this customer worth to us?"

This enables ROI-weighted retention decisions:
    - Spending R$50 to save a customer worth R$2,000 → YES
    - Spending R$50 to save a customer worth R$30  → NO
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import sys
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import TEST_SIZE, RANDOM_STATE


# ──────────────────────────────────────────────
# 1. CLV TARGET ENGINEERING
# ──────────────────────────────────────────────

def engineer_clv_target(customer_df: pd.DataFrame) -> pd.DataFrame:
    """
    (DEPRECATED) 
    Previously used a mathematical formula for CLV.
    Now, the target 'clv_target' is generated directly in the data_engine
    using a pure Time-Based Out-Of-Time (OOT) Split.
    
    The model now predicts the ACTUAL revenue generated in the target window,
    rather than a heuristic formula.
    """
    return customer_df
    # Function maintained to prevent breaking API signatures, but does nothing.
    return customer_df


# ──────────────────────────────────────────────
# 2. FEATURE PREPARATION
# ──────────────────────────────────────────────

CLV_FEATURE_COLS = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "avg_review_score",
    "total_items",
    "unique_categories",
    "avg_delivery_delay",
    "purchase_span_days",
    "avg_days_between_purchases",
]


def prepare_clv_features(customer_df: pd.DataFrame) -> dict:
    """
    Prepare features and CLV target, then split into train/test.
    """
    X = customer_df[CLV_FEATURE_COLS].copy()
    y = customer_df["clv_target"].copy()

    # Handle NaN/Inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": CLV_FEATURE_COLS,
    }


# ──────────────────────────────────────────────
# 3. MODEL TRAINING
# ──────────────────────────────────────────────

def train_clv_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    """
    Train an XGBoost Regressor for continuous CLV prediction.
    """
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=0,
        objective="reg:squarederror",
    )

    model.fit(X_train, y_train)
    return model


def save_clv_model(model: xgb.XGBRegressor, path: str = "models/clv_model.json"):
    """Save the CLV model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
    print(f"💾 CLV model saved to {path}")


def load_clv_model(path: str = "models/clv_model.json") -> xgb.XGBRegressor:
    """Load a trained CLV model from disk."""
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model


# ──────────────────────────────────────────────
# 4. PREDICTION & EVALUATION
# ──────────────────────────────────────────────

def predict_clv(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """Generate CLV predictions. Returns array of predicted R$ values."""
    predictions = model.predict(X)
    # CLV can't be negative
    predictions = np.maximum(predictions, 0)
    return np.round(predictions, 2)


def get_clv_metrics(model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute regression metrics for the CLV model."""
    y_pred = predict_clv(model, X_test)

    return {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "r2_score": round(float(r2_score(y_test, y_pred)), 4),
        "median_predicted_clv": round(float(np.median(y_pred)), 2),
        "mean_predicted_clv": round(float(np.mean(y_pred)), 2),
    }


# ──────────────────────────────────────────────
# 5. CLV TIER ASSIGNMENT
# ──────────────────────────────────────────────

def assign_clv_tier(predicted_clv: float, thresholds: dict = None) -> str:
    """
    Assign a human-readable CLV tier based on predicted value.

    Tiers:
        Platinum: Top 5%  (highest predicted CLV)
        Gold:     Top 20%
        Silver:   Top 50%
        Bronze:   Bottom 50%
    """
    if thresholds is None:
        # Default thresholds (will be overridden by percentile-based)
        thresholds = {"platinum": 500, "gold": 200, "silver": 100}

    if predicted_clv >= thresholds["platinum"]:
        return "Platinum"
    elif predicted_clv >= thresholds["gold"]:
        return "Gold"
    elif predicted_clv >= thresholds["silver"]:
        return "Silver"
    else:
        return "Bronze"


def compute_clv_thresholds(predictions: np.ndarray) -> dict:
    """Compute percentile-based CLV tier thresholds."""
    return {
        "platinum": float(np.percentile(predictions, 95)),
        "gold": float(np.percentile(predictions, 80)),
        "silver": float(np.percentile(predictions, 50)),
    }


# ──────────────────────────────────────────────
# 6. FULL CLV PIPELINE
# ──────────────────────────────────────────────

def run_clv_pipeline(customer_df: pd.DataFrame) -> dict:
    """
    Execute the full CLV prediction pipeline.

    Args:
        customer_df: Customer-level DataFrame (must have RFM + features)

    Returns:
        Dict with model, metrics, updated customer_df with clv columns
    """
    print("💰 Preparing CLV targets (provided by Time-Based Split engine)...")
    # Target already engineered by data_engine.py
    # customer_df = engineer_clv_target(customer_df)

    print("🔧 Preparing CLV features...")
    splits = prepare_clv_features(customer_df)

    print("🚀 Training XGBoost CLV Regressor...")
    model = train_clv_model(splits["X_train"], splits["y_train"])

    print("📏 Evaluating CLV model performance...")
    metrics = get_clv_metrics(model, splits["X_test"], splits["y_test"])
    print(f"   → MAE:      R${metrics['mae']:.2f}")
    print(f"   → RMSE:     R${metrics['rmse']:.2f}")
    print(f"   → R² Score: {metrics['r2_score']:.4f}")
    print(f"   → Mean CLV: R${metrics['mean_predicted_clv']:.2f}")

    print("🔮 Predicting CLV for all customers...")
    all_features = customer_df[CLV_FEATURE_COLS].copy()
    all_features = all_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    clv_predictions = predict_clv(model, all_features)

    customer_df = customer_df.copy()
    customer_df["predicted_clv"] = clv_predictions

    # Assign tiers
    thresholds = compute_clv_thresholds(clv_predictions)
    customer_df["clv_tier"] = customer_df["predicted_clv"].apply(
        lambda x: assign_clv_tier(x, thresholds)
    )

    print(f"📊 CLV Tier Distribution:")
    print(customer_df["clv_tier"].value_counts().to_string())

    # Save model
    save_clv_model(model)

    print("✅ CLV pipeline complete!")

    return {
        "model": model,
        "metrics": metrics,
        "customer_df": customer_df,
        "thresholds": thresholds,
    }


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from data_engine import run_data_pipeline
    from segmentation import run_segmentation_pipeline

    customer_data = run_data_pipeline()
    seg_results = run_segmentation_pipeline(customer_data)
    customer_data = seg_results["customer_df"]

    clv_results = run_clv_pipeline(customer_data)

    print("\n💎 Top 10 Highest CLV Customers:")
    top_clv = clv_results["customer_df"].nlargest(10, "predicted_clv")
    print(top_clv[["customer_unique_id", "predicted_clv", "clv_tier", "monetary", "frequency"]].to_string(index=False))

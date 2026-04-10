"""
RetainIQ — Churn Prediction Engine
=====================================
XGBoost-based churn classifier with SHAP explainability.

This module:
1. Prepares features and splits data for training.
2. Trains an XGBoost classifier with optimized hyperparameters.
3. Generates per-user churn probabilities.
4. Uses SHAP TreeExplainer to produce human-readable explanations.
5. Computes model performance metrics (ROC-AUC, F1, Confusion Matrix).
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from utils import TEST_SIZE, RANDOM_STATE, SEGMENT_COLORS


# ──────────────────────────────────────────────
# 1. FEATURE PREPARATION
# ──────────────────────────────────────────────

# Columns used as features for churn prediction
FEATURE_COLS = [
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

TARGET_COL = "churned"


def prepare_features(customer_df: pd.DataFrame) -> dict:
    """
    Prepare feature matrix and target vector, then split into train/test.

    Args:
        customer_df: Customer-level DataFrame with features and 'churned' column.

    Returns:
        Dict with X_train, X_test, y_train, y_test, and feature_names.
    """
    X = customer_df[FEATURE_COLS].copy()
    y = customer_df[TARGET_COL].copy()

    # Handle any remaining NaN/Inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": FEATURE_COLS,
    }


# ──────────────────────────────────────────────
# 2. MODEL TRAINING
# ──────────────────────────────────────────────

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier with tuned hyperparameters.

    The scale_pos_weight is automatically calculated to handle class imbalance
    (since churned users significantly outnumber active ones in Olist).

    Returns:
        Trained XGBClassifier.
    """
    # Handle class imbalance
    n_positive = y_train.sum()
    n_negative = len(y_train) - n_positive
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )

    model.fit(X_train, y_train)

    return model


def save_model(model: xgb.XGBClassifier, path: str = "models/churn_model.json"):
    """Save the trained model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
    print(f"💾 Model saved to {path}")


def load_model(path: str = "models/churn_model.json") -> xgb.XGBClassifier:
    """Load a trained model from disk."""
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model


# ──────────────────────────────────────────────
# 3. PREDICTION
# ──────────────────────────────────────────────

def predict_churn(model: xgb.XGBClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """
    Generate churn predictions and probabilities for all customers.

    Returns:
        DataFrame with columns: 'churn_prediction' (0/1) and 'churn_probability' (0.0–1.0).
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        "churn_prediction": predictions,
        "churn_probability": np.round(probabilities, 4),
    }, index=X.index)

    return result


# ──────────────────────────────────────────────
# 4. MODEL EVALUATION
# ──────────────────────────────────────────────

def get_model_metrics(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Compute comprehensive model performance metrics.

    Returns:
        Dict with accuracy, f1, precision, recall, roc_auc, confusion_matrix,
        and classification_report.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    return metrics


# ──────────────────────────────────────────────
# 5. SHAP EXPLAINABILITY
# ──────────────────────────────────────────────

def explain_predictions(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    top_n: int = 100,
) -> dict:
    """
    Generate SHAP explanations for the top N highest-risk users.

    For each flagged user, extracts the top 3 features driving their
    churn prediction with plain-English descriptions.

    Args:
        model: Trained XGBClassifier.
        X: Feature matrix for all customers.
        top_n: Number of highest-risk users to explain.

    Returns:
        Dict with:
            - 'shap_values': Full SHAP value matrix
            - 'top_reasons': List of dicts with customer index + top 3 reasons
            - 'feature_importance': Global feature importance from SHAP
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Global feature importance (mean absolute SHAP value per feature)
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": np.abs(shap_values).mean(axis=0),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Get churn probabilities to find highest-risk users
    probas = model.predict_proba(X)[:, 1]
    high_risk_indices = np.argsort(probas)[-top_n:][::-1]

    # Feature name to human-readable description
    feature_descriptions = {
        "recency": "Days since last purchase",
        "frequency": "Number of orders placed",
        "monetary": "Total amount spent",
        "avg_order_value": "Average order value",
        "avg_review_score": "Average review rating given",
        "total_items": "Total items purchased",
        "unique_categories": "Number of product categories explored",
        "avg_delivery_delay": "Average delivery delay (days)",
        "purchase_span_days": "Time span of purchasing history",
        "avg_days_between_purchases": "Average gap between purchases",
    }

    # Extract top 3 reasons per high-risk user
    top_reasons = []
    for idx in high_risk_indices:
        user_shap = shap_values[idx]
        # Get indices of top 3 features by absolute SHAP value
        top_3_indices = np.argsort(np.abs(user_shap))[-3:][::-1]

        reasons = []
        for feat_idx in top_3_indices:
            feature_name = X.columns[feat_idx]
            shap_val = user_shap[feat_idx]
            feature_val = X.iloc[idx][feature_name]
            direction = "increases" if shap_val > 0 else "decreases"
            human_name = feature_descriptions.get(feature_name, feature_name)

            reasons.append({
                "feature": feature_name,
                "human_description": human_name,
                "feature_value": round(float(feature_val), 2),
                "shap_value": round(float(shap_val), 4),
                "direction": direction,
                "explanation": f"{human_name} = {feature_val:.1f} ({direction} churn risk)",
            })

        top_reasons.append({
            "customer_index": int(X.index[idx]),
            "churn_probability": round(float(probas[idx]), 4),
            "reasons": reasons,
        })

    return {
        "shap_values": shap_values,
        "top_reasons": top_reasons,
        "feature_importance": feature_importance,
    }


# ──────────────────────────────────────────────
# 6. VISUALIZATION FUNCTIONS
# ──────────────────────────────────────────────

def plot_feature_importance(feature_importance: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of global SHAP feature importance.
    """
    fi = feature_importance.sort_values("importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=fi["importance"],
            y=fi["feature"],
            orientation="h",
            marker_color="#6C63FF",
            text=fi["importance"].apply(lambda x: f"{x:.3f}"),
            textposition="outside",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Global Feature Importance (Mean |SHAP Value|)",
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="",
        font=dict(family="Inter", color="white"),
        height=400,
        margin=dict(l=200),
    )

    return fig


def plot_confusion_matrix(metrics: dict) -> go.Figure:
    """
    Heatmap visualization of the confusion matrix.
    """
    cm = np.array(metrics["confusion_matrix"])
    labels = ["Active", "Churned"]

    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=[[str(val) for val in row] for row in cm],
            texttemplate="%{text}",
            textfont=dict(size=18),
            colorscale=[[0, "#1a1a2e"], [1, "#6C63FF"]],
            showscale=False,
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        font=dict(family="Inter", color="white"),
        height=350,
        width=400,
    )

    return fig


def plot_churn_distribution(customer_df: pd.DataFrame) -> go.Figure:
    """
    Histogram of churn probability distribution, colored by segment.
    """
    fig = px.histogram(
        customer_df,
        x="churn_probability",
        color="segment",
        color_discrete_map=SEGMENT_COLORS,
        nbins=50,
        title="Churn Probability Distribution by Segment",
        labels={"churn_probability": "Churn Probability", "segment": "Segment"},
        opacity=0.7,
        barmode="overlay",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="white"),
        height=400,
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
    )

    return fig


def plot_action_matrix(customer_df: pd.DataFrame) -> go.Figure:
    """
    The 2x2 Action Matrix scatter plot:
    X = Monetary Value, Y = Churn Probability.
    Colored by recommended action quadrant.
    """
    fig = go.Figure()

    action_colors = {
        "SAVE_NOW": "#FF6B6B",
        "LET_GO": "#A0AEC0",
        "NURTURE": "#6C63FF",
        "MONITOR": "#00C9A7",
    }

    action_labels = {
        "SAVE_NOW": "🚨 Save Now",
        "LET_GO": "🚫 Let Go",
        "NURTURE": "👑 Nurture",
        "MONITOR": "👀 Monitor",
    }

    for action, color in action_colors.items():
        mask = customer_df["action"] == action
        subset = customer_df[mask]

        if len(subset) > 0:
            fig.add_trace(
                go.Scatter(
                    x=subset["monetary"],
                    y=subset["churn_probability"],
                    mode="markers",
                    marker=dict(color=color, size=5, opacity=0.6),
                    name=action_labels.get(action, action),
                    hovertemplate=(
                        "<b>Customer:</b> %{customdata[0]}<br>"
                        "<b>Monetary:</b> R$%{x:,.2f}<br>"
                        "<b>Churn Risk:</b> %{y:.1%}<br>"
                        "<b>Segment:</b> %{customdata[1]}<br>"
                        "<extra></extra>"
                    ),
                    customdata=subset[["customer_unique_id", "segment"]].values,
                )
            )

    # Add quadrant lines
    monetary_median = customer_df["monetary"].median()
    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig.add_vline(x=monetary_median, line_dash="dash", line_color="rgba(255,255,255,0.3)")

    # Add quadrant labels
    x_max = customer_df["monetary"].quantile(0.95)
    fig.add_annotation(x=x_max * 0.7, y=0.85, text="🚨 SAVE NOW", showarrow=False, font=dict(size=14, color="#FF6B6B"))
    fig.add_annotation(x=monetary_median * 0.3, y=0.85, text="🚫 LET GO", showarrow=False, font=dict(size=14, color="#A0AEC0"))
    fig.add_annotation(x=x_max * 0.7, y=0.15, text="👑 NURTURE", showarrow=False, font=dict(size=14, color="#6C63FF"))
    fig.add_annotation(x=monetary_median * 0.3, y=0.15, text="👀 MONITOR", showarrow=False, font=dict(size=14, color="#00C9A7"))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Action Matrix — Value vs. Risk",
        xaxis_title="Monetary Value (R$)",
        yaxis_title="Churn Probability",
        font=dict(family="Inter", color="white"),
        height=550,
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        xaxis=dict(range=[0, x_max]),
    )

    return fig


# ──────────────────────────────────────────────
# 7. FULL CHURN PIPELINE
# ──────────────────────────────────────────────

def run_churn_pipeline(customer_df: pd.DataFrame) -> dict:
    """
    Execute the full churn prediction pipeline.

    Args:
        customer_df: Customer-level DataFrame with features, churn labels, and segments.

    Returns:
        Dict containing:
            - 'model': Trained XGBClassifier
            - 'metrics': Performance metrics dict
            - 'customer_df': DataFrame with churn_probability and action columns
            - 'shap_results': SHAP explanations dict
            - 'feature_splits': Train/test split data
    """
    print("🔧 Preparing features...")
    splits = prepare_features(customer_df)

    print("🚀 Training XGBoost churn model...")
    model = train_model(splits["X_train"], splits["y_train"])

    print("📏 Evaluating model performance...")
    metrics = get_model_metrics(model, splits["X_test"], splits["y_test"])
    print(f"   → Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   → F1 Score:  {metrics['f1_score']:.4f}")
    print(f"   → Precision: {metrics['precision']:.4f}")
    print(f"   → Recall:    {metrics['recall']:.4f}")
    print(f"   → ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("🔮 Generating predictions for all customers...")
    all_features = customer_df[FEATURE_COLS].copy()
    all_features = all_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    predictions = predict_churn(model, all_features)
    customer_df = customer_df.copy()
    customer_df["churn_probability"] = predictions["churn_probability"].values
    customer_df["churn_prediction"] = predictions["churn_prediction"].values

    # Assign action quadrant
    from utils import assign_action
    monetary_median = customer_df["monetary"].median()
    customer_df["monetary_percentile"] = customer_df["monetary"].rank(pct=True)
    customer_df["action"] = customer_df.apply(
        lambda row: assign_action(row["churn_probability"], row["monetary_percentile"]),
        axis=1,
    )

    print("🧠 Computing SHAP explanations (top 100 high-risk users)...")
    shap_results = explain_predictions(model, all_features, top_n=100)

    # Save model
    save_model(model)

    print("✅ Churn prediction pipeline complete!")

    return {
        "model": model,
        "metrics": metrics,
        "customer_df": customer_df,
        "shap_results": shap_results,
        "feature_splits": splits,
    }


# ──────────────────────────────────────────────
# CLI entry point for standalone testing
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from data_engine import run_data_pipeline
    from segmentation import run_segmentation_pipeline

    # Run data pipeline
    customer_data = run_data_pipeline()

    # Run segmentation
    seg_results = run_segmentation_pipeline(customer_data)
    customer_data = seg_results["customer_df"]

    # Run churn prediction
    churn_results = run_churn_pipeline(customer_data)

    # Print action distribution
    print("\n🎯 Action Matrix Distribution:")
    print(churn_results["customer_df"]["action"].value_counts().to_string())

    # Print sample SHAP reasons
    print("\n🧠 Sample SHAP Explanations (Top 3 high-risk users):")
    for user in churn_results["shap_results"]["top_reasons"][:3]:
        print(f"\n  Customer Index: {user['customer_index']} | Churn Prob: {user['churn_probability']:.1%}")
        for r in user["reasons"]:
            print(f"    → {r['explanation']}")

"""
RetainIQ — Backend API Routes (Phase 2)
==========================================
Dashboard data-fetching endpoints and ML training trigger.

These endpoints serve pre-computed data to the Next.js frontend
via JSON, completely decoupling the ML engine from the UI layer.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func

from .database import get_db, Customer, Order

router = APIRouter(prefix="/api", tags=["Dashboard"])

# ──────────────────────────────────────────────
# Global in-memory state for ML results
# (Loaded once on training, served on every request)
# ──────────────────────────────────────────────

_ml_state = {
    "model": None,
    "metrics": None,
    "shap_results": None,
    "segment_summary": None,
    "customer_df": None,
    "last_trained": None,
    "is_training": False,
    # Phase 3: CLV + Activation
    "clv_model": None,
    "clv_metrics": None,
    "clv_thresholds": None,
    "activation_results": None,
}


def get_ml_state():
    """Return the global ML state dict."""
    return _ml_state


# ──────────────────────────────────────────────
# HELPER: Import ML modules from project root
# ──────────────────────────────────────────────

# Add project root to path so we can import data_engine, segmentation, churn_model
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _run_full_pipeline():
    """
    Execute the complete ML pipeline:
        Data → Segmentation → Churn → CLV → Activation
    and store results in the global _ml_state dict.
    
    This runs in a background thread so the API stays responsive.
    """
    from data_engine import run_data_pipeline
    from segmentation import run_segmentation_pipeline
    from churn_model import run_churn_pipeline
    from backend.clv_model import run_clv_pipeline
    from backend.activation import run_batch_activation

    _ml_state["is_training"] = True

    try:
        # Step 1: Data Engine
        customer_df = run_data_pipeline()

        # Step 2: Segmentation
        seg_results = run_segmentation_pipeline(customer_df)
        customer_df = seg_results["customer_df"]

        # Step 3: Churn Prediction
        churn_results = run_churn_pipeline(customer_df)
        customer_df = churn_results["customer_df"]

        # Step 4: CLV Prediction
        clv_results = run_clv_pipeline(customer_df)
        customer_df = clv_results["customer_df"]

        # Step 5: Batch Activation (simulate automated actions)
        activation_results = run_batch_activation(customer_df)

        # Store everything in memory
        _ml_state["model"] = churn_results["model"]
        _ml_state["metrics"] = churn_results["metrics"]
        _ml_state["shap_results"] = {
            "top_reasons": churn_results["shap_results"]["top_reasons"],
            "feature_importance": churn_results["shap_results"]["feature_importance"].to_dict(orient="records"),
        }
        _ml_state["segment_summary"] = seg_results["segment_summary"].to_dict(orient="records")
        _ml_state["customer_df"] = customer_df
        _ml_state["last_trained"] = datetime.utcnow().isoformat()

        # Phase 3 state
        _ml_state["clv_model"] = clv_results["model"]
        _ml_state["clv_metrics"] = clv_results["metrics"]
        _ml_state["clv_thresholds"] = clv_results["thresholds"]
        _ml_state["activation_results"] = activation_results

        print("✅ Full Pipeline (incl. CLV + Activation) complete.")

    except Exception as e:
        print(f"❌ ML Pipeline failed: {e}")
        raise

    finally:
        _ml_state["is_training"] = False


# ──────────────────────────────────────────────
# API ROUTES
# ──────────────────────────────────────────────


@router.get("/status")
def api_status():
    """Returns the current state of the ML pipeline."""
    return {
        "model_loaded": _ml_state["model"] is not None,
        "is_training": _ml_state["is_training"],
        "last_trained": _ml_state["last_trained"],
        "total_customers": len(_ml_state["customer_df"]) if _ml_state["customer_df"] is not None else 0,
    }


@router.post("/trigger-training")
def trigger_training(background_tasks: BackgroundTasks):
    """
    Manually trigger a full ML pipeline re-training.
    In production this would be called by a CRON/Airflow job.
    """
    if _ml_state["is_training"]:
        raise HTTPException(status_code=409, detail="Training already in progress")

    background_tasks.add_task(_run_full_pipeline)

    return {
        "status": "training_started",
        "message": "Full ML pipeline triggered in background. Check /api/status for progress.",
    }


@router.get("/dashboard/stats")
def get_dashboard_stats():
    """
    KPI cards for the Overview tab.
    Returns total customers, revenue, avg order value, churn rate, and model metrics.
    """
    if _ml_state["customer_df"] is None:
        raise HTTPException(status_code=503, detail="Model not yet trained. POST /api/trigger-training first.")

    df = _ml_state["customer_df"]
    metrics = _ml_state["metrics"]

    # CLV stats (if available)
    clv_stats = None
    if "predicted_clv" in df.columns:
        clv_stats = {
            "mean_clv": round(float(df["predicted_clv"].mean()), 2),
            "median_clv": round(float(df["predicted_clv"].median()), 2),
            "total_predicted_value": round(float(df["predicted_clv"].sum()), 2),
        }

    return {
        "total_customers": int(len(df)),
        "total_revenue": round(float(df["monetary"].sum()), 2),
        "avg_order_value": round(float(df["avg_order_value"].mean()), 2),
        "churn_rate": round(float(df["churned"].mean() * 100), 1),
        "model_metrics": {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "roc_auc": metrics["roc_auc"],
        },
        "clv_stats": clv_stats,
        "clv_metrics": _ml_state.get("clv_metrics"),
        "last_trained": _ml_state["last_trained"],
    }


@router.get("/segments")
def get_segments():
    """
    Segment data for the Segments tab.
    Returns segment summary + per-customer segment assignments.
    """
    if _ml_state["customer_df"] is None:
        raise HTTPException(status_code=503, detail="Model not yet trained.")

    df = _ml_state["customer_df"]

    # Segment distribution
    distribution = df["segment"].value_counts().to_dict()

    # Per-customer data (limited to keep payload manageable)
    customers = df[[
        "customer_unique_id", "segment", "recency", "frequency", "monetary",
        "avg_review_score", "churn_probability", "action"
    ]].to_dict(orient="records")

    return {
        "segment_summary": _ml_state["segment_summary"],
        "segment_distribution": distribution,
        "customers": customers[:500],  # Limit to 500 for frontend performance
        "total_customers": len(df),
    }


@router.get("/churn-risk")
def get_churn_risk():
    """
    Churn risk data for the Churn Risk tab.
    Returns model metrics, feature importance, and SHAP explanations.
    """
    if _ml_state["customer_df"] is None:
        raise HTTPException(status_code=503, detail="Model not yet trained.")

    df = _ml_state["customer_df"]
    metrics = _ml_state["metrics"]

    # Churn probability distribution (binned for histogram)
    hist_data = df["churn_probability"].value_counts(bins=50, sort=False).reset_index()
    hist_data.columns = ["bin", "count"]
    hist_data["bin"] = hist_data["bin"].apply(lambda x: round(x.mid, 2))

    return {
        "metrics": {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "roc_auc": metrics["roc_auc"],
            "confusion_matrix": metrics["confusion_matrix"],
        },
        "feature_importance": _ml_state["shap_results"]["feature_importance"],
        "top_high_risk_users": _ml_state["shap_results"]["top_reasons"][:20],
        "churn_distribution": hist_data.to_dict(orient="records"),
    }


@router.get("/action-matrix")
def get_action_matrix():
    """
    Action Matrix data for the Action Matrix tab.
    Returns action distribution and per-customer action assignments.
    """
    if _ml_state["customer_df"] is None:
        raise HTTPException(status_code=503, detail="Model not yet trained.")

    df = _ml_state["customer_df"]

    # Action distribution
    action_dist = df["action"].value_counts().to_dict()

    # Scatter data: monetary vs churn_probability colored by action
    scatter_data = df[[
        "customer_unique_id", "monetary", "churn_probability", "segment", "action"
    ]].to_dict(orient="records")

    return {
        "action_distribution": action_dist,
        "scatter_data": scatter_data[:2000],  # Limit for performance
        "total_customers": len(df),
    }


@router.get("/export/{action_key}")
def export_action_list(action_key: str):
    """
    Export a filtered customer list by action quadrant.
    Returns full customer data for the selected action group.
    """
    if _ml_state["customer_df"] is None:
        raise HTTPException(status_code=503, detail="Model not yet trained.")

    valid_actions = ["SAVE_NOW", "LET_GO", "NURTURE", "MONITOR"]
    if action_key.upper() not in valid_actions:
        raise HTTPException(status_code=400, detail=f"Invalid action. Valid: {valid_actions}")

    df = _ml_state["customer_df"]
    filtered = df[df["action"] == action_key.upper()]

    export_data = filtered[[
        "customer_unique_id", "segment", "churn_probability", "monetary",
        "recency", "frequency", "avg_review_score", "action"
    ]].sort_values("churn_probability", ascending=False).to_dict(orient="records")

    return {
        "action": action_key.upper(),
        "count": len(export_data),
        "customers": export_data,
    }


@router.get("/export-customers")
def export_filtered_customers(status: Optional[str] = "ALL", segment: Optional[str] = "ALL"):
    """
    Export a filtered customer list specifically built for CSV download on the frontend.
    Filters by churn status and customer segment.
    """
    if _ml_state["customer_df"] is None:
        raise HTTPException(status_code=503, detail="Model not yet trained.")

    df = _ml_state["customer_df"]

    # Apply Churn Status Filter
    if status.upper() == "CHURNED":
        df = df[df["churn_prediction"] == 1]
    elif status.upper() == "RETAINED":
        df = df[df["churn_prediction"] == 0]

    # Apply Segment Filter
    if segment.upper() != "ALL":
        df = df[df["segment"].str.upper() == segment.upper()]

    # Select columns useful for offline analysis
    export_cols = [
        "customer_unique_id", "segment", "churned", "churn_prediction", "churn_probability",
        "action", "predicted_clv", "clv_tier", "monetary", "frequency", "recency",
        "avg_review_score", "avg_delivery_delay"
    ]

    # Handle case where certain columns might not exist if pipeline failed midway
    available_cols = [col for col in export_cols if col in df.columns]

    export_data = df[available_cols].sort_values("churn_probability", ascending=False).to_dict(orient="records")

    return {
        "status_filter": status,
        "segment_filter": segment,
        "count": len(export_data),
        "customers": export_data
    }


# ──────────────────────────────────────────────
# DB-BACKED ROUTES (from ingestion layer)
# ──────────────────────────────────────────────

@router.get("/db/stats")
def get_db_stats(db: Session = Depends(get_db)):
    """
    Real-time stats from the SQLite database (orders received via webhook).
    """
    customer_count = db.query(func.count(Customer.id)).scalar()
    order_count = db.query(func.count(Order.order_id)).scalar()
    total_revenue = db.query(func.sum(Order.total_value)).scalar() or 0

    return {
        "db_customers": customer_count,
        "db_orders": order_count,
        "db_total_revenue": round(total_revenue, 2),
    }


# ──────────────────────────────────────────────
# Phase 3: CLV & Activation Endpoints
# ──────────────────────────────────────────────

@router.get("/clv")
def get_clv_data():
    """
    CLV predictions, tier distribution, and model metrics.
    """
    if _ml_state["customer_df"] is None:
        raise HTTPException(status_code=503, detail="Model not yet trained.")

    df = _ml_state["customer_df"]

    if "predicted_clv" not in df.columns:
        raise HTTPException(status_code=503, detail="CLV model not yet trained.")

    # Tier distribution
    tier_dist = df["clv_tier"].value_counts().to_dict()

    # Top 20 highest CLV customers
    top_customers = (
        df.nlargest(20, "predicted_clv")
        [["customer_unique_id", "predicted_clv", "clv_tier", "segment",
          "churn_probability", "monetary", "frequency", "action"]]
        .to_dict(orient="records")
    )

    return {
        "clv_metrics": _ml_state["clv_metrics"],
        "clv_thresholds": _ml_state["clv_thresholds"],
        "tier_distribution": tier_dist,
        "top_customers": top_customers,
        "total_predicted_value": round(float(df["predicted_clv"].sum()), 2),
    }


@router.get("/activations")
def get_activations():
    """
    Activation events log — automated marketing actions that were fired.
    """
    if _ml_state["activation_results"] is None:
        raise HTTPException(status_code=503, detail="Activation engine not yet run.")

    results = _ml_state["activation_results"]

    return {
        "total_evaluated": results["total_evaluated"],
        "total_activated": results["total_activated"],
        "actions_fired": results["actions_fired"],
        "estimated_discount_budget": results["estimated_discount_budget"],
        "recent_events": results["events"][:50],
    }

"""
RetainIQ — Data Engine
=======================
Handles all data ingestion, cleaning, merging, and feature engineering
for the Olist Brazilian E-Commerce dataset.

The Olist dataset ships as multiple CSV files. This module:
1. Loads and merges the relational tables into a single order-level DataFrame.
2. Cleans invalid/cancelled orders and missing customer IDs.
3. Engineers customer-level features for segmentation and churn modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from utils import CHURN_THRESHOLD_DAYS


# ──────────────────────────────────────────────
# 1. DATA LOADING & MERGING
# ──────────────────────────────────────────────

def load_olist_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Load all Olist CSV files and merge them into a single order-level DataFrame.

    Expected files in data_dir:
        - olist_orders_dataset.csv
        - olist_order_items_dataset.csv
        - olist_order_payments_dataset.csv
        - olist_order_reviews_dataset.csv
        - olist_customers_dataset.csv
        - olist_products_dataset.csv

    Returns:
        pd.DataFrame: Merged order-level DataFrame with customer, payment,
                       review, and product information.
    """
    data_path = Path(data_dir)

    # Load core tables
    orders = pd.read_csv(data_path / "olist_orders_dataset.csv")
    items = pd.read_csv(data_path / "olist_order_items_dataset.csv")
    payments = pd.read_csv(data_path / "olist_order_payments_dataset.csv")
    reviews = pd.read_csv(data_path / "olist_order_reviews_dataset.csv")
    customers = pd.read_csv(data_path / "olist_customers_dataset.csv")
    products = pd.read_csv(data_path / "olist_products_dataset.csv")

    # Aggregate payments per order (a single order can have multiple payment rows)
    payments_agg = (
        payments.groupby("order_id")
        .agg(total_payment=("payment_value", "sum"))
        .reset_index()
    )

    # Take the first review per order (some orders have duplicates)
    reviews_dedup = (
        reviews.sort_values("review_creation_date")
        .drop_duplicates(subset="order_id", keep="last")
        [["order_id", "review_score"]]
    )

    # Count items per order and get unique product categories
    items_agg = (
        items.groupby("order_id")
        .agg(
            n_items=("order_item_id", "count"),
            total_freight=("freight_value", "sum"),
        )
        .reset_index()
    )

    # Merge product category onto items, then aggregate unique categories per order
    items_with_cat = items.merge(
        products[["product_id", "product_category_name"]], on="product_id", how="left"
    )
    unique_cats = (
        items_with_cat.groupby("order_id")["product_category_name"]
        .nunique()
        .reset_index()
        .rename(columns={"product_category_name": "n_unique_categories"})
    )

    # === Master merge ===
    df = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(payments_agg, on="order_id", how="left")
        .merge(reviews_dedup, on="order_id", how="left")
        .merge(items_agg, on="order_id", how="left")
        .merge(unique_cats, on="order_id", how="left")
    )

    return df


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """
    Load a user-uploaded CSV file (from Streamlit file_uploader).

    This is the alternative ingestion path for non-Olist data.
    The CSV must contain at minimum: a customer ID, a date, and a monetary value column.

    Returns:
        pd.DataFrame: Raw uploaded data.
    """
    return pd.read_csv(uploaded_file)


# ──────────────────────────────────────────────
# 2. DATA CLEANING
# ──────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the merged Olist DataFrame.

    Steps:
        1. Keep only delivered orders (remove cancelled, unavailable, etc.)
        2. Drop rows with missing customer_unique_id (can't track them)
        3. Parse date columns to datetime
        4. Remove orders with non-positive payment
        5. Fill missing review scores with the dataset median

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for feature engineering.
    """
    df = df.copy()

    # 1. Keep only delivered orders
    df = df[df["order_status"] == "delivered"].copy()

    # 2. Drop rows without a trackable customer ID
    df = df.dropna(subset=["customer_unique_id"])

    # 3. Parse dates
    date_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 4. Remove non-positive payments
    df = df[df["total_payment"] > 0].copy()

    # 5. Fill missing review scores with median
    median_review = df["review_score"].median()
    df["review_score"] = df["review_score"].fillna(median_review)

    # 6. Fill missing item/freight counts
    df["n_items"] = df["n_items"].fillna(1).astype(int)
    df["total_freight"] = df["total_freight"].fillna(0)
    df["n_unique_categories"] = df["n_unique_categories"].fillna(1).astype(int)

    # 7. Compute delivery delay (days): positive = late, negative = early
    if "order_delivered_customer_date" in df.columns and "order_estimated_delivery_date" in df.columns:
        df["delivery_delay_days"] = (
            df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
        ).dt.days
        df["delivery_delay_days"] = df["delivery_delay_days"].fillna(0)

    return df.reset_index(drop=True)


# ──────────────────────────────────────────────
# 3. FEATURE & TARGET ENGINEERING (Time-Based Split)
# ──────────────────────────────────────────────

def engineer_features_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implements a rigorous Time-Based Out-Of-Time (OOT) Validation Split to prevent target leakage.
    
    Instead of using the entire dataset to calculate features AND labels concurrently, we split 
    the data temporally at a 'cutoff_date' (90 days before the maximum date in the dataset).
    
    Features (RFM + behavior): Computed strictly on orders BEFORE the cutoff date.
    Target (Churn + CLV): Computed strictly by observing behavior AFTER the cutoff date.
    This ensures models learn to forecast the future, rather than cheating by observing it.
    """
    df = df.copy()

    # Define the 90-day prediction cutoff threshold
    max_date = df["order_purchase_timestamp"].max()
    reference_date = max_date + pd.Timedelta(days=1)
    cutoff_date = reference_date - pd.Timedelta(days=90)
    
    # === SPLIT 1: FEATURE WINDOW (Historical data) ===
    feature_df = df[df["order_purchase_timestamp"] < cutoff_date].copy()
    
    # === SPLIT 2: TARGET WINDOW (Future behaviors) ===
    target_df = df[df["order_purchase_timestamp"] >= cutoff_date].copy()

    # 1. Compute Features (using only Feature Window)
    customer_df = (
        feature_df.groupby("customer_unique_id")
        .agg(
            recency=("order_purchase_timestamp", lambda x: (cutoff_date - x.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("total_payment", "sum"),
            avg_order_value=("total_payment", "mean"),
            avg_review_score=("review_score", "mean"),
            total_items=("n_items", "sum"),
            unique_categories=("n_unique_categories", "sum"),
            avg_delivery_delay=("delivery_delay_days", "mean"),
            first_purchase=("order_purchase_timestamp", "min"),
            last_purchase=("order_purchase_timestamp", "max"),
        )
        .reset_index()
    )

    # Calculate relational temporal features
    customer_df["purchase_span_days"] = (
        customer_df["last_purchase"] - customer_df["first_purchase"]
    ).dt.days

    customer_df["avg_days_between_purchases"] = np.where(
        customer_df["frequency"] > 1,
        customer_df["purchase_span_days"] / (customer_df["frequency"] - 1),
        0,
    )
    customer_df = customer_df.drop(columns=["first_purchase", "last_purchase"])

    # 2. Compute Targets (using only Target Window)
    future_behaviors = target_df.groupby("customer_unique_id").agg(
        target_purchases=("order_id", "nunique"),
        target_revenue=("total_payment", "sum")
    ).reset_index()

    # Merge historical traits with future outcomes
    customer_df = customer_df.merge(future_behaviors, on="customer_unique_id", how="left")
    
    # Fill NaN for users who made NO purchases in the target window (i.e. they churned)
    customer_df["target_purchases"] = customer_df["target_purchases"].fillna(0)
    customer_df["target_revenue"] = customer_df["target_revenue"].fillna(0)
    
    # DEFINING THE TRUE TARGETS:
    # Churn Label: 1 if ZERO purchases in target window, else 0 (Leakage resolved)
    customer_df["churned"] = (customer_df["target_purchases"] == 0).astype(int)
    
    # CLV Target: Exact continuous revenue generated in the target window
    customer_df["clv_target"] = customer_df["target_revenue"]

    # Round numerics
    numeric_cols = customer_df.select_dtypes(include=[np.number]).columns
    customer_df[numeric_cols] = customer_df[numeric_cols].round(2)

    return customer_df


# ──────────────────────────────────────────────
# 5. FULL PIPELINE
# ──────────────────────────────────────────────

def run_data_pipeline(data_dir: str = "data") -> pd.DataFrame:
    """
    Execute the complete data pipeline end-to-end.

    Steps:
        1. Load and merge Olist data
        2. Clean the merged DataFrame
        3. Engineer customer-level features
        4. Define churn labels

    Args:
        data_dir: Path to the directory containing Olist CSV files.

    Returns:
        pd.DataFrame: Final customer-level feature matrix with churn labels.
    """
    print("📦 Loading Olist dataset...")
    raw_df = load_olist_data(data_dir)
    print(f"   → Loaded {len(raw_df):,} raw orders")

    print("🧹 Cleaning data...")
    clean_df = clean_data(raw_df)
    print(f"   → {len(clean_df):,} valid delivered orders after cleaning")

    print("⚙️  Engineering features & Time-Based Split Target Variables...")
    customer_df = engineer_features_and_labels(clean_df)
    print(f"   → {len(customer_df):,} targetable customers")

    churn_rate = customer_df["churned"].mean() * 100
    print(f"   → Strict Time-split Churn rate: {churn_rate:.1f}%")

    print("✅ Data pipeline complete!")
    return customer_df


# ──────────────────────────────────────────────
# CLI entry point for standalone testing
# ──────────────────────────────────────────────

if __name__ == "__main__":
    customer_data = run_data_pipeline()
    print("\n📊 Feature Matrix Preview:")
    print(customer_data.head(10).to_string())
    print(f"\n📐 Shape: {customer_data.shape}")
    print(f"\n📈 Column stats:")
    print(customer_data.describe().round(2).to_string())

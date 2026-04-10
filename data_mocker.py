"""
RetainIQ — Data Mocker (Olist Drip-Feed Simulator)
=====================================================
Reads historical Olist CSV files and sends them one-by-one as HTTP POST
requests to the /api/webhooks/orders endpoint, simulating a live Shopify
or Stripe integration feeding real-time transactions.

Usage:
    # Terminal 1: Start the FastAPI server
    uvicorn backend.main:app --reload

    # Terminal 2: Run the mocker
    python data_mocker.py --limit 500 --delay 0.05
"""

import argparse
import time
import sys
import requests
import pandas as pd
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

API_URL = "http://127.0.0.1:8000/api/webhooks/orders"

DATA_DIR = Path(__file__).parent / "data"

ORDERS_FILE = DATA_DIR / "olist_orders_dataset.csv"
ITEMS_FILE = DATA_DIR / "olist_order_items_dataset.csv"
REVIEWS_FILE = DATA_DIR / "olist_order_reviews_dataset.csv"
CUSTOMERS_FILE = DATA_DIR / "olist_customers_dataset.csv"


# ──────────────────────────────────────────────
# DATA PREPARATION
# ──────────────────────────────────────────────

def prepare_feed_data(limit: int) -> pd.DataFrame:
    """
    Merges the Olist relational tables into a flat feed of order-level
    records ready to be sent as JSON payloads to the webhook.

    Each row becomes one API call with:
        - order_id
        - customer_id (customer_unique_id)
        - total_value (sum of price + freight per order)
        - order_timestamp
        - review_score (if available)
    """
    print("📦 Loading Olist CSVs for mock feed...")

    orders = pd.read_csv(ORDERS_FILE)
    items = pd.read_csv(ITEMS_FILE)
    reviews = pd.read_csv(REVIEWS_FILE)
    customers = pd.read_csv(CUSTOMERS_FILE)

    # Filter to delivered orders only
    orders = orders[orders["order_status"] == "delivered"].copy()

    # Merge customer_unique_id (the true customer identity)
    orders = orders.merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left",
    )

    # Aggregate item-level prices to order-level totals
    order_totals = (
        items.groupby("order_id")
        .agg(total_value=("price", "sum"))
        .reset_index()
    )
    orders = orders.merge(order_totals, on="order_id", how="left")

    # Merge review scores (take the first review per order)
    review_scores = (
        reviews.groupby("order_id")["review_score"]
        .first()
        .reset_index()
    )
    orders = orders.merge(review_scores, on="order_id", how="left")

    # Clean up
    orders = orders.dropna(subset=["total_value", "order_purchase_timestamp"])
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])

    # Sort chronologically (oldest first) to simulate real-time arrival
    orders = orders.sort_values("order_purchase_timestamp").reset_index(drop=True)

    # Select and rename columns to match the webhook payload schema
    feed = orders[[
        "order_id",
        "customer_unique_id",
        "total_value",
        "order_purchase_timestamp",
        "review_score",
    ]].rename(columns={
        "customer_unique_id": "customer_id",
        "order_purchase_timestamp": "order_timestamp",
    })

    # Limit the number of records
    if limit and limit < len(feed):
        feed = feed.head(limit)

    print(f"   → Prepared {len(feed):,} order payloads for drip-feed")
    return feed


# ──────────────────────────────────────────────
# DRIP-FEED ENGINE
# ──────────────────────────────────────────────

def drip_feed(feed: pd.DataFrame, delay: float = 0.05):
    """
    Sends each row as an HTTP POST to the webhook endpoint,
    one at a time, with a configurable delay between requests.
    """
    total = len(feed)
    success = 0
    duplicates = 0
    errors = 0

    print(f"\n🚀 Starting drip-feed to {API_URL}")
    print(f"   → {total} orders | {delay}s delay between requests\n")

    start_time = time.time()

    for i, row in feed.iterrows():
        payload = {
            "order_id": str(row["order_id"]),
            "customer_id": str(row["customer_id"]),
            "total_value": float(row["total_value"]),
            "order_timestamp": row["order_timestamp"].isoformat(),
        }

        # Add review_score only if it exists
        if pd.notna(row.get("review_score")):
            payload["review_score"] = float(row["review_score"])

        try:
            resp = requests.post(API_URL, json=payload, timeout=10)

            if resp.status_code == 201:
                success += 1
            elif resp.status_code == 409:
                duplicates += 1
            else:
                errors += 1
                print(f"   ⚠️  Order {payload['order_id']}: HTTP {resp.status_code} — {resp.text[:100]}")

        except requests.exceptions.ConnectionError:
            print(f"\n❌ Connection refused. Is the FastAPI server running?")
            print(f"   Start it with: uvicorn backend.main:app --reload")
            sys.exit(1)
        except Exception as e:
            errors += 1
            print(f"   ❌ Order {payload['order_id']}: {e}")

        # Progress bar every 100 records
        processed = success + duplicates + errors
        if processed % 100 == 0 or processed == total:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            bar_len = 30
            filled = int(bar_len * processed / total)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r   [{bar}] {processed}/{total}  ({rate:.0f} req/s)  ✅{success} ⏭️{duplicates} ❌{errors}", end="", flush=True)

        time.sleep(delay)

    elapsed = time.time() - start_time
    print(f"\n\n✅ Drip-feed complete in {elapsed:.1f}s")
    print(f"   → Successful:  {success:,}")
    print(f"   → Duplicates:  {duplicates:,}")
    print(f"   → Errors:      {errors:,}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RetainIQ Data Mocker — Olist Drip-Feed Simulator")
    parser.add_argument("--limit", type=int, default=500, help="Max number of orders to send (default: 500)")
    parser.add_argument("--delay", type=float, default=0.05, help="Seconds between requests (default: 0.05)")
    parser.add_argument("--url", type=str, default=API_URL, help="Target webhook URL")
    args = parser.parse_args()

    API_URL = args.url

    feed = prepare_feed_data(args.limit)
    drip_feed(feed, delay=args.delay)

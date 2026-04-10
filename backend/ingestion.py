from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional

from .database import get_db, Customer, Order

router = APIRouter()

# --- Pydantic Schemas for API Input ---
class OrderPayload(BaseModel):
    order_id: str
    customer_id: str
    total_value: float
    order_timestamp: datetime
    review_score: Optional[float] = None

# --- Business Logic ---
def update_rfm_metrics(db: Session, customer_id: str):
    """
    Recalculates Recency, Frequency, and Monetary values for a customer
    after a new order is received, then triggers ML inference.
    """
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        return
        
    # Recalculate metrics based on all orders
    all_orders = db.query(Order).filter(Order.customer_id == customer_id).all()
    
    # Frequency
    customer.frequency_count = len(all_orders)
    
    # Monetary
    customer.monetary_total = sum(order.total_value for order in all_orders)
    customer.avg_order_value = customer.monetary_total / customer.frequency_count if customer.frequency_count > 0 else 0
    
    # Recency
    latest_order = max(order.order_timestamp for order in all_orders)
    now = datetime.utcnow()
    # If the order is from the past (like Olist mock data), we calculate from 'now' 
    # but for true SaaS it would just be (now - latest_order).days
    customer.recency_days = (now - latest_order).days
    
    # Update Timestamps
    customer.last_purchase = latest_order
    if len(all_orders) == 1:
        customer.first_purchase = latest_order
        
    # TODO: Phase 2 - Trigger prediction logic via async task
    # e.g., trigger_ml_inference(customer_id)
        
    db.commit()


# --- API Routes ---
@router.post("/api/webhooks/orders", status_code=201)
def receive_order_webhook(
    payload: OrderPayload, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Webhook endpoint to receive live transactions from Shopify/Stripe.
    Upserts the customer and async recalculates RFM.
    """
    # 1. Upsert Customer
    customer = db.query(Customer).filter(Customer.id == payload.customer_id).first()
    if not customer:
        customer = Customer(id=payload.customer_id)
        db.add(customer)
        db.commit()
        db.refresh(customer)

    # 2. Check if Order already exists (idempotency)
    existing_order = db.query(Order).filter(Order.order_id == payload.order_id).first()
    if existing_order:
        raise HTTPException(status_code=409, detail="Order already processed")

    # 3. Create Order
    new_order = Order(
        order_id=payload.order_id,
        customer_id=payload.customer_id,
        total_value=payload.total_value,
        order_timestamp=payload.order_timestamp,
        review_score=payload.review_score
    )
    db.add(new_order)
    db.commit()

    # 4. Fire Async RFM Recalculation so webhook returns 201 instantly
    background_tasks.add_task(update_rfm_metrics, db, payload.customer_id)

    return {"status": "success", "message": "Order ingested and RFM update queued", "order_id": payload.order_id}

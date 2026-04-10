import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

# Define database location (root directory of project)
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "retainiq.db"))
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# Connect to SQLite, allow multi-threading
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Customer(Base):
    """
    Core customer profile representing aggregated RFM metrics and ML predictions.
    """
    __tablename__ = "customers"

    id = Column(String, primary_key=True, index=True) # customer_unique_id
    
    # Raw RFM Metrics
    recency_days = Column(Integer, default=0)
    frequency_count = Column(Integer, default=0)
    monetary_total = Column(Float, default=0.0)
    
    # Advanced Features
    avg_order_value = Column(Float, default=0.0)
    avg_review_score = Column(Float, default=0.0)
    
    # ML Predictions
    segment = Column(String, nullable=True) # e.g., 'Champions'
    churn_probability = Column(Float, nullable=True)
    predicted_clv = Column(Float, nullable=True) # Customer Lifetime Value ($)
    action_quadrant = Column(String, nullable=True) # e.g., 'SAVE_NOW'
    
    # Timestamps
    first_purchase = Column(DateTime, nullable=True)
    last_purchase = Column(DateTime, nullable=True)
    last_ml_update = Column(DateTime, default=datetime.utcnow)

    # Relationships
    orders = relationship("Order", back_populates="customer", cascade="all, delete-orphan")


class Order(Base):
    """
    Individual transactional records. Used to recalculate RFM when new data arrives.
    """
    __tablename__ = "orders"

    order_id = Column(String, primary_key=True, index=True)
    customer_id = Column(String, ForeignKey("customers.id"), index=True)
    
    order_timestamp = Column(DateTime, nullable=False, index=True)
    total_value = Column(Float, nullable=False)
    review_score = Column(Float, nullable=True)
    
    # If the database grows too large, we could remove this relationship
    customer = relationship("Customer", back_populates="orders")


# Central dependency to inject DB session into FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

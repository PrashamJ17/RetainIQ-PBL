# 🧠 RetainIQ: Enterprise Customer Retention Engine

## Description
RetainIQ is a production-ready, microservices-based SaaS analytics platform designed to reverse customer churn before it happens. By ingesting raw e-commerce transaction data, RetainIQ leverages advanced Machine Learning to transform logs into instantly actionable marketing insights. 

It completely bridges the gap between raw Big Data and executive marketing decisions by not only predicting *who* will churn, but calculating exactly *how much* revenue is at stake, explaining *why* they are leaving, and prescribing exactly *what* to do about it.

## Features
- 📊 **Dynamic RFM Segmentation:** Evaluates Recency, Frequency, and Monetary value to autonomously group customers using K-Means clustering into distinct financial tribes (e.g., Champions, At-Risk).
- 📉 **Predictive Churn Modeling:** Utilizes a highly accurate XGBoost classification engine to assign a calculated `0–100%` churn risk probability to every individual user.
- 💰 **Customer Lifetime Value (CLV):** Continuous regression modeling estimating specific future revenue generated per user.
- 🧠 **SHAP Explainability (Explainable AI):** Opens the Machine Learning "black box" by computing the Top 3 mathematical reasons an individual was flagged for churn, providing complete confidence to human operators.
- ⚡ **Real-Time Action Matrix:** Automatically visualizes all users on a Value-vs-Risk scatter plot, prescribing precise intervention strategies: *Save Now, Let Go, Nurture, Monitor*.
- 🔔 **Automated API Activations:** Built-in simulation layer for CRM webhook integrations (like Mailchimp or Klaviyo), programmed to trigger targeted recovery emails instantaneously when users cross churn thresholds.

## Tech Stack
### Frontend (Dashboard UI)
- **Framework:** Next.js 14, React 18
- **Styling:** Tailwind CSS, Glassmorphism UI
- **Data Visualization:** Recharts

### Backend (REST API)
- **Framework:** FastAPI, Uvicorn
- **Database:** SQLite, SQLAlchemy ORM
- **Python Version:** 3.13

### Machine Learning
- **Core Analytics:** Pandas, NumPy
- **Algorithms:** Scikit-Learn (K-Means), XGBoost (Classifier & Regressor)
- **Explainability:** SHAP

### Infrastructure
- **Containerization:** Docker, Docker Compose

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/PrashamJ17/RetainIQ-PBL.git
cd RetainIQ-PBL
```

### 2. Prepare the Initial Dataset
RetainIQ utilizes the [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) to mock historical initialization. Download the archive and extract the required CSVs into the root `data/` directory:

```text
data/
├── olist_orders_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_customers_dataset.csv
└── olist_products_dataset.csv
```

## Usage

RetainIQ is mapped into a secure, cross-communicating Docker ecosystem. Launching the entire stack requires only a single command:

```bash
docker-compose up --build
```

**Access Points:**
- **Frontend Dashboard:** Navigate to `http://localhost:3000` to interact with the premium Next.js UI.
- **FastAPI Sandbox:** Navigate to `http://localhost:8000/docs` to view and interact with the automatically generated Swagger API documentation.

## Project Structure

```text
RetainIQ/
├── backend/
│   ├── api.py               # Asynchronous FastAPI routing and state management
│   ├── database.py          # SQLAlchemy SQLite integration
│   ├── ingestion.py         # Mock streaming and webhook transaction simulator
│   ├── churn_model.py       # XGBoost classifier and SHAP generation
│   ├── clv_model.py         # Continuous XGBRegressor for lifetime value estimation
│   ├── activation.py        # CRM integration rules and execution engine
│   └── Dockerfile           # Backend Python microservice container
├── frontend/
│   ├── app/                 # Next.js App Router (Layouts, Pages)
│   ├── components/          # Recharts visualization modules and Tailwind components
│   ├── lib/api.js           # Centralized fetching utilizing internal Docker DNS logic
│   └── Dockerfile           # Frontend Node.js microservice container
├── docker-compose.yml       # Ecosystem orchestration and volume persistence
└── PROJECT_REPORT.md        # Comprehensive technical defense and architecture breakdown
```

## Screenshots / Demo

*Note: Replace the below placeholders with actual image URLs.*

![Dashboard Overview]([Add details here: URL to Overview Image])
*The Overview executive dashboard rendering dynamic KPIs and pipeline performance status.*

![Action Matrix]([Add details here: URL to Action Matrix Image])
*The Action Matrix isolating High-Value/High-Risk customers for immediate 'Save Now' interventions.*

## Example Input/Output

### Input (Raw Webhook Data)
The ingestion layer receives standard JSON e-commerce payloads:
```json
{
  "customer_id": "cust_8829",
  "order_value": 150.40,
  "timestamp": "2026-04-10T14:30:00Z"
}
```

### Output (ML Evaluation & Action)
The backend calculates RFM, triggers the XGBoost pipeline, calculates lifetime value, and outputs absolute commands:
```text
🤖 AI Evaluation Complete for cust_8829:
- Churn Risk: 92.4%
- Predicted Future Value (CLV): $430
- Primary Reason: "Average delivery delay exceeded 14 days"
- Action Matrix Assignment: 🚨 SAVE NOW
- System Command: Firing Webhook -> Send 20% Recovery Discount via Klaviyo.
```

## Future Improvements
- **Generative AI (LLMs):** Injecting calculated RFM parameters into LLMs to autonomously write highly personalized email copy rather than relying on static discount templates.
- **Event-Streaming (Kafka):** Scaling from REST webhooks to Apache Kafka to evaluate real-time browsing behaviors (like cart abandonment) with sub-second latency.
- **Uplift Modeling:** Upgrading from standard Churn prediction to Causal Inference models to determine if a marketing intervention will *actually* alter a customer's behavior (avoiding 'Sleeping Dogs').
- **Transformer Sequence Models:** Transitioning from tabular machine learning to deep learning sequence models like LSTMs to track chronological user journeys over time.

## Author / Contact
**Prasham Jain**  
[GitHub Profile](https://github.com/PrashamJ17) | [LinkedIn](https://www.linkedin.com/in/prasham-jain-774652314/)

## Mentor 
**Dr Rishi Gupta**

B.Tech Computer Science and Engineering  
Manipal University Jaipur (Class of 2028)

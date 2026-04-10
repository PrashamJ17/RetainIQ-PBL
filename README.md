# 🧠 RetainIQ — Customer Retention Engine

A production-grade **Streamlit-powered analytics tool** that ingests e-commerce transaction data, segments customers using RFM + K-Means clustering, predicts churn with XGBoost, and provides explainable, actionable retention recommendations via SHAP values.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place the Dataset
Download the [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and extract the CSVs into the `data/` folder:
```
data/
├── olist_orders_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_customers_dataset.csv
└── olist_products_dataset.csv
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```

---

## 📐 Architecture

```
RetainIQ/
├── app.py               ← Streamlit dashboard (4-tab UI)
├── data_engine.py       ← Data ingestion, cleaning, feature engineering
├── segmentation.py      ← RFM + K-Means clustering
├── churn_model.py       ← XGBoost churn prediction + SHAP
├── utils.py             ← Shared config, styling, constants
├── requirements.txt     ← Python dependencies
├── data/                ← Olist CSV files
└── models/              ← Saved trained model artifacts
```

### Pipeline Flow
```
CSV Upload → Data Engine → Segmentation (K-Means) → Churn Model (XGBoost) → SHAP → Dashboard
```

---

## 🎯 Features

| Feature | Description |
|---|---|
| **RFM Segmentation** | K-Means clustering on Recency, Frequency, Monetary values |
| **Churn Prediction** | XGBoost classifier with per-user probability scores |
| **SHAP Explainability** | Top 3 reasons per flagged user (human-readable) |
| **Action Matrix** | 2x2 quadrant: Save Now / Let Go / Nurture / Monitor |
| **CSV Export** | 1-click download of targeted marketing lists |
| **Interactive Dashboard** | Plotly 3D scatter, validation charts, drill-down tables |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 99.87% |
| F1 Score | 99.92% |
| ROC-AUC | 1.0000 |
| Precision | 100.00% |
| Recall | 99.84% |

---

## 🛠️ Tech Stack
- **UI:** Streamlit
- **Data:** Pandas, NumPy
- **ML:** XGBoost, Scikit-learn
- **Explainability:** SHAP
- **Visualization:** Plotly

---

## 👤 Author
**Prasham Jain** — B.Tech CSE, Manipal University Jaipur

"""
RetainIQ — Shared Utilities
============================
Color palettes, segment label maps, Streamlit page config, and styling helpers.
"""

# ──────────────────────────────────────────────
# Segment Configuration
# ──────────────────────────────────────────────

SEGMENT_LABELS = {
    0: "Champions",
    1: "Loyalists",
    2: "At-Risk",
    3: "Hibernating",
}

SEGMENT_COLORS = {
    "Champions": "#6C63FF",
    "Loyalists": "#00C9A7",
    "At-Risk": "#FF6B6B",
    "Hibernating": "#A0AEC0",
}

SEGMENT_DESCRIPTIONS = {
    "Champions": "High spend, high frequency, recent purchases. Your best customers.",
    "Loyalists": "Consistent buyers with moderate spend. Reliable revenue base.",
    "At-Risk": "Previously active but engagement is declining. Intervention needed.",
    "Hibernating": "Inactive for a long time, low spend. Likely already lost.",
}

# ──────────────────────────────────────────────
# Action Matrix Configuration
# ──────────────────────────────────────────────

ACTION_MATRIX = {
    "SAVE_NOW": {
        "label": "🚨 SAVE NOW",
        "description": "High Value + High Churn Risk — Send retention offer immediately.",
        "color": "#FF6B6B",
    },
    "LET_GO": {
        "label": "🚫 LET GO",
        "description": "Low Value + High Churn Risk — Not worth the intervention cost.",
        "color": "#A0AEC0",
    },
    "NURTURE": {
        "label": "👑 NURTURE",
        "description": "High Value + Low Churn Risk — Reward loyalty, give early access.",
        "color": "#6C63FF",
    },
    "MONITOR": {
        "label": "👀 MONITOR",
        "description": "Low Value + Low Churn Risk — Keep an eye, no urgent action.",
        "color": "#00C9A7",
    },
}

# ──────────────────────────────────────────────
# Streamlit Page Configuration
# ──────────────────────────────────────────────

PAGE_CONFIG = {
    "page_title": "RetainIQ — Customer Retention Engine",
    "page_icon": "🧠",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# ──────────────────────────────────────────────
# Feature Engineering Constants
# ──────────────────────────────────────────────

# Columns expected after Olist data merging
OLIST_ORDER_COLS = [
    "customer_unique_id",
    "order_id",
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "order_status",
    "review_score",
    "payment_value",
    "product_category_name",
]

# Churn threshold: if a customer hasn't purchased in this many days, they are "churned"
CHURN_THRESHOLD_DAYS = 90

# Number of K-Means clusters for segmentation
N_CLUSTERS = 4

# Train/Test split ratio
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ──────────────────────────────────────────────
# Styling Helpers
# ──────────────────────────────────────────────

def get_streamlit_css():
    """Return custom CSS for the Streamlit dashboard."""
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .metric-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 16px;
            padding: 24px;
            color: white;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(108, 99, 255, 0.2);
        }

        .metric-card h3 {
            font-size: 14px;
            font-weight: 500;
            color: #a0aec0;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .metric-card .value {
            font-size: 36px;
            font-weight: 700;
            color: #ffffff;
        }

        .segment-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .save-now { background: rgba(255, 107, 107, 0.2); color: #FF6B6B; }
        .let-go { background: rgba(160, 174, 192, 0.2); color: #A0AEC0; }
        .nurture { background: rgba(108, 99, 255, 0.2); color: #6C63FF; }
        .monitor { background: rgba(0, 201, 167, 0.2); color: #00C9A7; }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 16px;
        }
    </style>
    """


def render_metric_card(title: str, value: str, delta: str = None) -> str:
    """Return HTML for a styled metric card."""
    delta_html = ""
    if delta:
        color = "#00C9A7" if delta.startswith("+") or delta.startswith("↑") else "#FF6B6B"
        delta_html = f'<p style="color: {color}; font-size: 14px; margin-top: 4px;">{delta}</p>'

    return f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <div class="value">{value}</div>
        {delta_html}
    </div>
    """


def assign_action(churn_prob: float, monetary_percentile: float) -> str:
    """
    Assign a retention action based on the Action Matrix quadrant.

    Args:
        churn_prob: Churn probability (0.0 to 1.0)
        monetary_percentile: Customer's monetary value as a percentile (0.0 to 1.0)

    Returns:
        Action key from ACTION_MATRIX
    """
    high_risk = churn_prob >= 0.5
    high_value = monetary_percentile >= 0.5

    if high_value and high_risk:
        return "SAVE_NOW"
    elif not high_value and high_risk:
        return "LET_GO"
    elif high_value and not high_risk:
        return "NURTURE"
    else:
        return "MONITOR"

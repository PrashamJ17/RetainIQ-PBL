"""
RetainIQ — Activation Layer (Webhook Simulator)
=================================================
Simulates automated marketing actions that would fire in a production
environment when a customer crosses a churn risk threshold.

In production, these would be real API calls to:
    - Klaviyo (email marketing)
    - Twilio (SMS alerts)
    - Slack (internal team notifications)
    - Intercom (in-app messages)

For the MVP, we log all "fired" actions to an in-memory ledger
and expose them via the API so the frontend can display a real-time
activity feed proving the system works end-to-end.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict

PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ──────────────────────────────────────────────
# 1. ACTION TEMPLATES
# ──────────────────────────────────────────────

ACTION_TEMPLATES = {
    "SAVE_NOW": {
        "channel": "email",
        "template": "win_back_vip",
        "subject": "We miss you! Here's 20% off your next order 💜",
        "discount_pct": 20,
        "urgency": "HIGH",
        "description": "VIP recovery email with 20% discount code",
    },
    "NURTURE": {
        "channel": "email",
        "template": "loyalty_reward",
        "subject": "You're a star ⭐ Unlock early access to new arrivals!",
        "discount_pct": 0,
        "urgency": "LOW",
        "description": "Loyalty nurture — early access, no discount needed",
    },
    "MONITOR": {
        "channel": "none",
        "template": "no_action",
        "subject": "",
        "discount_pct": 0,
        "urgency": "NONE",
        "description": "No action required — customer is stable",
    },
    "LET_GO": {
        "channel": "none",
        "template": "no_action",
        "subject": "",
        "discount_pct": 0,
        "urgency": "NONE",
        "description": "Low-value churned customer — do not spend budget",
    },
}


# ──────────────────────────────────────────────
# 2. ACTIVATION EVENT LOG
# ──────────────────────────────────────────────

@dataclass
class ActivationEvent:
    """A single automated marketing action that was 'fired'."""
    customer_id: str
    action: str
    channel: str
    template: str
    subject: str
    discount_pct: int
    urgency: str
    churn_probability: float
    predicted_clv: float
    segment: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "SIMULATED"  # Would be "SENT" in production


# Global event ledger
_activation_log: list[ActivationEvent] = []


def get_activation_log() -> list[dict]:
    """Return the full activation event log as a list of dicts."""
    return [asdict(event) for event in _activation_log]


def clear_activation_log():
    """Clear the activation log (for testing)."""
    _activation_log.clear()


# ──────────────────────────────────────────────
# 3. ACTIVATION ENGINE
# ──────────────────────────────────────────────

def evaluate_and_activate(
    customer_id: str,
    churn_probability: float,
    predicted_clv: float,
    segment: str,
    action_quadrant: str,
    churn_threshold: float = 0.85,
    clv_threshold: float = 100.0,
) -> Optional[ActivationEvent]:
    """
    Evaluate whether a customer should receive an automated action.

    Rules:
        1. Only fire for SAVE_NOW or NURTURE quadrants
        2. SAVE_NOW fires only if churn_prob >= churn_threshold
        3. Additional filter: only for customers with CLV >= clv_threshold
           (don't waste marketing spend on low-value users)

    In production, this would call the Klaviyo/Twilio API.
    For the MVP, it logs the event to the activation ledger.

    Returns:
        ActivationEvent if action was triggered, None otherwise.
    """
    template = ACTION_TEMPLATES.get(action_quadrant)
    if not template:
        return None

    # Rule: Only activate for actionable quadrants
    if action_quadrant not in ["SAVE_NOW", "NURTURE"]:
        return None

    # Rule: SAVE_NOW requires high churn probability
    if action_quadrant == "SAVE_NOW" and churn_probability < churn_threshold:
        return None

    # Rule: Only activate for customers worth saving
    if predicted_clv < clv_threshold:
        return None

    # === SIMULATE THE ACTION ===
    event = ActivationEvent(
        customer_id=customer_id,
        action=action_quadrant,
        channel=template["channel"],
        template=template["template"],
        subject=template["subject"],
        discount_pct=template["discount_pct"],
        urgency=template["urgency"],
        churn_probability=round(churn_probability, 4),
        predicted_clv=round(predicted_clv, 2),
        segment=segment,
    )

    # In production: requests.post("https://api.klaviyo.com/v2/...", json={...})
    # For MVP: Log to in-memory ledger
    _activation_log.append(event)

    return event


# ──────────────────────────────────────────────
# 4. BATCH ACTIVATION
# ──────────────────────────────────────────────

def run_batch_activation(customer_df, churn_threshold: float = 0.85, clv_threshold: float = 100.0) -> dict:
    """
    Scan all customers and fire automated actions for eligible ones.

    Args:
        customer_df: DataFrame with churn_probability, predicted_clv, segment, action columns.
        churn_threshold: Minimum churn probability to trigger SAVE_NOW.
        clv_threshold: Minimum predicted CLV to justify marketing spend.

    Returns:
        Dict with activation stats and event log.
    """
    clear_activation_log()

    total_evaluated = 0
    total_activated = 0
    total_discount_value = 0.0

    actions_fired = {"SAVE_NOW": 0, "NURTURE": 0}

    for _, row in customer_df.iterrows():
        total_evaluated += 1

        clv = row.get("predicted_clv", 0)
        churn_prob = row.get("churn_probability", 0)

        event = evaluate_and_activate(
            customer_id=str(row["customer_unique_id"]),
            churn_probability=float(churn_prob),
            predicted_clv=float(clv),
            segment=str(row.get("segment", "Unknown")),
            action_quadrant=str(row.get("action", "MONITOR")),
            churn_threshold=churn_threshold,
            clv_threshold=clv_threshold,
        )

        if event:
            total_activated += 1
            actions_fired[event.action] = actions_fired.get(event.action, 0) + 1
            # Estimate discount cost (discount_pct of CLV)
            if event.discount_pct > 0:
                total_discount_value += clv * (event.discount_pct / 100)

    print(f"\n🚀 Batch Activation Complete!")
    print(f"   → Evaluated:  {total_evaluated:,} customers")
    print(f"   → Activated:  {total_activated:,} actions")
    print(f"   → SAVE_NOW:   {actions_fired.get('SAVE_NOW', 0):,}")
    print(f"   → NURTURE:    {actions_fired.get('NURTURE', 0):,}")
    print(f"   → Est. Discount Budget: R${total_discount_value:,.2f}")

    return {
        "total_evaluated": total_evaluated,
        "total_activated": total_activated,
        "actions_fired": actions_fired,
        "estimated_discount_budget": round(total_discount_value, 2),
        "events": get_activation_log()[:100],  # Return first 100 for API
    }


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from data_engine import run_data_pipeline
    from segmentation import run_segmentation_pipeline
    from churn_model import run_churn_pipeline
    from clv_model import run_clv_pipeline

    # Full pipeline
    customer_data = run_data_pipeline()
    seg_results = run_segmentation_pipeline(customer_data)
    customer_data = seg_results["customer_df"]
    churn_results = run_churn_pipeline(customer_data)
    customer_data = churn_results["customer_df"]
    clv_results = run_clv_pipeline(customer_data)
    customer_data = clv_results["customer_df"]

    # Run activation
    activation_results = run_batch_activation(customer_data)

    # Show sample events
    print("\n📧 Sample Activation Events:")
    for event in activation_results["events"][:5]:
        print(f"   → {event['customer_id'][:20]}... | {event['action']} | CLV: R${event['predicted_clv']:.2f} | Churn: {event['churn_probability']:.0%}")

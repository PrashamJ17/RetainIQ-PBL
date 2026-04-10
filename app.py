"""
RetainIQ — Customer Retention Engine Dashboard
================================================
A Streamlit-powered interactive dashboard for marketing managers.

Tabs:
    1. 📊 Overview — Dataset summary, key metrics
    2. 🎯 Segments — 3D RFM scatter, segment distribution, drill-down
    3. ⚠️ Churn Risk — Churn predictions, SHAP explanations, model metrics
    4. 🚀 Action Matrix — 2x2 quadrant, actionable export
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils import (
    PAGE_CONFIG,
    SEGMENT_COLORS,
    SEGMENT_DESCRIPTIONS,
    ACTION_MATRIX,
    get_streamlit_css,
    render_metric_card,
)
from data_engine import run_data_pipeline
from segmentation import (
    run_segmentation_pipeline,
    plot_segments_3d,
    plot_segment_distribution,
    plot_rfm_boxplots,
    plot_elbow_silhouette,
)
from churn_model import (
    run_churn_pipeline,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_churn_distribution,
    plot_action_matrix,
)

# ──────────────────────────────────────────────
# PAGE SETUP
# ──────────────────────────────────────────────

st.set_page_config(**PAGE_CONFIG)
st.markdown(get_streamlit_css(), unsafe_allow_html=True)


# ──────────────────────────────────────────────
# CACHED DATA LOADING
# ──────────────────────────────────────────────

@st.cache_data(show_spinner="🔄 Running data pipeline...")
def load_and_process_data():
    """Run the full pipeline and cache results."""
    # Phase 1: Data Engine
    customer_df = run_data_pipeline()

    # Phase 2: Segmentation
    seg_results = run_segmentation_pipeline(customer_df)
    customer_df = seg_results["customer_df"]

    # Phase 3: Churn Prediction
    churn_results = run_churn_pipeline(customer_df)

    return {
        "customer_df": churn_results["customer_df"],
        "segment_summary": seg_results["segment_summary"],
        "cluster_validation": seg_results["cluster_validation"],
        "metrics": churn_results["metrics"],
        "shap_results": churn_results["shap_results"],
    }


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🧠 RetainIQ")
    st.markdown("#### Customer Retention Engine")
    st.divider()
    st.markdown(
        "Analyze customer behavior, predict churn risk, "
        "and generate actionable retention strategies."
    )
    st.divider()

    # Load data
    with st.spinner("Processing pipeline..."):
        try:
            data = load_and_process_data()
            st.success("✅ Pipeline loaded!")
        except Exception as e:
            st.error(f"❌ Pipeline error: {e}")
            st.stop()

    customer_df = data["customer_df"]
    segment_summary = data["segment_summary"]
    cluster_validation = data["cluster_validation"]
    metrics = data["metrics"]
    shap_results = data["shap_results"]

    st.divider()
    st.markdown("### 📊 Quick Stats")
    st.metric("Total Customers", f"{len(customer_df):,}")
    st.metric("Churn Rate", f"{customer_df['churned'].mean():.1%}")
    st.metric("Model ROC-AUC", f"{metrics['roc_auc']:.4f}")

    st.divider()
    st.markdown(
        "<p style='color: #a0aec0; font-size: 12px;'>"
        "Built by Prasham Jain • B.Tech CSE<br>"
        "Manipal University Jaipur</p>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# MAIN CONTENT — TABS
# ──────────────────────────────────────────────

tab_overview, tab_segments, tab_churn, tab_actions = st.tabs([
    "📊 Overview",
    "🎯 Segments",
    "⚠️ Churn Risk",
    "🚀 Action Matrix",
])


# ──────────────────────────────────────────────
# TAB 1: OVERVIEW
# ──────────────────────────────────────────────

with tab_overview:
    st.markdown("## 📊 Dashboard Overview")
    st.markdown("A high-level summary of your customer base and model performance.")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            render_metric_card("Total Customers", f"{len(customer_df):,}"),
            unsafe_allow_html=True,
        )
    with col2:
        total_revenue = customer_df["monetary"].sum()
        st.markdown(
            render_metric_card("Total Revenue", f"R${total_revenue:,.0f}"),
            unsafe_allow_html=True,
        )
    with col3:
        avg_order = customer_df["avg_order_value"].mean()
        st.markdown(
            render_metric_card("Avg Order Value", f"R${avg_order:,.2f}"),
            unsafe_allow_html=True,
        )
    with col4:
        churn_pct = customer_df["churned"].mean() * 100
        st.markdown(
            render_metric_card("Overall Churn Rate", f"{churn_pct:.1f}%"),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Model Performance Summary
    st.markdown("### 🤖 Model Performance")
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col_m2.metric("F1 Score", f"{metrics['f1_score']:.2%}")
    col_m3.metric("Precision", f"{metrics['precision']:.2%}")
    col_m4.metric("Recall", f"{metrics['recall']:.2%}")
    col_m5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

    st.markdown("---")

    # Segment Overview Table
    st.markdown("### 🎯 Segment Overview")
    styled_summary = segment_summary.style.format({
        "avg_monetary": "R${:,.2f}",
        "avg_recency": "{:.0f} days",
        "avg_frequency": "{:.2f}",
        "churn_rate": "{:.1f}%",
    })
    st.dataframe(styled_summary, use_container_width=True, hide_index=True)

    # Segment descriptions
    st.markdown("### 📖 Segment Definitions")
    for segment, desc in SEGMENT_DESCRIPTIONS.items():
        color = SEGMENT_COLORS.get(segment, "#888")
        st.markdown(
            f'<span style="color: {color}; font-weight: 600;">● {segment}:</span> {desc}',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# TAB 2: SEGMENTS
# ──────────────────────────────────────────────

with tab_segments:
    st.markdown("## 🎯 Customer Segmentation")
    st.markdown("Behavioral clustering using RFM (Recency, Frequency, Monetary) + K-Means.")

    # Cluster Validation
    st.markdown("### 🔬 Cluster Validation")
    st.plotly_chart(plot_elbow_silhouette(cluster_validation), use_container_width=True)

    st.markdown("---")

    # 3D Scatter
    st.markdown("### 🌐 3D RFM Visualization")
    st.plotly_chart(plot_segments_3d(customer_df), use_container_width=True)

    # Side by side: Distribution + Boxplots
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("### 📊 Distribution")
        st.plotly_chart(plot_segment_distribution(customer_df), use_container_width=True)

    with col_right:
        st.markdown("### 📦 RFM Characteristics")
        st.plotly_chart(plot_rfm_boxplots(customer_df), use_container_width=True)

    # Drill-down table
    st.markdown("### 🔍 Segment Drill-Down")
    selected_segment = st.selectbox(
        "Select a segment to explore:",
        ["All"] + list(SEGMENT_DESCRIPTIONS.keys()),
    )

    if selected_segment == "All":
        drill_df = customer_df
    else:
        drill_df = customer_df[customer_df["segment"] == selected_segment]

    st.dataframe(
        drill_df[["customer_unique_id", "segment", "recency", "frequency", "monetary",
                   "avg_review_score", "churn_probability", "action"]]
        .sort_values("monetary", ascending=False)
        .head(100),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Showing top 100 by monetary value. Total in segment: {len(drill_df):,}")


# ──────────────────────────────────────────────
# TAB 3: CHURN RISK
# ──────────────────────────────────────────────

with tab_churn:
    st.markdown("## ⚠️ Churn Risk Analysis")
    st.markdown("XGBoost predictions with SHAP-powered explainability.")

    # Churn Distribution
    st.markdown("### 📈 Churn Probability Distribution")
    st.plotly_chart(plot_churn_distribution(customer_df), use_container_width=True)

    col_fi, col_cm = st.columns([2, 1])

    with col_fi:
        st.markdown("### 🏆 Feature Importance (SHAP)")
        st.plotly_chart(
            plot_feature_importance(shap_results["feature_importance"]),
            use_container_width=True,
        )

    with col_cm:
        st.markdown("### 📐 Confusion Matrix")
        st.plotly_chart(
            plot_confusion_matrix(metrics),
            use_container_width=True,
        )

    st.markdown("---")

    # Top High-Risk Users with SHAP Reasons
    st.markdown("### 🚨 Top High-Risk Users (with AI Explanations)")
    st.markdown("Each user is explained by the top 3 features driving their churn prediction.")

    n_show = st.slider("Number of users to show:", 5, 50, 10)

    for i, user in enumerate(shap_results["top_reasons"][:n_show]):
        idx = user["customer_index"]
        prob = user["churn_probability"]
        user_row = customer_df.iloc[idx] if idx < len(customer_df) else None

        segment = user_row["segment"] if user_row is not None else "Unknown"
        seg_color = SEGMENT_COLORS.get(segment, "#888")
        monetary = user_row["monetary"] if user_row is not None else 0

        with st.expander(
            f"#{i+1} — Churn Risk: {prob:.0%} | Segment: {segment} | Spend: R${monetary:,.2f}",
            expanded=(i < 3),
        ):
            for r in user["reasons"]:
                icon = "🔴" if r["direction"] == "increases" else "🟢"
                st.markdown(f"{icon} **{r['human_description']}** = `{r['feature_value']}` → {r['direction']} churn risk")


# ──────────────────────────────────────────────
# TAB 4: ACTION MATRIX
# ──────────────────────────────────────────────

with tab_actions:
    st.markdown("## 🚀 Action Matrix")
    st.markdown("Cross-referencing customer value with churn risk to generate targeted strategies.")

    # The Quadrant Plot
    st.plotly_chart(plot_action_matrix(customer_df), use_container_width=True)

    # Action Distribution
    st.markdown("### 📊 Action Distribution")
    action_counts = customer_df["action"].value_counts()

    cols = st.columns(4)
    for i, (action_key, action_info) in enumerate(ACTION_MATRIX.items()):
        count = action_counts.get(action_key, 0)
        pct = count / len(customer_df) * 100
        with cols[i]:
            st.markdown(
                f"""
                <div style="background: rgba(26,26,46,0.8); border-radius: 12px; padding: 16px;
                            border-left: 4px solid {action_info['color']}; margin-bottom: 8px;">
                    <div style="font-size: 24px; font-weight: 700; color: {action_info['color']};">{count:,}</div>
                    <div style="font-size: 14px; color: white; font-weight: 600;">{action_info['label']}</div>
                    <div style="font-size: 12px; color: #a0aec0;">{pct:.1f}% of total</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(action_info["description"])

    st.markdown("---")

    # Filtered Export
    st.markdown("### 📥 Export Targeted Lists")
    st.markdown("Select an action group to download a ready-to-use marketing CSV.")

    export_action = st.selectbox(
        "Select action to export:",
        list(ACTION_MATRIX.keys()),
        format_func=lambda x: ACTION_MATRIX[x]["label"],
    )

    export_df = customer_df[customer_df["action"] == export_action][
        ["customer_unique_id", "segment", "churn_probability", "monetary",
         "recency", "frequency", "avg_review_score", "action"]
    ].sort_values("churn_probability", ascending=False)

    st.dataframe(export_df.head(50), use_container_width=True, hide_index=True)
    st.caption(f"Total users in this group: {len(export_df):,}")

    # Download button
    csv_data = export_df.to_csv(index=False)
    st.download_button(
        label=f"⬇️ Download {ACTION_MATRIX[export_action]['label']} List ({len(export_df):,} users)",
        data=csv_data,
        file_name=f"retainiq_{export_action.lower()}_list.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # Full Export
    st.markdown("### 📦 Full Dataset Export")
    full_csv = customer_df.to_csv(index=False)
    st.download_button(
        label=f"⬇️ Download Complete Dataset ({len(customer_df):,} customers)",
        data=full_csv,
        file_name="retainiq_full_customer_data.csv",
        mime="text/csv",
    )

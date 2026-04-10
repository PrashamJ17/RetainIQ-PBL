"""
RetainIQ — Segmentation Engine
================================
Customer segmentation using RFM (Recency, Frequency, Monetary) analysis
combined with K-Means clustering.

This module:
1. Normalizes RFM features using StandardScaler.
2. Determines optimal cluster count using the Elbow Method + Silhouette Score.
3. Runs K-Means clustering to group customers into behavioral segments.
4. Maps cluster IDs to business-readable labels based on centroid analysis.
5. Provides Plotly visualization functions for the dashboard.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import SEGMENT_COLORS, N_CLUSTERS, RANDOM_STATE


# ──────────────────────────────────────────────
# 1. RFM NORMALIZATION
# ──────────────────────────────────────────────

def normalize_rfm(customer_df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Extract and normalize the R, F, M columns using StandardScaler.

    Args:
        customer_df: Customer-level DataFrame with 'recency', 'frequency', 'monetary'.

    Returns:
        Tuple of (normalized DataFrame with R/F/M columns, fitted scaler).
    """
    rfm_cols = ["recency", "frequency", "monetary"]
    rfm_data = customer_df[rfm_cols].copy()

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data)
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=rfm_cols, index=customer_df.index)

    return rfm_scaled_df, scaler


# ──────────────────────────────────────────────
# 2. OPTIMAL CLUSTER SELECTION
# ──────────────────────────────────────────────

def find_optimal_clusters(rfm_scaled: pd.DataFrame, k_range: range = range(2, 9)) -> dict:
    """
    Evaluate cluster quality using Inertia (Elbow) and Silhouette Score.

    Args:
        rfm_scaled: Normalized RFM DataFrame.
        k_range: Range of k values to test.

    Returns:
        Dict with 'inertias', 'silhouette_scores', and 'best_k'.
    """
    inertias = []
    silhouette_scores_list = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(rfm_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores_list.append(silhouette_score(rfm_scaled, labels))

    best_k = list(k_range)[np.argmax(silhouette_scores_list)]

    return {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouette_scores": silhouette_scores_list,
        "best_k": best_k,
    }


# ──────────────────────────────────────────────
# 3. K-MEANS CLUSTERING
# ──────────────────────────────────────────────

def run_kmeans(customer_df: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> pd.DataFrame:
    """
    Run K-Means clustering on normalized RFM features and assign
    business-readable segment labels.

    The labeling logic:
        - Sort clusters by their mean Monetary value (descending).
        - Highest monetary cluster → "Champions"
        - Second highest → "Loyalists"
        - Third → "At-Risk"
        - Lowest → "Hibernating"

    Args:
        customer_df: Customer-level DataFrame with RFM features.
        n_clusters: Number of clusters.

    Returns:
        pd.DataFrame: Original DataFrame with added 'cluster' and 'segment' columns.
    """
    customer_df = customer_df.copy()

    # Normalize
    rfm_scaled, scaler = normalize_rfm(customer_df)

    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    customer_df["cluster"] = kmeans.fit_predict(rfm_scaled)

    # Compute cluster centroids in original scale for labeling
    cluster_summary = (
        customer_df.groupby("cluster")
        .agg(
            mean_recency=("recency", "mean"),
            mean_frequency=("frequency", "mean"),
            mean_monetary=("monetary", "mean"),
            count=("customer_unique_id", "count"),
        )
        .reset_index()
    )

    # Score each cluster: high monetary + high frequency + low recency = best
    cluster_summary["score"] = (
        cluster_summary["mean_monetary"].rank(ascending=True)
        + cluster_summary["mean_frequency"].rank(ascending=True)
        + cluster_summary["mean_recency"].rank(ascending=False)  # Lower recency = better
    )

    # Sort by composite score (highest = Champions)
    cluster_summary = cluster_summary.sort_values("score", ascending=False).reset_index(drop=True)

    # Build label map
    segment_names = ["Champions", "Loyalists", "At-Risk", "Hibernating"]
    # If more clusters than labels, extend with numbered labels
    while len(segment_names) < n_clusters:
        segment_names.append(f"Segment-{len(segment_names)}")

    label_map = {}
    for i, row in cluster_summary.iterrows():
        label_map[row["cluster"]] = segment_names[i]

    customer_df["segment"] = customer_df["cluster"].map(label_map)

    return customer_df


# ──────────────────────────────────────────────
# 4. SEGMENT SUMMARY STATISTICS
# ──────────────────────────────────────────────

def get_segment_summary(customer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for each segment.

    Returns:
        pd.DataFrame with one row per segment: count, mean R/F/M,
        avg review score, and churn rate.
    """
    summary = (
        customer_df.groupby("segment")
        .agg(
            customer_count=("customer_unique_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            avg_review_score=("avg_review_score", "mean"),
            churn_rate=("churned", "mean"),
        )
        .reset_index()
    )

    summary["churn_rate"] = (summary["churn_rate"] * 100).round(1)
    summary[["avg_recency", "avg_frequency", "avg_monetary", "avg_review_score"]] = (
        summary[["avg_recency", "avg_frequency", "avg_monetary", "avg_review_score"]].round(2)
    )

    # Sort by avg_monetary descending (Champions first)
    summary = summary.sort_values("avg_monetary", ascending=False).reset_index(drop=True)

    return summary


# ──────────────────────────────────────────────
# 5. VISUALIZATION FUNCTIONS
# ──────────────────────────────────────────────

def plot_segments_3d(customer_df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive 3D scatter plot of customers colored by segment.
    Axes: Recency, Frequency, Monetary.
    """
    fig = px.scatter_3d(
        customer_df,
        x="recency",
        y="frequency",
        z="monetary",
        color="segment",
        color_discrete_map=SEGMENT_COLORS,
        opacity=0.6,
        hover_data=["customer_unique_id", "avg_review_score"],
        title="Customer Segments — 3D RFM Distribution",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="white"),
        scene=dict(
            xaxis_title="Recency (days)",
            yaxis_title="Frequency (orders)",
            zaxis_title="Monetary (R$)",
        ),
        legend=dict(
            title="Segment",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(108,99,255,0.3)",
            borderwidth=1,
        ),
        height=600,
    )

    return fig


def plot_segment_distribution(customer_df: pd.DataFrame) -> go.Figure:
    """
    Create a horizontal bar chart showing the number of customers per segment.
    """
    segment_counts = (
        customer_df["segment"]
        .value_counts()
        .reset_index()
    )
    segment_counts.columns = ["segment", "count"]

    colors = [SEGMENT_COLORS.get(seg, "#888") for seg in segment_counts["segment"]]

    fig = go.Figure(
        go.Bar(
            x=segment_counts["count"],
            y=segment_counts["segment"],
            orientation="h",
            marker_color=colors,
            text=segment_counts["count"].apply(lambda x: f"{x:,}"),
            textposition="outside",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Customers per Segment",
        xaxis_title="Number of Customers",
        yaxis_title="",
        font=dict(family="Inter", color="white"),
        height=350,
        margin=dict(l=120),
    )

    return fig


def plot_rfm_boxplots(customer_df: pd.DataFrame) -> go.Figure:
    """
    Create box plots comparing R, F, and M distributions across segments.
    Useful for understanding segment characteristics at a glance.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Recency (days)", "Frequency (orders)", "Monetary (R$)"),
        horizontal_spacing=0.08,
    )

    for segment in ["Champions", "Loyalists", "At-Risk", "Hibernating"]:
        seg_data = customer_df[customer_df["segment"] == segment]
        color = SEGMENT_COLORS.get(segment, "#888")

        fig.add_trace(
            go.Box(y=seg_data["recency"], name=segment, marker_color=color, showlegend=True),
            row=1, col=1,
        )
        fig.add_trace(
            go.Box(y=seg_data["frequency"], name=segment, marker_color=color, showlegend=False),
            row=1, col=2,
        )
        fig.add_trace(
            go.Box(y=seg_data["monetary"], name=segment, marker_color=color, showlegend=False),
            row=1, col=3,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="RFM Distribution by Segment",
        font=dict(family="Inter", color="white"),
        height=450,
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
    )

    return fig


def plot_elbow_silhouette(cluster_results: dict) -> go.Figure:
    """
    Plot the Elbow Method (inertia) and Silhouette Score side by side
    to justify the chosen number of clusters.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Elbow Method (Inertia)", "Silhouette Score"),
        horizontal_spacing=0.12,
    )

    k_range = cluster_results["k_range"]

    # Elbow
    fig.add_trace(
        go.Scatter(
            x=k_range,
            y=cluster_results["inertias"],
            mode="lines+markers",
            line=dict(color="#6C63FF", width=2),
            marker=dict(size=8),
            name="Inertia",
        ),
        row=1, col=1,
    )

    # Silhouette
    fig.add_trace(
        go.Scatter(
            x=k_range,
            y=cluster_results["silhouette_scores"],
            mode="lines+markers",
            line=dict(color="#00C9A7", width=2),
            marker=dict(size=8),
            name="Silhouette",
        ),
        row=1, col=2,
    )

    # Mark best k
    best_k = cluster_results["best_k"]
    best_idx = k_range.index(best_k)
    fig.add_trace(
        go.Scatter(
            x=[best_k],
            y=[cluster_results["silhouette_scores"][best_idx]],
            mode="markers",
            marker=dict(size=14, color="#FF6B6B", symbol="star"),
            name=f"Best k={best_k}",
            showlegend=True,
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=1)
    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=2)
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Cluster Validation: Elbow Method & Silhouette Analysis",
        font=dict(family="Inter", color="white"),
        height=400,
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
    )

    return fig


# ──────────────────────────────────────────────
# 6. FULL SEGMENTATION PIPELINE
# ──────────────────────────────────────────────

def run_segmentation_pipeline(customer_df: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> dict:
    """
    Execute the full segmentation pipeline.

    Args:
        customer_df: Customer-level feature matrix (from data_engine).
        n_clusters: Number of K-Means clusters.

    Returns:
        Dict containing:
            - 'customer_df': DataFrame with segment assignments
            - 'segment_summary': Summary stats per segment
            - 'cluster_validation': Elbow + silhouette results
    """
    print("🔍 Finding optimal clusters...")
    rfm_scaled, _ = normalize_rfm(customer_df)
    cluster_validation = find_optimal_clusters(rfm_scaled)
    print(f"   → Best k by silhouette: {cluster_validation['best_k']}")

    print(f"🎯 Running K-Means with k={n_clusters}...")
    customer_df = run_kmeans(customer_df, n_clusters=n_clusters)

    print("📊 Computing segment summaries...")
    segment_summary = get_segment_summary(customer_df)
    print(segment_summary.to_string(index=False))

    print("✅ Segmentation complete!")

    return {
        "customer_df": customer_df,
        "segment_summary": segment_summary,
        "cluster_validation": cluster_validation,
    }


# ──────────────────────────────────────────────
# CLI entry point for standalone testing
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from data_engine import run_data_pipeline

    customer_data = run_data_pipeline()
    results = run_segmentation_pipeline(customer_data)

    print("\n📈 Segment Distribution:")
    print(results["customer_df"]["segment"].value_counts().to_string())

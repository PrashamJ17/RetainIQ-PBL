"use client";

import { MetricCard, SectionTitle } from "./ui";

export default function OverviewTab({ data }) {
  if (!data) return null;

  const { total_customers, total_revenue, avg_order_value, churn_rate, model_metrics, clv_stats, clv_metrics } = data;

  return (
    <div className="fade-in">
      <SectionTitle subtitle="Executive summary of business health and model performance">
        📊 Business Overview
      </SectionTitle>

      {/* KPI Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <MetricCard
          title="Total Customers" value={total_customers?.toLocaleString()} icon="👥"
          subtitle="Unique tracked users"
        />
        <MetricCard
          title="Total Revenue" value={`R$${(total_revenue / 1e6).toFixed(2)}M`} icon="💰"
          color="var(--accent-secondary)" subtitle="Lifetime transaction value"
        />
        <MetricCard
          title="Avg Order Value" value={`R$${avg_order_value?.toFixed(2)}`} icon="🛒"
          subtitle="Per checkout average"
        />
        <MetricCard
          title="Churn Rate" value={`${churn_rate}%`} icon="⚠️"
          color="var(--accent-danger)" subtitle="90-day inactivity threshold"
        />
      </div>

      {/* Model Metrics */}
      <SectionTitle subtitle="XGBoost churn classifier performance on held-out test set">
        🤖 Model Performance
      </SectionTitle>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
        {model_metrics && Object.entries(model_metrics).map(([key, val]) => (
          <div key={key} className="glass-card p-4 text-center">
            <p className="text-xs uppercase tracking-wider mb-1" style={{ color: "var(--text-secondary)" }}>
              {key.replace("_", " ")}
            </p>
            <p className="text-2xl font-bold" style={{
              color: val >= 0.99 ? "var(--accent-secondary)" : "var(--accent-primary)"
            }}>
              {typeof val === "number" ? (val * 100).toFixed(2) + "%" : val}
            </p>
          </div>
        ))}
      </div>

      {/* CLV Stats */}
      {clv_stats && (
        <>
          <SectionTitle subtitle="XGBoost CLV regressor — predicted future customer value">
            💎 Customer Lifetime Value
          </SectionTitle>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <MetricCard
              title="Mean CLV" value={`R$${clv_stats.mean_clv?.toFixed(2)}`} icon="📈"
              color="var(--accent-primary)" subtitle="Average predicted value"
            />
            <MetricCard
              title="Median CLV" value={`R$${clv_stats.median_clv?.toFixed(2)}`} icon="📊"
              subtitle="50th percentile value"
            />
            <MetricCard
              title="Total Portfolio" value={`R$${(clv_stats.total_predicted_value / 1e6).toFixed(2)}M`}
              icon="🏦" color="var(--accent-secondary)" subtitle="Sum of all predicted CLV"
            />
          </div>
          {clv_metrics && (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="glass-card p-4 text-center">
                <p className="text-xs uppercase tracking-wider mb-1" style={{ color: "var(--text-secondary)" }}>MAE</p>
                <p className="text-xl font-bold">R${clv_metrics.mae}</p>
              </div>
              <div className="glass-card p-4 text-center">
                <p className="text-xs uppercase tracking-wider mb-1" style={{ color: "var(--text-secondary)" }}>RMSE</p>
                <p className="text-xl font-bold">R${clv_metrics.rmse}</p>
              </div>
              <div className="glass-card p-4 text-center">
                <p className="text-xs uppercase tracking-wider mb-1" style={{ color: "var(--text-secondary)" }}>R² Score</p>
                <p className="text-xl font-bold" style={{ color: "var(--accent-secondary)" }}>
                  {(clv_metrics.r2_score * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

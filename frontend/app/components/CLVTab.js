"use client";

import { SectionTitle, MetricCard, Badge } from "./ui";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";

const TIER_COLORS = {
  Platinum: "#6C63FF",
  Gold: "#FFD700",
  Silver: "#C0C0C0",
  Bronze: "#CD7F32",
};

export default function CLVTab({ data }) {
  if (!data) return null;

  const { clv_metrics, clv_thresholds, tier_distribution, top_customers, total_predicted_value } = data;

  const pieData = Object.entries(tier_distribution || {}).map(([name, value]) => ({
    name, value
  }));

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload?.length) {
      const d = payload[0].payload;
      return (
        <div className="glass-card p-3 text-sm" style={{ border: "1px solid var(--glass-border)" }}>
          <p className="font-bold" style={{ color: TIER_COLORS[d.name] }}>{d.name}</p>
          <p>{d.value.toLocaleString()} customers</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="fade-in">
      <SectionTitle subtitle="XGBRegressor predicting continuous monetary value per customer">
        💎 Customer Lifetime Value
      </SectionTitle>

      {/* CLV Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <MetricCard title="Total Portfolio" value={`R$${(total_predicted_value / 1e6).toFixed(2)}M`}
                    icon="🏦" color="var(--accent-secondary)" />
        <MetricCard title="R² Score" value={`${(clv_metrics?.r2_score * 100).toFixed(1)}%`}
                    icon="📊" subtitle="Variance explained" />
        <MetricCard title="MAE" value={`R$${clv_metrics?.mae}`}
                    icon="📏" subtitle="Avg prediction error" />
        <MetricCard title="Mean CLV" value={`R$${clv_metrics?.mean_predicted_clv}`}
                    icon="💰" color="var(--accent-primary)" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Tier Distribution Pie */}
        <div className="glass-card p-6">
          <h3 className="text-sm font-semibold uppercase tracking-wider mb-4"
              style={{ color: "var(--text-secondary)" }}>
            CLV Tier Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                   outerRadius={100} innerRadius={55} paddingAngle={3} stroke="none"
                   label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                {pieData.map((entry) => (
                  <Cell key={entry.name} fill={TIER_COLORS[entry.name] || "#888"} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Tier Thresholds */}
        <div className="glass-card p-6">
          <h3 className="text-sm font-semibold uppercase tracking-wider mb-4"
              style={{ color: "var(--text-secondary)" }}>
            Tier Thresholds (Percentile-Based)
          </h3>
          <div className="space-y-4 mt-6">
            {[
              { tier: "Platinum", pct: "Top 5%", threshold: clv_thresholds?.platinum },
              { tier: "Gold", pct: "Top 20%", threshold: clv_thresholds?.gold },
              { tier: "Silver", pct: "Top 50%", threshold: clv_thresholds?.silver },
              { tier: "Bronze", pct: "Bottom 50%", threshold: 0 },
            ].map(t => (
              <div key={t.tier} className="flex items-center justify-between p-3 rounded-lg"
                   style={{ background: TIER_COLORS[t.tier] + "10", borderLeft: `3px solid ${TIER_COLORS[t.tier]}` }}>
                <div>
                  <Badge type={t.tier.toLowerCase()}>{t.tier}</Badge>
                  <span className="ml-3 text-sm" style={{ color: "var(--text-secondary)" }}>{t.pct}</span>
                </div>
                <span className="font-bold" style={{ color: TIER_COLORS[t.tier] }}>
                  ≥ R${t.threshold?.toFixed(2) || "0.00"}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Top Customers Table */}
      <div className="glass-card p-6">
        <h3 className="text-sm font-semibold uppercase tracking-wider mb-4"
            style={{ color: "var(--text-secondary)" }}>
          Top 20 Highest-Value Customers
        </h3>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Customer ID</th>
                <th>Predicted CLV</th>
                <th>Tier</th>
                <th>Segment</th>
                <th>Churn Risk</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {top_customers?.map((c, i) => (
                <tr key={i}>
                  <td className="font-mono text-sm">{i + 1}</td>
                  <td className="font-mono text-sm">{c.customer_unique_id?.substring(0, 16)}...</td>
                  <td className="font-bold" style={{ color: "var(--accent-secondary)" }}>
                    R${c.predicted_clv?.toFixed(2)}
                  </td>
                  <td><Badge type={c.clv_tier?.toLowerCase()}>{c.clv_tier}</Badge></td>
                  <td>{c.segment}</td>
                  <td style={{ color: c.churn_probability > 0.5 ? "var(--accent-danger)" : "var(--accent-secondary)" }}>
                    {(c.churn_probability * 100).toFixed(1)}%
                  </td>
                  <td>
                    <Badge type={c.action === "SAVE_NOW" ? "save" : c.action === "NURTURE" ? "nurture" : "monitor"}>
                      {c.action}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

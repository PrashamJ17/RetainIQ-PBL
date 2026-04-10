"use client";

import { SectionTitle, Badge } from "./ui";
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";

const SEGMENT_COLORS = {
  Champions: "#6C63FF",
  Loyalists: "#00C9A7",
  "At-Risk": "#FF6B6B",
  Hibernating: "#A0AEC0",
};

export default function SegmentsTab({ data }) {
  if (!data) return null;

  const { segment_summary, segment_distribution } = data;

  const pieData = Object.entries(segment_distribution).map(([name, value]) => ({
    name, value
  }));

  const barData = segment_summary?.map(s => ({
    segment: s.segment,
    customers: s.customer_count,
    avg_monetary: Math.round(s.avg_monetary),
    churn_rate: s.churn_rate,
  })) || [];

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload?.length) {
      const d = payload[0].payload;
      return (
        <div className="glass-card p-3 text-sm" style={{ border: "1px solid var(--glass-border)" }}>
          <p className="font-bold">{d.name || d.segment}</p>
          {d.value && <p>Count: {d.value.toLocaleString()}</p>}
          {d.avg_monetary && <p>Avg Spend: R${d.avg_monetary}</p>}
          {d.churn_rate && <p>Churn Rate: {d.churn_rate}%</p>}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="fade-in">
      <SectionTitle subtitle="K-Means clustering on normalized RFM features — 4 behavioral segments">
        🎯 Customer Segments
      </SectionTitle>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Pie Chart */}
        <div className="glass-card p-6">
          <h3 className="text-sm font-semibold uppercase tracking-wider mb-4" style={{ color: "var(--text-secondary)" }}>
            Segment Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                   outerRadius={100} innerRadius={50} paddingAngle={3}
                   stroke="none" label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                {pieData.map((entry) => (
                  <Cell key={entry.name} fill={SEGMENT_COLORS[entry.name] || "#888"} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Bar Chart: Avg Monetary by Segment */}
        <div className="glass-card p-6">
          <h3 className="text-sm font-semibold uppercase tracking-wider mb-4" style={{ color: "var(--text-secondary)" }}>
            Average Spend by Segment (R$)
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData} layout="vertical" margin={{ left: 80 }}>
              <XAxis type="number" stroke="var(--text-secondary)" fontSize={12} />
              <YAxis type="category" dataKey="segment" stroke="var(--text-secondary)" fontSize={12} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="avg_monetary" radius={[0, 6, 6, 0]}>
                {barData.map((entry) => (
                  <Cell key={entry.segment} fill={SEGMENT_COLORS[entry.segment] || "#888"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Segment Summary Table */}
      <div className="glass-card p-6">
        <h3 className="text-sm font-semibold uppercase tracking-wider mb-4" style={{ color: "var(--text-secondary)" }}>
          Segment Details
        </h3>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Segment</th>
                <th>Customers</th>
                <th>Avg Recency</th>
                <th>Avg Frequency</th>
                <th>Avg Monetary</th>
                <th>Avg Review</th>
                <th>Churn Rate</th>
              </tr>
            </thead>
            <tbody>
              {segment_summary?.map((s) => (
                <tr key={s.segment}>
                  <td>
                    <span className="font-semibold" style={{ color: SEGMENT_COLORS[s.segment] }}>
                      {s.segment}
                    </span>
                  </td>
                  <td>{s.customer_count.toLocaleString()}</td>
                  <td>{s.avg_recency} days</td>
                  <td>{s.avg_frequency}</td>
                  <td>R${s.avg_monetary.toLocaleString()}</td>
                  <td>⭐ {s.avg_review_score}</td>
                  <td>
                    <span style={{ color: s.churn_rate > 80 ? "var(--accent-danger)" : "var(--accent-secondary)" }}>
                      {s.churn_rate}%
                    </span>
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

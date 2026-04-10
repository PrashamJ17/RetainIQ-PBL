"use client";

import { SectionTitle, MetricCard } from "./ui";
import {
  ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, Legend,
} from "recharts";

const ACTION_CONFIG = {
  SAVE_NOW: { color: "#FF6B6B", label: "🚨 Save Now", desc: "High Value + High Risk" },
  LET_GO: { color: "#A0AEC0", label: "🚫 Let Go", desc: "Low Value + High Risk" },
  NURTURE: { color: "#6C63FF", label: "👑 Nurture", desc: "High Value + Low Risk" },
  MONITOR: { color: "#00C9A7", label: "👀 Monitor", desc: "Low Value + Low Risk" },
};

export default function ActionTab({ data }) {
  if (!data) return null;

  const { action_distribution, scatter_data } = data;

  // Prepare scatter data — sample for performance
  const scatterByAction = {};
  scatter_data?.forEach(d => {
    if (!scatterByAction[d.action]) scatterByAction[d.action] = [];
    // Lower sample size strictly to 120 per category so distinct dots are visible rather than a solid line
    if (scatterByAction[d.action].length < 120) {
      scatterByAction[d.action].push({
        x: Math.min(d.monetary, 2000), // Cap at 2000 to prevent extreme compression
        actual_x: d.monetary,
        y: d.churn_probability,
        id: d.customer_unique_id?.substring(0, 8),
        segment: d.segment,
      });
    }
  });

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload?.length) {
      const d = payload[0]?.payload;
      return (
        <div className="glass-card p-3 text-sm" style={{ border: "1px solid var(--glass-border)" }}>
          <p className="font-mono">{d?.id}...</p>
          <p>Monetary: R${(d?.actual_x || d?.x)?.toFixed(2)}</p>
          <p>Churn Risk: {(d?.y * 100).toFixed(1)}%</p>
          <p>Segment: {d?.segment}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="fade-in">
      <SectionTitle subtitle="Value vs. Risk — translating AI predictions into direct financial actions">
        🚀 Action Matrix
      </SectionTitle>

      {/* Action Distribution Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {Object.entries(ACTION_CONFIG).map(([key, cfg]) => (
          <div key={key} className="metric-card" style={{ borderColor: cfg.color + "30" }}>
            <p className="text-xs font-semibold uppercase tracking-wider mb-1"
               style={{ color: "var(--text-secondary)" }}>
              {cfg.label}
            </p>
            <p className="text-3xl font-bold" style={{ color: cfg.color }}>
              {(action_distribution?.[key] || 0).toLocaleString()}
            </p>
            <p className="text-xs mt-1" style={{ color: "var(--text-secondary)" }}>{cfg.desc}</p>
          </div>
        ))}
      </div>

      {/* Scatter Plot */}
      <div className="glass-card p-6 mb-8">
        <h3 className="text-sm font-semibold uppercase tracking-wider mb-4"
            style={{ color: "var(--text-secondary)" }}>
          Customer Value vs. Churn Risk Quadrant
        </h3>
        <ResponsiveContainer width="100%" height={500}>
          <ScatterChart margin={{ top: 40, right: 30, bottom: 40, left: 20 }}>
            <XAxis type="number" dataKey="x" name="Monetary" stroke="var(--text-secondary)"
                   fontSize={12} label={{ value: "Monetary Value (R$)", position: "bottom", offset: 15, fill: "var(--text-secondary)" }}
                   domain={[0, 2000]} />
            <YAxis type="number" dataKey="y" name="Churn Risk" stroke="var(--text-secondary)"
                   fontSize={12} label={{ value: "Churn Probability", angle: -90, position: "insideLeft", offset: -5, fill: "var(--text-secondary)" }}
                   domain={[-0.05, 1.05]} />
            <Tooltip content={<CustomTooltip />} />
            <Legend verticalAlign="top" height={36} wrapperStyle={{ top: -10 }} />
            {Object.entries(scatterByAction).map(([action, points]) => (
              <Scatter key={action} name={ACTION_CONFIG[action]?.label || action}
                       data={points} fill={ACTION_CONFIG[action]?.color || "#888"} 
                       opacity={0.85} stroke="rgba(255, 255, 255, 0.2)" strokeWidth={1} />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Quick Guide */}
      <div className="glass-card p-6">
        <h3 className="text-sm font-semibold uppercase tracking-wider mb-4"
            style={{ color: "var(--text-secondary)" }}>
          Strategy Guide
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(ACTION_CONFIG).map(([key, cfg]) => (
            <div key={key} className="p-4 rounded-xl" style={{ background: cfg.color + "10", borderLeft: `3px solid ${cfg.color}` }}>
              <p className="font-bold mb-1" style={{ color: cfg.color }}>{cfg.label}</p>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                {key === "SAVE_NOW" && "Send 20% discount immediately. These are your best customers who are about to leave."}
                {key === "NURTURE" && "Give early access to VIP features. Don't waste discounts — they're already happy."}
                {key === "MONITOR" && "No action needed. Keep an eye on their behavior."}
                {key === "LET_GO" && "Do not spend budget. These customers generate too little value to justify intervention."}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

"use client";

import { SectionTitle, MetricCard, Badge } from "./ui";

export default function ActivationsTab({ data }) {
  if (!data) return null;

  const { total_evaluated, total_activated, actions_fired, estimated_discount_budget, recent_events } = data;

  const conversionRate = total_evaluated > 0 ? ((total_activated / total_evaluated) * 100).toFixed(1) : 0;

  return (
    <div className="fade-in">
      <SectionTitle subtitle="Simulated automated marketing actions — Klaviyo/Twilio webhook triggers">
        ⚡ Activation Engine
      </SectionTitle>

      {/* Stats Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <MetricCard title="Customers Evaluated" value={total_evaluated?.toLocaleString()} icon="👥" />
        <MetricCard title="Actions Fired" value={total_activated?.toLocaleString()} icon="🚀"
                    color="var(--accent-primary)" subtitle={`${conversionRate}% activation rate`} />
        <MetricCard title="SAVE_NOW Emails" value={actions_fired?.SAVE_NOW?.toLocaleString() || "0"}
                    icon="🚨" color="var(--accent-danger)" />
        <MetricCard title="Est. Discount Budget" value={`R$${(estimated_discount_budget / 1000).toFixed(1)}K`}
                    icon="💸" color="var(--accent-secondary)"
                    subtitle="20% discount cost estimate" />
      </div>

      {/* Action breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
        <div className="glass-card p-6">
          <h3 className="text-sm font-semibold uppercase tracking-wider mb-4"
              style={{ color: "var(--text-secondary)" }}>
            Actions Breakdown
          </h3>
          <div className="space-y-3">
            {Object.entries(actions_fired || {}).map(([action, count]) => {
              const pct = total_activated > 0 ? (count / total_activated * 100).toFixed(1) : 0;
              const color = action === "SAVE_NOW" ? "var(--accent-danger)" : "var(--accent-primary)";
              return (
                <div key={action}>
                  <div className="flex justify-between text-sm mb-1">
                    <span>{action === "SAVE_NOW" ? "🚨 Save Now" : "👑 Nurture"}</span>
                    <span style={{ color }}>{count.toLocaleString()} ({pct}%)</span>
                  </div>
                  <div className="w-full h-2 rounded-full" style={{ background: "rgba(108,99,255,0.1)" }}>
                    <div className="h-full rounded-full transition-all duration-500"
                         style={{ width: `${pct}%`, background: color }}></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="glass-card p-6">
          <h3 className="text-sm font-semibold uppercase tracking-wider mb-4"
              style={{ color: "var(--text-secondary)" }}>
            ROI Analysis
          </h3>
          <div className="space-y-4">
            <div className="p-4 rounded-xl" style={{ background: "rgba(108,99,255,0.05)" }}>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                If even <span className="font-bold text-white">5%</span> of SAVE_NOW customers are retained:
              </p>
              <p className="text-2xl font-bold mt-2" style={{ color: "var(--accent-secondary)" }}>
                R${((actions_fired?.SAVE_NOW || 0) * 0.05 * 140).toLocaleString()} potential revenue saved
              </p>
              <p className="text-xs mt-1" style={{ color: "var(--text-secondary)" }}>
                Based on avg order value of R$140
              </p>
            </div>
            <div className="p-4 rounded-xl" style={{ background: "rgba(255,107,107,0.05)" }}>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Marketing spend (discount cost):
              </p>
              <p className="text-xl font-bold mt-1" style={{ color: "var(--accent-danger)" }}>
                R${estimated_discount_budget?.toLocaleString()}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Events Feed */}
      <div className="glass-card p-6">
        <h3 className="text-sm font-semibold uppercase tracking-wider mb-4"
            style={{ color: "var(--text-secondary)" }}>
          📧 Recent Activation Events
        </h3>
        <div className="max-h-96 overflow-y-auto space-y-2">
          {recent_events?.slice(0, 30).map((evt, i) => (
            <div key={i} className="flex items-center gap-4 p-3 rounded-lg hover:bg-[rgba(108,99,255,0.05)] transition-colors">
              <span className="text-lg">{evt.action === "SAVE_NOW" ? "🚨" : "👑"}</span>
              <div className="flex-1 min-w-0">
                <p className="font-mono text-sm truncate">{evt.customer_id}</p>
                <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
                  {evt.subject}
                </p>
              </div>
              <div className="text-right shrink-0">
                <p className="text-sm font-bold" style={{ color: "var(--accent-secondary)" }}>
                  R${evt.predicted_clv?.toFixed(0)}
                </p>
                <p className="text-xs" style={{ color: "var(--accent-danger)" }}>
                  {(evt.churn_probability * 100).toFixed(0)}% risk
                </p>
              </div>
              <Badge type={evt.action === "SAVE_NOW" ? "save" : "nurture"}>
                {evt.status}
              </Badge>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

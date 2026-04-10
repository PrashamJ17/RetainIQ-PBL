"use client";

export function MetricCard({ title, value, subtitle, icon, color = "var(--accent-primary)" }) {
  return (
    <div className="metric-card fade-in">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest mb-2"
             style={{ color: "var(--text-secondary)" }}>
            {title}
          </p>
          <p className="text-3xl font-bold" style={{ color }}>{value}</p>
          {subtitle && (
            <p className="text-sm mt-1" style={{ color: "var(--text-secondary)" }}>{subtitle}</p>
          )}
        </div>
        {icon && <span className="text-3xl opacity-50">{icon}</span>}
      </div>
    </div>
  );
}

export function SectionTitle({ children, subtitle }) {
  return (
    <div className="mb-6">
      <h2 className="text-xl font-bold gradient-text">{children}</h2>
      {subtitle && (
        <p className="text-sm mt-1" style={{ color: "var(--text-secondary)" }}>{subtitle}</p>
      )}
    </div>
  );
}

export function Badge({ type, children }) {
  const classMap = {
    save: "badge-save", nurture: "badge-nurture",
    monitor: "badge-monitor", letgo: "badge-letgo",
    platinum: "badge-platinum", gold: "badge-gold",
    silver: "badge-silver", bronze: "badge-bronze",
  };
  return <span className={`badge ${classMap[type] || "badge-monitor"}`}>{children}</span>;
}

export function LoadingState({ message = "Loading data..." }) {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <div className="spinner mb-4"></div>
      <p className="text-sm" style={{ color: "var(--text-secondary)" }}>{message}</p>
    </div>
  );
}

export function EmptyState({ title, message, action }) {
  return (
    <div className="glass-card p-12 text-center fade-in">
      <p className="text-6xl mb-4">🧠</p>
      <h3 className="text-xl font-bold mb-2">{title}</h3>
      <p className="text-sm mb-6" style={{ color: "var(--text-secondary)" }}>{message}</p>
      {action}
    </div>
  );
}

"use client";

import { useState } from "react";
import { exportFilteredCustomers } from "../lib/api";
import { SectionTitle, MetricCard } from "./ui";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis, Cell,
} from "recharts";

export default function ChurnTab({ data }) {
  if (!data) return null;

  const { metrics, feature_importance, top_high_risk_users, churn_distribution } = data;

  const [exporting, setExporting] = useState(false);
  const [filterStatus, setFilterStatus] = useState("CHURNED");
  const [filterSegment, setFilterSegment] = useState("ALL");

  const handleExport = async () => {
    setExporting(true);
    try {
      const res = await exportFilteredCustomers(filterStatus, filterSegment);
      if (!res || !res.customers || res.customers.length === 0) {
        alert("No customers found for these filters.");
        setExporting(false);
        return;
      }
      
      const exportData = res.customers;
      const headers = Object.keys(exportData[0]);
      const csvRows = [];
      
      // Header row
      csvRows.push(headers.join(","));
      
      // Data rows
      for (const row of exportData) {
        const values = headers.map(h => {
          const val = row[h] !== null && row[h] !== undefined ? row[h] : "";
          return `"${String(val).replace(/"/g, '""')}"`;
        });
        csvRows.push(values.join(","));
      }
      
      const csvString = csvRows.join("\n");
      const blob = new Blob([csvString], { type: "text/csv;charset=utf-8;" });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.setAttribute("hidden", "");
      a.setAttribute("href", url);
      a.setAttribute("download", `customers_${filterStatus.toLowerCase()}_${filterSegment.toLowerCase()}.csv`);
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } catch (e) {
      console.error(e);
      alert("Failed to export data.");
    }
    setExporting(false);
  };

  const fiData = feature_importance?.slice(0, 10).map(f => ({
    feature: f.feature.replace(/_/g, " "),
    importance: parseFloat(f.importance.toFixed(3)),
  })).reverse() || [];

  // Confusion matrix
  const cm = metrics?.confusion_matrix || [[0, 0], [0, 0]];

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload?.length) {
      return (
        <div className="glass-card p-3 text-sm" style={{ border: "1px solid var(--glass-border)" }}>
          <p>{payload[0]?.payload?.feature || ""}</p>
          <p className="font-bold" style={{ color: "var(--accent-primary)" }}>
            {payload[0]?.value?.toFixed(4)}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="fade-in">
      <SectionTitle subtitle="XGBoost predictions with SHAP explainability">
        ⚠️ Churn Risk Analysis
      </SectionTitle>

      {/* CSV Export Panel */}
      <div className="glass-card p-6 mb-8 flex flex-col md:flex-row items-center justify-between gap-4" style={{ background: "linear-gradient(145deg, rgba(108,99,255,0.05), rgba(0,201,167,0.05))" }}>
        <div>
          <h3 className="text-lg font-bold gradient-text mb-1">Customer Data Export</h3>
          <p className="text-xs" style={{ color: "var(--text-secondary)" }}>Download raw customer data with ML predictions for offline analysis</p>
        </div>
        
        <div className="flex flex-col md:flex-row items-center gap-4">
          <select 
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-3 py-2 rounded-lg text-sm bg-transparent border focus:outline-none"
            style={{ borderColor: "var(--border-subtle)", color: "var(--text-primary)" }}
          >
            <option value="ALL" style={{background: "#1a1a2e"}}>All Statuses</option>
            <option value="CHURNED" style={{background: "#1a1a2e"}}>Predicted Churned</option>
            <option value="RETAINED" style={{background: "#1a1a2e"}}>Predicted Retained</option>
          </select>

          <select 
            value={filterSegment}
            onChange={(e) => setFilterSegment(e.target.value)}
            className="px-3 py-2 rounded-lg text-sm bg-transparent border focus:outline-none"
            style={{ borderColor: "var(--border-subtle)", color: "var(--text-primary)" }}
          >
            <option value="ALL" style={{background: "#1a1a2e"}}>All Segments</option>
            <option value="Champions" style={{background: "#1a1a2e"}}>Champions</option>
            <option value="Loyal" style={{background: "#1a1a2e"}}>Loyal</option>
            <option value="At-Risk" style={{background: "#1a1a2e"}}>At-Risk</option>
            <option value="Hibernating" style={{background: "#1a1a2e"}}>Hibernating</option>
          </select>
          
          <button 
            onClick={handleExport}
            disabled={exporting}
            className={`px-4 py-2 rounded-xl text-sm font-bold text-white transition-all ${exporting ? "opacity-50 cursor-not-allowed" : "hover:scale-105"}`}
            style={{ background: "var(--accent-primary)", boxShadow: "0 4px 14px 0 rgba(108, 99, 255, 0.39)" }}
          >
            {exporting ? "⏳ Exporting..." : "📥 Download CSV"}
          </button>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
        <MetricCard title="Accuracy" value={`${(metrics.accuracy * 100).toFixed(2)}%`} icon="🎯" />
        <MetricCard title="F1 Score" value={`${(metrics.f1_score * 100).toFixed(2)}%`} icon="⚖️" />
        <MetricCard title="Precision" value={`${(metrics.precision * 100).toFixed(1)}%`} icon="🔬" />
        <MetricCard title="Recall" value={`${(metrics.recall * 100).toFixed(2)}%`} icon="🔍" />
        <MetricCard title="ROC-AUC" value={metrics.roc_auc.toFixed(4)} icon="📈"
                    color="var(--accent-secondary)" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Feature Importance */}
        <div className="glass-card p-6">
          <h3 className="text-sm font-semibold uppercase tracking-wider mb-4"
              style={{ color: "var(--text-secondary)" }}>
            Global Feature Importance (SHAP)
          </h3>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={fiData} layout="vertical" margin={{ left: 120 }}>
              <XAxis type="number" stroke="var(--text-secondary)" fontSize={12} />
              <YAxis type="category" dataKey="feature" stroke="var(--text-secondary)" fontSize={11} width={120} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="importance" fill="var(--accent-primary)" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Confusion Matrix */}
        <div className="glass-card p-6">
          <h3 className="text-sm font-semibold uppercase tracking-wider mb-4"
              style={{ color: "var(--text-secondary)" }}>
            Confusion Matrix
          </h3>
          <div className="flex items-center justify-center h-[350px]">
            <div className="text-center">
              <div className="grid grid-cols-2 gap-2 max-w-xs mx-auto">
                <div className="p-6 rounded-xl" style={{ background: "rgba(0, 201, 167, 0.15)" }}>
                  <p className="text-xs mb-1" style={{ color: "var(--text-secondary)" }}>True Negative</p>
                  <p className="text-3xl font-bold" style={{ color: "var(--accent-secondary)" }}>
                    {cm[0][0]?.toLocaleString()}
                  </p>
                </div>
                <div className="p-6 rounded-xl" style={{ background: "rgba(255, 107, 107, 0.1)" }}>
                  <p className="text-xs mb-1" style={{ color: "var(--text-secondary)" }}>False Positive</p>
                  <p className="text-3xl font-bold" style={{ color: "var(--accent-danger)" }}>
                    {cm[0][1]?.toLocaleString()}
                  </p>
                </div>
                <div className="p-6 rounded-xl" style={{ background: "rgba(255, 107, 107, 0.1)" }}>
                  <p className="text-xs mb-1" style={{ color: "var(--text-secondary)" }}>False Negative</p>
                  <p className="text-3xl font-bold" style={{ color: "var(--accent-danger)" }}>
                    {cm[1][0]?.toLocaleString()}
                  </p>
                </div>
                <div className="p-6 rounded-xl" style={{ background: "rgba(108, 99, 255, 0.15)" }}>
                  <p className="text-xs mb-1" style={{ color: "var(--text-secondary)" }}>True Positive</p>
                  <p className="text-3xl font-bold" style={{ color: "var(--accent-primary)" }}>
                    {cm[1][1]?.toLocaleString()}
                  </p>
                </div>
              </div>
              <div className="mt-4 flex justify-center gap-8 text-xs" style={{ color: "var(--text-secondary)" }}>
                <span>← Predicted →</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Top High-Risk Users with SHAP Explanations */}
      <div className="glass-card p-6">
        <h3 className="text-sm font-semibold uppercase tracking-wider mb-4"
            style={{ color: "var(--text-secondary)" }}>
          🧠 AI Explanations — Top High-Risk Users
        </h3>
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {top_high_risk_users?.slice(0, 10).map((user, i) => (
            <details key={i} className="group">
              <summary className="cursor-pointer p-3 rounded-lg hover:bg-[rgba(108,99,255,0.05)] flex justify-between items-center">
                <span>
                  <span className="font-mono text-sm">Customer #{user.customer_index}</span>
                  <span className="ml-3 text-sm font-bold" style={{ color: "var(--accent-danger)" }}>
                    {(user.churn_probability * 100).toFixed(1)}% risk
                  </span>
                </span>
                <span className="text-xs" style={{ color: "var(--text-secondary)" }}>▼ expand</span>
              </summary>
              <div className="pl-6 pb-3 space-y-1">
                {user.reasons?.map((r, j) => (
                  <p key={j} className="text-sm" style={{ color: "var(--text-secondary)" }}>
                    →{" "}
                    <span style={{ color: r.direction === "increases" ? "var(--accent-danger)" : "var(--accent-secondary)" }}>
                      {r.explanation}
                    </span>
                  </p>
                ))}
              </div>
            </details>
          ))}
        </div>
      </div>
    </div>
  );
}

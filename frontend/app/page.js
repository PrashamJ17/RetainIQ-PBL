"use client";

import { useState, useEffect, useCallback } from "react";
import { LoadingState, EmptyState } from "./components/ui";
import OverviewTab from "./components/OverviewTab";
import SegmentsTab from "./components/SegmentsTab";
import ChurnTab from "./components/ChurnTab";
import ActionTab from "./components/ActionTab";
import CLVTab from "./components/CLVTab";
import ActivationsTab from "./components/ActivationsTab";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const TABS = [
  { id: "overview", label: "📊 Overview", icon: "📊" },
  { id: "segments", label: "🎯 Segments", icon: "🎯" },
  { id: "churn", label: "⚠️ Churn Risk", icon: "⚠️" },
  { id: "action", label: "🚀 Actions", icon: "🚀" },
  { id: "clv", label: "💎 CLV", icon: "💎" },
  { id: "activations", label: "⚡ Activations", icon: "⚡" },
];

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState("overview");
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [data, setData] = useState({});
  const [error, setError] = useState(null);

  // Fetch API status
  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      const json = await res.json();
      setStatus(json);
      setError(null);
      return json;
    } catch (e) {
      setError("Cannot connect to API. Is the FastAPI server running on port 8000?");
      return null;
    }
  }, []);

  // Fetch data for a specific tab
  const fetchTabData = useCallback(async (tab) => {
    const endpoints = {
      overview: "/api/dashboard/stats",
      segments: "/api/segments",
      churn: "/api/churn-risk",
      action: "/api/action-matrix",
      clv: "/api/clv",
      activations: "/api/activations",
    };

    const endpoint = endpoints[tab];
    if (!endpoint) return;

    try {
      const res = await fetch(`${API_BASE}${endpoint}`);
      if (res.status === 503) return; // Model not trained
      const json = await res.json();
      setData(prev => ({ ...prev, [tab]: json }));
    } catch (e) {
      console.error(`Failed to fetch ${tab}:`, e);
    }
  }, []);

  // Initial load
  useEffect(() => {
    async function init() {
      const s = await fetchStatus();
      if (s?.model_loaded) {
        await fetchTabData("overview");
      }
      setLoading(false);
    }
    init();
  }, [fetchStatus, fetchTabData]);

  // Fetch data when tab changes
  useEffect(() => {
    if (status?.model_loaded && !data[activeTab]) {
      fetchTabData(activeTab);
    }
  }, [activeTab, status, data, fetchTabData]);

  // Trigger training
  const handleTrain = async () => {
    setTraining(true);
    try {
      await fetch(`${API_BASE}/api/trigger-training`, { method: "POST" });

      // Poll status every 5 seconds
      const poll = setInterval(async () => {
        const s = await fetchStatus();
        if (s && !s.is_training && s.model_loaded) {
          clearInterval(poll);
          setTraining(false);
          // Reload all data
          setData({});
          await fetchTabData("overview");
        }
      }, 5000);
    } catch (e) {
      setTraining(false);
      setError("Failed to trigger training");
    }
  };

  // Error screen
  if (error && !status) {
    return (
      <div className="min-h-screen flex items-center justify-center p-8">
        <EmptyState
          title="Connection Error"
          message={error}
          action={
            <button onClick={() => { setError(null); setLoading(true); fetchStatus().then(() => setLoading(false)); }}
                    className="px-6 py-3 rounded-xl font-semibold text-white"
                    style={{ background: "var(--accent-primary)" }}>
              Retry Connection
            </button>
          }
        />
      </div>
    );
  }

  // Loading screen
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingState message="Connecting to RetainIQ API..." />
      </div>
    );
  }

  // Training overlay
  if (training) {
    return (
      <div className="training-overlay">
        <div className="spinner mb-6" style={{ width: 60, height: 60 }}></div>
        <h2 className="text-2xl font-bold mb-2 gradient-text">Training ML Pipeline</h2>
        <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
          Data → Segmentation → Churn → CLV → Activation
        </p>
        <p className="text-xs mt-4" style={{ color: "var(--text-secondary)" }}>
          Processing 93,000+ customers... this takes ~5 minutes
        </p>
      </div>
    );
  }

  return (
    <div className="min-h-screen" style={{ background: "var(--bg-primary)" }}>
      {/* Header */}
      <header className="sticky top-0 z-50 px-6 py-4" style={{
        background: "rgba(10, 10, 26, 0.85)",
        backdropFilter: "blur(12px)",
        borderBottom: "1px solid var(--border-subtle)",
      }}>
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🧠</span>
            <div>
              <h1 className="text-lg font-bold gradient-text">RetainIQ</h1>
              <p className="text-xs" style={{ color: "var(--text-secondary)" }}>Customer Retention Engine</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {status?.model_loaded ? (
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span>
                <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
                  {status.total_customers?.toLocaleString()} customers loaded
                </span>
              </div>
            ) : (
              <button onClick={handleTrain}
                      className="px-4 py-2 rounded-xl font-semibold text-sm text-white transition-all hover:scale-105"
                      style={{ background: "linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))" }}>
                🚀 Train Model
              </button>
            )}

            {status?.model_loaded && (
              <button onClick={handleTrain}
                      className="px-3 py-2 rounded-lg text-xs font-medium transition-all hover:bg-[rgba(108,99,255,0.1)]"
                      style={{ color: "var(--text-secondary)", border: "1px solid var(--border-subtle)" }}>
                ↻ Retrain
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <nav className="px-6 py-3" style={{ borderBottom: "1px solid var(--border-subtle)" }}>
        <div className="max-w-7xl mx-auto flex gap-2 overflow-x-auto">
          {TABS.map(tab => (
            <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                    className={`tab-btn whitespace-nowrap ${activeTab === tab.id ? "active" : ""}`}>
              {tab.label}
            </button>
          ))}
        </div>
      </nav>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {!status?.model_loaded ? (
          <EmptyState
            title="No Model Trained Yet"
            message="Click 'Train Model' to run the full pipeline: Data ingestion → K-Means Segmentation → XGBoost Churn → CLV Prediction → Activation Engine"
            action={
              <button onClick={handleTrain}
                      className="px-8 py-3 rounded-xl font-semibold text-white text-lg transition-all hover:scale-105"
                      style={{ background: "linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))" }}>
                🚀 Train ML Pipeline
              </button>
            }
          />
        ) : (
          <>
            {activeTab === "overview" && (data.overview ? <OverviewTab data={data.overview} /> : <LoadingState />)}
            {activeTab === "segments" && (data.segments ? <SegmentsTab data={data.segments} /> : <LoadingState />)}
            {activeTab === "churn" && (data.churn ? <ChurnTab data={data.churn} /> : <LoadingState />)}
            {activeTab === "action" && (data.action ? <ActionTab data={data.action} /> : <LoadingState />)}
            {activeTab === "clv" && (data.clv ? <CLVTab data={data.clv} /> : <LoadingState />)}
            {activeTab === "activations" && (data.activations ? <ActivationsTab data={data.activations} /> : <LoadingState />)}
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="px-6 py-4 text-center text-xs" style={{ color: "var(--text-secondary)", borderTop: "1px solid var(--border-subtle)" }}>
        RetainIQ v2.0 — Built with XGBoost, SHAP, FastAPI & Next.js
        {status?.last_trained && (
          <span className="ml-4">Last trained: {new Date(status.last_trained).toLocaleString()}</span>
        )}
      </footer>
    </div>
  );
}

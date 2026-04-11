const API_BASE = (typeof window === "undefined") 
  ? (process.env.INTERNAL_API_URL || process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000")
  : (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000");

async function fetchAPI(endpoint) {
  const res = await fetch(`${API_BASE}${endpoint}`, { cache: "no-store" });
  if (!res.ok) {
    if (res.status === 503) return null; // Model not trained yet
    throw new Error(`API error: ${res.status}`);
  }
  return res.json();
}

export async function getStatus() {
  return fetchAPI("/api/status");
}

export async function triggerTraining() {
  const res = await fetch(`${API_BASE}/api/trigger-training`, { method: "POST" });
  return res.json();
}

export async function getDashboardStats() {
  return fetchAPI("/api/dashboard/stats");
}

export async function getSegments() {
  return fetchAPI("/api/segments");
}

export async function getChurnRisk() {
  return fetchAPI("/api/churn-risk");
}

export async function getActionMatrix() {
  return fetchAPI("/api/action-matrix");
}

export async function getCLV() {
  return fetchAPI("/api/clv");
}

export async function getActivations() {
  return fetchAPI("/api/activations");
}

export async function getExport(actionKey) {
  return fetchAPI(`/api/export/${actionKey}`);
}

export async function exportFilteredCustomers(status = "ALL", segment = "ALL") {
  const params = new URLSearchParams();
  if (status) params.append("status", status);
  if (segment) params.append("segment", segment);
  return fetchAPI(`/api/export-customers?${params.toString()}`);
}

// Production server — proxies Anthropic API calls and serves the built frontend
import express from "express";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3001;

app.use(express.json({ limit: "1mb" }));

// Serve built frontend
app.use(express.static(path.join(__dirname, "dist")));

// Proxy Anthropic API calls (avoids browser CORS)
app.post("/api/anthropic/v1/messages", async (req, res) => {
  const apiKey = req.headers["x-api-key"];
  if (!apiKey) {
    return res.status(400).json({ error: { message: "Missing x-api-key header" } });
  }
  try {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify(req.body),
    });
    const data = await response.json();
    res.status(response.status).json(data);
  } catch (e) {
    res.status(502).json({ error: { message: "Proxy error: " + e.message } });
  }
});

// Proxy admin restore endpoints — raw binary passthrough (no JSON parsing)
app.post("/api/ml/restore/*", express.raw({ type: "*/*", limit: "50mb" }), async (req, res) => {
  const mlPath = req.path.replace(/^\/api\/ml/, "");
  try {
    const response = await fetch(`http://localhost:8000${mlPath}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
        "X-Admin-Secret": req.headers["x-admin-secret"] || "",
      },
      body: req.body,
    });
    const data = await response.json();
    res.status(response.status).json(data);
  } catch (e) {
    res.status(502).json({ error: { message: "ML server unavailable: " + e.message } });
  }
});

// Proxy ML API calls to Python FastAPI server
app.all("/api/ml/*", async (req, res) => {
  const mlPath = req.path.replace(/^\/api\/ml/, "");
  try {
    const response = await fetch(`http://localhost:8000${mlPath}`, {
      method: req.method,
      headers: { "Content-Type": "application/json" },
      body: ["POST", "PUT", "PATCH"].includes(req.method) ? JSON.stringify(req.body) : undefined,
    });
    const data = await response.json();
    res.status(response.status).json(data);
  } catch (e) {
    res.status(502).json({ error: { message: "ML server unavailable: " + e.message } });
  }
});

// Health check — Railway uses this to verify the service is up
app.get("/health", async (req, res) => {
  let mlStatus = "starting";
  try {
    const r = await fetch("http://localhost:8000/health", {
      signal: AbortSignal.timeout(3000),
    });
    const data = await r.json();
    mlStatus = data.status || "ok";
  } catch {
    mlStatus = "unavailable";
  }
  res.json({ status: "ok", ml_server: mlStatus });
});

// SPA fallback
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "dist", "index.html"));
});

app.listen(PORT, () => {
  console.log(`ICT Backtest Terminal running on http://localhost:${PORT}`);
});

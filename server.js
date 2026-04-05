// Production server — proxies Anthropic API calls and serves the built frontend
import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import Database from "better-sqlite3";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3001;

// Scanner DB path — Railway mounts volume at DATA_DIR, local uses ml/models/
const SCANNER_DB_PATH = process.env.DATA_DIR
  ? path.join(process.env.DATA_DIR, "models", "scanner.db")
  : path.join(__dirname, "ml", "models", "scanner.db");

app.use(express.json({ limit: "1mb" }));

// Serve built frontend
app.use(express.static(path.join(__dirname, "dist")));

// ═══════════════════════════════════════════════════════════
//  BAYESIAN WIN RATE — computed from all resolved scanner_setups
// ═══════════════════════════════════════════════════════════

app.get("/api/bayesian", (req, res) => {
  let db;
  try {
    db = new Database(SCANNER_DB_PATH, { readonly: true });

    // All resolved trades (wins + losses), excluding expired/pending
    const rows = db.prepare(`
      SELECT outcome, killzone, setup_quality, direction, mfe_atr, mae_atr
      FROM scanner_setups
      WHERE outcome IN ('tp1', 'tp2', 'tp3', 'stopped_out')
    `).all();

    const wins = rows.filter(r => ["tp1", "tp2", "tp3"].includes(r.outcome));
    const losses = rows.filter(r => r.outcome === "stopped_out");
    const totalWins = wins.length;
    const totalLosses = losses.length;
    const totalTrades = totalWins + totalLosses;

    // Beta(1,1) uninformative prior + observed data
    const winAlpha = 1 + totalWins;
    const winBeta = 1 + totalLosses;
    const winRateMean = winAlpha / (winAlpha + winBeta);

    // 95% credible interval via normal approximation to Beta
    const variance = (winAlpha * winBeta) /
      ((winAlpha + winBeta) ** 2 * (winAlpha + winBeta + 1));
    const std = Math.sqrt(variance);
    const lower95 = Math.max(0, winRateMean - 1.96 * std);
    const upper95 = Math.min(1, winRateMean + 1.96 * std);

    // Per-killzone stats (normalize keys: Asian→asia, NY_AM→ny_am, etc.)
    const kzKeyMap = { Asian: "asia", London: "london", NY_AM: "ny_am", NY_PM: "ny_pm", Off: "off" };
    const sessionStats = {};
    for (const kzDb of ["Asian", "London", "NY_AM", "NY_PM", "Off"]) {
      const key = kzKeyMap[kzDb] || kzDb.toLowerCase();
      const kzRows = rows.filter(r => r.killzone === kzDb);
      const kzWins = kzRows.filter(r => ["tp1", "tp2", "tp3"].includes(r.outcome));
      const kzTotal = kzRows.length;
      const kzWinCount = kzWins.length;

      // MFE/MAE for this killzone's winners (where available)
      const kzMfe = kzWins.filter(r => r.mfe_atr != null).map(r => r.mfe_atr);
      const kzMae = kzWins.filter(r => r.mae_atr != null).map(r => r.mae_atr);

      sessionStats[key] = {
        trades: kzTotal,
        wins: kzWinCount,
        win_rate: kzTotal > 0 ? kzWinCount / kzTotal : 0,
        median_drawdown: median(kzMae),
        p95_drawdown: percentile(kzMae, 95),
        median_favorable: median(kzMfe),
      };
    }

    // Overall MFE/MAE from all winners
    const allMfe = wins.filter(r => r.mfe_atr != null).map(r => r.mfe_atr);
    const allMae = wins.filter(r => r.mae_atr != null).map(r => r.mae_atr);

    // Outcome breakdown (including expired for dataset completeness)
    const outcomeCounts = db.prepare(`
      SELECT outcome, count(*) as cnt
      FROM scanner_setups
      WHERE outcome IS NOT NULL
      GROUP BY outcome
    `).all();
    const byOutcome = {};
    for (const r of outcomeCounts) byOutcome[r.outcome] = r.cnt;
    const totalAll = db.prepare("SELECT count(*) as cnt FROM scanner_setups").get().cnt;

    // Setup quality stats
    const qualityRows = db.prepare(`
      SELECT setup_quality,
             count(*) as trades,
             sum(CASE WHEN outcome IN ('tp1','tp2','tp3') THEN 1 ELSE 0 END) as wins
      FROM scanner_setups
      WHERE outcome IN ('tp1','tp2','tp3','stopped_out')
        AND setup_quality IS NOT NULL
        AND setup_quality != 'no_trade'
      GROUP BY setup_quality
    `).all();
    const setupTypeStats = {};
    for (const r of qualityRows) {
      setupTypeStats[r.setup_quality] = {
        trades: r.trades,
        wins: r.wins,
        win_rate: r.trades > 0 ? r.wins / r.trades : 0,
      };
    }

    db.close();

    res.json({
      bayesian: {
        win_rate_mean: winRateMean,
        win_rate_lower_95: lower95,
        win_rate_upper_95: upper95,
        total_trades: totalTrades,
        consecutive_losses: 0,
        max_consecutive_losses: 0,
        max_drawdown: 0,
        current_drawdown: 0,
      },
      sessionStats: {
        session_stats: sessionStats,
        bayesian_priors: {
          win_alpha: winAlpha,
          win_beta: winBeta,
          overall_win_rate: winRateMean,
          total_trades: totalTrades,
          drawdown_mu: median(allMae),
          favorable_mu: median(allMfe),
        },
        dataset_stats: {
          total: totalAll,
          resolved: totalTrades,
          by_outcome: byOutcome,
        },
      },
      setup_type_stats: setupTypeStats,
    });
  } catch (e) {
    if (db) try { db.close(); } catch {}
    res.status(500).json({ error: "Bayesian computation failed: " + e.message });
  }
});

// Helpers for MFE/MAE percentile computation
function median(arr) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function percentile(arr, p) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const idx = (p / 100) * (s.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (idx - lo);
}

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

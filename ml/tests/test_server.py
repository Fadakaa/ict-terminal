"""Tests for FastAPI ML server endpoints."""
from datetime import datetime, timezone
from pathlib import Path

import pytest
from unittest.mock import patch, MagicMock
from ml.server import app, get_db, get_dataset_manager
from ml.database import TradeLogger
from ml.config import make_test_config


_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ff_calendar_sample.xml"


@pytest.fixture
def tmp_logger(tmp_path):
    db = TradeLogger(db_path=str(tmp_path / "test.db"))
    return db


@pytest.fixture
def empty_dataset_manager():
    """Default empty dataset manager for test isolation."""
    dm = MagicMock()
    dm.get_stats.return_value = {
        "total": 0, "wfo_count": 0, "live_count": 0,
        "regime_distribution": {}, "outcome_distribution": {},
    }
    return dm


@pytest.fixture
def client(tmp_logger, tmp_path, empty_dataset_manager):
    """Override the DB and dataset_manager dependencies for tests."""
    app.dependency_overrides[get_db] = lambda: tmp_logger
    app.dependency_overrides[get_dataset_manager] = lambda: empty_dataset_manager
    test_cfg = make_test_config(
        db_path=str(tmp_path / "test.db"),
        model_dir=str(tmp_path / "models"),
    )
    from starlette.testclient import TestClient
    with patch("ml.server.get_config", return_value=test_cfg):
        yield TestClient(app)
    app.dependency_overrides.clear()


class TestHealthEndpoint:
    def test_health_returns_checks(self, client):
        r = client.get("/health")
        data = r.json()
        # Deep health check returns checks dict; may be 503 in test env
        # (no scheduler, no API keys) — that's expected
        assert r.status_code in (200, 503)
        assert "checks" in data
        assert data["checks"]["server"] == "ok"
        assert data["checks"]["database"] == "ok"
        assert data["status"] in ("healthy", "degraded")


class TestStatusEndpoint:
    def test_status_returns_model_info(self, client):
        r = client.get("/status")
        assert r.status_code == 200
        data = r.json()
        assert data["classifier_trained"] is False
        assert data["completed_trades"] == 0


class TestPredictEndpoint:
    def test_predict_cold_start(self, client):
        payload = {
            "analysis": {"bias": "bullish", "orderBlocks": [], "fvgs": [],
                         "liquidity": [], "takeProfits": [], "confluences": []},
            "candles": [{"datetime": f"2026-03-10 {i:02d}:00:00", "open": 2600 + i,
                         "high": 2603 + i, "low": 2598 + i, "close": 2601 + i}
                        for i in range(30)],
            "timeframe": "1h",
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        assert r.json()["model_status"] == "cold_start"

    def test_predict_validates_input(self, client):
        r = client.post("/predict", json={"bad": "data"})
        assert r.status_code == 422


class TestLogSetupEndpoint:
    def test_log_setup_success(self, client):
        payload = {
            "setup_id": "test-001",
            "analysis": {"bias": "bullish", "orderBlocks": [], "fvgs": [],
                         "liquidity": [], "takeProfits": [], "confluences": [],
                         "entry": {"price": 2650, "direction": "long", "rationale": ""},
                         "stopLoss": {"price": 2643, "rationale": ""}},
            "candles": [{"datetime": f"2026-03-10 {i:02d}:00:00", "open": 2600 + i,
                         "high": 2603 + i, "low": 2598 + i, "close": 2601 + i}
                        for i in range(30)],
            "timeframe": "1h",
        }
        r = client.post("/log-setup", json=payload)
        assert r.status_code == 201
        assert r.json()["setup_id"] == "test-001"


class TestLogOutcomeEndpoint:
    def test_log_outcome_success(self, client):
        # First log a setup
        setup = {
            "setup_id": "test-002",
            "analysis": {"bias": "bullish", "orderBlocks": [], "fvgs": [],
                         "liquidity": [], "takeProfits": [], "confluences": []},
            "candles": [{"datetime": f"2026-03-10 {i:02d}:00:00", "open": 2600 + i,
                         "high": 2603 + i, "low": 2598 + i, "close": 2601 + i}
                        for i in range(30)],
            "timeframe": "1h",
        }
        client.post("/log-setup", json=setup)

        outcome = {
            "setup_id": "test-002",
            "result": "tp1_hit",
            "max_favorable_excursion": 35,
            "max_adverse_excursion": 5,
            "pnl_pips": 30,
        }
        r = client.post("/log-outcome", json=outcome)
        assert r.status_code == 200

    def test_log_outcome_unknown_setup(self, client):
        outcome = {"setup_id": "nonexistent", "result": "stopped_out"}
        r = client.post("/log-outcome", json=outcome)
        assert r.status_code == 404


class TestPendingOutcomesEndpoint:
    def test_pending_outcomes_empty(self, client):
        r = client.get("/pending-outcomes")
        assert r.status_code == 200
        assert r.json() == []


class TestBeliefsEndpoint:
    def test_beliefs_null_initially(self, client):
        r = client.get("/beliefs")
        assert r.status_code == 200
        assert r.json() is None

    def test_beliefs_populated_after_outcome(self, client):
        # Log a setup + outcome first
        setup = {
            "setup_id": "b-001",
            "analysis": {"bias": "bullish", "orderBlocks": [], "fvgs": [],
                         "liquidity": [], "takeProfits": [], "confluences": []},
            "candles": [{"datetime": f"2026-03-10 {i:02d}:00:00", "open": 2600 + i,
                         "high": 2603 + i, "low": 2598 + i, "close": 2601 + i}
                        for i in range(30)],
            "timeframe": "1h",
        }
        client.post("/log-setup", json=setup)
        client.post("/log-outcome", json={
            "setup_id": "b-001", "result": "tp1_hit", "pnl_pips": 50,
        })
        r = client.get("/beliefs")
        assert r.status_code == 200
        data = r.json()
        assert data is not None
        assert "win_rate_mean" in data
        assert data["total_trades"] == 1

    def test_log_outcome_updates_bayesian(self, client, tmp_logger):
        """Log-outcome should trigger Bayesian update (alpha incremented for win)."""
        setup = {
            "setup_id": "b-002",
            "analysis": {"bias": "bullish", "orderBlocks": [], "fvgs": [],
                         "liquidity": [], "takeProfits": [], "confluences": []},
            "candles": [{"datetime": f"2026-03-10 {i:02d}:00:00", "open": 2600 + i,
                         "high": 2603 + i, "low": 2598 + i, "close": 2601 + i}
                        for i in range(30)],
            "timeframe": "1h",
        }
        client.post("/log-setup", json=setup)
        client.post("/log-outcome", json={
            "setup_id": "b-002", "result": "tp1_hit", "pnl_pips": 40,
        })
        state = tmp_logger.get_bayesian_state()
        assert state is not None
        # Default prior alpha=1, after one win alpha=2
        assert state["alpha"] == 2.0
        assert state["total_wins"] == 1


class TestRetrainEndpoint:
    def test_retrain_insufficient_data(self, client):
        r = client.post("/retrain")
        assert r.status_code == 400
        assert "insufficient" in r.json()["detail"].lower() or "0" in r.json()["detail"]


class TestDatasetStatsEndpoint:
    """Step 10: Tests for /dataset/stats endpoint."""

    def test_dataset_stats_empty(self, client):
        """Empty dataset returns zero counts."""
        mock_dm = MagicMock()
        mock_dm.get_stats.return_value = {
            "total": 0, "wfo_count": 0, "live_count": 0,
            "regime_distribution": {}, "outcome_distribution": {},
        }
        app.dependency_overrides[get_dataset_manager] = lambda: mock_dm
        r = client.get("/dataset/stats")
        assert r.status_code == 200
        assert r.json()["total"] == 0
        app.dependency_overrides.pop(get_dataset_manager, None)

    def test_dataset_stats_with_data(self, client):
        """Dataset with trades returns correct counts."""
        mock_dm = MagicMock()
        mock_dm.get_stats.return_value = {
            "total": 100, "wfo_count": 80, "live_count": 20,
            "regime_distribution": {"trending_up": 50, "ranging": 50},
            "outcome_distribution": {"tp1_hit": 60, "stopped_out": 40},
        }
        app.dependency_overrides[get_dataset_manager] = lambda: mock_dm
        r = client.get("/dataset/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 100
        assert data["wfo_count"] == 80
        app.dependency_overrides.pop(get_dataset_manager, None)


class TestBayesianDriftEndpoint:
    """Step 10: Tests for /bayesian/drift endpoint."""

    def test_drift_no_state(self, client):
        """No Bayesian state returns no drift."""
        r = client.get("/bayesian/drift")
        assert r.status_code == 200
        data = r.json()
        assert data["level"] == "none"

    def test_drift_after_trades(self, client):
        """After recording trades, drift check returns valid result."""
        # Log setup + outcome to create Bayesian state
        setup = {
            "setup_id": "drift-001",
            "analysis": {"bias": "bullish", "orderBlocks": [], "fvgs": [],
                         "liquidity": [], "takeProfits": [], "confluences": []},
            "candles": [{"datetime": f"2026-03-10 {i:02d}:00:00", "open": 2600 + i,
                         "high": 2603 + i, "low": 2598 + i, "close": 2601 + i}
                        for i in range(30)],
            "timeframe": "1h",
        }
        client.post("/log-setup", json=setup)
        client.post("/log-outcome", json={
            "setup_id": "drift-001", "result": "tp1_hit", "pnl_pips": 30,
        })
        r = client.get("/bayesian/drift")
        assert r.status_code == 200
        data = r.json()
        assert "drift_sd" in data
        assert "level" in data


class TestPredictWithDataset:
    """predict endpoint should pass dataset_manager for effective count."""

    def test_predict_not_cold_start_with_dataset_trades(self, client):
        """Predict endpoint should not be cold_start when dataset has trades."""
        mock_dm = MagicMock()
        mock_dm.get_stats.return_value = {
            "total": 50, "wfo_count": 50, "live_count": 0,
            "regime_distribution": {"trending_up": 25, "ranging": 25},
            "outcome_distribution": {"tp1_hit": 30, "stopped_out": 20},
        }
        app.dependency_overrides[get_dataset_manager] = lambda: mock_dm
        payload = {
            "analysis": {"bias": "bullish", "orderBlocks": [], "fvgs": [],
                         "liquidity": [], "takeProfits": [], "confluences": []},
            "candles": [{"datetime": f"2026-03-10 {i:02d}:00:00", "open": 2600 + i,
                         "high": 2603 + i, "low": 2598 + i, "close": 2601 + i}
                        for i in range(30)],
            "timeframe": "1h",
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        # With 50 dataset trades, should NOT be cold_start
        assert r.json()["model_status"] != "cold_start"
        app.dependency_overrides.pop(get_dataset_manager, None)


class TestStatusWithDataset:
    """Status endpoint should include dataset counts."""

    def test_status_includes_dataset_total(self, client):
        mock_dm = MagicMock()
        mock_dm.get_stats.return_value = {
            "total": 80, "wfo_count": 75, "live_count": 5,
            "regime_distribution": {}, "outcome_distribution": {},
        }
        app.dependency_overrides[get_dataset_manager] = lambda: mock_dm
        r = client.get("/status")
        assert r.status_code == 200
        data = r.json()
        # Should include dataset_trades field
        assert "dataset_trades" in data
        assert data["dataset_trades"] == 80
        app.dependency_overrides.pop(get_dataset_manager, None)


class TestTradesHistoryEndpoint:
    def test_trades_history_empty(self, client):
        r = client.get("/trades/history")
        assert r.status_code == 200
        assert r.json() == []

    def test_trades_history_with_data(self, client):
        setup = {
            "setup_id": "hist-001",
            "analysis": {"bias": "bullish", "orderBlocks": [], "fvgs": [],
                         "liquidity": [], "takeProfits": [], "confluences": [],
                         "entry": {"price": 2650, "direction": "long", "rationale": ""},
                         "stopLoss": {"price": 2643, "rationale": ""}},
            "candles": [{"datetime": f"2026-03-10 {i:02d}:00:00", "open": 2600 + i,
                         "high": 2603 + i, "low": 2598 + i, "close": 2601 + i}
                        for i in range(30)],
            "timeframe": "1h",
        }
        client.post("/log-setup", json=setup)
        client.post("/log-outcome", json={
            "setup_id": "hist-001", "result": "tp1_hit",
            "max_favorable_excursion": 35, "max_adverse_excursion": 5, "pnl_pips": 30,
        })
        r = client.get("/trades/history")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["setup_id"] == "hist-001"
        assert data[0]["cumulative_pnl"] == 30.0


class TestDeleteSetupEndpoint:
    def test_delete_setup_success(self, client):
        setup = {
            "setup_id": "del-001",
            "analysis": {"bias": "bullish", "orderBlocks": [], "fvgs": [],
                         "liquidity": [], "takeProfits": [], "confluences": []},
            "candles": [{"datetime": f"2026-03-10 {i:02d}:00:00", "open": 2600 + i,
                         "high": 2603 + i, "low": 2598 + i, "close": 2601 + i}
                        for i in range(30)],
            "timeframe": "1h",
        }
        client.post("/log-setup", json=setup)
        r = client.delete("/delete-setup/del-001")
        assert r.status_code == 200

    def test_delete_setup_not_found(self, client):
        r = client.delete("/delete-setup/nonexistent")
        assert r.status_code == 404


# ── Task 14 — Forex Calendar endpoints ────────────────────────


@pytest.fixture
def calendar_client(tmp_logger, tmp_path, empty_dataset_manager):
    """Variant of ``client`` that points the shared scanner at an isolated DB
    and wires a fixture-backed calendar store onto it."""
    from ml.calendar import CalendarStore, ForexFactorySource
    from ml.scanner_db import ScannerDB
    from ml.scanner import ScannerEngine

    app.dependency_overrides[get_db] = lambda: tmp_logger
    app.dependency_overrides[get_dataset_manager] = lambda: empty_dataset_manager

    test_cfg = make_test_config(
        db_path=str(tmp_path / "test_scanner.db"),
        model_dir=str(tmp_path / "models"),
    )
    scanner_db = ScannerDB(db_path=str(tmp_path / "test_scanner.db"))
    engine = ScannerEngine(db=scanner_db)
    store = CalendarStore(
        source=ForexFactorySource(_offline_path=str(_FIXTURE_PATH)),
        db_path=scanner_db.db_path,
    )
    store.refresh(force=True)
    engine._calendar_store = store

    from starlette.testclient import TestClient
    with patch("ml.server.get_config", return_value=test_cfg), \
         patch("ml.server._get_scanner", return_value=engine):
        yield TestClient(app)
    app.dependency_overrides.clear()


def test_calendar_upcoming_endpoint(calendar_client):
    r = calendar_client.get("/calendar/upcoming?hours=72")
    assert r.status_code == 200
    body = r.json()
    assert "events" in body and "count" in body
    # Fixture has multiple USD high-impact events; at least one should land
    # in any 72h window straddling the fixture's date.
    # Just confirm the shape, not the count (depends on test runtime).
    if body["events"]:
        assert "title" in body["events"][0]
        assert "category" in body["events"][0]


def test_calendar_proximity_endpoint(calendar_client):
    r = calendar_client.get("/calendar/proximity")
    assert r.status_code == 200
    body = r.json()
    assert "state" in body
    assert body["state"] in {"clear", "caution", "imminent",
                             "post_event", "unavailable"}


def test_calendar_refresh_endpoint(calendar_client):
    r = calendar_client.post("/calendar/refresh")
    assert r.status_code == 200
    body = r.json()
    assert "updated" in body


def test_calendar_stats_endpoint(calendar_client):
    r = calendar_client.get("/calendar/stats?days=365")
    assert r.status_code == 200
    body = r.json()
    assert "by_category" in body
    assert "total" in body

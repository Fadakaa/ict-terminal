"""Tests for notification message builder — Task 10 (calendar warning render)."""
from ml.notifications import build_notification_message


def _base_setup():
    return {
        "direction": "long",
        "bias": "bullish",
        "entry_price": 2400.0,
        "sl_price": 2390.0,
        "calibrated_sl": 2388.0,
        "tps": [2415.0, 2425.0, 2440.0],
        "rr_ratios": [1.5, 2.5, 4.0],
        "setup_quality": "A",
        "killzone": "London",
        "timeframe": "1h",
        "current_price": 2400.0,
        "calibration_json": {"confidence": {"grade": "A", "autogluon_win_prob": 0.62}},
    }


def test_notification_includes_calendar_warning_imminent():
    setup = _base_setup()
    setup["calendar_proximity"] = {
        "state": "imminent",
        "warning": "NFP releases in 25 minutes",
        "next_event_title": "Non-Farm Employment Change",
        "next_event_minutes": 25,
        "next_event_category": "nfp",
    }
    msg = build_notification_message(setup)
    assert ("CAUTION" in msg or "WARNING" in msg or "⚠" in msg)
    assert ("NFP" in msg or "Non-Farm" in msg)


def test_notification_includes_calendar_warning_caution():
    setup = _base_setup()
    setup["calendar_proximity"] = {
        "state": "caution",
        "warning": "FOMC in 75 minutes",
        "next_event_title": "FOMC Statement",
        "next_event_minutes": 75,
        "next_event_category": "fomc",
    }
    msg = build_notification_message(setup)
    assert ("CAUTION" in msg or "⚠" in msg)
    assert "FOMC" in msg


def test_notification_no_warning_when_clear():
    setup = _base_setup()
    setup["calendar_proximity"] = {"state": "clear", "warning": None}
    msg = build_notification_message(setup)
    assert "CAUTION" not in msg
    assert "WARNING" not in msg


def test_notification_no_warning_when_proximity_missing():
    """A setup that pre-dates calendar integration must still render cleanly."""
    setup = _base_setup()
    msg = build_notification_message(setup)
    assert "CAUTION" not in msg
    assert "WARNING" not in msg
    assert "Direction: LONG" in msg


def test_notification_post_event_state_does_not_warn():
    """post_event is informational only — design choice: do not flag in alert.

    The setup is already past the event window; the trader is in normal price
    discovery. Only caution/imminent get inline warnings."""
    setup = _base_setup()
    setup["calendar_proximity"] = {
        "state": "post_event",
        "warning": "FOMC printed 60 minutes ago",
        "next_event_title": None,
        "next_event_minutes": None,
        "next_event_category": None,
    }
    msg = build_notification_message(setup)
    assert "CAUTION" not in msg
    assert "WARNING" not in msg

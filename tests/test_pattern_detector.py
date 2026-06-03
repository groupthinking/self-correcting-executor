#!/usr/bin/env python3
"""Unit tests for analyzers.pattern_detector.PatternDetector.

These tests exercise the pattern-detection and insight-generation logic
without a live PostgreSQL server by mocking the ``utils.db_tracker``
connection helpers. They are written as plain synchronous functions that
drive the coroutines via ``asyncio.run`` so they run under stock pytest
regardless of pytest-asyncio configuration.
"""

import asyncio
from unittest.mock import MagicMock, patch

import analyzers.pattern_detector as pd


class _FakeCursor:
    """Returns canned result sets for the three SELECTs in detect_patterns()."""

    def __init__(self, results):
        self._results = list(results)
        self._last = []

    def execute(self, sql, params=None):
        # Hand back the next canned result set for each query in order.
        self._last = self._results.pop(0)

    def fetchall(self):
        return self._last

    def close(self):
        pass


def _make_conn(results):
    conn = MagicMock()
    conn.cursor.return_value = _FakeCursor(results)
    return conn


def test_analyze_builds_patterns_insights_and_recommendations():
    failures = [("data_processor", 5), ("api_health_checker", 2)]
    slow = [("multimodal_llm_analyzer", 3.4, 4)]
    usage = [("data_processor", 12), ("api_health_checker", 9)]

    with patch.object(pd, "ensure_tables_exist", lambda: None), patch.object(
        pd, "get_db_connection", lambda: _make_conn([failures, slow, usage])
    ):
        result = asyncio.run(pd.PatternDetector().analyze())

    assert result["success"] is True

    patterns = result["patterns"]
    repeated = patterns["failure_patterns"]["repeated_failures"]
    assert repeated[0] == {
        "protocol": "data_processor",
        "failure_count": 5,
        "severity": "high",
    }
    # count of 2 (> 1 but <= 3) is "medium" severity
    assert repeated[1]["severity"] == "medium"

    slow_protocols = patterns["performance_patterns"]["slow_protocols"]
    assert slow_protocols == [
        {
            "protocol": "multimodal_llm_analyzer",
            "avg_duration": 3.4,
            "execution_count": 4,
        }
    ]

    top = patterns["usage_patterns"]["top_protocols"]
    assert top[0] == {"protocol": "data_processor", "usage_count": 12}

    insight_types = sorted({i["type"] for i in result["insights"]})
    assert insight_types == ["high_usage", "performance_issue", "repeated_failure"]

    recs = result["recommendations"]
    assert {r["mutation_type"] for r in recs} == {
        "error_handling",
        "performance_optimization",
    }
    # One error-handling recommendation per repeated failure, with the protocol
    # name correctly extracted from each insight message.
    error_handling = {
        r["protocol"] for r in recs if r["mutation_type"] == "error_handling"
    }
    assert error_handling == {"data_processor", "api_health_checker"}
    perf = [r for r in recs if r["mutation_type"] == "performance_optimization"]
    assert [r["protocol"] for r in perf] == ["multimodal_llm_analyzer"]


def test_slow_protocol_below_threshold_is_ignored():
    # avg_duration <= 1.0s should not be reported as a slow protocol.
    with patch.object(pd, "ensure_tables_exist", lambda: None), patch.object(
        pd,
        "get_db_connection",
        lambda: _make_conn([[], [("fast_protocol", 0.5, 10)], []]),
    ):
        patterns = asyncio.run(pd.PatternDetector().detect_patterns())

    assert patterns["performance_patterns"]["slow_protocols"] == []


def test_detect_patterns_degrades_gracefully_on_db_error():
    def boom():
        raise RuntimeError("connection refused")

    with patch.object(pd, "ensure_tables_exist", lambda: None), patch.object(
        pd, "get_db_connection", boom
    ):
        result = asyncio.run(pd.PatternDetector().detect_patterns())

    assert result["failure_patterns"]["repeated_failures"] == []
    assert result["performance_patterns"]["slow_protocols"] == []
    assert result["usage_patterns"]["top_protocols"] == []
    assert "error" in result

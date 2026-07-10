"""Stage 2: Technique validation -- Speculative decoding accelerator."""
import importlib
import os
import pathlib
import sys

import pytest
from unittest.mock import patch

# Ensure src/ is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def _make_client(env_overrides: dict):
    """Create a fresh TestClient with the given environment.

    The app module creates module-level singletons (engine, stats) at
    import time, so we must reload it to pick up env changes.
    """
    with patch.dict(os.environ, env_overrides, clear=False):
        import src.speculative as app_module

        importlib.reload(app_module)
        from fastapi.testclient import TestClient

        return TestClient(app_module.app)


@pytest.fixture
def demo_client():
    """Test client running in demo mode (no real models)."""
    return _make_client({"DEMO_MODE": "true"})


@pytest.fixture
def live_client():
    """Test client configured with model URLs (will fall back to demo)."""
    return _make_client({
        "DEMO_MODE": "false",
        "TARGET_MODEL_URL": "",
        "DRAFT_MODEL_URL": "",
        "SPECULATIVE_MODEL_URL": "",
    })


# -- Target model (baseline) ------------------------------------------------


class TestTargetModel:
    def test_target_model_responds(self, demo_client):
        """Baseline inference works via speculative run in demo mode."""
        resp = demo_client.post(
            "/api/v1/speculative/run",
            json={"text": "What is speculative decoding?", "max_tokens": 64},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "baseline" in data
        assert data["baseline"]["latency_ms"] > 0
        assert isinstance(data["baseline"]["output"], str)
        assert len(data["baseline"]["output"]) > 0
        assert data["baseline"]["tokens"] > 0


# -- Draft model -------------------------------------------------------------


class TestDraftModel:
    def test_draft_model_responds(self, demo_client):
        """Draft model (speculative) inference works in demo mode."""
        resp = demo_client.post(
            "/api/v1/speculative/run",
            json={"text": "Explain draft models.", "max_tokens": 64},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "speculative" in data
        assert data["speculative"]["latency_ms"] > 0
        assert isinstance(data["speculative"]["output"], str)
        assert len(data["speculative"]["output"]) > 0
        assert data["speculative"]["tokens"] > 0


# -- Speculative comparison --------------------------------------------------


class TestSpeculativeComparison:
    def test_speculative_returns_both_latencies(self, demo_client):
        """Both baseline and speculative latencies are returned."""
        resp = demo_client.post(
            "/api/v1/speculative/run",
            json={"text": "Summarize healthcare AI benefits.", "max_tokens": 128},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "baseline" in data
        assert "speculative" in data
        assert isinstance(data["baseline"]["latency_ms"], (int, float))
        assert isinstance(data["speculative"]["latency_ms"], (int, float))
        assert data["baseline"]["latency_ms"] > 0
        assert data["speculative"]["latency_ms"] > 0

    def test_speculative_faster_than_baseline(self, demo_client):
        """Speculative decoding shows speedup > 1.0 in demo mode."""
        resp = demo_client.post(
            "/api/v1/speculative/run",
            json={"text": "Compare inference strategies.", "max_tokens": 128},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "speedup" in data
        assert isinstance(data["speedup"], (int, float))
        assert data["speedup"] > 1.0, (
            f"Expected speedup > 1.0, got {data['speedup']}"
        )

    def test_speedup_threshold_check(self, demo_client):
        """speedup_meets_threshold field is correctly computed."""
        resp = demo_client.post(
            "/api/v1/speculative/run",
            json={"text": "Test threshold check.", "max_tokens": 64},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "speedup_meets_threshold" in data
        assert isinstance(data["speedup_meets_threshold"], bool)
        # Verify consistency: threshold check matches speedup vs 1.5
        if data["speedup"] >= 1.5:
            assert data["speedup_meets_threshold"] is True
        else:
            assert data["speedup_meets_threshold"] is False


# -- Status endpoint ---------------------------------------------------------


class TestStatusEndpoint:
    def test_status_endpoint(self, demo_client):
        """Status endpoint reports configuration."""
        resp = demo_client.get("/api/v1/speculative/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "speculative_configured" in data
        assert isinstance(data["speculative_configured"], bool)
        assert "target_model" in data
        assert "draft_model" in data
        assert "mode" in data
        assert data["mode"] in ("vllm_native", "app_layer", "demo")
        assert "claim_threshold" in data
        assert isinstance(data["claim_threshold"], (int, float))
        assert "num_speculative_tokens" in data


# -- Demo mode ---------------------------------------------------------------


class TestDemoMode:
    def test_demo_mode_works(self, demo_client):
        """Demo mode works without real models."""
        # Health check
        health_resp = demo_client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["mode"] == "demo"

        # Speculative run
        run_resp = demo_client.post(
            "/api/v1/speculative/run",
            json={"text": "Demo mode test.", "max_tokens": 32},
        )
        assert run_resp.status_code == 200
        data = run_resp.json()
        assert data["baseline"]["latency_ms"] > 0
        assert data["speculative"]["latency_ms"] > 0
        assert data["speedup"] > 0

        # Inference
        inf_resp = demo_client.post(
            "/api/v1/inference",
            json={"text": "Demo inference.", "max_tokens": 32},
        )
        assert inf_resp.status_code == 200
        assert inf_resp.json()["output"]

    def test_demo_mode_no_real_model_urls_needed(self):
        """Demo mode starts without any model URLs configured."""
        client = _make_client({
            "DEMO_MODE": "true",
            "TARGET_MODEL_URL": "",
            "DRAFT_MODEL_URL": "",
            "SPECULATIVE_MODEL_URL": "",
        })
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


# -- AI disclaimer -----------------------------------------------------------


class TestAIDisclaimer:
    def test_ai_disclaimer_present(self, demo_client):
        """AI disclaimer is present in speculative run response."""
        resp = demo_client.post(
            "/api/v1/speculative/run",
            json={"text": "Check disclaimer.", "max_tokens": 32},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "ai_disclaimer" in data
        assert isinstance(data["ai_disclaimer"], str)
        assert len(data["ai_disclaimer"]) > 0
        assert "environment-specific" in data["ai_disclaimer"]

    def test_ai_disclaimer_in_inference(self, demo_client):
        """AI disclaimer is present in standard inference response."""
        resp = demo_client.post(
            "/api/v1/inference",
            json={"text": "Inference disclaimer check.", "max_tokens": 32},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "ai_disclaimer" in data
        assert "environment-specific" in data["ai_disclaimer"]


# -- Stats endpoint ----------------------------------------------------------


class TestStatsEndpoint:
    def test_stats_returns_fields(self, demo_client):
        """Stats endpoint returns expected fields."""
        resp = demo_client.get("/api/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "request_count" in data
        assert "average_latency_ms" in data
        assert "uptime_seconds" in data
        assert "mode" in data
        assert "version" in data

"""Speculative Decoding Accelerator Service.

Benchmarks speculative decoding on Intel Xeon CPUs by pairing a small
draft model (350M parameters) with a larger target model (2B parameters).
Supports vLLM native speculative decoding and app-layer draft model
comparison.

Adapted from triforce healthcare-agent speculative decoding module.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AI_DISCLAIMER = (
    "Speedup measurements are environment-specific "
    "-- run on your target hardware for accurate results."
)

logger = logging.getLogger("speculative-decoding")

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class SpeculativeRunRequest(BaseModel):
    text: str
    max_tokens: int = Field(default=128, ge=1, le=4096)


class InferenceRequest(BaseModel):
    text: str
    max_tokens: int = Field(default=128, ge=1, le=4096)


class ModelResult(BaseModel):
    latency_ms: float
    output: str
    tokens: int


class SpeculativeResult(BaseModel):
    latency_ms: float
    output: str
    tokens: int
    method: str  # "vllm_native" or "app_layer"


class SpeculativeRunResponse(BaseModel):
    baseline: ModelResult
    speculative: SpeculativeResult
    speedup: float
    speedup_meets_threshold: bool
    token_acceptance_rate: Optional[float] = None
    ai_disclaimer: str


class InferenceResponse(BaseModel):
    output: str
    model: str
    latency_ms: float
    tokens: int
    method: str
    ai_disclaimer: str


class StatsResponse(BaseModel):
    request_count: int
    average_latency_ms: float
    average_speedup: Optional[float] = None
    average_acceptance_rate: Optional[float] = None
    uptime_seconds: float
    mode: str
    version: str


# ---------------------------------------------------------------------------
# SpeculativeEngine
# ---------------------------------------------------------------------------


class SpeculativeEngine:
    """Runs speculative decoding comparisons.

    Supports three operating modes:
    - vllm_native: Dedicated vLLM pod with --speculative-config
    - app_layer: Separate draft and target model calls via Ollama or
      any OpenAI-compatible endpoint
    - demo: Simulates speculative decoding with ~2x speedup

    Falls back from vllm_native -> app_layer -> demo as needed.
    """

    def __init__(self) -> None:
        self._target_url = os.environ.get("TARGET_MODEL_URL", "")
        self._target_model_name = os.environ.get("TARGET_MODEL_NAME", "")
        self._draft_url = os.environ.get("DRAFT_MODEL_URL", "")
        self._draft_model_name = os.environ.get("DRAFT_MODEL_NAME", "")
        self._speculative_url = os.environ.get("SPECULATIVE_MODEL_URL", "")
        self._api_key = os.environ.get("MODEL_API_KEY", "")
        self._claim_threshold = float(
            os.environ.get("SPECULATIVE_CLAIM_THRESHOLD", "1.5")
        )
        self._num_speculative_tokens = int(
            os.environ.get("NUM_SPECULATIVE_TOKENS", "5")
        )
        self._demo_mode = os.environ.get("DEMO_MODE", "true").lower() == "true"

    @property
    def mode(self) -> str:
        """Current operating mode."""
        if self._demo_mode:
            return "demo"
        if self._speculative_url:
            return "vllm_native"
        if self._draft_url and self._target_url:
            return "app_layer"
        return "demo"

    @property
    def speculative_configured(self) -> bool:
        """Whether any speculative decoding backend is configured."""
        return bool(self._speculative_url or (self._draft_url and self._target_url))

    def status(self) -> dict:
        """Return configuration status."""
        return {
            "speculative_configured": self.speculative_configured or self._demo_mode,
            "target_model": self._target_url or "demo-target-2b",
            "target_model_name": self._target_model_name or "demo-target",
            "draft_model": self._draft_url or "demo-draft-350m",
            "draft_model_name": self._draft_model_name or "demo-draft",
            "speculative_model": self._speculative_url or None,
            "mode": self.mode,
            "num_speculative_tokens": self._num_speculative_tokens,
            "claim_threshold": self._claim_threshold,
        }

    # -- Model calling -------------------------------------------------------

    async def _call_model(
        self, url: str, text: str, max_tokens: int, model_name: str = "default"
    ) -> dict:
        """Call a model endpoint using the OpenAI-compatible API.

        When model URLs are configured (e.g. Ollama at /v1), makes real
        HTTP calls. The url should already include the /v1 prefix if needed;
        chat completions are posted to {url}/chat/completions.
        """
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }

        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                headers = {"Content-Type": "application/json"}
                if self._api_key:
                    headers["Authorization"] = f"Bearer {self._api_key}"

                # Build the completions URL. If url already ends with /v1,
                # append /chat/completions; otherwise append /v1/chat/completions.
                base = url.rstrip("/")
                if base.endswith("/v1"):
                    endpoint = f"{base}/chat/completions"
                else:
                    endpoint = f"{base}/v1/chat/completions"

                resp = await client.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            return {
                "latency_ms": round((time.monotonic() - start) * 1000, 2),
                "output": "",
                "tokens": 0,
                "error": str(e),
            }

        choice = data["choices"][0]["message"]
        usage = data.get("usage", {})
        return {
            "latency_ms": round((time.monotonic() - start) * 1000, 2),
            "output": choice.get("content", ""),
            "tokens": usage.get("completion_tokens", 0),
        }

    # -- Demo simulation -----------------------------------------------------

    async def _simulate_baseline(
        self, text: str, max_tokens: int
    ) -> dict:
        """Simulate target model inference (slower)."""
        delay = random.uniform(0.300, 0.500)  # 300-500 ms
        await asyncio.sleep(delay)
        tokens = min(max_tokens, random.randint(20, 60))
        return {
            "latency_ms": round(delay * 1000, 2),
            "output": (
                f"[Demo baseline response for: '{text[:60]}...' "
                f"({tokens} tokens generated)]"
            ),
            "tokens": tokens,
        }

    async def _simulate_speculative(
        self, text: str, max_tokens: int
    ) -> dict:
        """Simulate speculative decoding (faster due to draft model)."""
        # Speculative decoding typically achieves 1.5-2.5x speedup
        delay = random.uniform(0.130, 0.220)  # 130-220 ms
        await asyncio.sleep(delay)
        tokens = min(max_tokens, random.randint(20, 60))
        return {
            "latency_ms": round(delay * 1000, 2),
            "output": (
                f"[Demo speculative response for: '{text[:60]}...' "
                f"({tokens} tokens generated)]"
            ),
            "tokens": tokens,
            "method": "app_layer",
        }

    # -- Token acceptance rate -----------------------------------------------

    def _estimate_acceptance_rate(
        self, draft_output: str, target_output: str
    ) -> float:
        """Estimate token acceptance rate by comparing draft vs target output.

        In real speculative decoding the target model verifies each draft
        token.  Here we approximate by comparing overlapping token sequences
        between draft and target outputs.
        """
        if not draft_output or not target_output:
            return 0.0
        draft_tokens = draft_output.split()
        target_tokens = target_output.split()
        if not draft_tokens:
            return 0.0
        matches = sum(
            1 for d, t in zip(draft_tokens, target_tokens) if d == t
        )
        return round(matches / len(draft_tokens), 4)

    # -- Run comparison ------------------------------------------------------

    async def run(
        self, text: str, max_tokens: int = 128
    ) -> SpeculativeRunResponse:
        """Run speculative decoding comparison.

        1. Call target model for baseline timing
        2. Try vLLM speculative model; fall back to draft model
        3. Compute speedup and check threshold
        """
        if self._demo_mode or self.mode == "demo":
            return await self._run_demo(text, max_tokens)

        # Live mode: call real models
        target_model = self._target_model_name or "default"
        draft_model = self._draft_model_name or "default"

        baseline = await self._call_model(
            self._target_url, text, max_tokens, model_name=target_model
        )

        # Try vLLM speculative first, fall back to app-layer draft
        method = "vllm_native"
        if self._speculative_url:
            speculative = await self._call_model(
                self._speculative_url, text, max_tokens, model_name=target_model
            )
            if "error" in speculative and self._draft_url:
                speculative = await self._call_model(
                    self._draft_url, text, max_tokens, model_name=draft_model
                )
                method = "app_layer"
        elif self._draft_url:
            speculative = await self._call_model(
                self._draft_url, text, max_tokens, model_name=draft_model
            )
            method = "app_layer"
        else:
            return await self._run_demo(text, max_tokens)

        if "error" in baseline or "error" in speculative:
            raise HTTPException(
                status_code=500,
                detail="Model call failed: "
                + (baseline.get("error") or speculative.get("error", "unknown")),
            )

        speedup = round(
            baseline["latency_ms"] / speculative["latency_ms"], 2
        ) if speculative["latency_ms"] > 0 else 0.0

        acceptance_rate = self._estimate_acceptance_rate(
            speculative["output"], baseline["output"]
        )

        return SpeculativeRunResponse(
            baseline=ModelResult(**{k: baseline[k] for k in ("latency_ms", "output", "tokens")}),
            speculative=SpeculativeResult(
                latency_ms=speculative["latency_ms"],
                output=speculative["output"],
                tokens=speculative["tokens"],
                method=method,
            ),
            speedup=speedup,
            speedup_meets_threshold=speedup >= self._claim_threshold,
            token_acceptance_rate=acceptance_rate,
            ai_disclaimer=AI_DISCLAIMER,
        )

    async def _run_demo(
        self, text: str, max_tokens: int
    ) -> SpeculativeRunResponse:
        """Run comparison in demo mode with simulated latencies."""
        baseline = await self._simulate_baseline(text, max_tokens)
        speculative = await self._simulate_speculative(text, max_tokens)

        speedup = round(
            baseline["latency_ms"] / speculative["latency_ms"], 2
        ) if speculative["latency_ms"] > 0 else 0.0

        # Simulated acceptance rate for demo mode
        acceptance_rate = round(random.uniform(0.70, 0.92), 4)

        return SpeculativeRunResponse(
            baseline=ModelResult(**baseline),
            speculative=SpeculativeResult(**speculative),
            speedup=speedup,
            speedup_meets_threshold=speedup >= self._claim_threshold,
            token_acceptance_rate=acceptance_rate,
            ai_disclaimer=AI_DISCLAIMER,
        )

    # -- Single inference ----------------------------------------------------

    async def inference(
        self, text: str, max_tokens: int = 128
    ) -> InferenceResponse:
        """Run single inference, preferring speculative model when available."""
        if self._demo_mode or self.mode == "demo":
            result = await self._simulate_speculative(text, max_tokens)
            return InferenceResponse(
                output=result["output"],
                model="demo-speculative",
                latency_ms=result["latency_ms"],
                tokens=result["tokens"],
                method=result["method"],
                ai_disclaimer=AI_DISCLAIMER,
            )

        # Live mode: use speculative model if available, else target
        if self._speculative_url:
            model_name = self._target_model_name or "default"
            result = await self._call_model(
                self._speculative_url, text, max_tokens, model_name=model_name
            )
            method = "vllm_native"
        elif self._draft_url:
            model_name = self._draft_model_name or "default"
            result = await self._call_model(
                self._draft_url, text, max_tokens, model_name=model_name
            )
            method = "app_layer"
        elif self._target_url:
            model_name = self._target_model_name or "default"
            result = await self._call_model(
                self._target_url, text, max_tokens, model_name=model_name
            )
            method = "app_layer"
        else:
            result = await self._simulate_speculative(text, max_tokens)
            model_name = "demo-speculative"
            method = "app_layer"

        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {result['error']}",
            )

        return InferenceResponse(
            output=result["output"],
            model=model_name,
            latency_ms=result["latency_ms"],
            tokens=result["tokens"],
            method=method,
            ai_disclaimer=AI_DISCLAIMER,
        )


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Speculative Decoding Accelerator",
    description=(
        "Benchmark speculative decoding on Intel Xeon CPUs. "
        "Pairs a 350M draft model with a 2B target model for "
        "faster LLM inference with vLLM and bfloat16."
    ),
    version="1.0.0",
)

engine = SpeculativeEngine()

# Runtime stats
start_time = time.time()
request_count = 0
total_latency = 0.0
speedup_history: list[float] = []
acceptance_history: list[float] = []

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Basic health-check endpoint."""
    return {
        "status": "healthy",
        "mode": engine.mode,
        "version": "1.0.0",
    }


@app.get("/api/v1/speculative/status")
async def speculative_status():
    """Report speculative decoding configuration status."""
    return engine.status()


@app.post("/api/v1/speculative/run", response_model=SpeculativeRunResponse)
async def speculative_run(request: SpeculativeRunRequest):
    """Run speculative decoding comparison."""
    global request_count, total_latency

    result = await engine.run(request.text, request.max_tokens)

    request_count += 1
    total_latency += result.baseline.latency_ms
    if result.speedup > 0:
        speedup_history.append(result.speedup)
    if result.token_acceptance_rate is not None:
        acceptance_history.append(result.token_acceptance_rate)

    return result


@app.post("/api/v1/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Run inference using speculative decoding when available."""
    global request_count, total_latency

    result = await engine.inference(request.text, request.max_tokens)

    request_count += 1
    total_latency += result.latency_ms

    return result


@app.get("/api/v1/stats", response_model=StatsResponse)
async def stats():
    """Return runtime statistics."""
    avg_latency = total_latency / request_count if request_count > 0 else 0.0
    avg_speedup = (
        round(sum(speedup_history) / len(speedup_history), 2)
        if speedup_history
        else None
    )
    avg_acceptance = (
        round(sum(acceptance_history) / len(acceptance_history), 4)
        if acceptance_history
        else None
    )
    return StatsResponse(
        request_count=request_count,
        average_latency_ms=round(avg_latency, 2),
        average_speedup=avg_speedup,
        average_acceptance_rate=avg_acceptance,
        uptime_seconds=round(time.time() - start_time, 2),
        mode=engine.mode,
        version="1.0.0",
    )

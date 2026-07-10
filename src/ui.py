"""Gradio UI for Speculative Decoding Accelerator.

Provides a browser-based interface with three tabs:
1. Speculative Decoding  -- run comparisons and view results
2. Configuration         -- inspect current model and engine settings
3. Statistics            -- runtime performance metrics

Launch alongside the FastAPI service or standalone.
"""

from __future__ import annotations

import os

import gradio as gr
import httpx

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

DISCLAIMER = (
    "⚠️ Speedup measurements are environment-specific "
    "-- run on your target hardware for accurate results."
)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _post_run(text: str, max_tokens: int) -> dict:
    """POST /api/v1/speculative/run and return the JSON body."""
    resp = httpx.post(
        f"{API_BASE}/api/v1/speculative/run",
        json={"text": text, "max_tokens": int(max_tokens)},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def _get_status() -> dict:
    """GET /api/v1/speculative/status."""
    resp = httpx.get(f"{API_BASE}/api/v1/speculative/status", timeout=10)
    resp.raise_for_status()
    return resp.json()


def _get_stats() -> dict:
    """GET /api/v1/stats."""
    resp = httpx.get(f"{API_BASE}/api/v1/stats", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Tab 1: Speculative Decoding
# ---------------------------------------------------------------------------

def run_comparison(text: str, max_tokens: int):
    """Execute a speculative decoding comparison and format the results."""
    if not text.strip():
        return "Please enter a prompt.", "", "", "", ""
    try:
        data = _post_run(text, max_tokens)
    except Exception as exc:
        return f"Error: {exc}", "", "", "", ""

    baseline_ms = data["baseline"]["latency_ms"]
    spec_ms = data["speculative"]["latency_ms"]
    speedup = data["speedup"]
    meets = data["speedup_meets_threshold"]
    acceptance = data.get("token_acceptance_rate")

    baseline_info = (
        f"Latency: {baseline_ms:.1f} ms\n"
        f"Tokens:  {data['baseline']['tokens']}\n\n"
        f"{data['baseline']['output']}"
    )
    spec_info = (
        f"Latency: {spec_ms:.1f} ms\n"
        f"Tokens:  {data['speculative']['tokens']}\n"
        f"Method:  {data['speculative']['method']}\n\n"
        f"{data['speculative']['output']}"
    )
    speedup_info = (
        f"{speedup:.2f}x {'(meets threshold)' if meets else '(below threshold)'}"
    )
    acceptance_info = (
        f"{acceptance:.1%}" if acceptance is not None else "N/A"
    )
    return baseline_info, spec_info, speedup_info, acceptance_info, DISCLAIMER


# ---------------------------------------------------------------------------
# Tab 2: Configuration
# ---------------------------------------------------------------------------

def load_config():
    """Fetch current configuration from the API."""
    try:
        status = _get_status()
    except Exception as exc:
        return f"Error loading configuration: {exc}"

    lines = [
        f"Mode:                   {status.get('mode', 'unknown')}",
        f"Target Model URL:       {status.get('target_model', 'N/A')}",
        f"Target Model Name:      {status.get('target_model_name', 'N/A')}",
        f"Draft Model URL:        {status.get('draft_model', 'N/A')}",
        f"Draft Model Name:       {status.get('draft_model_name', 'N/A')}",
        f"Speculative Model URL:  {status.get('speculative_model', 'N/A')}",
        f"Speculative Configured: {status.get('speculative_configured', False)}",
        f"Claim Threshold:        {status.get('claim_threshold', 'N/A')}x",
        f"Num Speculative Tokens: {status.get('num_speculative_tokens', 'N/A')}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 3: Statistics
# ---------------------------------------------------------------------------

def load_stats():
    """Fetch runtime statistics from the API."""
    try:
        data = _get_stats()
    except Exception as exc:
        return f"Error loading statistics: {exc}"

    avg_speedup = data.get("average_speedup")
    avg_acceptance = data.get("average_acceptance_rate")
    lines = [
        f"Total Requests:          {data['request_count']}",
        f"Average Latency:         {data['average_latency_ms']:.1f} ms",
        f"Average Speedup:         {f'{avg_speedup:.2f}x' if avg_speedup else 'N/A'}",
        f"Average Acceptance Rate: {f'{avg_acceptance:.1%}' if avg_acceptance else 'N/A'}",
        f"Uptime:                  {data['uptime_seconds']:.0f} s",
        f"Mode:                    {data['mode']}",
        f"Version:                 {data['version']}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------

def create_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""
    with gr.Blocks(
        title="Speculative Decoding Accelerator",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# Speculative Decoding Accelerator")
        gr.Markdown(
            "Compare baseline vs speculative decoding latency using "
            "a fast draft model and a verification target model."
        )

        # -- Tab 1: Speculative Decoding ------------------------------------
        with gr.Tab("Speculative Decoding"):
            with gr.Row():
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter text to compare inference strategies...",
                    lines=3,
                )
                max_tokens_input = gr.Slider(
                    minimum=16, maximum=512, value=128, step=16,
                    label="Max Tokens",
                )
            run_btn = gr.Button("Run Comparison", variant="primary")

            with gr.Row():
                baseline_output = gr.Textbox(label="Baseline (Target Model)", lines=8)
                speculative_output = gr.Textbox(label="Speculative Decoding", lines=8)

            with gr.Row():
                speedup_output = gr.Textbox(label="Speedup Ratio")
                acceptance_output = gr.Textbox(label="Token Acceptance Rate")

            disclaimer_output = gr.Markdown(value=DISCLAIMER)

            run_btn.click(
                fn=run_comparison,
                inputs=[prompt_input, max_tokens_input],
                outputs=[
                    baseline_output,
                    speculative_output,
                    speedup_output,
                    acceptance_output,
                    disclaimer_output,
                ],
            )

        # -- Tab 2: Configuration -------------------------------------------
        with gr.Tab("Configuration"):
            config_output = gr.Textbox(label="Current Configuration", lines=12)
            refresh_config_btn = gr.Button("Refresh Configuration")
            refresh_config_btn.click(fn=load_config, outputs=[config_output])

        # -- Tab 3: Statistics ----------------------------------------------
        with gr.Tab("Statistics"):
            stats_output = gr.Textbox(label="Runtime Statistics", lines=10)
            refresh_stats_btn = gr.Button("Refresh Statistics")
            refresh_stats_btn.click(fn=load_stats, outputs=[stats_output])

    return demo


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ui = create_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)

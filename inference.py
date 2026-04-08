"""
inference.py
============
Baseline inference script for the Data Cleaning Agent Environment.

MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your HuggingFace / API key.

Optional:
    ENV_BASE_URL   URL of the running environment server.
                   Default: http://localhost:7860

STDOUT FORMAT (mandatory — evaluated by judges):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
    export HF_TOKEN="hf_..."
    export ENV_BASE_URL="http://localhost:7860"
    python inference.py
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME: str = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:7860")

if not API_KEY:
    print("[ERROR] No API key found. Set HF_TOKEN or API_KEY environment variable.", flush=True)
    print("  export HF_TOKEN='hf_...'", flush=True)
    sys.exit(1)

# Episode settings
MAX_STEPS: int = 5
TEMPERATURE: float = 0.1
MAX_TOKENS: int = 2048
SUCCESS_SCORE_THRESHOLD: float = 0.5  # score >= 0.5 counts as success

# Benchmark metadata
BENCHMARK: str = "data-cleaning-env"
TASK_NAMES: Dict[int, str] = {
    1: "column-cleaner",
    2: "data-sanitizer",
    3: "data-reconstructor",
}

# ─────────────────────────────────────────────────────────────────────────────
# Mandatory stdout logging functions
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line — one per episode."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    """Emit [STEP] line — once per step immediately after env.step() returns."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Keep action on single line — strip newlines
    action_clean = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    """Emit [END] line — always emitted, even on exception."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT: str = textwrap.dedent("""
You are an expert data cleaning engineer.

You will receive:
1. A task description explaining exactly what cleaning rules to apply
2. A schema hint showing expected column names and types
3. The dirty dataset (a JSON array of objects)
4. Feedback from your previous submission (if any)

Your job: return the fully cleaned dataset.

STRICT OUTPUT RULES:
- Return ONLY a valid JSON array of objects. Nothing else.
- No markdown. No backticks. No explanations. No "Here is the cleaned data:".
- Every element must be a JSON object (one row of the table).
- Follow the schema_hint EXACTLY — use the exact column names and types specified.
- Apply EVERY rule in the task_description. Do not skip any rule.
- If the task says to return a float, return a float (e.g., 45.0 not "45.00").
- If the task says to return a boolean, return true or false (not "true"/"false").
- If the task says to return an integer, return an integer (e.g., 200 not "200").

Example of correct output format:
[
  {"column_a": "value", "column_b": 42, "column_c": true},
  {"column_a": "other", "column_b": 15, "column_c": false}
]

Start your response with [ and end with ]. Nothing before or after.
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Environment HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def env_health_check() -> bool:
    """Returns True if the environment server is reachable and healthy."""
    try:
        resp = requests.get(f"{ENV_BASE_URL}/health", timeout=10)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def env_reset(task_id: int) -> Dict[str, Any]:
    """Call POST /reset on the environment."""
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        params={"task_id": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(task_id: int, cleaned_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Call POST /step with the agent's cleaned dataset."""
    payload = {
        "task_id": task_id,
        "cleaned_data": cleaned_data,
        "metadata": {},
    }
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# LLM agent
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(
    client: OpenAI,
    observation: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """
    Ask the LLM to produce a cleaned version of the dirty dataset.
    Returns parsed list of row dicts, or None if response could not be parsed.
    """
    user_message = (
        f"TASK DESCRIPTION:\n{observation['task_description']}\n\n"
        f"EXPECTED SCHEMA (use these exact column names and types):\n"
        f"{json.dumps(observation['schema_hint'], indent=2)}\n\n"
        f"DIRTY DATA TO CLEAN:\n"
        f"{json.dumps(observation['dirty_data'], indent=2)}\n\n"
        f"PREVIOUS FEEDBACK (from your last submission):\n"
        f"{observation.get('feedback', 'None — this is your first attempt.')}\n\n"
        f"PREVIOUS SCORE BREAKDOWN:\n"
        f"{json.dumps(observation.get('score_breakdown', {}), indent=2)}\n\n"
        f"Now return the cleaned dataset as a JSON array. "
        f"Start immediately with [ — no preamble."
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM API call failed: {exc}", flush=True)
        return None

    # Strip markdown code fences if model added them despite instructions
    if raw.startswith("```"):
        lines = raw.splitlines()
        start = 1
        end = len(lines) - 1
        raw = "\n".join(lines[start:end])

    # Find the JSON array
    start_idx = raw.find("[")
    end_idx = raw.rfind("]")
    if start_idx == -1 or end_idx == -1:
        print(f"[DEBUG] No JSON array found in LLM response.", flush=True)
        return None
    raw = raw[start_idx: end_idx + 1]

    try:
        cleaned = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[DEBUG] JSON parse error: {exc}", flush=True)
        return None

    if not isinstance(cleaned, list) or len(cleaned) == 0:
        print(f"[DEBUG] LLM returned empty or non-list response.", flush=True)
        return None

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner — one task
# ─────────────────────────────────────────────────────────────────────────────

async def run_episode(client: OpenAI, task_id: int) -> float:
    """
    Run one full episode for a given task.

    Emits [START], [STEP]x N, [END] in mandatory format.
    Returns best raw score achieved (0.0–1.0).
    """
    task_name = TASK_NAMES[task_id]
    rewards: List[float] = []
    steps_taken: int = 0
    best_score: float = 0.0
    success: bool = False

    # Emit [START]
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset episode
        result = env_reset(task_id=task_id)
        observation = result["observation"]
        done: bool = result.get("done", False)

        if done:
            print(f"[DEBUG] Environment returned done=True on reset. Skipping.", flush=True)
            return 0.0

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Ask LLM to clean the data
            cleaned_data = call_llm(client, observation)

            if cleaned_data is None:
                # LLM failed — submit empty list as fallback
                cleaned_data = []
                action_str = "[]"
            else:
                # Compact single-line action string for logging
                action_str = json.dumps(cleaned_data, separators=(",", ":"))

            # Submit to environment
            result = env_step(task_id=task_id, cleaned_data=cleaned_data)
            observation = result["observation"]
            reward: float = float(result.get("reward", 0.0))
            done = result.get("done", False)
            info = result.get("info", {})
            raw_score: float = float(info.get("raw_score", reward))

            rewards.append(reward)
            steps_taken = step

            if raw_score > best_score:
                best_score = raw_score

            # Emit [STEP] — mandatory format
            error_val: Optional[str] = None  # no last_action_error in this env
            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error_val,
            )

            if done:
                break

        # Compute normalized score and success
        score: float = min(max(best_score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        score = min(max(best_score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        # Emit [END] — always, even on exception
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return best_score


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("=" * 62, flush=True)
    print("  Data Cleaning Agent — Baseline Inference", flush=True)
    print("=" * 62, flush=True)
    print(f"  Model   : {MODEL_NAME}", flush=True)
    print(f"  API URL : {API_BASE_URL}", flush=True)
    print(f"  Env URL : {ENV_BASE_URL}", flush=True)
    print("=" * 62, flush=True)

    # Verify environment is running
    print("\nChecking environment health...", flush=True)
    if not env_health_check():
        print(f"\n[ERROR] Cannot reach environment at {ENV_BASE_URL}", flush=True)
        print("Make sure the server is running:", flush=True)
        print("  uv run server", flush=True)
        print("  OR", flush=True)
        print("  docker run -p 7860:7860 data-cleaning-env", flush=True)
        sys.exit(1)
    print("  Environment is healthy ✓", flush=True)

    # Create LLM client
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Run all three tasks — each gets its own [START]..[END] block
    scores: Dict[int, float] = {}
    for task_id in [1, 2, 3]:
        print(f"\n{'─' * 62}", flush=True)
        score = await run_episode(client, task_id)
        scores[task_id] = score

    # Final summary
    print(f"\n\n{'=' * 62}", flush=True)
    print("  FINAL BASELINE SCORES", flush=True)
    print("=" * 62, flush=True)
    task_labels = {
        1: "Column Cleaner      (Easy)",
        2: "Data Sanitizer    (Medium)",
        3: "Data Reconstructor  (Hard)",
    }
    for task_id, score in scores.items():
        bar = "█" * int(score * 20)
        spaces = " " * (20 - int(score * 20))
        print(f"  Task {task_id} — {task_labels[task_id]}  {score:.4f}  [{bar}{spaces}]", flush=True)

    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average score : {avg:.4f}", flush=True)
    print("=" * 62, flush=True)

    # Validate all scores
    for tid, s in scores.items():
        assert 0.0 <= s <= 1.0, f"Task {tid} score {s} is out of range [0.0, 1.0]!"

    print("\n  All scores in valid range [0.0, 1.0] ✓", flush=True)
    print("  Ready to submit!\n", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user", flush=True)
    except Exception as e:
        print(f"[FATAL] Unhandled exception in main: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
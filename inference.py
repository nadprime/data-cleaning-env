"""
inference.py
============
Baseline inference script for the Data Cleaning Agent Environment.

MANDATORY environment variables (set these before running):
    API_BASE_URL  : LLM API endpoint
    MODEL_NAME    : Model identifier
    HF_TOKEN      : Your HuggingFace token (used as API key)

Optional:
    ENV_BASE_URL  : URL of the running environment server
                    Default: http://localhost:7860

Usage:
    python inference.py

The script:
  1. Connects to the environment server
  2. Runs one episode per task (tasks 1, 2, 3)
  3. On each step: shows dirty data + instructions to the LLM, submits answer
  4. Prints final scores for all three tasks
"""
import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — read from environment variables
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
if not API_KEY:
    print("[ERROR] No API key found. Set HF_TOKEN or API_KEY as an environment variable.")
    print("  export HF_TOKEN='hf_...'")
    sys.exit(1)
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS: int = 5
TEMPERATURE: float = 0.1   # low temperature → more deterministic output
MAX_TOKENS: int = 2048      # enough tokens for a 10-row dataset as JSON

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — tells the LLM its role and output format
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT: str = """You are an expert data cleaning engineer.

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

Start your response with [ and end with ]. Nothing before or after."""

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
    """
    Call POST /reset on the environment.
    Returns the full response dict (observation, reward, done, info).
    """
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        params={"task_id": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(task_id: int, cleaned_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Call POST /step with the agent's cleaned dataset.
    Returns the full response dict (observation, reward, done, info).
    """
    payload = {
        "task_id": task_id,
        "cleaned_data": cleaned_data,
        "metadata": {},
    }
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json=payload,
        timeout=60,  # longer timeout — grader + LLM call
    )
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# LLM agent
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(
    client: OpenAI,
    observation: Dict[str, Any],
    step_num: int,
) -> Optional[List[Dict[str, Any]]]:
    """
    Ask the LLM to produce a cleaned version of the dirty dataset.

    Args:
        client     : OpenAI client instance
        observation: the current observation dict from the environment
        step_num   : current step (used for logging)

    Returns:
        Parsed list of row dicts, or None if the LLM response could not be parsed.
    """
    # Build the user message from the observation
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
                {"role": "user",   "content": user_message},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"      [LLM ERROR] API call failed: {exc}")
        return None

    # Strip markdown code fences if the model added them despite instructions
    if raw.startswith("```"):
        lines = raw.splitlines()
        # Find the opening fence and closing fence
        start = 1
        end = len(lines) - 1
        if lines[0].startswith("```json"):
            start = 1
        raw = "\n".join(lines[start:end])

    # Find the JSON array (in case model adds text before/after)
    start_idx = raw.find("[")
    end_idx = raw.rfind("]")
    if start_idx == -1 or end_idx == -1:
        print(f"      [PARSE ERROR] No JSON array found in response.")
        print(f"      Response preview: {raw[:200]!r}")
        return None
    raw = raw[start_idx : end_idx + 1]

    try:
        cleaned = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"      [JSON ERROR] {exc}")
        print(f"      Raw (first 300 chars): {raw[:300]!r}")
        return None

    if not isinstance(cleaned, list):
        print(f"      [TYPE ERROR] Expected list, got {type(cleaned).__name__}")
        return None

    if len(cleaned) == 0:
        print(f"      [EMPTY ERROR] LLM returned empty list.")
        return None

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(client: OpenAI, task_id: int) -> float:
    """
    Run one full episode on a given task.

    The agent keeps submitting until done=True or max_steps.
    Uses the feedback from each step to improve the next submission.

    Returns:
        best raw score achieved (0.0–1.0)
    """
    print(f"\n  ── Task {task_id} ──────────────────────────────────────────")

    result = env_reset(task_id=task_id)
    observation = result["observation"]
    best_score: float = 0.0
    done: bool = result.get("done", False)

    if done:
        print("  [WARNING] Environment returned done=True on reset. Skipping.")
        return 0.0

    for step in range(1, MAX_STEPS + 1):
        print(f"\n  Step {step}/{MAX_STEPS}")
        print(f"    Asking {MODEL_NAME} to clean the data...")

        cleaned_data = call_llm(client, observation, step)

        if cleaned_data is None:
            print(f"    LLM produced no parseable output. Skipping step.")
            continue

        print(f"    LLM returned {len(cleaned_data)} rows. Submitting...")

        result = env_step(task_id=task_id, cleaned_data=cleaned_data)
        observation = result["observation"]
        reward = result["reward"]
        done = result["done"]
        info = result.get("info", {})

        raw_score = info.get("raw_score", 0.0)
        if raw_score > best_score:
            best_score = raw_score

        print(f"    reward={reward:.4f}  raw_score={raw_score:.4f}  best_so_far={best_score:.4f}")
        print(f"    feedback: {observation['feedback'][:150]}")

        if observation.get("score_breakdown"):
            bd = observation["score_breakdown"]
            print(f"    breakdown: {bd}")

        if done:
            if raw_score >= 0.99:
                print(f"    ✓ Perfect score achieved at step {step}!")
            else:
                print(f"    Episode ended (max steps reached).")
            break

    return best_score


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 62)
    print("  Data Cleaning Agent — Baseline Inference")
    print("=" * 62)
    print(f"  Model      : {MODEL_NAME}")
    print(f"  API URL    : {API_BASE_URL}")
    print(f"  Env URL    : {ENV_BASE_URL}")
    print("=" * 62)

    # Step 1: Verify environment is running
    print("\nChecking environment health...")
    if not env_health_check():
        print(f"\n[ERROR] Cannot reach environment at {ENV_BASE_URL}")
        print("Make sure the server is running:")
        print("  docker run -p 7860:7860 data-cleaning-env")
        print("  OR")
        print("  PYTHONPATH=. uvicorn server.app:app --port 7860")
        sys.exit(1)
    print("  Environment is healthy ✓")

    # Step 2: Create LLM client
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    task_names = {
        1: "Column Cleaner      (Easy)",
        2: "Data Sanitizer    (Medium)",
        3: "Data Reconstructor  (Hard)",
    }

    # Step 3: Run all three tasks
    scores: Dict[int, float] = {}
    for task_id in [1, 2, 3]:
        score = run_episode(client, task_id)
        scores[task_id] = score

    # Step 4: Print results
    print("\n\n" + "=" * 62)
    print("  FINAL BASELINE SCORES")
    print("=" * 62)
    for task_id, score in scores.items():
        bar = "█" * int(score * 20)
        spaces = " " * (20 - int(score * 20))
        print(f"  Task {task_id} — {task_names[task_id]}  {score:.4f}  [{bar}{spaces}]")

    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average score : {avg:.4f}")
    print("=" * 62)

    # Step 5: Validate all scores are in range [0, 1]
    for tid, s in scores.items():
        assert 0.0 <= s <= 1.0, f"Task {tid} score {s} is out of range [0.0, 1.0]!"

    print("\n  All scores in valid range [0.0, 1.0] ✓")
    print("  Ready to submit!\n")


if __name__ == "__main__":
    main()
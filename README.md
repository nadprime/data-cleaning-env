---
title: Data Cleaning Agent Environment
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - data-cleaning
  - reinforcement-learning
license: mit
---

# 🧹 Data Cleaning Agent Environment

An **OpenEnv-compliant** environment where AI agents learn to clean messy tabular datasets.
This simulates a real-world task that data engineers perform daily — and costs companies
thousands of hours of manual work each year.

## Why This Environment Exists

Every company that uses data has dirty data. Data engineers spend 60–80% of their time
cleaning it before it can be used. This environment trains agents to automate:
- Column name normalization
- Missing value handling
- Deduplication
- Type casting (strings → float/int/bool)
- Outlier detection and capping
- Category normalization and typo fixing

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | int (1–3) | Which task to solve |
| `cleaned_data` | list[dict] | The agent's cleaned rows |
| `metadata` | dict | Optional extra info |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | int | Current task |
| `task_description` | str | Instructions in plain English |
| `dirty_data` | list[dict] | The raw, messy input data |
| `schema_hint` | dict | Expected column names and types |
| `feedback` | str | Grader feedback on last submission |
| `score_breakdown` | dict | Per-metric partial scores |
| `reward` | float | Reward from last action |
| `done` | bool | Whether episode is finished |

## Tasks

| Task | Difficulty | What to Fix | Expected Score (LLM baseline) |
|------|-----------|-------------|-------------------------------|
| 1 | Easy | Column names: lowercase, underscores, strip spaces | ~0.95 |
| 2 | Medium | Missing values, duplicates, blank strings | ~0.75 |
| 3 | Hard | Types, outliers, category typos, booleans | ~0.55 |

## Reward Function

reward = raw_score + improvement_bonus - step_penalty
where improvement_bonus = 0.1 if score > previous_best else 0.0
step_penalty      = 0.05 × max(0, step_count - 1)
clamped to [0.0, 1.0]

Dense reward: partial progress is rewarded on every step.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check → `{"status": "healthy"}` |
| `/reset` | POST | Start episode → `?task_id=1` |
| `/step` | POST | Submit cleaned data |
| `/state` | GET | Episode metadata |
| `/docs` | GET | Interactive API documentation |

## Setup

### Run with Docker (recommended)
```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

Then open http://localhost:7860/docs for the interactive API.

### Run locally (for development)
```bash
pip install fastapi uvicorn pydantic
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Run the inference script
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_BASE_URL="http://localhost:7860"
python inference.py
```

## Baseline Scores

These scores were measured using Llama-3.3-70B-Instruct via HuggingFace Router:

| Task | Raw Score | Notes |
|------|-----------|-------|
| Task 1 (Easy) | ~0.95 | LLM handles column renaming well |
| Task 2 (Medium) | ~0.75 | Struggles with exact fill values |
| Task 3 (Hard) | ~0.55 | Typo correction is challenging |
| **Average** | **~0.75** | |
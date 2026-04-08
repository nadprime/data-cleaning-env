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

---

## Why This Environment Matters

Data engineers spend **60–80% of their time cleaning data** before it can be used for
analytics or ML. This is not a toy problem — it costs companies millions annually in
engineering hours.

This environment trains agents to automate the exact operations a junior data engineer
performs daily:

| Operation | Real-World Frequency | Task in This Env |
|---|---|---|
| Column name normalization | Every ETL pipeline | Task 1 |
| Missing value imputation | Every real dataset | Task 2 |
| Deduplication | Every data warehouse load | Task 2 |
| Type casting from raw strings | Every API ingestion | Task 3 |
| Outlier detection and capping | Every ML feature pipeline | Task 3 |
| Category normalization and typo fixing | Every CRM/ERP integration | Task 3 |

An agent scoring above 0.85 average across all tasks could replace a significant
portion of manual data cleaning work in production ETL pipelines.

---

## Tasks

| Task | Difficulty | What to Fix | Expected Score (LLM baseline) |
|------|-----------|-------------|-------------------------------|
| 1 | Easy | Column names: lowercase, underscores, strip spaces | ~0.95 |
| 2 | Medium | Missing values, duplicates, blank strings | ~0.75 |
| 3 | Hard | Types, outliers, category typos, booleans | ~0.55 |

### Task 1 — Column Cleaner (Easy)

Fix malformed column names in a tabular dataset. Apply three rules to every column:
- Convert to lowercase
- Replace spaces with underscores
- Strip leading and trailing whitespace

Do not change any data values — only fix the column names.

**Example:**

| Before | After |
|---|---|
| `First Name` | `first_name` |
| `Last  Name` | `last_name` |
| ` Age ` | `age` |
| `Email Address` | `email_address` |

---

### Task 2 — Data Sanitizer (Medium)

Handle missing values and remove duplicate rows from a dataset. Apply these rules in order:
- Remove exact duplicate rows (all columns must match)
- Fill missing `age` values with `'30'` (the median)
- Fill missing `city` values with `'Unknown'`
- Fill missing `score` values with `'82'` (the median)
- Fill blank or empty `name` values with `'Unknown'`

The final output must have exactly 6 rows (one duplicate is present in the input).

---

### Task 3 — Data Reconstructor (Hard)

Fully reconstruct a dataset with multiple data quality issues across every column:
- `product`: convert to lowercase
- `price`: strip `$` and commas, convert to float
- `category`: convert to lowercase, fix typos (`ELECTRONIC` → `electronics`, `electonics` → `electronics`)
- `stock`: convert to integer, cap values above 500 to the median value 200
- `rating`: convert to float, cap values above 5.0 to exactly 5.0
- `in_stock`: convert to boolean (`yes/YES/1/true` → `True`, `no/false/0` → `False`)

---

## Reward Function

```
reward = raw_score
       + 0.1  × (1 if score > previous_best else 0)
       - 0.05 × max(0, step_count - 1)
       clamped to [0.0, 1.0]
```

| Component | Purpose |
|---|---|
| `raw_score` | Dense partial progress signal — every improvement is rewarded |
| `improvement_bonus` | Encourages the agent to keep trying when it is making progress |
| `step_penalty` | Rewards efficiency — a perfect first attempt scores higher than a perfect fifth |
| `clamped` | Reward is clamped to `[0.0, 1.0]` — agent always has incentive to improve |

Grader raw scores are additionally clamped to the strict interval `(0.01, 0.99)`
to satisfy validator requirements that scores stay strictly between 0 and 1.

---

## Action Space

| Field | Type | Description |
|---|---|---|
| `task_id` | int (1–3) | Which task to solve |
| `cleaned_data` | list[dict] | The agent's cleaned rows as a list of objects |
| `metadata` | dict | Optional extra information |

### Example Action

```json
{
  "task_id": 1,
  "cleaned_data": [
    {"first_name": "Alice", "last_name": "Smith", "age": "30", "email_address": "alice@example.com"},
    {"first_name": "Bob",   "last_name": "Jones", "age": "25", "email_address": "bob@example.com"}
  ],
  "metadata": {}
}
```

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | int | Current task |
| `task_description` | str | Plain-English cleaning instructions |
| `dirty_data` | list[dict] | The original messy input — always shown, never modified |
| `schema_hint` | dict | Expected column names and types |
| `feedback` | str | Grader feedback on the last submission |
| `score_breakdown` | dict | Per-metric partial scores |
| `reward` | float | Reward from the last action |
| `done` | bool | Whether the episode has finished |
| `step_count` | int | How many steps have been taken this episode |

### Example Observation

```json
{
  "task_id": 1,
  "task_description": "Fix the column names...",
  "dirty_data": [
    {"First Name": "Alice", "Last  Name": "Smith", " Age ": "30", "Email Address": "alice@example.com"}
  ],
  "schema_hint": {
    "first_name": "string",
    "last_name": "string",
    "age": "string",
    "email_address": "string"
  },
  "feedback": "All 4 column names correct! | Data values: 20/20 cells correct.",
  "score_breakdown": {"column_names": 1.0, "data_values": 1.0},
  "reward": 1.0,
  "done": true,
  "step_count": 1
}
```

---

## Episode Lifecycle

```
POST /reset?task_id=1  →  receive initial observation with dirty data
POST /step             →  submit cleaned data, receive reward + feedback
                       →  repeat until done=True
```

Episodes end when:
- The agent achieves a perfect score (raw_score >= 0.99), or
- The agent reaches the maximum of 5 steps

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check → `{"status": "healthy"}` |
| `/reset` | POST | Start a new episode → `?task_id=1` |
| `/step` | POST | Submit cleaned data, receive reward |
| `/state` | GET | Episode metadata |
| `/schema` | GET | JSON schemas for action/observation/state (validator-required) |
| `/metadata` | GET | Environment metadata (validator-required) |
| `/openapi.json` | GET | OpenAPI spec (validator-required) |
| `/docs` | GET | Interactive API documentation |
| `/redoc` | GET | ReDoc API documentation |

---

## Grader Design

Each task has a deterministic programmatic grader that produces scores strictly between 0 and 1.
The same input always produces the same score — no randomness.

Note: exact boundary values `0.0` and `1.0` are intentionally avoided; scores are
clamped into `(0.01, 0.99)`.

### Task 1 Grader

| Metric | Weight | What It Checks |
|---|---|---|
| Column names | 50% | Proportion of 4 expected columns present |
| Data values | 50% | Cell-level accuracy across all rows |

### Task 2 Grader

| Metric | Weight | What It Checks |
|---|---|---|
| Row count | 25% | Output has exactly 6 rows (penalty per wrong row) |
| Cell accuracy | 75% | Cell-level accuracy across all expected values |

### Task 3 Grader

| Metric | Weight | What It Checks |
|---|---|---|
| product | 15% | Lowercase string match |
| price | 20% | Float within 0.01 of expected |
| category | 20% | Lowercase match, typos fixed |
| stock | 20% | Integer exact match, outliers capped |
| rating | 15% | Float within 0.05, capped at 5.0 |
| in_stock | 10% | Boolean exact match |

---

## Baseline Scores

Measured using `meta-llama/Llama-3.3-70B-Instruct` via HuggingFace Router:

| Task | Raw Score | Notes |
|---|---|---|
| Task 1 — Easy | ~0.95 | LLM handles column renaming well |
| Task 2 — Medium | ~0.75 | Struggles with exact fill values |
| Task 3 — Hard | ~0.55 | Typo correction and type casting are challenging |
| **Average** | **~0.75** | |

---

## Setup and Usage

### Run With Docker (Recommended)

```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

Then open http://localhost:7860/docs for the interactive API.

### Run Locally With uv

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Start the server
uv run server
```

### Run Locally With pip

```bash
pip install fastapi uvicorn pydantic openenv-core python-multipart
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

---

## Run the Inference Script

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

The script runs one episode per task (Tasks 1, 2, 3) and prints final scores.
Runtime is under 5 minutes on a standard machine.

---

## Run the Test Suite

```bash
uv run pytest tests/ -v
```

All tests must pass before deployment.

---

## Project Structure

```
data-cleaning-env/
├── Dockerfile              ← Docker build instructions
├── requirements.txt        ← Python packages
├── pyproject.toml          ← Project config and uv dependencies
├── openenv.yaml            ← OpenEnv metadata
├── README.md               ← This file
├── models.py               ← Typed data contracts (Action, Observation, State)
├── inference.py            ← LLM agent baseline script
├── run_server.py           ← Server entry point for uv
├── deploy_to_hf.py         ← HuggingFace deployment helper
├── uv.lock                 ← Resolved dependency lockfile
├── tests/
│   ├── __init__.py
│   └── test_environment.py ← Full test suite
└── server/
    ├── __init__.py
    ├── environment.py      ← Game logic, tasks, and graders
    └── app.py              ← FastAPI HTTP server
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | LLM API endpoint |
| `MODEL_NAME` | Yes | Model identifier |
| `HF_TOKEN` | Yes | HuggingFace API token |
| `ENV_BASE_URL` | No | Environment server URL (default: http://localhost:7860) |

Never hardcode these values. Always load from environment variables or a `.env` file.

---

## Troubleshooting

**Server fails with `ModuleNotFoundError: No module named 'models'`**
```bash
# Always run with PYTHONPATH set
PYTHONPATH=. uvicorn server.app:app --port 7860
# Or use uv which handles this automatically
uv run server
```

**Docker daemon not running**
```bash
sudo service docker start
docker ps  # verify
```

**Port 7860 already in use**
```bash
lsof -i :7860       # find what's using it
kill -9 <PID>       # stop it
```

**HuggingFace token invalid**
```bash
curl -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami
# Should print your username — if not, generate a new token
```
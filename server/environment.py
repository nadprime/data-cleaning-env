"""
server/environment.py
=====================
The Data Cleaning Agent Environment.

Three tasks of increasing difficulty:
  Task 1 (Easy)   — Fix malformed column names
  Task 2 (Medium) — Handle missing values and remove duplicates
  Task 3 (Hard)   — Full data reconstruction: types, outliers, normalization

Each task has:
  - dirty_data   : the messy input
  - clean_data   : the ground truth (what a perfect submission looks like)
  - description  : plain-English instructions for the agent
  - schema_hint  : column names and expected types

Graders are deterministic: same input always gives same score.
Scores range from 0.0 (completely wrong) to 1.0 (perfect).
"""
import copy
import uuid
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Easy: Fix column names
# Rules: lowercase, replace spaces with underscores, strip whitespace
# ─────────────────────────────────────────────────────────────────────────────

TASK_1_DIRTY: List[Dict[str, Any]] = [
    {"First Name": "Alice",  "Last  Name": "Smith",  " Age ": "30", "Email Address": "alice@example.com"},
    {"First Name": "Bob",    "Last  Name": "Jones",  " Age ": "25", "Email Address": "bob@example.com"},
    {"First Name": "Carol",  "Last  Name": "White",  " Age ": "35", "Email Address": "carol@example.com"},
    {"First Name": "David",  "Last  Name": "Brown",  " Age ": "28", "Email Address": "david@example.com"},
    {"First Name": "Eve",    "Last  Name": "Davis",  " Age ": "22", "Email Address": "eve@example.com"},
]

TASK_1_CLEAN: List[Dict[str, Any]] = [
    {"first_name": "Alice",  "last_name": "Smith",  "age": "30", "email_address": "alice@example.com"},
    {"first_name": "Bob",    "last_name": "Jones",  "age": "25", "email_address": "bob@example.com"},
    {"first_name": "Carol",  "last_name": "White",  "age": "35", "email_address": "carol@example.com"},
    {"first_name": "David",  "last_name": "Brown",  "age": "28", "email_address": "david@example.com"},
    {"first_name": "Eve",    "last_name": "Davis",  "age": "22", "email_address": "eve@example.com"},
]

TASK_1_DESCRIPTION: str = (
    "Fix the column names in this dataset. "
    "Apply these rules to EVERY column name: "
    "(1) convert the column name to lowercase, "
    "(2) replace spaces with underscores, "
    "(3) strip leading and trailing whitespace. "
    "Do NOT change any of the data values — only fix the column names. "
    "The correct output columns are: first_name, last_name, age, email_address."
)

TASK_1_SCHEMA: Dict[str, str] = {
    "first_name": "string",
    "last_name": "string",
    "age": "string",
    "email_address": "string",
}

# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Medium: Missing values and deduplication
# Rules: remove duplicates, fill nulls, fill blank strings
# ─────────────────────────────────────────────────────────────────────────────

TASK_2_DIRTY: List[Dict[str, Any]] = [
    {"name": "Alice",  "age": None,  "city": "New York",    "score": "85"},
    {"name": "Bob",    "age": "25",  "city": None,           "score": "90"},
    {"name": "Carol",  "age": "35",  "city": "Chicago",      "score": None},
    {"name": "Bob",    "age": "25",  "city": None,           "score": "90"},  # exact duplicate of row 2
    {"name": "David",  "age": "28",  "city": "Houston",      "score": "78"},
    {"name": "",       "age": "22",  "city": "Phoenix",      "score": "65"},  # blank name
    {"name": "Eve",    "age": "31",  "city": "Philadelphia", "score": "92"},
]

TASK_2_CLEAN: List[Dict[str, Any]] = [
    {"name": "Alice",   "age": "30",  "city": "New York",    "score": "85"},  # age filled (median=30)
    {"name": "Bob",     "age": "25",  "city": "Unknown",     "score": "90"},  # city filled
    {"name": "Carol",   "age": "35",  "city": "Chicago",     "score": "82"},  # score filled (median=82)
    {"name": "David",   "age": "28",  "city": "Houston",     "score": "78"},  # duplicate removed (was row 4)
    {"name": "Unknown", "age": "22",  "city": "Phoenix",     "score": "65"},  # blank name filled
    {"name": "Eve",     "age": "31",  "city": "Philadelphia","score": "92"},
]

TASK_2_DESCRIPTION: str = (
    "Clean this dataset by applying these rules in order: "
    "(1) Remove exact duplicate rows (a row is duplicate if ALL columns match). "
    "(2) Fill missing (None/null) 'age' values with the string '30' (the median age). "
    "(3) Fill missing (None/null) 'city' values with the string 'Unknown'. "
    "(4) Fill missing (None/null) 'score' values with the string '82' (the median score). "
    "(5) Fill blank or empty string 'name' values with the string 'Unknown'. "
    "The final output must have exactly 6 rows (one duplicate was removed)."
)

TASK_2_SCHEMA: Dict[str, str] = {
    "name": "string",
    "age": "string (fill nulls with '30')",
    "city": "string (fill nulls with 'Unknown')",
    "score": "string (fill nulls with '82')",
}

# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Hard: Full data reconstruction
# Rules: type casting, outlier capping, category normalization, bool parsing
# ─────────────────────────────────────────────────────────────────────────────

TASK_3_DIRTY: List[Dict[str, Any]] = [
    {"product": "laptop",   "price": "$1,200.00", "category": "ELECTRONICS", "stock": "150",  "rating": "4.5",  "in_stock": "yes"},
    {"product": "MOUSE",    "price": "$25.99",    "category": "electronics",  "stock": "500",  "rating": "4.2",  "in_stock": "YES"},
    {"product": "Keyboard", "price": "45.00",     "category": "Electronics",  "stock": "9999", "rating": "4.8",  "in_stock": "1"},
    {"product": "monitor",  "price": "$350",      "category": "ELECTRONIC",   "stock": "75",   "rating": "4.7",  "in_stock": "true"},
    {"product": "Webcam",   "price": "$89.95",    "category": "electonics",   "stock": "200",  "rating": "3.9",  "in_stock": "no"},
    {"product": "headset",  "price": "$120.00",   "category": "Electronics",  "stock": "300",  "rating": "6.0",  "in_stock": "false"},
    {"product": "USB Hub",  "price": "$35.00",    "category": "accessories",  "stock": "400",  "rating": "4.1",  "in_stock": "0"},
]

TASK_3_CLEAN: List[Dict[str, Any]] = [
    {"product": "laptop",   "price": 1200.00, "category": "electronics", "stock": 150,  "rating": 4.5, "in_stock": True},
    {"product": "mouse",    "price": 25.99,   "category": "electronics", "stock": 500,  "rating": 4.2, "in_stock": True},
    {"product": "keyboard", "price": 45.00,   "category": "electronics", "stock": 200,  "rating": 4.8, "in_stock": True},   # 9999 outlier → capped to median 200
    {"product": "monitor",  "price": 350.00,  "category": "electronics", "stock": 75,   "rating": 4.7, "in_stock": True},   # ELECTRONIC typo fixed
    {"product": "webcam",   "price": 89.95,   "category": "electronics", "stock": 200,  "rating": 3.9, "in_stock": False},  # electonics typo fixed
    {"product": "headset",  "price": 120.00,  "category": "electronics", "stock": 300,  "rating": 5.0, "in_stock": False},  # rating 6.0 → capped to 5.0
    {"product": "usb hub",  "price": 35.00,   "category": "accessories", "stock": 400,  "rating": 4.1, "in_stock": False},
]

TASK_3_DESCRIPTION: str = (
    "This dataset has many data quality issues. Fix ALL of them precisely: "
    "(1) 'product': convert to lowercase. "
    "(2) 'price': parse to float — remove dollar signs ($) and commas, then convert to float. "
    "(3) 'category': convert to lowercase AND fix typos: "
    "    'ELECTRONIC' → 'electronics', 'electonics' → 'electronics'. "
    "    'accessories' stays as 'accessories'. "
    "(4) 'stock': convert to integer. Cap any value above 500 to the median value 200. "
    "(5) 'rating': convert to float. Cap any value above 5.0 to exactly 5.0. "
    "(6) 'in_stock': convert to boolean. "
    "    True values: 'yes', 'YES', '1', 'true'. "
    "    False values: 'no', 'false', '0'. "
    "Output the 'price' field as a float, 'stock' as an integer, "
    "'rating' as a float, 'in_stock' as a boolean (not a string)."
)

TASK_3_SCHEMA: Dict[str, str] = {
    "product": "string (lowercase)",
    "price": "float (parsed from string, remove $ and commas)",
    "category": "string (lowercase, typos corrected)",
    "stock": "int (outliers above 500 → set to 200)",
    "rating": "float (max 5.0)",
    "in_stock": "boolean (True or False, not a string)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Registry — maps task IDs to their data
# ─────────────────────────────────────────────────────────────────────────────

TASKS: Dict[int, Dict[str, Any]] = {
    1: {
        "dirty": TASK_1_DIRTY,
        "clean": TASK_1_CLEAN,
        "description": TASK_1_DESCRIPTION,
        "schema": TASK_1_SCHEMA,
    },
    2: {
        "dirty": TASK_2_DIRTY,
        "clean": TASK_2_CLEAN,
        "description": TASK_2_DESCRIPTION,
        "schema": TASK_2_SCHEMA,
    },
    3: {
        "dirty": TASK_3_DIRTY,
        "clean": TASK_3_CLEAN,
        "description": TASK_3_DESCRIPTION,
        "schema": TASK_3_SCHEMA,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_str(v: Any) -> str:
    """
    Normalize a value to a lowercase stripped string for comparison.
    Returns empty string for None.
    """
    if v is None:
        return ""
    return str(v).strip().lower()


def _clamp_score(score: float) -> float:
    """
    Clamp score to strictly open interval (0.01, 0.99).
    Validator requires scores strictly between 0 and 1 — not 0.0 or 1.0.
    """
    return round(max(0.01, min(0.99, score)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Grader: Task 1
# ─────────────────────────────────────────────────────────────────────────────

def grade_task_1(
    submitted: List[Dict[str, Any]],
) -> Tuple[float, str, Dict[str, float]]:
    """
    Grade Task 1 (column name fixing).

    Scoring breakdown:
      50% — correct column names (proportion of 4 expected columns present)
      50% — data values unchanged (cell-level accuracy)

    Returns:
        (total_score 0.0–1.0, feedback string, breakdown dict)
    """
    expected = TASK_1_CLEAN
    expected_cols = set(expected[0].keys())  # {first_name, last_name, age, email_address}

    if not submitted:
        return 0.01, "No data submitted.", {"column_names": 0.01, "data_values": 0.01}

    if not isinstance(submitted[0], dict):
        return 0.01, "Each row must be a dict.", {"column_names": 0.01, "data_values": 0.01}

    submitted_cols = set(submitted[0].keys())

    # Score 1: column names
    correct_cols = expected_cols & submitted_cols
    col_score = len(correct_cols) / len(expected_cols)

    # Score 2: data values — check only for matched columns
    rows_to_check = min(len(expected), len(submitted))
    total_cells = rows_to_check * len(expected_cols)
    correct_cells = 0

    for i in range(rows_to_check):
        exp_row = expected[i]
        sub_row = submitted[i]
        for col in expected_cols:
            if col in sub_row:
                if _normalize_str(sub_row[col]) == _normalize_str(exp_row[col]):
                    correct_cells += 1

    val_score = correct_cells / total_cells if total_cells > 0 else 0.0

    total = round(0.5 * col_score + 0.5 * val_score, 4)

    # Build feedback
    parts = []
    if col_score < 1.0:
        wrong = expected_cols - submitted_cols
        extra = submitted_cols - expected_cols
        if wrong:
            parts.append(f"Missing columns: {sorted(wrong)}")
        if extra:
            parts.append(f"Unexpected columns: {sorted(extra)}")
    else:
        parts.append("All 4 column names correct!")

    parts.append(f"Data values: {correct_cells}/{total_cells} cells correct.")

    breakdown = {
        "column_names": round(col_score, 4),
        "data_values": round(val_score, 4),
    }
    return _clamp_score(total), " | ".join(parts), breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Grader: Task 2
# ─────────────────────────────────────────────────────────────────────────────

def grade_task_2(
    submitted: List[Dict[str, Any]],
) -> Tuple[float, str, Dict[str, float]]:
    """
    Grade Task 2 (missing values + deduplication).

    Scoring breakdown:
      25% — correct row count (should be 6, not 7)
      75% — cell-level accuracy across all expected values

    Returns:
        (total_score 0.0–1.0, feedback string, breakdown dict)
    """
    expected = TASK_2_CLEAN

    if not submitted:
        return 0.01, "No data submitted.", {"row_count": 0.01, "cell_accuracy": 0.01}

    # Score 1: row count
    # Perfect = 6 rows. Penalty 0.2 per row off, minimum 0.
    expected_rows = len(expected)
    submitted_rows = len(submitted)
    row_diff = abs(submitted_rows - expected_rows)
    row_score = max(0.0, 1.0 - row_diff * 0.2)

    # Score 2: cell accuracy (compare row by row, column by column)
    cols = list(expected[0].keys())
    total_cells = expected_rows * len(cols)
    correct_cells = 0

    for i in range(min(expected_rows, submitted_rows)):
        exp_row = expected[i]
        sub_row = submitted[i]
        for col in cols:
            exp_val = _normalize_str(exp_row.get(col))
            sub_val = _normalize_str(sub_row.get(col))
            if sub_val == exp_val:
                correct_cells += 1

    cell_score = correct_cells / total_cells if total_cells > 0 else 0.0

    total = round(0.25 * row_score + 0.75 * cell_score, 4)

    feedback = (
        f"Rows: got {submitted_rows}, expected {expected_rows}. "
        f"Cell accuracy: {correct_cells}/{total_cells} ({cell_score:.0%})."
    )

    breakdown = {
        "row_count": round(row_score, 4),
        "cell_accuracy": round(cell_score, 4),
    }
    return _clamp_score(total), feedback, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Grader: Task 3
# ─────────────────────────────────────────────────────────────────────────────

def _to_bool_strict(v: Any) -> Optional[bool]:
    """
    Convert various truthy/falsy representations to Python bool.
    Returns None if unrecognized.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("yes", "true", "1"):
            return True
        if s in ("no", "false", "0"):
            return False
    return None


def grade_task_3(
    submitted: List[Dict[str, Any]],
) -> Tuple[float, str, Dict[str, float]]:
    """
    Grade Task 3 (full data reconstruction).

    Each column type is graded separately with its own weight:
      product  → 15%  (lowercase string match)
      price    → 20%  (float within 0.01)
      category → 20%  (lowercase string match, typos fixed)
      stock    → 20%  (integer exact match, outlier capped)
      rating   → 15%  (float within 0.05, capped at 5.0)
      in_stock → 10%  (boolean exact match)

    Returns:
        (total_score 0.0–1.0, feedback string, breakdown dict)
    """
    expected = TASK_3_CLEAN
    n = len(expected)

    weights: Dict[str, float] = {
        "product":  0.15,
        "price":    0.20,
        "category": 0.20,
        "stock":    0.20,
        "rating":   0.15,
        "in_stock": 0.10,
    }

    if not submitted or len(submitted) < n:
        msg = f"Expected {n} rows, got {len(submitted)}."
        return 0.01, msg, {k: 0.01 for k in weights}

    col_scores: Dict[str, float] = {}

    for col in weights:
        correct = 0
        for i in range(n):
            exp_val = expected[i][col]
            sub_val = submitted[i].get(col)

            if col == "in_stock":
                if _to_bool_strict(sub_val) == exp_val:
                    correct += 1

            elif col == "price":
                try:
                    if abs(float(sub_val) - float(exp_val)) < 0.01:
                        correct += 1
                except (ValueError, TypeError):
                    pass

            elif col == "stock":
                try:
                    if int(sub_val) == int(exp_val):
                        correct += 1
                except (ValueError, TypeError):
                    pass

            elif col == "rating":
                try:
                    if abs(float(sub_val) - float(exp_val)) < 0.05:
                        correct += 1
                except (ValueError, TypeError):
                    pass

            else:  # product, category (string columns)
                if str(sub_val).strip().lower() == str(exp_val).strip().lower():
                    correct += 1

        col_scores[col] = correct / n

    total = sum(weights[col] * col_scores[col] for col in weights)
    total = round(total, 4)

    parts = []
    for col, score in col_scores.items():
        status = "✓" if score >= 0.9 else ("~" if score >= 0.5 else "✗")
        parts.append(f"{col}={score:.0%}{status}")

    breakdown = {k: round(v, 4) for k, v in col_scores.items()}
    return _clamp_score(total), " | ".join(parts), breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Grader registry
# ─────────────────────────────────────────────────────────────────────────────

GRADERS = {
    1: grade_task_1,
    2: grade_task_2,
    3: grade_task_3,
}

MAX_STEPS = 5  # agent gets up to 5 attempts per episode


# ─────────────────────────────────────────────────────────────────────────────
# The Environment class
# ─────────────────────────────────────────────────────────────────────────────

class DataCleaningEnvironment:
    """
    OpenEnv-compliant Data Cleaning Agent Environment.

    Episode lifecycle:
      1. Call reset(task_id) → receives initial observation with dirty data
      2. Call step(action)   → submits cleaned data, receives reward + feedback
      3. Episode ends when done=True (perfect score OR max steps reached)

    Reward design:
      reward = raw_score + improvement_bonus - step_penalty
      - raw_score       : grader output (0.0–1.0)
      - improvement_bonus : +0.1 if score beats previous best (progress signal)
      - step_penalty    : -0.05 × max(0, step_count - 1) (efficiency incentive)
      - final reward is clamped to [0.0, 1.0]
    """

    def __init__(self) -> None:
        self._episode_id: str = ""
        self._task_id: int = 1
        self._step_count: int = 0
        self._best_score: float = 0.0
        self._done: bool = True  # must call reset() before step()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, task_id: int = 1) -> "DataCleaningObservation":
        """
        Start a new episode. Resets all state.

        Args:
            task_id: 1 (Easy), 2 (Medium), or 3 (Hard)

        Returns:
            Initial observation containing the dirty data and task description.

        Raises:
            ValueError: if task_id is not 1, 2, or 3
        """
        from models import DataCleaningObservation  # local import avoids circular issues

        if task_id not in TASKS:
            raise ValueError(f"task_id must be 1, 2, or 3. Received: {task_id!r}")

        self._episode_id = str(uuid.uuid4())[:8]
        self._task_id = task_id
        self._step_count = 0
        self._best_score = 0.0
        self._done = False

        task = TASKS[task_id]

        return DataCleaningObservation(
            task_id=task_id,
            task_description=task["description"],
            dirty_data=copy.deepcopy(task["dirty"]),
            schema_hint=task["schema"],
            step_count=0,
            done=False,
            reward=0.0,
            feedback="Episode started. Submit your cleaned version of dirty_data.",
            score_breakdown={},
        )

    def step(
        self, action: "DataCleaningAction"
    ) -> Tuple["DataCleaningObservation", float, bool, Dict[str, Any]]:
        """
        Agent submits cleaned data. The grader evaluates it.

        Args:
            action: DataCleaningAction with the agent's cleaned_data

        Returns:
            (observation, reward, done, info) tuple

        Raises:
            RuntimeError: if step() is called before reset()
        """
        from models import DataCleaningAction, DataCleaningObservation

        if self._done:
            raise RuntimeError(
                "Episode is already done. Call reset() before step()."
            )

        self._step_count += 1

        # Run the grader for this task
        grader = GRADERS[self._task_id]
        raw_score, feedback, breakdown = grader(action.cleaned_data)

        # Shaped reward
        improvement_bonus = 0.1 if raw_score > self._best_score else 0.0
        step_penalty = 0.05 * max(0, self._step_count - 1)
        reward = raw_score + improvement_bonus - step_penalty
        reward = round(max(0.0, min(1.0, reward)), 4)

        # Track best score
        if raw_score > self._best_score:
            self._best_score = raw_score

        # End episode on perfect score or step limit
        done = (raw_score >= 0.99) or (self._step_count >= MAX_STEPS)
        self._done = done

        task = TASKS[self._task_id]
        obs = DataCleaningObservation(
            task_id=self._task_id,
            task_description=task["description"],
            dirty_data=copy.deepcopy(task["dirty"]),  # always show original dirty data
            schema_hint=task["schema"],
            step_count=self._step_count,
            done=done,
            reward=reward,
            feedback=feedback,
            score_breakdown=breakdown,
        )

        info: Dict[str, Any] = {
            "episode_id": self._episode_id,
            "raw_score": raw_score,
            "best_score": self._best_score,
            "step_count": self._step_count,
            "max_steps": MAX_STEPS,
        }

        return obs, reward, done, info

    def state(self) -> "DataCleaningState":
        """Return episode-level metadata."""
        from models import DataCleaningState

        return DataCleaningState(
            episode_id=self._episode_id or "not-started",
            step_count=self._step_count,
            task_id=self._task_id,
            max_steps=MAX_STEPS,
            best_score=self._best_score,
        )
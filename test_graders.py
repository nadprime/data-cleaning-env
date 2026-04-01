"""
Quick sanity check for the grader functions.
Run: python test_graders.py
Expected: all assertions pass, prints "All tests passed!"
"""
import sys
sys.path.insert(0, '.')

from server.environment import (
    grade_task_1, grade_task_2, grade_task_3,
    TASK_1_CLEAN, TASK_2_CLEAN, TASK_3_CLEAN,
    TASK_1_DIRTY, TASK_2_DIRTY,
)

print("Testing graders...")

# ── Task 1 ────────────────────────────────────────────────────────────────────

# Perfect submission → should score ~1.0
score, feedback, bd = grade_task_1(TASK_1_CLEAN)
print(f"Task 1 perfect: {score:.4f}  | {feedback[:80]}")
assert score >= 0.99, f"Expected ~1.0, got {score}"

# Empty submission → should score 0.0
score, feedback, bd = grade_task_1([])
print(f"Task 1 empty  : {score:.4f}  | {feedback}")
assert score == 0.0, f"Expected 0.0, got {score}"

# Dirty data submitted as-is → column names wrong → should score < 0.5
score, feedback, bd = grade_task_1(TASK_1_DIRTY)
print(f"Task 1 dirty  : {score:.4f}  | {feedback[:80]}")
assert score < 0.5, f"Dirty data should score low, got {score}"

# ── Task 2 ────────────────────────────────────────────────────────────────────

score, feedback, bd = grade_task_2(TASK_2_CLEAN)
print(f"Task 2 perfect: {score:.4f}  | {feedback[:80]}")
assert score >= 0.99, f"Expected ~1.0, got {score}"

score, feedback, bd = grade_task_2([])
print(f"Task 2 empty  : {score:.4f}  | {feedback}")
assert score == 0.0, f"Expected 0.0, got {score}"

# Wrong row count (7 rows instead of 6) → should penalize
score, feedback, bd = grade_task_2(TASK_2_DIRTY)
print(f"Task 2 dirty  : {score:.4f}  | {feedback[:80]}")
assert score < 0.5, f"Dirty data should score low, got {score}"

# ── Task 3 ────────────────────────────────────────────────────────────────────

score, feedback, bd = grade_task_3(TASK_3_CLEAN)
print(f"Task 3 perfect: {score:.4f}  | {feedback[:80]}")
assert score >= 0.99, f"Expected ~1.0, got {score}"

score, feedback, bd = grade_task_3([])
print(f"Task 3 empty  : {score:.4f}  | {feedback}")
assert score == 0.0, f"Expected 0.0, got {score}"

# ── Score range checks ────────────────────────────────────────────────────────
for grade_fn, label in [(grade_task_1, "T1"), (grade_task_2, "T2"), (grade_task_3, "T3")]:
    for data in [[], [{"bad": "data"}], TASK_1_CLEAN]:
        s, _, _ = grade_fn(data)
        assert 0.0 <= s <= 1.0, f"{label}: score {s} is out of range [0, 1]"

print()
print("=" * 40)
print("All tests passed! Graders work correctly.")
print("=" * 40)

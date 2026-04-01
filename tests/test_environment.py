"""
Automated test suite.
Run with: uv run pytest tests/ -v
"""
import pytest
import sys
sys.path.insert(0, ".")

from server.environment import (
    DataCleaningEnvironment,
    grade_task_1, grade_task_2, grade_task_3,
    TASK_1_CLEAN, TASK_2_CLEAN, TASK_3_CLEAN,
    TASK_1_DIRTY, TASK_2_DIRTY, TASK_3_DIRTY,
)
from models import DataCleaningAction


# ── Grader tests ──────────────────────────────────────────────────────────────

class TestGraders:

    def test_task1_perfect(self):
        score, _, _ = grade_task_1(TASK_1_CLEAN)
        assert score >= 0.99

    def test_task1_empty(self):
        score, _, _ = grade_task_1([])
        assert score == 0.0

    def test_task1_dirty_scores_low(self):
        score, _, _ = grade_task_1(TASK_1_DIRTY)
        assert score < 0.5

    def test_task2_perfect(self):
        score, _, _ = grade_task_2(TASK_2_CLEAN)
        assert score >= 0.99

    def test_task2_empty(self):
        score, _, _ = grade_task_2([])
        assert score == 0.0

    def test_task3_perfect(self):
        score, _, _ = grade_task_3(TASK_3_CLEAN)
        assert score >= 0.99

    def test_task3_empty(self):
        score, _, _ = grade_task_3([])
        assert score == 0.0

    def test_all_scores_in_range(self):
        """No grader should ever produce a score outside [0.0, 1.0]."""
        for fn in [grade_task_1, grade_task_2, grade_task_3]:
            for data in [[], [{"bad": "data"}], TASK_1_CLEAN]:
                score, _, _ = fn(data)
                assert 0.0 <= score <= 1.0, (
                    f"{fn.__name__} score {score} out of range"
                )

    def test_difficulty_progression(self):
        """
        Difficulty progression: completely wrong data scores low on all tasks.
        Perfect data scores ~1.0 on its own grader.
        """
        wrong_data = [{"wrong_column": "wrong_value"}]

        s1, _, _ = grade_task_1(wrong_data)
        s2, _, _ = grade_task_2(wrong_data)
        s3, _, _ = grade_task_3(wrong_data)

        assert s1 <= 0.6, f"Task 1 wrong data should score low, got {s1}"
        assert s2 <= 0.6, f"Task 2 wrong data should score low, got {s2}"
        assert s3 <= 0.6, f"Task 3 wrong data should score low, got {s3}"

        p1, _, _ = grade_task_1(TASK_1_CLEAN)
        p2, _, _ = grade_task_2(TASK_2_CLEAN)
        p3, _, _ = grade_task_3(TASK_3_CLEAN)

        assert p1 >= 0.99, f"Task 1 perfect data should score ~1.0, got {p1}"
        assert p2 >= 0.99, f"Task 2 perfect data should score ~1.0, got {p2}"
        assert p3 >= 0.99, f"Task 3 perfect data should score ~1.0, got {p3}"

    def test_feedback_is_string(self):
        """Every grader must return a non-empty feedback string."""
        for fn in [grade_task_1, grade_task_2, grade_task_3]:
            _, feedback, _ = fn(TASK_1_CLEAN)
            assert isinstance(feedback, str)
            assert len(feedback) > 0

    def test_breakdown_is_dict(self):
        """Every grader must return a breakdown dict."""
        _, _, bd1 = grade_task_1(TASK_1_CLEAN)
        assert isinstance(bd1, dict)
        assert "column_names" in bd1
        assert "data_values" in bd1

        _, _, bd2 = grade_task_2(TASK_2_CLEAN)
        assert isinstance(bd2, dict)
        assert "row_count" in bd2
        assert "cell_accuracy" in bd2

        _, _, bd3 = grade_task_3(TASK_3_CLEAN)
        assert isinstance(bd3, dict)
        assert "product" in bd3
        assert "price" in bd3


# ── Environment integration tests ─────────────────────────────────────────────

class TestEnvironment:

    def setup_method(self):
        """Fresh environment before each test."""
        self.env = DataCleaningEnvironment()

    def test_reset_returns_clean_state(self):
        obs = self.env.reset(task_id=1)
        assert obs.task_id == 1
        assert obs.step_count == 0
        assert obs.done == False
        assert obs.reward == 0.0
        assert len(obs.dirty_data) > 0

    def test_reset_all_tasks(self):
        for task_id in [1, 2, 3]:
            obs = self.env.reset(task_id=task_id)
            assert obs.task_id == task_id
            assert obs.done == False
            assert obs.step_count == 0

    def test_reset_clears_previous_state(self):
        """Calling reset() twice should give a clean state."""
        self.env.reset(task_id=1)
        self.env.step(DataCleaningAction(
            task_id=1, cleaned_data=[], metadata={}
        ))
        # Reset again — step count should go back to 0
        obs = self.env.reset(task_id=2)
        assert obs.step_count == 0
        assert obs.task_id == 2

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            self.env.reset(task_id=99)

    def test_invalid_task_zero_raises(self):
        with pytest.raises(ValueError):
            self.env.reset(task_id=0)

    def test_step_before_reset_raises(self):
        with pytest.raises(RuntimeError):
            self.env.step(DataCleaningAction(
                task_id=1,
                cleaned_data=[],
                metadata={}
            ))

    def test_step_returns_correct_types(self):
        self.env.reset(task_id=1)
        obs, reward, done, info = self.env.step(
            DataCleaningAction(
                task_id=1,
                cleaned_data=TASK_1_CLEAN,
                metadata={}
            )
        )
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_perfect_submission_ends_episode(self):
        self.env.reset(task_id=1)
        obs, reward, done, info = self.env.step(
            DataCleaningAction(
                task_id=1,
                cleaned_data=TASK_1_CLEAN,
                metadata={}
            )
        )
        assert done == True
        assert info["raw_score"] >= 0.99

    def test_reward_always_in_range(self):
        for task_id in [1, 2, 3]:
            self.env.reset(task_id=task_id)
            _, reward, _, _ = self.env.step(
                DataCleaningAction(
                    task_id=task_id,
                    cleaned_data=[{"bad": "data"}],
                    metadata={}
                )
            )
            assert 0.0 <= reward <= 1.0, (
                f"Task {task_id} reward {reward} out of range"
            )

    def test_state_before_reset(self):
        """state() should not crash before reset() is called."""
        state = self.env.state()
        assert isinstance(state.episode_id, str)
        assert isinstance(state.step_count, int)

    def test_state_returns_correct_metadata(self):
        self.env.reset(task_id=2)
        state = self.env.state()
        assert state.task_id == 2
        assert state.max_steps == 5
        assert state.step_count == 0
        assert isinstance(state.episode_id, str)
        assert len(state.episode_id) > 0

    def test_step_increments_count(self):
        self.env.reset(task_id=1)
        self.env.step(DataCleaningAction(
            task_id=1,
            cleaned_data=[],
            metadata={}
        ))
        assert self.env._step_count == 1

    def test_episode_ends_at_max_steps(self):
        self.env.reset(task_id=1)
        done = False
        for _ in range(5):
            _, _, done, _ = self.env.step(
                DataCleaningAction(
                    task_id=1,
                    cleaned_data=[],
                    metadata={}
                )
            )
        assert done == True

    def test_info_contains_required_keys(self):
        self.env.reset(task_id=1)
        _, _, _, info = self.env.step(
            DataCleaningAction(
                task_id=1,
                cleaned_data=[],
                metadata={}
            )
        )
        assert "episode_id" in info
        assert "raw_score" in info
        assert "best_score" in info
        assert "step_count" in info
        assert "max_steps" in info

    def test_best_score_tracks_improvement(self):
        self.env.reset(task_id=1)
        # First step with bad data
        _, _, _, info1 = self.env.step(
            DataCleaningAction(
                task_id=1,
                cleaned_data=[{"bad": "data"}],
                metadata={}
            )
        )
        # Second step with perfect data
        _, _, _, info2 = self.env.step(
            DataCleaningAction(
                task_id=1,
                cleaned_data=TASK_1_CLEAN,
                metadata={}
            )
        )
        assert info2["best_score"] >= info1["best_score"]

    def test_observation_always_contains_dirty_data(self):
        """dirty_data in observation should always be the original — never modified."""
        self.env.reset(task_id=1)
        obs, _, _, _ = self.env.step(
            DataCleaningAction(
                task_id=1,
                cleaned_data=TASK_1_CLEAN,
                metadata={}
            )
        )
        # dirty_data should still be the original messy data
        assert len(obs.dirty_data) > 0
        # First row should have the original messy column names
        first_row_keys = list(obs.dirty_data[0].keys())
        assert any(" " in k or k != k.lower() for k in first_row_keys), (
            "dirty_data should contain original messy column names"
        )
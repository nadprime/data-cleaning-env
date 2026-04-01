"""
models.py
=========
Typed data contracts for the Data Cleaning Environment.

OpenEnv requires three model types:
  - Action      : what the agent sends to the environment
  - Observation : what the agent receives back
  - State       : episode-level metadata
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DataCleaningAction(BaseModel):
    """
    What the agent submits each step.

    Fields:
        task_id      : which task (1, 2, or 3)
        cleaned_data : the agent's cleaned version of the dataset (list of row dicts)
        metadata     : any optional extra info the agent wants to pass
    """

    task_id: int = Field(..., ge=1, le=3, description="Task ID: 1=Easy, 2=Medium, 3=Hard")
    cleaned_data: List[Dict[str, Any]] = Field(
        default_factory=list, description="The cleaned dataset as a list of row dicts"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class DataCleaningObservation(BaseModel):
    """
    What the agent sees after each step (and at reset).

    Fields:
        task_id          : which task is active
        task_description : plain-English instructions for cleaning
        dirty_data       : the original messy input data (always shown)
        schema_hint      : expected column names and types
        step_count       : how many steps have been taken this episode
        done             : whether the episode is finished
        reward           : reward from the last action (0.0 at reset)
        feedback         : human-readable feedback from the grader
        score_breakdown  : per-metric partial scores (for partial progress)
    """

    task_id: int
    task_description: str
    dirty_data: List[Dict[str, Any]]
    schema_hint: Dict[str, str]
    step_count: int
    done: bool
    reward: float
    feedback: str
    score_breakdown: Dict[str, float] = Field(default_factory=dict)


class DataCleaningState(BaseModel):
    """
    Episode-level metadata (returned by GET /state).

    Fields:
        episode_id  : unique ID for this episode run
        step_count  : current step number (0 to max_steps)
        task_id     : which task is loaded
        max_steps   : maximum steps allowed per episode
        best_score  : best raw grader score achieved so far this episode
    """

    episode_id: str
    step_count: int
    task_id: int
    max_steps: int
    best_score: float
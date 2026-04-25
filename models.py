# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Disaster Response Environment.
Flood disaster simulation with time-based resource management.
"""

from typing import List, Optional, Dict, Any
from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────

class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class Severity(str, Enum):
    MILD     = "mild"
    MODERATE = "moderate"
    CRITICAL = "critical"


class ResourceType(str, Enum):
    RESCUE_TEAM  = "rescue_team"
    HELICOPTER   = "helicopter"
    MEDICAL_UNIT = "medical_unit"


class ResourceState(str, Enum):
    AVAILABLE  = "available"
    TRAVELLING = "travelling"
    WORKING    = "working"
    RETURNING  = "returning"
    RESTOCKING = "restocking"


# ─── Action ───────────────────────────────────────────────────────────────────

class DisasterAction(Action):
    """
    Parallel deployment action.
    Agent deploys ALL available resources in one step.

    Example:
    {
        "deployments": [
            {"resource_id": "rescue_team_1", "targets": ["Zone_A"]},
            {"resource_id": "rescue_team_2", "targets": ["Zone_B"]},
            {"resource_id": "helicopter",    "targets": ["Zone_A", "Zone_C"]},
            {"resource_id": "medical_unit",  "targets": ["Zone_A"]}
        ]
    }
    """
    deployments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of resource deployments for this step"
    )


# ─── Observation ──────────────────────────────────────────────────────────────

class DisasterObservation(Observation):
    """Full state observation returned to agent each step."""

    # Time
    current_step:       int   = Field(default=0,      description="Current step")
    max_steps:          int   = Field(default=200,     description="Max steps")
    time_elapsed:       int   = Field(default=0,      description="Minutes elapsed")

    # Scenario
    difficulty:         str   = Field(default="easy", description="easy/medium/hard")

    # State summaries
    zones_summary:      str   = Field(default="",     description="Zone status text")
    resources_summary:  str   = Field(default="",     description="Resource status text")

    # Feedback
    last_action_result: str   = Field(default="",     description="Last action outcome")

    # Scoring
    total_casualties:   int   = Field(default=0,      description="Total casualties")
    total_rescued:      int   = Field(default=0,      description="Total rescued")

    # Actions hint
    available_actions:  str   = Field(default="",     description="Available actions")

    # Episode state
    done:               bool  = Field(default=False,  description="Episode done")
    reward:             float = Field(default=0.0,    description="Current reward")

    # Sub-score breakdown (populated at milestone/done steps, None otherwise)
    reward_breakdown:   Optional[Dict[str, float]] = Field(default=None, description="rescue_ratio, time_efficiency, critical_zone, severity sub-scores")
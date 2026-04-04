# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Disaster Response Environment Client."""

from typing import Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import DisasterAction, DisasterObservation
except ImportError:
    from models import DisasterAction, DisasterObservation


class DisasterEnv(
    EnvClient[DisasterAction, DisasterObservation, State]
):
    """
    Client for the Disaster Response Environment.

    Example — simple deploy:
        >>> with DisasterEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.zones_summary)
        ...
        ...     action = DisasterAction(resource_id="rescue_team_1", targets=["Zone_A"])
        ...     result = client.step(action)
        ...     print(result.observation.last_action_result)

    Example — helicopter chain:
        ...     action = DisasterAction(resource_id="helicopter", targets=["Zone_A", "Zone_C"])
        ...     result = client.step(action)

    Example — hold:
        ...     action = DisasterAction(resource_id="hold", targets=[])
        ...     result = client.step(action)

    Example with Docker:
        >>> client = DisasterEnv.from_docker_image("disaster_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(DisasterAction(resource_id="hold", targets=[]))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: DisasterAction) -> Dict:
        """Convert DisasterAction to JSON payload."""
        return {
            "deployments": action.deployments,
        }

    def _parse_result(self, payload: Dict) -> StepResult[DisasterObservation]:
        """Parse server response into StepResult[DisasterObservation]."""
        obs_data = payload.get("observation", {})

        observation = DisasterObservation(
            current_step        = obs_data.get("current_step", 0),
            max_steps           = obs_data.get("max_steps", 60),
            time_elapsed        = obs_data.get("time_elapsed", 0),
            difficulty          = obs_data.get("difficulty", "easy"),
            zones_summary       = obs_data.get("zones_summary", ""),
            resources_summary   = obs_data.get("resources_summary", ""),
            last_action_result  = obs_data.get("last_action_result", ""),
            total_casualties    = obs_data.get("total_casualties", 0),
            total_rescued       = obs_data.get("total_rescued", 0),
            available_actions   = obs_data.get("available_actions", ""),
            done                = obs_data.get("done", False),
            reward              = obs_data.get("reward", 0.0),
            metadata            = obs_data.get("metadata", {}),
        )

        return StepResult(
            observation = observation,
            reward      = payload.get("reward"),
            done        = payload.get("done", False),
        )
    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id  = payload.get("episode_id"),
            step_count  = payload.get("step_count", 0),
        )
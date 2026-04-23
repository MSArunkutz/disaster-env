# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Disaster Response Environment.
"""

import os
from dotenv import load_dotenv
load_dotenv()

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import DisasterAction, DisasterObservation
    from .disaster_env_environment import DisasterEnvEnvironment
except (ImportError, ModuleNotFoundError):
    from models import DisasterAction, DisasterObservation
    from server.disaster_env_environment import DisasterEnvEnvironment


# Read difficulty from environment variable — defaults to easy
DIFFICULTY = os.environ.get("DIFFICULTY", "easy")


# Wrap with difficulty injected
def DisasterEnvFactory():
    return DisasterEnvEnvironment(difficulty=DIFFICULTY)


app = create_app(
    DisasterEnvFactory,
    DisasterAction,
    DisasterObservation,
    env_name="disaster_env",
    max_concurrent_envs=1,
)


def main():
    """
    Entry point for direct execution via uv run or python -m.
    This function enables running the server without Docker:
        uv run --project . server
        python -m disaster_env.server.app
        openenv serve disaster_env
    """
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
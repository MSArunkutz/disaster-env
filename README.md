---
title: Disaster Response RL Environment
emoji: 🌊
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# Disaster Response — Long-Horizon Planning Environment

> An LLM agent must plan 5–20 steps ahead to save lives in a cascading flood disaster.
> Greedy, reactive agents fail. Only agents that reason about future states succeed.

Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) spec — Theme #2: Long-Horizon Planning & Instruction Following.

---

## Why This Tests Long-Horizon Planning

**Multi-step dependency chains:** The optimal strategy is never "rescue now." It's:
`medical_unit treats CRITICAL zone` → `severity drops to MODERATE` → `rescue teams extract 67% faster` → `resources freed sooner` → `cascade prevented in next zone`

This 4-step dependency chain takes ~15 steps to play out. Agents that can't plan ahead miss it entirely.

**Cascading uncertainty:** Every 12–20 steps, unattended zones gain more casualties and escalate severity. The agent must anticipate *which zone cascades next* and pre-position resources — not react after the fact.

**Irreversibility:** Once a zone escalates to CRITICAL and stays neglected for 12+ steps, the scoring penalty is permanent. There's no catching up — only planning ahead avoids it.

**Resource contention:** 4 resources share 3–8 zones. Every assignment has opportunity cost over multiple future steps (travel time, working time, return time). Short-sighted greedy allocation consistently underperforms.

---

## Scoring & Milestones

The environment uses **sparse milestone rewards** to incentivize long-horizon planning:

| Milestone | Step | Reward |
|-----------|------|--------|
| Quarter   | 50   | `compute_score(env)` |
| Midpoint  | 100  | `compute_score(env)` + early commitment bonus |
| Three-quarter | 150 | `compute_score(env)` |
| Final     | 200  | `compute_score(env)` (submitted to OpenEnv) |

All other steps return `0.0`. Agents must plan across 50-step windows, not optimize myopically.

### Final Score Weights
| Component | Weight | Description |
|-----------|--------|-------------|
| Rescue ratio | 45% | People saved / total casualties |
| Time efficiency | 20% | Steps used vs max_steps |
| Critical zone response | 20% | How quickly critical zones were attended |
| Severity reduction | 15% | Medical unit effectiveness |

---

## Overview

A flood has hit a region. The agent must coordinate 4 rescue resources across multiple zones to save as many people as possible before time runs out.

### Resources
| Resource | Speed | Capability |
|---|---|---|
| rescue_team_1 | 60 km/h | Extracts people from zones |
| rescue_team_2 | 60 km/h | Extracts people from zones |
| helicopter | 180 km/h | Fast extraction, can chain multiple zones per trip |
| medical_unit | 48 km/h | Treats injured — reduces zone severity (CRITICAL→MODERATE→MILD) |

### Difficulty Levels (3 Tasks)
| Difficulty | Zones | Max Steps | Cascade Every |
|---|---|---|---|
| Easy | 3 | 200 | 20 steps |
| Medium | 5 | 200 | 15 steps |
| Hard | 8 | 200 | 12 steps |

### Scoring (0.0 – 1.0)

rescued_score  × 0.45   (people saved / total casualties)
time_score     × 0.20   (steps used vs max steps)
response_score × 0.20   (critical zones attended quickly)
severity_bonus × 0.15   (medical unit effectiveness)

---

## Setup

### Prerequisites
- Docker
- Python 3.10+
- uv

### 1. Clone and install
```bash
git clone https://huggingface.co/spaces/your-username/disaster-env
cd disaster-env
uv sync
```

### 2. Configure environment
```bash
cp .env.example .env
# Fill in your credentials
```

### 3. Run locally
```bash
# Start server
uv run server

# In another terminal, run inference
python inference.py
```

### 4. Run with Docker
```bash
# Linux/Mac
./start.sh

# Windows
start.bat

# Then run inference
python inference.py
```

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | Yes |
| `MODEL_NAME` | Model identifier | Yes |
| `HF_TOKEN` | Hugging Face / API key | Yes |
| `DIFFICULTY` | Starting difficulty (easy/medium/hard) | No (default: easy) |
| `PORT` | Server port | No (default: 7860) |
| `MODE` | TEST uses Groq, PROD uses API_BASE_URL | No (default: PROD) |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment (pass seed=1/2/3 for easy/medium/hard) |
| `/step` | POST | Execute deployment action |
| `/state` | GET | Current environment state |
| `/ws` | WebSocket | Persistent session |
| `/docs` | GET | Interactive API documentation |

### Reset with difficulty
```json
{"seed": 1}   // easy
{"seed": 2}   // medium
{"seed": 3}   // hard
```

### Step action format
```json
{
    "action": {
        "deployments": [
            {"resource_id": "rescue_team_1", "targets": ["Zone_A"]},
            {"resource_id": "rescue_team_2", "targets": ["Zone_B"]},
            {"resource_id": "helicopter",    "targets": ["Zone_A", "Zone_C"]},
            {"resource_id": "medical_unit",  "targets": ["Zone_A"]}
        ]
    }
}
```

---

## Baseline Scores

Measured with `meta-llama/Llama-3.1-8B-Instruct`:

| Difficulty | Score | Steps | Time |
|---|---|---|---|
| Easy | 0.9888 | 9/200 | ~4s |
| Medium | 0.9525 | 38/200 | ~237s |
| Hard | 0.8638 | 109/200 | ~793s |
| **Average** | **0.9350** | | **~17 min** |

---

## Project Structure

disaster-env/
├── server/
│   ├── app.py                          # FastAPI server
│   └── disaster_env_environment.py     # Core simulation logic
├── models.py                           # Action/Observation dataclasses
├── client.py                           # HTTP/WebSocket client
├── scenario_generator.py               # Random scenario generation
├── graders.py                          # 3 graders returning 0.0–1.0
├── inference.py                        # Agent demo script
├── openenv.yaml                        # OpenEnv manifest
└── Dockerfile                          # Container definition

---

## Resources

- [OpenEnv Repository](https://github.com/meta-pytorch/OpenEnv)
- [TRL Integration](https://huggingface.co/docs/trl/openenv)
- [Environment Hub](https://huggingface.co/collections/openenv/environment-hub)

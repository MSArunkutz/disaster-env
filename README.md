---
title: Disaster Response RL Environment
emoji: 🌊
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# Disaster Response RL Environment

A flood disaster response simulation environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) spec.

An LLM agent acts as an emergency coordinator managing rescue resources across multiple flood-affected zones under time pressure.

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

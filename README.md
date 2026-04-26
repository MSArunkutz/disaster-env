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

This environment implements **RLVE (Reinforcement Learning with Verifiable Environment feedback)**: as the agent rescues more people, the remaining zones cascade faster — the environment dynamically raises difficulty in response to the agent's own success, making reward hacking structurally harder as the agent improves.

---

## Why This Tests Long-Horizon Planning

**Multi-step dependency chains:** The optimal strategy is never "rescue now." It's:
`medical_unit treats CRITICAL zone` → `severity drops to MODERATE` → `rescue teams extract 67% faster` → `resources freed sooner` → `cascade prevented in next zone`

This 4-step dependency chain takes ~15 steps to play out. Agents that can't plan ahead miss it entirely.

**Cascading uncertainty:** Every 12–20 steps, unattended zones gain more casualties and escalate severity. The agent must anticipate *which zone cascades next* and pre-position resources — not react after the fact.

**Irreversibility:** Once a zone escalates to CRITICAL and stays neglected for 12+ steps, the scoring penalty is permanent. There's no catching up — only planning ahead avoids it.

**Resource contention:** 4 resources share 3–8 zones. Every assignment has opportunity cost over multiple future steps (travel time, working time, return time). Short-sighted greedy allocation consistently underperforms.

---

## Reward Design

The reward signal is a **hybrid of dense step-level feedback and sparse milestone scores**, designed to train across 50-step planning windows without reward deserts.

### Dense reward (every step)
```
dense = rescued_delta × 0.002
```
Small immediate signal for each new person rescued this step. Keeps gradients alive between milestones.

### Sparse milestone reward (at checkpoints only)
```
milestone_reward = compute_score(env) + 0.1 commitment_bonus
```
Full scoring evaluation at each milestone checkpoint. The `+0.1` bonus at the midpoint rewards agents that commit to a plan and execute it, not ones that hedge until the end.

### Total step reward
```
non-milestone step:  reward = dense
milestone/done step: reward = compute_score(env) + 0.1 + dense
```

### Milestone checkpoints (difficulty-aware)

| Difficulty | Milestone Steps | Intervals |
|-----------|----------------|-----------|
| Easy | 50, 100, 150, 200 | 4 |
| Medium | 40, 80, 120, 160, 200 | 5 |
| Hard | 29, 57, 86, 114, 143, 171, 200 | 7 |

Hard has more checkpoints because the 8-zone cascade pressure demands tighter feedback loops.

### Final score weights

| Component | Weight | Description |
|-----------|--------|-------------|
| Rescue ratio | 45% | People saved / total casualties |
| Time efficiency | 20% | Steps used vs max_steps |
| Critical zone response | 20% | How quickly critical zones were attended |
| Severity reduction | 15% | Medical unit effectiveness |
| Per-step rescue delta | — | `+0.002 × rescued_delta` every step — keeps gradients alive between milestones |
| Milestone commitment bonus | +0.1 per checkpoint | Rewards agents that commit to a plan and execute it, not ones that hedge until the end |

### Per-step reward breakdown

At every milestone and episode end, the environment computes and logs a full sub-score breakdown:
```json
{
  "rescue_ratio": 0.82,
  "time_efficiency": 0.91,
  "critical_zone": 0.75,
  "severity": 0.60,
  "total": 0.79
}
```
This enables per-component analysis during GRPO training — the optimizer can see which planning sub-skill is weakest.

---

## Reward Hacking Guards

The reward function is designed so common exploitation strategies fail:

| Strategy | Why it fails |
|----------|-------------|
| Rescue only the easiest zone | 45% rescue ratio weight penalizes zones left untouched; critical zone response (20%) penalizes ignoring high-severity zones |
| Hold all resources to get time efficiency | Time efficiency rewards steps *not used* — but rescue ratio drops to 0 if no one is saved, dominating the score |
| Max step-limit farming | `max_steps` caps the episode; `UNLIMITED_MODE=true` removes the wall-clock limit but not the step limit |
| Exploiting dense reward without milestones | Dense reward is `0.002 × delta` per step — negligible vs milestone `compute_score()`. A strategy that earns only dense reward scores near 0 overall |
| Ignoring cascades | Every 12–20 steps, unattended zones gain casualties. The rescue ratio denominator grows — total score falls even if the agent saves the same number of people |

---

## RLVE — Environment Difficulty Adapts to Agent Skill

This environment qualifies as **RLVE (Reinforcement Learning with Verifiable Environment feedback)**:

- Each rescue removes a person from the casualty denominator — *improving* the agent's rescue ratio
- But neglected zones cascade every 12–20 steps, adding casualties back — *undoing* the improvement
- As the agent gets better at rescuing, surviving zones escalate faster under the increased load
- A better agent faces a dynamically harder environment — strategies that work at easy difficulty are structurally penalized at higher difficulties

This is not a static curriculum. The cascade mechanic means the environment *responds* to the agent's own effectiveness, making the reward signal self-adjusting without any external difficulty controller.

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

---

## Baseline Scores

### Teacher model — `meta-llama/Llama-3.1-8B-Instruct` (used for SFT data collection)

| Difficulty | Score | Steps used |
|---|---|---|
| Easy | 0.9888 | 9 / 200 |
| Medium | 0.9525 | 38 / 200 |
| Hard | 0.8638 | 109 / 200 |
| **Average** | **0.9350** | |

This is the target: the Llama teacher solved each episode efficiently in a fraction of the allowed steps.

### Student model — `Qwen2.5-3B-Instruct` untuned (our training starting point)

| Difficulty | Score | Steps | Parsed | Notes |
|---|---|---|---|---|
| Easy | 0.0000 | 200 / 200 | 200 / 200 | Valid JSON every step — wrong field names, zero rescues |
| Medium | 0.0000 | 200 / 200 | 200 / 200 | Same pattern — all steps exhausted, no effect on env |
| Hard | 0.2160 | 5 / 200 | 5 / 5 | Episode crashed at step 4: model included an extra `free_resources` field that Pydantic rejected; `compute_score()` called on partial env state |
| **Average** | **0.0720** | | | Hard score is a crash artifact, not capability |

**Key insight:** `parsed=200/200` on easy/medium shows the untuned model can follow JSON format instructions — it just has no domain knowledge of which resource IDs and zone names are valid. The hard crash (`free_resources: Extra inputs are not permitted`) confirms the same: the model invents plausible-sounding fields that don't exist in `DisasterAction`.

This baseline gives maximum training signal: every improvement from SFT and GRPO is measurable against a true zero.

Raw scores, per-step logs, and bar chart: [`supporting_content/`](supporting_content/)

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

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint (PROD mode) | required |
| `MODEL_NAME` | Model identifier (PROD mode) | required |
| `HF_TOKEN` | HuggingFace / API key | required |
| `MODE` | `PROD` uses API_BASE_URL, `TEST` uses Groq | `PROD` |
| `PORT` | Server port | `8000` |
| `ENV_BASE_URL` | Client→server URL | `http://localhost:8000` |
| `DIFFICULTY` | Starting difficulty (`easy`/`medium`/`hard`) | `easy` |
| `UNLIMITED_MODE` | Remove 18-min wall-clock limit for training runs | `false` |
| `RUN_DIFFICULTIES` | Comma-separated list of difficulties to run | all three |
| `MAX_STEPS` | Override step limit (default 200) | `200` |
| `GROQ_API_BASE_URL` | Groq endpoint (TEST mode only) | Groq default |
| `GROQ_MODEL_NAME` | Groq model (TEST mode only) | `llama-3.1-8b-instant` |
| `GROQ_API_KEY` | Groq API key (TEST mode only) | required if TEST |

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

## Training Support

### SFT data collection
Each episode auto-saves step-level traces to `training_data/`:
```bash
python training/collect_sft_data.py --episodes 60 --min-score 0.85
```
Produces JSONL with: `step`, `observation`, `agent_prompt`, `agent_response`, `action`, `reward`, `reward_breakdown`, `done`.

### GRPO — snapshot / restore
The environment supports stateless parallel rollouts for GRPO training:
```python
snap = env.snapshot()    # deep-copy of all 16 mutable state fields
env.restore(snap)        # restore to exact prior state for counterfactual rollouts
```
This enables `GRPOTrainer` to replay candidate actions on frozen environment states without spinning up multiple server instances.

### Training notebook
`training/training_notebook.ipynb` — Unsloth + TRL pipeline for Qwen2.5-3B or Gemma-3-1B:
- SFT warm-start on filtered traces (score > 0.85)
- GRPO on top using `graders.compute_score()` as the reward signal
- Reward curve plots: baseline → SFT → GRPO

---

## Project Structure

```
disaster_env/
├── server/
│   ├── app.py                          # FastAPI server
│   └── disaster_env_environment.py     # Core simulation + reward logic
├── training/
│   ├── collect_sft_data.py             # SFT trace collection (distillation)
│   └── training_notebook.ipynb         # Unsloth + TRL training (SFT + GRPO)
├── training_data/                      # Auto-created; per-episode JSONL traces
├── supporting_content/                 # Judges: visualizer + eval artifacts
│   ├── visualizer.html                 # Step-by-step training data animator
│   ├── baseline/                       # Qwen2.5-3B untuned eval
│   │   ├── baseline_results.json
│   │   ├── baseline_log.json
│   │   └── baseline_chart.png
│   └── sft/                            # SFT warm-start eval
│       ├── sft_results.json
│       ├── sft_log.json
│       └── sft_chart.png
├── models.py                           # Action/Observation dataclasses
├── client.py                           # HTTP/WebSocket client
├── scenario_generator.py               # Random scenario generation
├── graders.py                          # compute_score() + compute_score_breakdown()
├── inference.py                        # Agent demo + data collection script
├── openenv.yaml                        # OpenEnv manifest
└── Dockerfile                          # Container definition
```

---

## Supporting Content

The `supporting_content/` folder contains artifacts for judges and reproducibility:

| File | Description |
|---|---|
| [`visualizer.html`](supporting_content/visualizer.html) | Self-contained step-by-step training data visualizer — load any JSONL from `training_data/` and animate the agent's decisions, resource movements, and zone states |
| [`baseline/baseline_results.json`](supporting_content/baseline/baseline_results.json) | Raw Qwen2.5-3B untuned scores: `{"easy": 0.0, "medium": 0.0, "hard": 0.216, "average": 0.072}` |
| [`baseline/baseline_log.json`](supporting_content/baseline/baseline_log.json) | Per-step logs for all three difficulties — includes raw model output, parse success flag, and the Pydantic error that terminated the hard episode early |
| [`baseline/baseline_chart.png`](supporting_content/baseline/baseline_chart.png) | Bar chart of untuned scores across difficulties |
| [`sft/sft_results.json`](supporting_content/sft/sft_results.json) | SFT warm-start scores: `{"easy": 0.0, "medium": 0.0, "hard": 0.218, "average": 0.073}` |
| [`sft/sft_log.json`](supporting_content/sft/sft_log.json) | Per-step logs for SFT checkpoint eval across all three difficulties |
| [`sft/sft_chart.png`](supporting_content/sft/sft_chart.png) | Side-by-side bar chart: untuned vs SFT warm-start |

---

## Submission Links

| | Link |
|---|---|
| HuggingFace Space | ⏳ (deploy pending) |
| Training Notebook (Colab) | [SFT + GRPO Pipeline](https://colab.research.google.com/drive/1mUqbYiCoYP8ngyQjH32XngFDH5thnccU?usp=sharing) |
| Code Repository | [MSArunkutz/disaster-env](https://github.com/MSArunkutz/disaster-env) |
| Blog Post | ⏳ (post pending) |

---

## Resources

- [OpenEnv Repository](https://github.com/meta-pytorch/OpenEnv)
- [TRL Integration](https://huggingface.co/docs/trl/openenv)
- [Environment Hub](https://huggingface.co/collections/openenv/environment-hub)

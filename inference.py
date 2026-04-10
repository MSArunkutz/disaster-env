"""
Inference Script — Disaster Response Environment.
===================================
MANDATORY VARIABLES:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
"""

# ─── Imports ───────────────────────────────────────────────────────────────────

import os
import re
import json
import time
import asyncio
import textwrap
from typing import List

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from client import DisasterEnv
from models import DisasterAction, DisasterObservation
from graders import grade

# ─── Log functions ───────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ─── Config ───────────────────────────────────────────────────────────────────

SUCCESS_SCORE_THRESHOLD = 0.5
BENCHMARK = "disaster_env"

TOTAL_TIME_BUDGET = 18 * 60  # 18 minutes in seconds
MAX_STEPS = 200              # high limit — time cut handles actual stopping
TEMPERATURE  = 0.2
MAX_TOKENS   = 200

API_BASE_URL = ""
API_KEY      = ""
MODEL_NAME   = ""

MODE = os.getenv("MODE", "PROD")

if MODE == "TEST":
    API_BASE_URL = os.getenv("GROQ_API_BASE_URL")
    API_KEY      = os.getenv("GROQ_API_KEY")
    MODEL_NAME   = os.getenv("GROQ_MODEL_NAME")
else:
    API_BASE_URL = os.getenv("API_BASE_URL","https://router.huggingface.co/v1")
    API_KEY      = os.getenv("HF_TOKEN")
    MODEL_NAME   = os.getenv("MODEL_NAME","meta-llama/Llama-3.1-8B-Instruct")

BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


DIFFICULTIES     = ["easy", "medium", "hard"]
DIFFICULTY_SEEDS = {"easy": 1, "medium": 2, "hard": 3}


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an emergency flood response coordinator.
Each step deploy ALL free resources simultaneously to rescue maximum people.

RESOURCES:
- rescue_team_1, rescue_team_2: extract people, speed 60km/h
  Rate: MILD=14/min, MODERATE=10/min, CRITICAL=6/min
- helicopter: fast extraction, speed 180km/h, capacity 15, can chain zones
- medical_unit: treats injured → reduces severity (CRITICAL→MODERATE→MILD)
  Has 50 supplies. Must restock at base when empty.

STRATEGY:
- Send medical_unit to CRITICAL zones first to reduce severity
- Then rescue teams extract faster from treated zones
- Use helicopter to chain nearby zones in one trip
- Deploy ALL free resources every step — never leave resources idle

RESPOND WITH JSON ONLY:
{
    "deployments": [
        {"resource_id": "rescue_team_1", "targets": ["Zone_A"]},
        {"resource_id": "rescue_team_2", "targets": ["Zone_B"]},
        {"resource_id": "helicopter",    "targets": ["Zone_A", "Zone_C"]},
        {"resource_id": "medical_unit",  "targets": ["Zone_A"]}
    ]
}
Only include FREE resources. Skip busy ones."""


# ─── Action Parser ────────────────────────────────────────────────────────────

def parse_action(response_text: str) -> DisasterAction:
    try:
        clean = re.sub(r"```json|```", "", response_text).strip()
        data  = json.loads(clean)
        deployments = data.get("deployments", [])
        
        # Deduplicate — keep only first occurrence of each resource_id
        seen = set()
        unique_deployments = []
        for d in deployments:
            rid = d.get("resource_id")
            if rid and rid not in seen:
                seen.add(rid)
                unique_deployments.append(d)
        
        return DisasterAction(deployments=unique_deployments)
    except Exception:
        return DisasterAction(deployments=[])


# ─── User Prompt ──────────────────────────────────────────────────────────────

def build_user_prompt(obs: DisasterObservation) -> str:
    return f"""Step {obs.current_step}/{obs.max_steps} | Rescued:{obs.total_rescued}/{obs.total_casualties}

{obs.zones_summary}
{obs.resources_summary}
{obs.available_actions}

Deploy all free resources now. JSON:"""


# ─── Episode Runner ───────────────────────────────────────────────────────────

async def run_episode(client: OpenAI, env: DisasterEnv, difficulty: str, time_remaining: float) -> float:
    start_time = time.time()
    deadline   = start_time + time_remaining - 30  # 30s safety buffer
    rewards_list = []

    seed   = DIFFICULTY_SEEDS[difficulty]
    result = await env.reset(seed=seed)
    obs    = result.observation

    # Structured start log
    log_start(task=difficulty, env=BENCHMARK, model=MODEL_NAME)

    print(f"\n{'='*50}")
    print(f"Starting episode — Difficulty: {difficulty.upper()}")
    print(f"{'='*50}")

    last_step = 0
    for step in range(1, MAX_STEPS + 1):
        last_step = step

        # Time cut check
        if time.time() > deadline:
            print(f"\n⏱️  Time budget hit at step {step} — force stopping episode.")
            break

        if result.done:
            print(f"Episode complete at step {step}.")
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(obs)},
        ]

        try:
            completion = client.chat.completions.create(
                model       = MODEL_NAME,
                messages    = messages,
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            if "429" in str(exc) or "rate_limit" in str(exc).lower():
                print("Rate limit — waiting 15s...")
                time.sleep(15)
            else:
                print(f"LLM failed ({exc}). Skipping.")
            response_text = '{"deployments": []}'

        action = parse_action(response_text)

        result = await env.step(action)
        obs    = result.observation

        # Structured log — required format
        action_str = json.dumps([d.get("resource_id") for d in action.deployments])
        reward_val = result.reward or 0.0
        rewards_list.append(reward_val)
        log_step(
            step   = step,
            action = action_str,
            reward = reward_val,
            done   = result.done,
            error  = None
        )

        # Human readable log — for debugging
        print(f"\n[Step {step}]")
        for line in obs.resources_summary.split("\n")[1:]:
            line = line.strip()
            if not line:
                continue
            res = line.split(":")[0].strip()
            if "FREE" in line:
                status = "at base, ready"
            elif "→" in line and "WORKING" not in line:
                zone  = line.split("→")[1].split("(")[0].strip()
                eta   = line.split("(")[1].replace("min)", "").strip() if "(" in line else "?"
                status = f"en route to {zone} — ETA {eta} steps"
            elif "WORKING" in line:
                zone  = line.split("@")[1].split("(")[0].strip() if "@" in line else "?"
                eta   = line.split("(")[1].replace("min)", "").strip() if "(" in line else "?"
                status = f"working at {zone} — done in {eta} steps"
            elif "RETURNING" in line:
                eta   = line.split("(")[1].replace("min)", "").strip() if "(" in line else "?"
                status = f"returning to base — ETA {eta} steps"
            elif "RESTOCKING" in line:
                eta   = line.split("(")[1].replace("min)", "").strip() if "(" in line else "?"
                status = f"restocking at base — {eta} steps remaining"
            else:
                status = line
            print(f"  {res:20s} {status}")

        print(f"  {'─'*46}")
        print(f"  Rescued: {obs.total_rescued}/{obs.total_casualties} | "
            f"Reward: {result.reward:.4f} | "
            f"Time left: {obs.max_steps - obs.current_step} steps")
    # Episode summary
    elapsed       = time.time() - start_time
    rescued       = obs.total_rescued
    total         = obs.total_casualties
    steps_used    = last_step
    max_s         = obs.max_steps

    rescued_score = rescued / total if total > 0 else 0.0
    time_score    = 1.0 - (steps_used / max_s) if max_s > 0 else 0.0
    final         = round(max(0.0, min(1.0, rescued_score * 0.75 + time_score * 0.25)), 4)

    success = final >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=steps_used, score=final, rewards=rewards_list)
    
    summary = {
        "steps":     steps_used,
        "max_steps": max_s,
        "rescued":   rescued,
        "total":     total,
        "score":     final,
        "elapsed":   elapsed,
    }
    return final, summary

# ─── Main ─────────────────────────────────────────────────────────────────────

async def main_async():
    total_start = time.time()
    
    print(f"\n{'='*50}")
    print(f"LLM CONFIG:")
    print(f"  API_BASE_URL: {API_BASE_URL}")
    print(f"  MODEL_NAME:   {MODEL_NAME}")
    print(f"  HF_TOKEN:     {'set' if API_KEY else 'MISSING'}")
    print(f"  IMAGE:        {LOCAL_IMAGE_NAME or 'using BASE_URL'}")
    print(f"{'='*50}\n")
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    scores = {}
    summaries = {}

    # Connect to environment — use docker image if provided
    if LOCAL_IMAGE_NAME:
        env = await DisasterEnv.from_docker_image(LOCAL_IMAGE_NAME)
        env_context = env
    else:
        env_context = DisasterEnv(base_url=BASE_URL)

    try:
        async with env_context as env:
            for difficulty in DIFFICULTIES:
                elapsed = time.time() - total_start
                remaining = TOTAL_TIME_BUDGET - elapsed

                if remaining < 60:
                    print(f"\nTime budget exhausted — skipping {difficulty.upper()}")
                    scores[difficulty] = 0.0
                    summaries[difficulty] = None
                    continue

                print(f"\n[Budget] {remaining:.0f}s remaining for {difficulty.upper()}")
                score, summary = await run_episode(client, env, difficulty, remaining)
                scores[difficulty] = score
                summaries[difficulty] = summary
    finally:
        total_elapsed = time.time() - total_start

        print(f"\n{'='*50}")
        print("ALL EPISODE SUMMARIES:")
        print(f"{'='*50}")
        for diff, s in summaries.items():
            if s:
                print(f"\n  {diff.upper()}")
                print(f"    Steps     : {s['steps']}/{s['max_steps']}")
                print(f"    Rescued   : {s['rescued']}/{s['total']}")
                print(f"    Score     : {s['score']:.4f}")
                print(f"    Time      : {s['elapsed']:.1f}s")

        print(f"\n{'='*50}")
        print("FINAL SCORES:")
        for diff, score in scores.items():
            print(f"  {diff.upper():10s}: {score:.4f}")
        print(f"  {'AVERAGE':10s}: {sum(scores.values())/len(scores):.4f}")
        print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
        print(f"{'='*50}")

def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
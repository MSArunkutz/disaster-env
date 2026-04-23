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
import glob
from pathlib import Path
from datetime import datetime
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

def log_metrics(lookahead: float, recovery: int, slope: float) -> None:
    print(f"[METRICS] lookahead={lookahead:.2f} recovery_cases={recovery} reward_slope={slope:+.4f}", flush=True)

def log_training_step(step, obs_dict, user_prompt, response_text, action_data, reward, done) -> dict:
    return {
        "step":           step,
        "observation":    obs_dict,
        "agent_prompt":   user_prompt,
        "agent_response": response_text,
        "action":         action_data,
        "reward":         reward,
        "done":           done,
    }

def merge_all_training_data() -> Path | None:
    combined_file = TRAINING_DATA_DIR / "combined_training.jsonl"
    run_files = sorted(glob.glob(str(TRAINING_DATA_DIR / "run_*.jsonl")))
    if not run_files:
        print("No training data files found to merge.")
        return None
    total = 0
    with open(combined_file, "w") as out:
        for rf in run_files:
            with open(rf, "r") as inp:
                for line in inp:
                    out.write(line)
                    total += 1
    print(f"Merged {len(run_files)} run files → {combined_file} ({total} steps)")
    return combined_file

def _compute_slope(values: list) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0

# ─── Config ───────────────────────────────────────────────────────────────────

SUCCESS_SCORE_THRESHOLD = 0.5
BENCHMARK = "disaster_env"

UNLIMITED_MODE = os.getenv("UNLIMITED_MODE", "false").lower() == "true"
TOTAL_TIME_BUDGET = 86400 if UNLIMITED_MODE else 18 * 60  # 24h or 18min
MAX_STEPS = int(os.getenv("MAX_STEPS", "200"))
TEMPERATURE  = 0.2
MAX_TOKENS   = 350

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
RUN_DIFFICULTY   = os.getenv("RUN_DIFFICULTY", "").lower()
if RUN_DIFFICULTY in ("easy", "medium", "hard"):
    DIFFICULTIES = [RUN_DIFFICULTY]
DIFFICULTY_SEEDS = {"easy": 1, "medium": 2, "hard": 3}

CASCADE_INTERVALS = {"easy": 20, "medium": 15, "hard": 12}

# ─── Training Data ────────────────────────────────────────────────────────────

TRAINING_DATA_DIR = Path("training_data")
TRAINING_DATA_DIR.mkdir(exist_ok=True)


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an emergency flood response coordinator managing a long-horizon rescue mission.

RESOURCES:
- rescue_team_1, rescue_team_2: extract people, speed 60km/h
  Rate: MILD=14/step, MODERATE=10/step, CRITICAL=6/step
- helicopter: fast extraction, speed 180km/h, capacity 15, can chain zones
- medical_unit: treats injured → reduces severity (CRITICAL→MODERATE→MILD)
  Has 50 supplies. Must restock at base when empty.

ENVIRONMENT RULES (use to plan ahead):
- CASCADE: every N steps, any unattended zone gains extra casualties — pre-position resources BEFORE cascade hits
- Travel time = ceil(distance / speed). A busy resource cannot be redirected until it returns.
- Severity reduction DOUBLES rescue rate — treat CRITICAL zones first for maximum efficiency
- CHAIN: helicopter targets list = multi-zone sequence in one deployment

PLANNING STRATEGY — think 5 steps ahead:
1. TRIAGE: Identify which zones will cascade soonest. Pre-position resources BEFORE cascade hits.
2. SEQUENCE: Treat CRITICAL zones with medical_unit FIRST — this doubles extraction speed later.
3. CHAIN: Route helicopter through multiple zones in one trip to cover ground efficiently.
4. RESTOCK: Track medical_unit supplies — send it to base BEFORE it empties to avoid wasted steps.
5. COMMIT: Once you assign a resource to a zone sequence, follow through — switching wastes travel time.

MULTI-STEP DEPENDENCY CHAIN (exploit this):
  medical_unit treats CRITICAL → zone becomes MODERATE → rescue teams extract 67% faster

RESPOND WITH:
<think>In the next 5 steps I will: [2 sentences describing your plan and why]</think>
{
    "deployments": [
        {"resource_id": "rescue_team_1", "targets": ["Zone_A"]},
        {"resource_id": "rescue_team_2", "targets": ["Zone_B"]},
        {"resource_id": "helicopter",    "targets": ["Zone_A", "Zone_C"]},
        {"resource_id": "medical_unit",  "targets": ["Zone_A"]}
    ]
}
Only include FREE resources. Skip busy ones. Always output valid JSON after </think>."""


# ─── Action Parser ────────────────────────────────────────────────────────────

def parse_action(response_text: str) -> DisasterAction:
    try:
        clean = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
        clean = re.sub(r"```json|```", "", clean).strip()
        data  = json.loads(clean)
        deployments = data.get("deployments", [])
        seen, unique = set(), []
        for d in deployments:
            rid = d.get("resource_id")
            if rid and rid not in seen:
                seen.add(rid)
                unique.append(d)
        return DisasterAction(deployments=unique)
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

def _print_step(step: int, obs: DisasterObservation, result) -> None:
    print(f"\n[Step {step}]")
    for line in obs.resources_summary.split("\n")[1:]:
        line = line.strip()
        if not line:
            continue
        res = line.split(":")[0].strip()
        if "FREE" in line:
            status = "at base, ready"
        elif "→" in line and "WORKING" not in line:
            zone   = line.split("→")[1].split("(")[0].strip()
            eta    = line.split("(")[1].replace("min)", "").strip() if "(" in line else "?"
            status = f"en route to {zone} — ETA {eta} steps"
        elif "WORKING" in line:
            zone   = line.split("@")[1].split("(")[0].strip() if "@" in line else "?"
            eta    = line.split("(")[1].replace("min)", "").strip() if "(" in line else "?"
            status = f"working at {zone} — done in {eta} steps"
        elif "RETURNING" in line:
            eta    = line.split("(")[1].replace("min)", "").strip() if "(" in line else "?"
            status = f"returning to base — ETA {eta} steps"
        elif "RESTOCKING" in line:
            eta    = line.split("(")[1].replace("min)", "").strip() if "(" in line else "?"
            status = f"restocking at base — {eta} steps remaining"
        else:
            status = line
        print(f"  {res:20s} {status}")
    print(f"  {'─'*46}")
    print(f"  Rescued: {obs.total_rescued}/{obs.total_casualties} | "
          f"Reward: {result.reward:.4f} | "
          f"Steps left: {obs.max_steps - obs.current_step}")


async def run_episode(client: OpenAI, env: DisasterEnv, difficulty: str, time_remaining: float) -> float:
    start_time = time.time()
    deadline   = start_time + time_remaining - 30  # 30s safety buffer
    rewards_list = []

    cascade_every = CASCADE_INTERVALS[difficulty]

    seed   = DIFFICULTY_SEEDS[difficulty]
    result = await env.reset(seed=seed)
    obs    = result.observation

    training_steps = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_start(task=difficulty, env=BENCHMARK, model=MODEL_NAME)
    print(f"\n{'='*50}")
    print(f"Starting episode — Difficulty: {difficulty.upper()}")
    print(f"{'='*50}")

    # Planning metrics
    cascade_preempted = 0
    cascade_total     = 0
    rescued_history   = []
    recovery_cases    = 0
    stall_length      = 0

    for step in range(1, MAX_STEPS + 1):
        if time.time() > deadline:
            print(f"\n⏱️  Time budget hit at step {step} — stopping.")
            break
        if result.done:
            print(f"Episode complete at step {step}.")
            break

        user_prompt = build_user_prompt(obs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
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
                print(f"LLM error ({exc}). Skipping.")
            response_text = '{"deployments": []}'

        action = parse_action(response_text)
        result = await env.step(action)
        obs    = result.observation

        # Training data
        try:
            obs_dict = obs.model_dump()
        except Exception:
            obs_dict = {"current_step": obs.current_step,
                        "total_rescued": obs.total_rescued,
                        "total_casualties": obs.total_casualties}
        try:
            action_data = action.model_dump().get("deployments", [])
        except Exception:
            action_data = [{"resource_id": d.get("resource_id"), "targets": d.get("targets", [])}
                           for d in action.deployments]
        training_steps.append(log_training_step(
            step          = step,
            obs_dict      = obs_dict,
            user_prompt   = user_prompt,
            response_text = response_text,
            action_data   = action_data,
            reward        = result.reward or 0.0,
            done          = result.done,
        ))

        # Planning metrics
        rescued_history.append(obs.total_rescued)
        if len(rescued_history) > 1:
            if rescued_history[-1] == rescued_history[-2]:
                stall_length += 1
            else:
                if stall_length >= 3:
                    recovery_cases += 1
                stall_length = 0
        if step % cascade_every == 0:
            cascade_total += 1
            attended = "✓" in obs.zones_summary
            deployed = bool(action.deployments)
            if attended or deployed:
                cascade_preempted += 1

        # Logs
        action_str = json.dumps([d.get("resource_id") for d in action.deployments])
        reward_val = result.reward or 0.0
        rewards_list.append(reward_val)
        log_step(step=step, action=action_str, reward=reward_val, done=result.done, error=None)
        _print_step(step, obs, result)
    # Save training data for this episode
    run_file = TRAINING_DATA_DIR / f"run_{difficulty}_{len(training_steps)}steps_{run_id}.jsonl"
    with open(run_file, "w") as f:
        for step_data in training_steps:
            f.write(json.dumps(step_data) + "\n")
    print(f"\n  Training data: {run_file} ({len(training_steps)} steps)")

    # Episode summary
    elapsed       = time.time() - start_time
    rescued       = obs.total_rescued
    total         = obs.total_casualties
    steps_used    = obs.current_step
    max_s         = obs.max_steps

    rescued_score = rescued / total if total > 0 else 0.0
    time_score    = 1.0 - (steps_used / max_s) if max_s > 0 else 0.0
    final         = round(max(0.0, min(1.0, rescued_score * 0.75 + time_score * 0.25)), 4)

    success = final >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=steps_used, score=final, rewards=rewards_list)

    lookahead_score = cascade_preempted / cascade_total if cascade_total > 0 else 1.0
    last_50 = rewards_list[-50:] if len(rewards_list) >= 50 else rewards_list
    reward_slope = _compute_slope(last_50)
    log_metrics(lookahead=lookahead_score, recovery=recovery_cases, slope=reward_slope)

    print(f"\n  Planning Quality:")
    print(f"    Lookahead score : {lookahead_score:.2f}  (pre-empted {cascade_preempted}/{cascade_total} cascade events)")
    print(f"    Recovery cases  : {recovery_cases}     (bounced back after 3+ step stalls)")
    print(f"    Reward slope    : {reward_slope:+.4f} ({'improving' if reward_slope >= 0 else 'declining'} over last {len(last_50)} steps)")

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
    print(f"  UNLIMITED:    {'YES — no step/time limits' if UNLIMITED_MODE else 'no'}")
    print(f"  DIFFICULTIES: {DIFFICULTIES}")
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
        if scores:
            print(f"  {'AVERAGE':10s}: {sum(scores.values())/len(scores):.4f}")
        else:
            print(f"  {'AVERAGE':10s}: no episodes completed")
        print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
        print(f"{'='*50}")

def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        merge_all_training_data()
        return
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
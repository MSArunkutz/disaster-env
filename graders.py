"""
Graders for Disaster Response Environment.
All return float in 0.0–1.0.
"""


def clamp(score: float) -> float:
    return round(max(0.0, min(1.0, score)), 4)


def compute_score(env) -> float:
    total    = env.total_casualties
    rescued  = env.total_rescued
    steps    = env.current_step
    max_steps= env.max_steps
    zones    = env.zones

    if total == 0:
        return 0.0

    # Rescue ratio
    rescued_score = rescued / total

    # Time efficiency
    time_score = 1.0 - (steps / max_steps)

    # Response quality on critical zones
    critical_zones = [
        z for z in zones.values()
        if z.get("original_severity") == "critical"
    ]
    if critical_zones:
        penalties = []
        for z in critical_zones:
            if z["neglect_steps"] <= 6:
                penalties.append(0.1)
            elif z["neglect_steps"] <= 12:
                penalties.append(0.0)
            else:
                penalties.append(-0.1)
        response_score = sum(penalties) / len(penalties)
    else:
        response_score = 0.0

    # Severity reduction bonus
    severity_bonus = min(getattr(env, "severity_changes", 0) * 0.05, 0.15)

    score = (
        rescued_score  * 0.45 +
        time_score     * 0.20 +
        response_score * 0.20 +
        severity_bonus * 0.15
    )
    return clamp(score)


def grade_easy(env=None) -> float:
    if env is None:
        return 0.0
    if env.difficulty != "easy":
        raise ValueError(f"Expected easy, got {env.difficulty}")
    return compute_score(env)


def grade_medium(env=None) -> float:
    if env is None:
        return 0.0
    if env.difficulty != "medium":
        raise ValueError(f"Expected medium, got {env.difficulty}")
    base = compute_score(env)
    ignored = sum(
        1 for z in env.zones.values()
        if z.get("original_severity") == "critical" and z["rescued"] == 0
    )
    return clamp(base - ignored * 0.05)


def grade_hard(env=None) -> float:
    if env is None:
        return 0.0
    if env.difficulty != "hard":
        raise ValueError(f"Expected hard, got {env.difficulty}")
    base = compute_score(env)
    ignored = sum(
        1 for z in env.zones.values()
        if z.get("original_severity") == "critical" and z["rescued"] == 0
    )
    return clamp(base - ignored * 0.03)


GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}

def grade(env=None) -> float:
    if env is None:
        return 0.0
    grader = GRADERS.get(env.difficulty)
    if not grader:
        raise ValueError(f"No grader for difficulty: {env.difficulty}")
    return grader(env)
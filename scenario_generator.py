"""
Scenario Generator for Disaster Response Environment.
Pure randomization — no LLM calls.
"""

import os
import random
import math
from typing import Dict, Any

# ─── Common Configurations ─────────────────────────────────────────────────
UNLIMITED_MODE = os.getenv("UNLIMITED_MODE", "false").lower() == "true"
_default_max = "9999" if UNLIMITED_MODE else "200"
MAX_STEPS = int(os.getenv("MAX_STEPS", _default_max))

# ─── Difficulty Configuration ─────────────────────────────────────────────────

DIFFICULTY_CONFIG = {
    "easy": {
        "zones":            3,
        "max_steps":        MAX_STEPS,
        "grid_size":        5,        
        "cas_range":        (5, 15),
        "severities":       ["mild", "moderate"],
        "cascade_every":    20,
        "cascade_amount":   2,
    },
    "medium": {
        "zones":            5,
        "max_steps":        MAX_STEPS,
        "grid_size":        8,        
        "cas_range":        (8, 20),
        "severities":       ["moderate", "critical"],
        "cascade_every":    15,
        "cascade_amount":   3,
    },
    "hard": {
        "zones":            8,
        "max_steps":        MAX_STEPS,
        "grid_size":        12,       
        "cas_range":        (10, 25),
        "severities":       ["moderate", "critical"],
        "cascade_every":    12,
        "cascade_amount":   5,
    }
}


# ─── Resource Configuration ───────────────────────────────────────────────────

RESOURCES_CONFIG = {
    "rescue_team_1": {
        "type":         "rescue_team",
        "speed":        1.0,
        "capacity":     None,
    },
    "rescue_team_2": {
        "type":         "rescue_team",
        "speed":        1.0,
        "capacity":     None,
    },
    "helicopter": {
        "type":         "helicopter",
        "speed":        3.0,
        "capacity":     15,
    },
    "medical_unit": {
        "type":         "medical_unit",
        "speed":        0.8,
        "treat_rate":   10,           # people treated per step
        "max_supplies": 50,           # total supplies
        "restock_steps": 5,           # steps to restock at base
    },
}

# Rescue rate depends on zone severity
RESCUE_RATES = {
    "mild":     14,
    "moderate": 10,
    "critical": 6,
}

# Severity reduction thresholds
SEVERITY_TREAT_THRESHOLDS = {
    "critical": 20,   # treat 20 people → CRITICAL becomes MODERATE
    "moderate": 15,   # treat 15 people → MODERATE becomes MILD
}


# ─── Utilities ────────────────────────────────────────────────────────────────

def euclidean_distance(p1: tuple, p2: tuple) -> float:
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def travel_steps(distance_km: float, speed_kmpm: float) -> int:
    return max(1, math.ceil(distance_km / speed_kmpm))


# ─── Zone Generator ───────────────────────────────────────────────────────────

def generate_zones(config: Dict) -> Dict[str, Any]:
    zones = {}
    for i in range(config["zones"]):
        zone_id  = f"Zone_{chr(65 + i)}"
        x        = round(random.uniform(1, config["grid_size"]), 1)
        y        = round(random.uniform(1, config["grid_size"]), 1)
        casualties = random.randint(*config["cas_range"])
        severity   = random.choice(config["severities"])

        zones[zone_id] = {
            "coords":               (x, y),
            "casualties":           casualties,
            "initial_casualties":   casualties,
            "severity":             severity,
            "original_severity":    severity,
            "rescued":              0,
            "treated":              0,        # people treated by medical unit
            "neglect_steps":        0,
            "is_attended":          False,
        }
    return zones


# ─── Resource Initializer ─────────────────────────────────────────────────────

def initialize_resources() -> Dict[str, Any]:
    resources = {}
    base = (0, 0)

    for resource_id, config in RESOURCES_CONFIG.items():
        r = {
            "type":             config["type"],
            "speed":            config["speed"],
            "capacity":         config.get("capacity"),
            "state":            "available",
            "current_location": base,
            "target_zone":      None,
            "chain_queue":      [],
            "steps_remaining":  0,
            "capacity_used":    0,
        }
        # Medical unit extras
        if config["type"] == "medical_unit":
            r["treat_rate"]    = config["treat_rate"]
            r["supplies"]      = config["max_supplies"]
            r["max_supplies"]  = config["max_supplies"]
            r["restock_steps"] = config["restock_steps"]

        resources[resource_id] = r

    return resources


# ─── Main Generator ───────────────────────────────────────────────────────────

def generate_scenario(difficulty: str) -> Dict[str, Any]:
    if difficulty not in DIFFICULTY_CONFIG:
        raise ValueError(f"Invalid difficulty: {difficulty}")

    config    = DIFFICULTY_CONFIG[difficulty]
    zones     = generate_zones(config)
    resources = initialize_resources()
    total     = sum(z["casualties"] for z in zones.values())

    return {
        "difficulty":       difficulty,
        "zones":            zones,
        "resources":        resources,
        "base_coords":      (0, 0),
        "max_steps":        config["max_steps"],
        "cascade_every":    config["cascade_every"],
        "cascade_amount":   config["cascade_amount"],
        "total_casualties": total,
    }
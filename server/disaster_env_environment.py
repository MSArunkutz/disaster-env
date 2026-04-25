"""
Disaster Response Environment — Core Simulation.
Flood disaster with time-based resource management,
parallel deployments, and medical unit severity reduction.
"""

import copy
import math
from typing import Dict, Any, List

try:
    from ..models import DisasterAction, DisasterObservation
    from ..scenario_generator import (
        generate_scenario, euclidean_distance,
        travel_steps, RESCUE_RATES, SEVERITY_TREAT_THRESHOLDS
    )
except (ImportError, ModuleNotFoundError):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import DisasterAction, DisasterObservation
    from scenario_generator import (
        generate_scenario, euclidean_distance,
        travel_steps, RESCUE_RATES, SEVERITY_TREAT_THRESHOLDS
    )

from openenv.core.env_server import Environment


_MILESTONE_INTERVALS = {"easy": 4, "medium": 5, "hard": 7}

def calculate_milestones(difficulty: str, max_steps: int) -> set:
    """Return milestone step numbers scaled to max_steps and difficulty."""
    n = _MILESTONE_INTERVALS.get(difficulty, 4)
    steps = {round(max_steps * i / n) for i in range(1, n)}
    steps.add(max_steps)
    return steps


class DisasterEnvEnvironment(Environment):

    def __init__(self, difficulty: str = "easy"):
        self.difficulty = difficulty
        self._reset_state()

    # ─── Reset ────────────────────────────────────────────────────────────────

    def reset(self, episode_id=None, seed=None) -> DisasterObservation:
        seed_map = {1: "easy", 2: "medium", 3: "hard"}
        if seed and seed in seed_map:
            self.difficulty = seed_map[seed]
        self._reset_state()
        return self._build_observation(
            last_action_result="Episode started. Deploy all available resources."
        )

    def _reset_state(self):
        self.scenario         = generate_scenario(self.difficulty)
        self.current_step     = 0
        self.total_rescued    = 0
        self.severity_changes = 0      # track severity reductions for grader
        self.zones            = self.scenario["zones"]
        self.resources        = self.scenario["resources"]
        self.base_coords      = self.scenario["base_coords"]
        self.max_steps        = self.scenario["max_steps"]
        self.cascade_every    = self.scenario["cascade_every"]
        self.cascade_amount   = self.scenario["cascade_amount"]
        self.total_casualties = self.scenario["total_casualties"]
        self.milestones         = calculate_milestones(self.difficulty, self.max_steps)
        self._last_rescued      = 0
        self._reward_breakdown  = None

    # ─── Step ─────────────────────────────────────────────────────────────────

    def step(self, action: DisasterAction) -> DisasterObservation:
        self.current_step += 1
        results = []

        # Process ALL deployments in parallel
        for deployment in action.deployments:
            resource_id = deployment.get("resource_id", "hold")
            targets     = deployment.get("targets", [])

            if resource_id == "hold":
                results.append("Hold.")
            else:
                result = self._process_single_deployment(resource_id, targets)
                results.append(result)

        if not action.deployments:
            results.append("No deployments this step.")

        # Advance resources
        self._advance_resources()

        # Cascade neglected zones
        self._apply_cascade()

        # Update neglect counters
        self._update_neglect()

        # Calculate reward
        reward = self._calculate_reward()

        # Check done
        done = self._check_done()

        return self._build_observation(
            last_action_result=" | ".join(results),
            reward=reward,
            done=done,
        )

    # ─── Single Deployment ────────────────────────────────────────────────────

    def _process_single_deployment(self, resource_id: str, targets: list) -> str:
        if resource_id not in self.resources:
            return f"Invalid resource: {resource_id}"

        resource = self.resources[resource_id]

        if resource["state"] != "available":
            return f"{resource_id} is {resource['state']} — skipped"

        if not targets:
            return f"No targets for {resource_id}"

        for t in targets:
            if t not in self.zones:
                return f"Invalid zone: {t}"

        # Medical unit restocking check
        if resource["type"] == "medical_unit" and resource.get("supplies", 0) <= 0:
            return f"medical_unit has no supplies — must restock first"

        first_target = targets[0]
        chain_queue  = targets[1:]

        dist  = euclidean_distance(
            resource["current_location"],
            self.zones[first_target]["coords"]
        )
        steps = travel_steps(dist, resource["speed"])

        resource["state"]           = "travelling"
        resource["target_zone"]     = first_target
        resource["chain_queue"]     = chain_queue
        resource["steps_remaining"] = steps
        resource["capacity_used"]   = 0

        chain_text = f" → chain {chain_queue}" if chain_queue else ""
        return f"{resource_id} → {first_target}{chain_text} ETA:{steps}min"

    # ─── Resource Advancement ─────────────────────────────────────────────────

    def _advance_resources(self):
        for resource_id, resource in self.resources.items():

            if resource["state"] == "available":
                continue

            # Restocking at base
            if resource["state"] == "restocking":
                resource["steps_remaining"] -= 1
                if resource["steps_remaining"] <= 0:
                    resource["supplies"]      = resource["max_supplies"]
                    resource["state"]         = "available"
                    resource["current_location"] = self.base_coords
                continue

            resource["steps_remaining"] -= 1

            # Travelling → Arrived
            if resource["state"] == "travelling" and resource["steps_remaining"] <= 0:
                zone_id  = resource["target_zone"]
                zone     = self.zones[zone_id]

                resource["state"]            = "working"
                resource["current_location"] = zone["coords"]
                zone["is_attended"]          = True
                zone["neglect_steps"]        = 0

                # Work duration based on severity
                work_steps = {"mild": 2, "moderate": 4, "critical": 6}
                resource["steps_remaining"] = work_steps.get(zone["severity"], 4)

            # Working
            elif resource["state"] == "working":
                zone_id = resource["target_zone"]
                zone    = self.zones[zone_id]

                if resource["type"] == "medical_unit":
                    self._advance_medical_unit(resource, zone)
                else:
                    self._advance_rescue_resource(resource, zone)
        self._check_returning()

    def _advance_rescue_resource(self, resource: Dict, zone: Dict):
        """Rescue teams and helicopter extract people."""
        rescuable = zone["casualties"] - zone["rescued"]
        
        if rescuable <= 0:
            zone["is_attended"] = False
            res_id = [k for k, v in self.resources.items() if v is resource][0]
            self._next_destination(res_id, resource)
            return

        # Rescue rate depends on zone severity
        rate = RESCUE_RATES.get(zone["severity"], 10)

        # Helicopter capacity check
        if resource["capacity"] is not None:
            remaining_cap = resource["capacity"] - resource["capacity_used"]
            rate = min(rate, remaining_cap)

        rescued_now        = min(rate, rescuable)
        zone["rescued"]    += rescued_now
        self.total_rescued += rescued_now

        if resource["capacity"] is not None:
            resource["capacity_used"] += rescued_now

        # Check if done — zone cleared OR capacity full OR work steps exhausted
        zone_cleared    = zone["rescued"] >= zone["casualties"]
        capacity_full   = (resource["capacity"] is not None and 
                        resource["capacity_used"] >= resource["capacity"])
        work_done       = resource["steps_remaining"] <= 0

        if zone_cleared or capacity_full or work_done:
            zone["is_attended"] = False
            res_id = [k for k, v in self.resources.items() if v is resource][0]
            # Reset capacity for next trip
            if capacity_full:
                resource["capacity_used"] = 0
            self._next_destination(res_id, resource)

    def _advance_medical_unit(self, resource: Dict, zone: Dict):
        """Medical unit treats injured — reduces zone severity."""
        if resource.get("supplies", 0) <= 0:
            # Out of supplies — return to base to restock
            zone["is_attended"] = False
            dist  = euclidean_distance(resource["current_location"], self.base_coords)
            steps = travel_steps(dist, resource["speed"])
            resource["state"]           = "returning"
            resource["steps_remaining"] = steps
            resource["target_zone"]     = None
            return

        treat_rate = resource.get("treat_rate", 10)
        treatable  = zone["casualties"] - zone.get("treated", 0)

        if treatable <= 0:
            zone["is_attended"] = False
            res_id = [k for k, v in self.resources.items() if v is resource][0]
            self._next_destination(res_id, resource)
            return

        treated_now         = min(treat_rate, treatable, resource["supplies"])
        zone["treated"]     = zone.get("treated", 0) + treated_now
        resource["supplies"] -= treated_now

        # Check severity reduction
        threshold = SEVERITY_TREAT_THRESHOLDS.get(zone["severity"])
        if threshold and zone["treated"] >= threshold:
            old_severity = zone["severity"]
            if zone["severity"] == "critical":
                zone["severity"] = "moderate"
                self.severity_changes += 1
            elif zone["severity"] == "moderate":
                zone["severity"] = "mild"
                self.severity_changes += 1
            zone["treated"] = 0  # reset treated counter for next reduction

        # Work steps done or supplies out
        if resource["steps_remaining"] <= 0 or resource["supplies"] <= 0:
            zone["is_attended"] = False
            res_id = [k for k, v in self.resources.items() if v is resource][0]
            if resource["supplies"] <= 0:
                # Go restock
                dist  = euclidean_distance(resource["current_location"], self.base_coords)
                steps = travel_steps(dist, resource["speed"])
                resource["state"]           = "returning"
                resource["steps_remaining"] = steps
                resource["target_zone"]     = None
            else:
                self._next_destination(res_id, resource)

    def _next_destination(self, resource_id: str, resource: Dict):
        if resource["chain_queue"]:
            next_zone = resource["chain_queue"].pop(0)
            dist  = euclidean_distance(
                resource["current_location"],
                self.zones[next_zone]["coords"]
            )
            steps = travel_steps(dist, resource["speed"])
            resource["state"]           = "travelling"
            resource["target_zone"]     = next_zone
            resource["steps_remaining"] = steps
        else:
            dist  = euclidean_distance(resource["current_location"], self.base_coords)
            steps = travel_steps(dist, resource["speed"])
            resource["state"]           = "returning"
            resource["target_zone"]     = None
            resource["steps_remaining"] = steps

    # ─── Returning → Available ────────────────────────────────────────────────

    def _check_returning(self):
        for resource in self.resources.values():
            if resource["state"] == "returning" and resource["steps_remaining"] <= 0:
                if resource["type"] == "medical_unit" and resource.get("supplies", 1) <= 0:
                    resource["state"]           = "restocking"
                    resource["steps_remaining"] = resource.get("restock_steps", 5)
                else:
                    resource["state"]           = "available"
                    resource["current_location"] = self.base_coords
                    resource["capacity_used"]    = 0

    # ─── Cascade ──────────────────────────────────────────────────────────────

    def _apply_cascade(self):
        for zone in self.zones.values():
            if not zone["is_attended"]:
                if zone["neglect_steps"] > 0 and \
                   zone["neglect_steps"] % self.cascade_every == 0:
                    remaining = zone["casualties"] - zone["rescued"]
                    if remaining > 0:
                        zone["casualties"]    += self.cascade_amount
                        self.total_casualties += self.cascade_amount

    def _update_neglect(self):
        for zone in self.zones.values():
            if not zone["is_attended"]:
                zone["neglect_steps"] += 1
            else:
                zone["neglect_steps"] = 0

    # ─── Scoring ──────────────────────────────────────────────────────────────

    def _calculate_reward(self) -> float:
        if self.total_casualties == 0:
            return 0.0

        # Dense: small reward for each new rescue this step
        rescued_delta      = self.total_rescued - self._last_rescued
        self._last_rescued = self.total_rescued
        dense              = round(rescued_delta * 0.002, 4)

        is_milestone = self.current_step in self.milestones
        is_done = self.current_step >= self.max_steps or self._check_done()

        if not (is_milestone or is_done):
            return dense

        try:
            from graders import compute_score, compute_score_breakdown
        except ImportError:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from graders import compute_score, compute_score_breakdown

        score = compute_score(self)
        breakdown = compute_score_breakdown(self)

        if is_milestone:
            score = min(1.0, score + 0.1)

        breakdown["total"] = round(score, 4)
        self._reward_breakdown = breakdown

        return round(score + dense, 4)

    def final_score(self) -> float:
        if self.total_casualties == 0:
            return 0.0

        rescued_score  = self.total_rescued / self.total_casualties
        time_score     = 1.0 - (self.current_step / self.max_steps)

        critical_zones = [z for z in self.zones.values() if z["original_severity"] == "critical"]
        if critical_zones:
            bonuses = []
            for z in critical_zones:
                if z["neglect_steps"] <= 6:
                    bonuses.append(0.1)
                elif z["neglect_steps"] <= 12:
                    bonuses.append(0.0)
                else:
                    bonuses.append(-0.1)
            response_score = sum(bonuses) / len(bonuses)
        else:
            response_score = 0.0

        severity_bonus = min(self.severity_changes * 0.05, 0.15)

        score = (
            rescued_score  * 0.45 +
            time_score     * 0.20 +
            response_score * 0.20 +
            severity_bonus * 0.15
        )
        return round(max(0.0, min(1.0, score)), 4)

    # ─── Done ─────────────────────────────────────────────────────────────────

    def _check_done(self) -> bool:
        all_rescued = all(
            z["rescued"] >= z["casualties"] for z in self.zones.values()
        )
        return all_rescued or self.current_step >= self.max_steps

    # ─── Observation Builder ──────────────────────────────────────────────────

    def _build_observation(
        self,
        last_action_result: str = "",
        reward: float = 0.0,
        done: bool = False
    ) -> DisasterObservation:
        return DisasterObservation(
            current_step        = self.current_step,
            max_steps           = self.max_steps,
            time_elapsed        = self.current_step,
            difficulty          = self.difficulty,
            zones_summary       = self._format_zones(),
            resources_summary   = self._format_resources(),
            last_action_result  = last_action_result,
            total_casualties    = self.total_casualties,
            total_rescued       = self.total_rescued,
            available_actions   = self._format_available_actions(),
            done                = done,
            reward              = reward,
            reward_breakdown    = self._reward_breakdown,
        )

    # ─── Formatters ───────────────────────────────────────────────────────────

    def _format_zones(self) -> str:
        lines = ["ZONES:"]
        for zone_id, z in self.zones.items():
            remaining = z["casualties"] - z["rescued"]
            attended  = "✓" if z["is_attended"] else f"✗({z['neglect_steps']}s)"
            treated   = f" treated:{z.get('treated',0)}" if z["severity"] in ["critical","moderate"] else ""
            lines.append(
                f"  {zone_id}: {remaining}left|{z['severity'][0].upper()}|"
                f"{z['coords']}|{attended}{treated}"
            )
        return "\n".join(lines)

    def _format_resources(self) -> str:
        lines = ["RESOURCES:"]
        for res_id, r in self.resources.items():
            if r["state"] == "available":
                if r["type"] == "medical_unit":
                    lines.append(f"  {res_id}: FREE|supplies:{r.get('supplies',0)}")
                else:
                    lines.append(f"  {res_id}: FREE")
            elif r["state"] == "travelling":
                lines.append(f"  {res_id}: →{r['target_zone']}({r['steps_remaining']}min)")
            elif r["state"] == "working":
                lines.append(f"  {res_id}: WORKING@{r['target_zone']}({r['steps_remaining']}min)")
            elif r["state"] == "returning":
                lines.append(f"  {res_id}: RETURNING({r['steps_remaining']}min)")
            elif r["state"] == "restocking":
                lines.append(f"  {res_id}: RESTOCKING({r['steps_remaining']}min)")
        return "\n".join(lines)

    def _format_available_actions(self) -> str:
        free  = [r_id for r_id, r in self.resources.items() if r["state"] == "available"]
        zones = list(self.zones.keys())
        return (
            f"FREE RESOURCES: {free}\n"
            f"ZONES: {zones}\n"
            f"Deploy all free resources in one step using deployments list."
        )

    # ─── Snapshot / Restore (required for GRPO parallel rollouts) ────────────

    def snapshot(self) -> Dict[str, Any]:
        """Return a deep-copy of all mutable env state."""
        return {
            "difficulty":       self.difficulty,
            "scenario":         copy.deepcopy(self.scenario),
            "current_step":     self.current_step,
            "total_rescued":    self.total_rescued,
            "severity_changes": self.severity_changes,
            "zones":            copy.deepcopy(self.zones),
            "resources":        copy.deepcopy(self.resources),
            "base_coords":      self.base_coords,
            "max_steps":        self.max_steps,
            "cascade_every":    self.cascade_every,
            "cascade_amount":   self.cascade_amount,
            "total_casualties": self.total_casualties,
            "milestones":       copy.copy(self.milestones),
            "_last_rescued":    self._last_rescued,
            "_reward_breakdown": copy.deepcopy(self._reward_breakdown),
        }

    def restore(self, snap: Dict[str, Any]) -> None:
        """Restore env state from a snapshot produced by snapshot()."""
        self.difficulty        = snap["difficulty"]
        self.scenario          = copy.deepcopy(snap["scenario"])
        self.current_step      = snap["current_step"]
        self.total_rescued     = snap["total_rescued"]
        self.severity_changes  = snap["severity_changes"]
        self.zones             = copy.deepcopy(snap["zones"])
        self.resources         = copy.deepcopy(snap["resources"])
        self.base_coords       = snap["base_coords"]
        self.max_steps         = snap["max_steps"]
        self.cascade_every     = snap["cascade_every"]
        self.cascade_amount    = snap["cascade_amount"]
        self.total_casualties  = snap["total_casualties"]
        self.milestones        = copy.copy(snap["milestones"])
        self._last_rescued     = snap["_last_rescued"]
        self._reward_breakdown = copy.deepcopy(snap["_reward_breakdown"])

    # ─── State ────────────────────────────────────────────────────────────────

    @property
    def state(self):
        return {
            "current_step":     self.current_step,
            "total_rescued":    self.total_rescued,
            "total_casualties": self.total_casualties,
            "difficulty":       self.difficulty,
            "severity_changes": self.severity_changes,
            "done":             self._check_done(),
        }
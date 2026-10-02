"""Audit completed local trajectories/events; no simulator, SDK, network or writes.

Usage: python experiments/E006-900s-fixture-baseline/analyze.py EXPERIMENT_DIR
JSON goes to stdout. Invalid/incomplete evidence exits 2 with an inconclusive
report; a complete failed hypothesis is a valid analysis and exits 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ABS_TOL = 1e-5
REL_TOL = 1e-6
STATE_TOL = 1e-6
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
E006_CONFIG_SHA256 = "dc2c6c56cb3e289f1dffff792df9fb1b16779aab0079f83e74da888619dae754"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value: Any) -> None:
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite JSON number")
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def decode(text: str | bytes) -> dict[str, Any]:
    value = json.loads(text, object_pairs_hook=unique_object)
    require(isinstance(value, dict), "Expected a JSON object")
    finite(value)
    return value


def regular(path: Path) -> Path:
    require(path.resolve() == path and path.is_file(), f"Missing/linked artifact: {path.name}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    with regular(path).open("rb") as stream:
        data = stream.read(MAX_JSON_BYTES + 1)
    require(len(data) <= MAX_JSON_BYTES, f"Oversized JSON: {path.name}")
    return decode(data)


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with regular(path).open("rb") as stream:
        while line := stream.readline(MAX_LINE_BYTES + 1):
            require(len(line) <= MAX_LINE_BYTES, f"Oversized JSONL row: {path.name}")
            rows.append(decode(line))
            require(len(rows) <= 1_000_000, "Too many JSONL rows")
    return rows


def integer(value: Any, name: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"Invalid integer: {name}")
    return value


def number(value: Any) -> float:
    require(type(value) in (float, int), "Expected numeric value, not boolean")
    result = float(value)
    require(math.isfinite(result), "Expected finite numeric value")
    return result


def vector(value: Any, length: int) -> list[float]:
    require(isinstance(value, list) and len(value) == length, "Invalid vector dimensions")
    return [number(item) for item in value]


def matrix(value: Any, rows: int, columns: int) -> list[list[float]]:
    require(isinstance(value, list) and len(value) == rows, "Invalid matrix row count")
    return [vector(row, columns) for row in value]


def close(actual: float, expected: Any, name: str, atol: float = ABS_TOL) -> None:
    require(
        math.isclose(actual, number(expected), rel_tol=REL_TOL, abs_tol=atol),
        f"Metric mismatch: {name}",
    )


def requested_matches_resolved(requested: Any, resolved: Any) -> None:
    """Resolved defaults may add fields, but cannot change any requested value."""
    if isinstance(requested, dict):
        require(isinstance(resolved, dict), "Requested/resolved config type mismatch")
        for key, value in requested.items():
            require(key in resolved, "Requested field missing from resolved config")
            requested_matches_resolved(value, resolved[key])
    elif isinstance(requested, list):
        require(
            isinstance(resolved, list) and len(requested) == len(resolved), "Config list mismatch"
        )
        for before, after in zip(requested, resolved, strict=True):
            requested_matches_resolved(before, after)
    else:
        if isinstance(requested, bool) or isinstance(resolved, bool):
            require(type(requested) is type(resolved), "Boolean config type mismatch")
        require(requested == resolved, "Requested/resolved config value mismatch")


def initial_state(root: Path, phase: str, cars: int) -> dict[str, list[list[float]]]:
    data = read_json(root / f"{phase}_initial_state.json")
    require(set(data) == {"root_state", "joint_pos", "joint_vel"}, "Invalid initial state keys")
    roots = matrix(data["root_state"], cars, 13)
    joints = data["joint_pos"]
    require(isinstance(joints, list) and len(joints) == cars, "Invalid joint state rows")
    require(isinstance(joints[0], list) and bool(joints[0]), "Invalid joint state width")
    width = len(joints[0])
    return {
        "root_state": roots,
        "joint_pos": matrix(joints, cars, width),
        "joint_vel": matrix(data["joint_vel"], cars, width),
    }


def evaluation(
    root: Path, phase: str, config: dict[str, Any], state: dict[str, Any], dt: float
) -> dict[str, Any]:
    cars = integer(config["num_envs"], "num_envs", 1)
    steps = integer(config["eval_steps"], "eval_steps", 1)
    summary = read_json(root / f"{phase}_evaluation.json")
    require(summary.get("phase") == phase, "Evaluation phase mismatch")
    require(summary.get("deterministic") is True, "Evaluation must be deterministic")
    require(integer(summary["seed"], "evaluation seed") == config["eval_seed"], "Seed mismatch")
    require(integer(summary["control_steps"], "evaluation steps", 1) == steps, "Step mismatch")
    close(steps * dt, summary["sim_seconds"], "evaluation simulated seconds")
    rows = read_rows(root / f"{phase}_trajectory.jsonl")
    require(len(rows) == steps, f"Incomplete/extra {phase} trajectory steps")
    previous = [row[:2] for row in state["root_state"]]
    returns, distances = [0.0] * cars, [0.0] * cars
    dones_per_car, segments, longest = [0] * cars, [0] * cars, [0] * cars
    for index, row in enumerate(rows):
        require(integer(row["control_step"], "control_step") == index, "Noncontiguous steps")
        close((index + 1) * dt, row["sim_time_s"], "trajectory simulated time")
        positions = matrix(row["positions_xy"], cars, 2)
        matrix(row["actions"], cars, 2)
        rewards = vector(row["rewards"], cars)
        dones = row["dones"]
        require(
            isinstance(dones, list) and len(dones) == cars and all(type(x) is bool for x in dones),
            "Invalid boolean done-vector dimensions",
        )
        for car in range(cars):
            returns[car] += rewards[car]
            if dones[car]:
                dones_per_car[car] += 1
                segments[car] = 0
            else:
                distances[car] += math.hypot(
                    positions[car][0] - previous[car][0],
                    positions[car][1] - previous[car][1],
                )
                segments[car] += 1
                longest[car] = max(longest[car], segments[car])
        previous = positions
    declared_returns = vector(summary["returns_per_car"], cars)
    declared_distances = vector(summary["nonreset_distance_per_car_m"], cars)
    for index in range(cars):
        close(returns[index], declared_returns[index], f"{phase} car return")
        close(distances[index], declared_distances[index], f"{phase} car reset-excluded travel")
    close(sum(returns) / cars, summary["mean_return"], f"{phase} mean return")
    close(sum(distances), summary["nonreset_distance_total_m"], f"{phase} total travel")
    total_dones = sum(dones_per_car)
    require(integer(summary["resets"], "resets") == total_dones, "Total done/reset mismatch")
    causes = summary["terminations"]
    require(isinstance(causes, dict) and "out_of_bounds" in causes, "Missing termination causes")
    for count in causes.values():
        require(integer(count, "termination count") <= total_dones, "Cause count exceeds dones")
    return {
        "control_steps": steps,
        "sim_seconds": steps * dt,
        "returns_per_car": returns,
        "mean_return": sum(returns) / cars,
        "dones_per_car": dones_per_car,
        "total_dones": total_dones,
        "reported_termination_counts": causes,
        "reset_excluded_distance_per_car_m": distances,
        "reset_excluded_distance_total_m": sum(distances),
        "verified_reported_distance_total_m": summary["nonreset_distance_total_m"],
        "longest_observed_reset_free_segment_steps_per_car": longest,
        "longest_observed_reset_free_segment_seconds_per_car": [steps * dt for steps in longest],
        "longest_observed_reset_free_segment_steps": max(longest),
        "longest_observed_reset_free_segment_seconds": max(longest) * dt,
        "segment_note": "Consecutive non-done transitions; excludes terminal/reset transitions. Window-censored, not completed laps.",
        "cause_note": "Cause labels are reported summaries; raw trajectory dones cannot reconstruct physical causes or their overlap.",
    }


def training(root: Path, config: dict[str, Any], dt: float) -> dict[str, Any]:
    cars = integer(config["num_envs"], "num_envs", 1)
    ppo = config["ppo"]
    rollout_steps = integer(ppo["n_steps"], "n_steps", 1)
    epochs = integer(ppo["n_epochs"], "n_epochs", 1)
    require(ppo.get("target_kl") is None, "This full-epoch audit does not support target_kl")
    summary = read_json(root / "training_summary.json")
    total = integer(summary["agent_timesteps"], "agent_timesteps", 1)
    updates = integer(summary["ppo_updates"], "ppo_updates")
    seconds = number(summary["wall_seconds"])
    require(seconds > 0, "Training wall seconds must be positive")
    events = [row for row in read_rows(root / "events.jsonl") if row.get("phase") == "train"]
    require(bool(events), "Missing training events")
    require(events[0].get("event") == "training_started", "Missing initial training event")
    require(events[-1].get("event") == "training_finished", "Missing final training event")
    require(sum(row.get("event") == "training_started" for row in events) == 1, "Repeated start")
    require(sum(row.get("event") == "training_finished" for row in events) == 1, "Repeated finish")
    require(events[0]["target_agent_timesteps"] == config["total_timesteps"], "Target mismatch")
    for key, value in summary.items():
        require(events[-1].get(key) == value, f"Training summary/finished event mismatch: {key}")
    full_rollouts, full_samples, partial_samples, last_count = 0, 0, 0, 0
    partial_events = []
    for row in events:
        kind = row.get("event")
        if kind not in {"rollout", "partial_rollout"}:
            continue
        require(not partial_events, "Collection event after terminal partial rollout")
        rollup = row["reward_terms_rollout"]
        count = integer(rollup["control_steps"], "rollout control_steps", 1)
        samples = integer(rollup["agent_steps"], "rollout agent_steps", 1)
        cumulative = integer(row["agent_timesteps"], "cumulative samples", 1)
        require(samples == count * cars, "Rollout sample/control-step mismatch")
        require(cumulative - last_count == samples, "Missing/duplicated rollout sample accounting")
        close(count * dt, rollup["sim_seconds_per_agent"], "rollout simulated time")
        for values in rollup["contribution_sum_per_agent"].values():
            vector(values, cars)
        if kind == "rollout":
            require(count == rollout_steps, "Completed rollout has wrong length")
            full_rollouts += 1
            full_samples += samples
        else:
            # The stop callback runs before buffer insertion/optimization, even
            # when this final collected transition reaches the nominal length.
            require(count <= rollout_steps, "Partial rollout exceeds configured length")
            require(row.get("optimized") is False, "Partial rollout must declare unoptimized")
            partial_samples += samples
            partial_events.append(
                {"control_steps": count, "agent_samples": samples, "reason": row.get("reason")}
            )
        last_count = cumulative
    require(full_samples + partial_samples == total == last_count, "Training sample mismatch")
    expected_epochs = full_rollouts * epochs
    require(updates == expected_epochs, "PPO epoch count does not match completed full rollouts")
    receipt = read_json(root.parent / "exit_status.json")
    require(
        integer(receipt["exit_code"], "job exit code") == 0, "Job did not complete successfully"
    )
    started = integer(receipt["started_epoch"], "started_epoch", 1)
    finished = integer(receipt["finished_epoch"], "finished_epoch", 1)
    require(finished > started, "Invalid parent job interval")
    whole_seconds = finished - started
    require(seconds <= whole_seconds + 1, "Training time exceeds whole-job clock interval")
    return {
        "collected_agent_samples": total,
        "complete_rollouts": full_rollouts,
        "fully_optimized_unique_agent_samples_inferred": full_samples,
        "unoptimized_partial_agent_samples": partial_samples,
        "partial_rollout_events": partial_events,
        "configured_epochs_per_full_rollout": epochs,
        "reported_ppo_epoch_count": updates,
        "expected_ppo_epoch_count": expected_epochs,
        "optimization_note": "Inferred from completed rollout events and matching epoch counter, not raw optimizer tracing; unique samples, not epoch-multiplied presentations.",
        "training_wall_seconds": seconds,
        "collected_agent_samples_per_training_second": total / seconds,
        "fully_optimized_unique_samples_per_training_second": full_samples / seconds,
        "whole_job_seconds": whole_seconds,
        "whole_job_minus_training_seconds": whole_seconds - seconds,
        "timing_note": "Training summary spans model.learn and final checkpoint save, including recording. Parent clock has one-second granularity; overhead includes startup/evaluation/other job work, not allocation/setup/backup.",
    }


def analyze(directory: Path) -> dict[str, Any]:
    root = directory.resolve(strict=True)
    require(root.is_dir(), "Experiment path must be a directory")
    config = read_json(root / "config.json")
    requested_matches_resolved(read_json(root / "requested_config.json"), config)
    cars = integer(config["num_envs"], "num_envs", 1)
    seed = integer(config["eval_seed"], "eval_seed")
    require(seed < 2**32, "Evaluation seed out of range")
    manifest = read_json(root / "manifest.json")
    require(manifest["config"] == config, "Runtime manifest/config mismatch")
    dt = number(manifest["control_dt_seconds"])
    require(dt > 0, "Control dt must be positive")
    states = {phase: initial_state(root, phase, cars) for phase in ("start", "end")}
    for key, rows in states["start"].items():
        other = states["end"][key]
        require(len(rows[0]) == len(other[0]), "Initial state width mismatch")
        for before_row, after_row in zip(rows, other, strict=True):
            for before, after in zip(before_row, after_row, strict=True):
                close(before, after, "Matched initial state", atol=STATE_TOL)
    phases = {
        phase: evaluation(root, phase, config, states[phase], dt) for phase in ("start", "end")
    }
    train = training(root, config, dt)
    start, end = phases["start"], phases["end"]
    fewer = (
        end["reported_termination_counts"]["out_of_bounds"]
        < start["reported_termination_counts"]["out_of_bounds"]
    )
    no_lower = (
        end["verified_reported_distance_total_m"] >= start["verified_reported_distance_total_m"]
    )
    config_hash = hashlib.sha256(regular(root / "requested_config.json").read_bytes()).hexdigest()
    protocol_recipe_matches = config_hash == E006_CONFIG_SHA256
    return {
        "schema_version": 1,
        "analysis_status": "complete",
        "consistency_passed": True,
        "experiment_directory": str(root),
        "comparison_tolerance": {
            "absolute": ABS_TOL,
            "relative": REL_TOL,
            "initial_state_absolute": STATE_TOL,
            "initial_state_relative": REL_TOL,
        },
        "tolerance_note": "Allows float32 producer norms vs Python float64 recomputation; does not relax hypothesis thresholds.",
        "control_dt_seconds": dt,
        "training": train,
        "evaluations": phases,
        "decision": {
            "rule": "Strictly fewer reported out_of_bounds events AND no lower verified reported reset-excluded total travel",
            "strictly_fewer_reported_out_of_bounds": fewer,
            "no_lower_verified_reported_travel": no_lower,
            "rule_passed_for_supplied_run": fewer and no_lower,
            "requested_config_sha256": config_hash,
            "matches_preregistered_e006_recipe": protocol_recipe_matches,
            "attribution_note": "This is the rule result for the supplied data, not E006 attribution. The report must separately establish frozen source, assets, runtime/hardware and prerequisite gates; recipe matching alone is insufficient.",
        },
        "limitations": [
            "Local arithmetic consistency, not authenticated backup, simulator rerun or checkpoint/media inspection",
            "Reported termination labels are not independently reconstructed from trajectory dones",
            "Reset transitions exclude physical pre-reset movement as well as teleport distance",
            "Reset-free observed segments are window-censored and are not completed laps or safe-driving proof",
            "One fixed seed and repeated generated arenas do not establish convergence or generalization",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_directory", type=Path)
    args = parser.parse_args()
    try:
        report = analyze(args.experiment_directory)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        print(
            json.dumps(
                {
                    "analysis_status": "inconclusive",
                    "consistency_passed": False,
                    "error": str(error),
                    "rule_passed_for_supplied_run": None,
                },
                allow_nan=False,
            )
        )
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

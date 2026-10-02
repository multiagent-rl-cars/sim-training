"""Synthetic trajectory/event arithmetic; no ignored outputs, simulator or SDK."""

import json
import runpy
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def analyzer() -> dict[str, Any]:
    script = (
        Path(__file__).resolve().parents[2] / "experiments/E006-900s-fixture-baseline/analyze.py"
    )
    return runpy.run_path(str(script))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    root = tmp_path / "experiment"
    root.mkdir()
    config = {
        "num_envs": 2,
        "eval_steps": 4,
        "eval_seed": 123,
        "total_timesteps": 8,
        "ppo": {"n_steps": 2, "n_epochs": 2},
    }
    write_json(root / "config.json", config)
    write_json(root / "requested_config.json", config)
    write_json(root / "manifest.json", {"config": config, "control_dt_seconds": 0.25})
    for phase in ("start", "end"):
        write_json(
            root / f"{phase}_initial_state.json",
            {
                "root_state": [[0.0] * 13, [10.0] + [0.0] * 12],
                "joint_pos": [[0.0, 0.0], [0.0, 0.0]],
                "joint_vel": [[0.0, 0.0], [0.0, 0.0]],
            },
        )
        # Car0: +1, reset to100, +1, reset to0. Car1: four +1 moves.
        # Expected travel is2+4=6, never the teleport-inclusive202+4.
        positions = [[[1, 0], [11, 0]], [[100, 0], [12, 0]], [[101, 0], [13, 0]], [[0, 0], [14, 0]]]
        write_rows(
            root / f"{phase}_trajectory.jsonl",
            [
                {
                    "control_step": step,
                    "sim_time_s": (step + 1) * 0.25,
                    "positions_xy": position,
                    "actions": [[0.0, 0.0], [0.0, 0.0]],
                    "rewards": [1.0, 2.0],
                    "dones": [step in (1, 3), False],
                }
                for step, position in enumerate(positions)
            ],
        )
        write_json(
            root / f"{phase}_evaluation.json",
            {
                "phase": phase,
                "deterministic": True,
                "seed": 123,
                "control_steps": 4,
                "sim_seconds": 1.0,
                "returns_per_car": [4.0, 8.0],
                "mean_return": 6.0,
                "resets": 2,
                "terminations": {"out_of_bounds": 2, "collision": 0},
                "nonreset_distance_per_car_m": [2.0, 4.0],
                "nonreset_distance_total_m": 6.0,
            },
        )
    summary = {"agent_timesteps": 8, "wall_seconds": 2.0, "ppo_updates": 4}
    write_json(root / "training_summary.json", summary)
    write_rows(
        root / "events.jsonl",
        [
            {"phase": "train", "event": "training_started", "target_agent_timesteps": 8},
            *[
                {
                    "phase": "train",
                    "event": "rollout",
                    "agent_timesteps": count,
                    "reward_terms_rollout": {
                        "control_steps": 2,
                        "agent_steps": 4,
                        "sim_seconds_per_agent": 0.5,
                        "contribution_sum_per_agent": {"toy": [1.0, 1.0]},
                    },
                }
                for count in (4, 8)
            ],
            {"phase": "train", "event": "training_finished", **summary},
        ],
    )
    write_json(
        root.parent / "exit_status.json",
        {"exit_code": 0, "started_epoch": 100, "finished_epoch": 110},
    )
    return root


def test_reset_teleports_excluded_and_segments_are_not_laps(
    experiment: Path, analyzer: dict[str, Any]
) -> None:
    result = analyzer["analyze"](experiment)
    start = result["evaluations"]["start"]
    assert start["returns_per_car"] == [4.0, 8.0]
    assert start["reset_excluded_distance_per_car_m"] == [2.0, 4.0]
    assert start["total_dones"] == 2
    assert start["dones_per_car"] == [2, 0]
    assert start["longest_observed_reset_free_segment_steps_per_car"] == [1, 4]
    assert start["longest_observed_reset_free_segment_seconds_per_car"] == [0.25, 1.0]
    assert "not completed laps" in start["segment_note"]
    train = result["training"]
    assert (
        train["collected_agent_samples"]
        == train["fully_optimized_unique_agent_samples_inferred"]
        == 8
    )
    assert train["unoptimized_partial_agent_samples"] == 0
    assert train["expected_ppo_epoch_count"] == train["reported_ppo_epoch_count"] == 4
    assert train["collected_agent_samples_per_training_second"] == 4.0
    assert train["whole_job_minus_training_seconds"] == 8.0
    assert result["decision"]["rule_passed_for_supplied_run"] is False
    assert "e006_hypothesis_supported" not in result["decision"]


def test_full_length_terminal_partial_is_not_optimized(
    experiment: Path, analyzer: dict[str, Any]
) -> None:
    events = rows(experiment / "events.jsonl")
    events[2].update(event="partial_rollout", optimized=False, reason="training_end")
    events[-1]["ppo_updates"] = 2
    write_rows(experiment / "events.jsonl", events)
    write_json(
        experiment / "training_summary.json",
        {"agent_timesteps": 8, "wall_seconds": 2.0, "ppo_updates": 2},
    )
    train = analyzer["analyze"](experiment)["training"]
    assert train["collected_agent_samples"] == 8
    assert train["fully_optimized_unique_agent_samples_inferred"] == 4
    assert train["unoptimized_partial_agent_samples"] == 4
    assert train["complete_rollouts"] == 1
    assert train["partial_rollout_events"][0]["control_steps"] == 2


@pytest.mark.parametrize(
    "bad",
    [
        "recipe",
        "bool_recipe",
        "step",
        "missing",
        "rewards",
        "dones",
        "nan",
        "dimensions",
        "state",
        "epochs",
        "sample_gap",
        "job_failed",
    ],
)
def test_inconsistent_evidence_is_rejected(
    experiment: Path, analyzer: dict[str, Any], bad: str
) -> None:
    if bad in {"recipe", "bool_recipe"}:
        value = json.loads((experiment / "requested_config.json").read_text())
        value["num_envs"] = 3 if bad == "recipe" else True
        write_json(experiment / "requested_config.json", value)
    elif bad in {"step", "missing", "rewards", "dones", "nan", "dimensions"}:
        values = rows(experiment / "end_trajectory.jsonl")
        if bad == "step":
            values[0]["control_step"] = 1
        elif bad == "missing":
            values.pop()
        elif bad == "rewards":
            values[0]["rewards"][0] += 1
        elif bad == "dones":
            values[0]["dones"][0] = 0
        elif bad == "nan":
            values[0]["positions_xy"][0][0] = float("nan")
        else:
            values[0]["actions"].pop()
        write_rows(experiment / "end_trajectory.jsonl", values)
    elif bad == "state":
        value = json.loads((experiment / "end_initial_state.json").read_text())
        value["root_state"][0][0] += 0.1
        write_json(experiment / "end_initial_state.json", value)
    elif bad in {"epochs", "sample_gap"}:
        events = rows(experiment / "events.jsonl")
        if bad == "epochs":
            events[-1]["ppo_updates"] = 3
            write_json(
                experiment / "training_summary.json",
                {"agent_timesteps": 8, "wall_seconds": 2.0, "ppo_updates": 3},
            )
        else:
            events[1]["agent_timesteps"] = 6
        write_rows(experiment / "events.jsonl", events)
    else:
        write_json(
            experiment.parent / "exit_status.json",
            {"exit_code": 124, "started_epoch": 100, "finished_epoch": 110},
        )
    with pytest.raises(ValueError):
        analyzer["analyze"](experiment)


def test_resolved_defaults_and_integer_float_values_are_allowed(analyzer: dict[str, Any]) -> None:
    analyzer["requested_matches_resolved"]({"budget": 60}, {"budget": 60.0, "default": 1})
    with pytest.raises(ValueError, match="Boolean"):
        analyzer["requested_matches_resolved"]({"budget": True}, {"budget": 1})


@pytest.mark.parametrize("distance", [6.0, 6.0 - 1e-7])
def test_rule_uses_strict_reported_threshold_without_tolerance_slack(
    experiment: Path, analyzer: dict[str, Any], distance: float
) -> None:
    # Dones cannot reconstruct physical cause labels. Keep two done transitions,
    # but report only one as out_of_bounds; another cause could explain the other.
    summary = json.loads((experiment / "end_evaluation.json").read_text())
    summary["terminations"]["out_of_bounds"] = 1
    summary["nonreset_distance_total_m"] = distance
    write_json(experiment / "end_evaluation.json", summary)
    result = analyzer["analyze"](experiment)["decision"]
    assert result["strictly_fewer_reported_out_of_bounds"] is True
    assert result["rule_passed_for_supplied_run"] is (distance == 6.0)
    assert result["matches_preregistered_e006_recipe"] is False


def test_cli_reports_incomplete_as_json(
    tmp_path: Path,
    analyzer: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import sys

    monkeypatch.setattr(sys, "argv", ["analyze.py", str(tmp_path)])
    assert analyzer["main"]() == 2
    report = json.loads(capsys.readouterr().out)
    assert report["analysis_status"] == "inconclusive"
    assert report["consistency_passed"] is False
    assert report["rule_passed_for_supplied_run"] is None

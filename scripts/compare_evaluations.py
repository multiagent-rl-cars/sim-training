"""Verify matched initial states and make a side-by-side start/end video.

Run with the Isaac environment's Python after both fresh-process evaluations.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess

import imageio_ffmpeg

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", type=Path)
args = parser.parse_args()
root = args.run_dir
start = json.loads((root / "start_evaluation.json").read_text())
end = json.loads((root / "end_evaluation.json").read_text())
a = json.loads((root / "start_initial_state.json").read_text())
b = json.loads((root / "end_initial_state.json").read_text())


def same(left, right):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(same(x, y) for x, y in zip(left, right))
    return math.isclose(left, right, abs_tol=1e-6, rel_tol=1e-6)


assert same(a, b), "Start/end initial states differ; videos are not a matched comparison"
assert start["seed"] == end["seed"]
assert start["control_steps"] == end["control_steps"]
assert start["deterministic"] and end["deterministic"]
summary = {"matched_initial_states": True, "tolerance": 1e-6, "start": start, "end": end,
           "interpretation": "Single-seed short generated-fixture evaluation, not a generalization result"}
(root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(root / "start.mp4"),
                "-i", str(root / "end.mp4"), "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
                "-map", "[v]", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(root / "start-vs-end.mp4")], check=True)
print(json.dumps(summary, indent=2))

# Static, exportable figure; all points come from this run's saved metrics.
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with (root / "metrics/progress.csv").open() as stream:
    rows = list(csv.DictReader(stream))
steps, returns = [], []
for row in rows:
    if row.get("time/total_timesteps") and row.get("rollout/ep_rew_mean"):
        steps.append(float(row["time/total_timesteps"]))
        returns.append(float(row["rollout/ep_rew_mean"]))
fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
axes[0].plot(steps, returns, marker="o", markersize=3)
axes[0].set(xlabel="Training agent-timesteps", ylabel="Mean completed episode return",
            title="Training reward (inherited defects)")
labels = ["Initial", "Final"]
offtrack = [start["terminations"].get("out_of_bounds", 0), end["terminations"].get("out_of_bounds", 0)]
collisions = [start["terminations"].get("collision", 0), end["terminations"].get("collision", 0)]
axes[1].bar(labels, offtrack, label="Off-track", color="#4b7aaf")
axes[1].bar(labels, collisions, bottom=offtrack, label="Car proximity", color="#e39a42")
axes[1].set(ylabel="Termination events across all environments", title=f"Matched {start['sim_seconds']:g}-second evaluation")
axes[1].legend()
axes[2].bar(labels, [start["nonreset_distance_total_m"], end["nonreset_distance_total_m"]], color=["#94a5ba", "#477d6c"])
axes[2].set(ylabel="Total vehicle travel (m; reset jumps excluded)", title="Travel is not route progress")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
fig.suptitle("One seed, generated circular-track fixture — no generalization claim", fontsize=12)
fig.savefig(root / "training-and-evaluation.png", dpi=170)

# E006 — 900-second frozen-fixture baseline

Status: hypothesis supported on the specified fixture; backup verified and
instance deleted. Date: 2026-09-27. The unchanged [preregistered protocol](protocol.md)
and [identity record](record.json) define the test before results. All four
[E001 integration gates](../E001-foundation-integration/README.md) passed first.

Question: does the final policy produce fewer off-track resets without reducing
reset-excluded travel in the fixed matched evaluation? This is not a claim of
historical reproduction, convergence, held-out generalization or real driving.

## Result and demonstrations

The final policy meets the preregistered rule on the matched seed-123 evaluation:
strictly fewer off-track events and no less reset-excluded travel.

| Across 12 separate repeated-circle arenas, 15 simulated seconds each | Initial | Final |
| --- | ---: | ---: |
| Reported off-track resets | 66 | 0 |
| Reset-excluded travel, all cars combined | 98.773 m | 197.148 m |
| Mean return per car over the evaluation window | 59.315 | 69.896 |

All 12 final-policy cars remained reset-free throughout this 15-second window.
Each covered about 0.73–0.81 signed revolutions, **not a full lap**. These are
repeated instances of one generated circle, not 12 distinct held-out tracks.

- [Actual before/after video](../../outputs/cloud-validation/20260927-run02/remote/outputs/baselines/basic-900s-01/experiment/start-vs-end.mp4),
  [midpoint still](../../outputs/cloud-validation/20260927-run02/baseline-comparison-mid-frame.png),
  [late still](../../outputs/cloud-validation/20260927-run02/baseline-comparison-last-frame.png).
  Only arena 0 is filmed; video counters aggregate all 12 cars. Video samples
  every second control step, so the last overlay precedes the final summary.
- [Training/evaluation plot](../../outputs/cloud-validation/20260927-run02/remote/outputs/baselines/basic-900s-01/experiment/training-and-evaluation.png),
  [raw comparison](../../outputs/cloud-validation/20260927-run02/remote/outputs/baselines/basic-900s-01/experiment/comparison.json),
  [trajectory audit](../../outputs/cloud-validation/20260927-run02/basic-900s-01-analysis.json),
  [checkpoint/video inspection](../../outputs/cloud-validation/20260927-run02/basic-900s-01-evidence.json).

The [analysis code](analyze.py) independently recomputes returns, done counts and
travel from initial states and raw trajectories. All consistency checks passed;
11 saved-evidence integration checks also passed. Source, USD asset hashes, clean
pinned Lab commit, lockfile, actual A6000 and exact requested recipe were checked
against the frozen reference before attributing this result to E006. Artifact
inspection does not by itself prove source authenticity or general driving quality.

## Time and sample accounting

Training took **900.051 s**, collecting **206,376 car-step samples** at 229.29/s.
Sixteen complete rollouts account for 196,608 unique optimized samples and 160
PPO optimization epochs. The final 814 vector steps (9,768 car samples, about
4.7% of collection) were unoptimized because the time callback ended collection.
These counts are inferred from recorded rollouts and matching epoch counters,
not a raw optimizer trace. Samples are not independent tracks or unique scenes.

Whole-job time was **1,029 s (17 min 9 s)**: about 129 seconds beyond training for
startup, fresh-process evaluations and other job work. Allocation, installation,
backup and analysis are outside that job time. At the quoted $0.68/hour, this job
alone corresponds to about $0.19 prorated compute, not the full-session invoice.
Recording was enabled; its isolated overhead was not measured.

## What this does not establish

The prior team's 15-minute convergence result remains **not strictly reproduced**:
original assets/final code/criterion are unresolved, hardware and control period
differ, and known inherited behavior defects remain. No held-out track, extra
seed, full-lap endurance, obstacle avoidance, multi-agent coordination or real
hardware was tested by E006. The input-amplitude correction in E003 is separate
and was not used here. Reward growth alone was not the acceptance rule.

W&B originally recorded offline as `434aa9828c404f999d1863242058f783` in
`2710-multiagent-rl/sim-training`. A later authorized CPU-only recovery uploaded
the run, repaired copied artifact paths, and verified the original 24 history rows,
eight training videos, evaluation media and exact model/evaluation artifact manifests.
[Open the verified online run](https://wandb.ai/2710-multiagent-rl/sim-training/runs/434aa9828c404f999d1863242058f783).
Receipts are in `outputs/wandb-upload-20260927/`; the original backup is unchanged. All raw
checkpoints, metrics and media remain in [the local session backup](../../outputs/cloud-validation/20260927-run02/).
All 240 session files match remote checksums and passed independent coverage
review, including 16 intermediate baseline checkpoints and eight training clips.
The instance was confirmed absent from Brev at 21:07:53 UTC.

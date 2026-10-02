"""Explicit CPU-only synthetic online W&B transport gate (does upload).

No simulator, real model, source archive, or credentials are uploaded. Run in a
fresh directory; this leaves a clearly labeled test run in the requested project.
Use an outer process deadline as SDK/service retries can outlive HTTP timeouts.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path

from rc_car.experiments import load_config
from rc_car.experiments.online_check import check_online
from rc_car.ops.offline_check import _generate_video


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, required=True, help="Generated recipe from config.yaml"
    )
    args = parser.parse_args()
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        {
            "WANDB_MODE": "online",
            "WANDB_DISABLE_GIT": "true",
            "WANDB_DISABLE_CODE": "true",
            "WANDB_ERROR_REPORTING": "false",
            "WANDB_CONSOLE": "off",
            "WANDB_SILENT": "true",
            "WANDB_X_DISABLE_MACHINE_INFO": "true",
            "WANDB_X_DISABLE_STATS": "true",
            "WANDB_INIT_TIMEOUT": "30",
            "WANDB_X_SERVICE_WAIT": "30",
        }
    )
    for name in ("WANDB_DIR", "WANDB_DATA_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR"):
        path = root / name.lower()
        path.mkdir()
        os.environ[name] = str(path)
    for name in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RESUME_FROM", "WANDB_FORK_FROM"):
        os.environ.pop(name, None)
    recipe = load_config(args.config)
    check_online(recipe)
    assert recipe.wandb is not None
    import wandb

    video, _ = _generate_video(root)
    marker = root / "synthetic-marker.json"
    marker.write_text(json.dumps({"test_only": True, "no_training": True}) + "\n")
    run = wandb.init(
        entity=recipe.wandb.entity,
        project=recipe.wandb.project,
        name="synthetic-online-upload-check",
        job_type="tracking-test",
        mode="online",
        config={"test_only": True, "no_training": True},
        dir=str(root),
        save_code=False,
        resume="never",
        reinit="create_new",
    )
    try:
        for step in range(3):
            run.log({"test/step": step, "test/scalar": step * 0.5})
        run.log({"test/synthetic_video": wandb.Video(str(video), format="mp4")})
        artifact = wandb.Artifact(run.id + "-synthetic-marker", type="tracking-test")
        artifact.add_file(str(marker), name=marker.name)
        uploaded = run.log_artifact(artifact).wait(timeout=60)
        artifact_id = uploaded.id
        artifact_name = uploaded.qualified_name
        run_id = run.id
        url = run.url
    except BaseException:
        run.finish(exit_code=1)
        raise
    run.finish()
    api = wandb.Api(timeout=20)
    remote = api.run(f"{recipe.wandb.entity}/{recipe.wandb.project}/{run_id}")
    rows = list(remote.scan_history())
    values = [row["test/scalar"] for row in rows if "test/scalar" in row]
    media = [row["test/synthetic_video"] for row in rows if "test/synthetic_video" in row]
    assert values == [0, 0.5, 1] and len(media) == 1
    remote_files = {entry.name: entry for entry in remote.files()}
    assert media[0]["path"] in remote_files
    assert remote_files[media[0]["path"]].size == video.stat().st_size
    saved = api.artifact(artifact_name)
    assert saved.id == artifact_id
    entry = saved.manifest.entries[marker.name]
    expected = base64.b64encode(hashlib.md5(marker.read_bytes()).digest()).decode()
    assert entry.digest == expected and entry.size == marker.stat().st_size
    assert remote.state == "finished"
    receipt = {
        "test_only": True,
        "passed": True,
        "url": url,
        "run_id": run_id,
        "scalar_rows_verified": 3,
        "video_files_verified": 1,
        "artifact_manifest_verified": True,
        "artifact": artifact_name,
    }
    (root / "online-check.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

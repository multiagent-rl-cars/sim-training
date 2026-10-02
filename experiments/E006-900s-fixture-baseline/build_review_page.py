"""Build this experiment's September peer-review page from actual recorded media.

No network. The receipt is produced by the local quality gate and upload audit;
this builder does not run tests or certify the claims itself.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    session = root / "outputs/cloud-validation/20260927-run02"
    experiment = session / "remote/outputs/baselines/basic-900s-01/experiment"
    receipt = json.loads((root / "outputs/wandb-upload-20260927/summary.json").read_text())
    page = (root / "docs/merge-review.template.html").read_text()
    media = {
        "COMPARISON_VIDEO": (experiment / "start-vs-end.mp4", "video/mp4"),
        "LATE_STILL": (session / "baseline-comparison-last-frame.png", "image/png"),
        "MID_STILL": (session / "baseline-comparison-mid-frame.png", "image/png"),
        "TRAINING_PLOT": (experiment / "training-and-evaluation.png", "image/png"),
    }
    for key, (path, mime) in media.items():
        value = base64.b64encode(path.read_bytes()).decode("ascii")
        page = page.replace(f"@@{key}@@", f"data:{mime};base64,{value}")
    page = page.replace("@@TEST_COUNT@@", str(receipt["local_tests_passed"]))
    links = []
    for run in receipt["runs"]:
        if not run["server_verified"]:
            raise ValueError("Cannot label an unverified run as uploaded")
        url = run["url"]
        if not url.startswith("https://wandb.ai/2710-multiagent-rl/sim-training/runs/"):
            raise ValueError("Unexpected review link destination")
        links.append(f'<a href="{html.escape(url, quote=True)}">{html.escape(run["label"])}</a>')
    status = (
        "<strong>Online evidence verified:</strong> all three recovered runs have their original "
        "metric histories, training videos, evaluation media and exact checkpoint/evaluation "
        "artifact manifests checked through the W&amp;B API. " + " · ".join(links)
    )
    page = page.replace("@@WANDB_STATUS@@", status)
    if "@@" in page:
        raise ValueError("Unresolved review-page template token")
    output = root / "scratch/merge-review.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page)
    print(output)


if __name__ == "__main__":
    main()

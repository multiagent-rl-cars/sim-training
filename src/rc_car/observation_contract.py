"""Versioned policy-image transport, independent of Isaac and learning imports.

Image helpers and reward/history buffers remain float32 binary masks in {0, 1}.
Only policy observation terms scale them to {0, 255}; the pinned Isaac SB3
wrapper casts to uint8 and SB3 divides by 255 once before either image encoder.
This is an intentional input-amplitude change, not a checkpoint conversion.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

BEHAVIOR_VERSION = "astra-foundation-v3-image-amplitude"
POLICY_IMAGE_ENCODING = "binary-255-sb3-normalized-v1"
IMAGE_ENCODING_TASKS = frozenset({"Basic-Car-Training-v0", "Multiagent-Car-Training-v0"})


def policy_image_contract() -> dict[str, object]:
    """Return a fresh JSON-safe description of the supported image boundary."""
    return {
        "encoding": POLICY_IMAGE_ENCODING,
        "helper_dtype": "float32",
        "helper_values": [0, 1],
        "policy_observation_dtype": "float32",
        "policy_observation_values": [0, 255],
        "sb3_transport_dtype": "uint8",
        "sb3_normalize_images": True,
        "network_dtype": "float32",
        "network_values": [0, 1],
        "channel_order": "NCHW",
    }


def require_policy_image_encoding(encoding: object, *, normalize_images: object = True) -> None:
    """Reject legacy/unknown checkpoints; matching shapes are not sufficient.

    The supported evaluator also verifies the training source snapshot. This
    explicit marker catches an old checkpoint copied into an otherwise current
    run. It is a compatibility guard, not authentication of a checkpoint file.
    """
    if (
        not isinstance(encoding, str)
        or encoding != POLICY_IMAGE_ENCODING
        or normalize_images is not True
    ):
        raise ValueError(
            "Policy image encoding mismatch; use the original source/encoding for legacy "
            "checkpoints. Automatic image-amplitude conversion is not supported."
        )


def reject_legacy_car_checkpoint(task: str | None, *, checkpoint_requested: bool) -> None:
    """Fail before Kit for unsupported car checkpoint loads in inherited scripts.

    New training in those scripts and unrelated tasks are unaffected. They do
    not implement the versioned experiment manifest/encoding load contract, so
    even a new car checkpoint must use the supported evaluator instead. Direct
    third-party ``PPO.load`` calls are outside these entry-point guarantees.
    """
    task_name = task.rsplit(":", 1)[-1].replace("-Play", "") if task is not None else None
    if checkpoint_requested and task_name in IMAGE_ENCODING_TASKS:
        raise ValueError(
            "Car checkpoint loading in inherited scripts/sb3 entrypoints is unsupported "
            "after the policy image encoding change. Use scripts/training_experiment.py "
            "with its matching manifest for new checkpoints, or the original frozen source "
            "for legacy checkpoints; automatic conversion is not supported."
        )


def policy_image_panel(
    track_image: NDArray[np.float32], obstacle_image: NDArray[np.float32]
) -> NDArray[np.uint8]:
    """Render one agent's CHW policy terms without rescaling their byte values.

    No value-based encoding guess is made: nonbinary and unit-valued pixels
    fail rather than silently producing a misleading PNG. Entirely black images
    are valid and cannot distinguish old/new encodings by values alone; the
    explicit marker and pipeline establish the encoding. Inputs are not modified.
    """
    if (
        track_image.ndim != 3
        or obstacle_image.ndim != 3
        or track_image.shape[0] != 1
        or obstacle_image.shape[0] < 1
        or track_image.shape[1:] != obstacle_image.shape[1:]
        or min(track_image.shape[1:]) < 1
    ):
        raise ValueError("Policy images must be CHW with one track channel and matching pixels")
    for value in (track_image, obstacle_image):
        if value.dtype != np.float32:
            raise ValueError("Policy image terms must be float32 before the SB3 wrapper")
        if not np.isfinite(value).all() or not np.isin(value, [0, 255]).all():
            raise ValueError("Policy image terms must contain only finite binary 0/255 pixels")
    return np.concatenate([track_image[0], *obstacle_image], axis=1).astype(np.uint8)

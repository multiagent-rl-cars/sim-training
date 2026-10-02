"""Image encoding metadata and PNG bytes, without a simulator or network."""

import json

import numpy as np
import pytest
from PIL import Image

from rc_car.observation_contract import (
    POLICY_IMAGE_ENCODING,
    policy_image_contract,
    policy_image_panel,
    reject_legacy_car_checkpoint,
    require_policy_image_encoding,
)


def test_metadata_is_json_safe_and_not_shared_mutable_state():
    contract = policy_image_contract()
    assert json.loads(json.dumps(contract))["encoding"] == POLICY_IMAGE_ENCODING
    contract["network_values"] = [0, 1 / 255]
    assert policy_image_contract()["network_values"] == [0, 1]


@pytest.mark.parametrize("encoding", [None, "legacy", "", {}, 1, True])
def test_old_or_unknown_encoding_is_rejected(encoding):
    with pytest.raises(ValueError, match="encoding mismatch"):
        require_policy_image_encoding(encoding)


def test_normalization_must_remain_enabled():
    require_policy_image_encoding(POLICY_IMAGE_ENCODING)
    for disabled in (False, None, 1, "true"):
        with pytest.raises(ValueError, match="encoding mismatch"):
            require_policy_image_encoding(POLICY_IMAGE_ENCODING, normalize_images=disabled)


@pytest.mark.parametrize(
    "task",
    [
        "Basic-Car-Training-v0",
        "Multiagent-Car-Training-v0",
        "car_training:Basic-Car-Training-v0",
        "Basic-Car-Training-Play-v0",
    ],
)
def test_inherited_car_checkpoint_loads_fail_without_stamping_or_conversion(task):
    with pytest.raises(ValueError, match="original frozen source"):
        reject_legacy_car_checkpoint(task, checkpoint_requested=True)
    reject_legacy_car_checkpoint(task, checkpoint_requested=False)


@pytest.mark.parametrize("task", [None, "Isaac-Cartpole-v0", "Reset-Agent"])
def test_unaffected_tasks_keep_inherited_checkpoint_behavior(task):
    reject_legacy_car_checkpoint(task, checkpoint_requested=True)


@pytest.mark.parametrize("channels", [1, 4])
def test_png_panel_preserves_black_white_pixels_and_inputs(channels, tmp_path):
    track = (np.indices((48, 48)).sum(0) % 2 * 255).astype(np.float32)[None]
    obstacles = np.repeat(255 - track, channels, axis=0)
    before = track.copy(), obstacles.copy()
    panel = policy_image_panel(track, obstacles)
    assert panel.shape == (48, 48 * (1 + channels))
    assert panel.dtype == np.uint8
    assert set(np.unique(panel)) == {0, 255}
    path = tmp_path / "policy_inputs.png"
    Image.fromarray(panel).resize((panel.shape[1] * 4, 192), Image.Resampling.NEAREST).save(path)
    with Image.open(path) as image:
        restored = np.asarray(image)
    assert set(np.unique(restored)) == {0, 255}
    np.testing.assert_array_equal(restored[::4, ::4], panel)
    np.testing.assert_array_equal(track, before[0])
    np.testing.assert_array_equal(obstacles, before[1])


@pytest.mark.parametrize("pixel", [1.0, 0.5, 256.0, -1.0, float("nan"), float("inf")])
def test_png_rejects_wrong_range_and_never_guesses_encoding(pixel):
    image = np.zeros((1, 3, 3), dtype=np.float32)
    image[0, 0, 0] = pixel
    with pytest.raises(ValueError, match="0/255 pixels"):
        policy_image_panel(image, image)


def test_png_rejects_dtype_and_shape():
    image = np.zeros((1, 3, 3), dtype=np.float32)
    assert not policy_image_panel(image, image).any()
    with pytest.raises(ValueError, match="float32"):
        policy_image_panel(image.astype(np.uint8), image)
    with pytest.raises(ValueError, match="CHW"):
        policy_image_panel(image[0], image)
    with pytest.raises(ValueError, match="matching pixels"):
        policy_image_panel(image, image[:, :2])

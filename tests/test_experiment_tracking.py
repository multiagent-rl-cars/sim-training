"""CPU-only regression checks; no Isaac installation or network required."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.experiment_tracking import log_video, start_tracking, validate_config


class ExperimentConfigTests(unittest.TestCase):
    def setUp(self):
        fixtures = Path(__file__).resolve().parent / "fixtures/legacy_recipes"
        self.config = json.loads((fixtures / "line_following_15min.json").read_text())

    def test_legacy_fixture_configs(self):
        fixtures = Path(__file__).resolve().parent / "fixtures/legacy_recipes"
        for path in fixtures.glob("*.json"):
            validate_config(json.loads(path.read_text()))

    def test_invalid_budgets_and_dimensions(self):
        for key in (
            "num_envs",
            "total_timesteps",
            "video_stride",
            "max_training_seconds",
            "training_video_interval_s",
        ):
            config = copy.deepcopy(self.config)
            config[key] = 0
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(config)

    def test_impossible_batch(self):
        self.config["ppo"]["batch_size"] = 10**9
        with self.assertRaises(ValueError):
            validate_config(self.config)

    def test_offline_override_is_honored(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict("os.environ", {"WANDB_MODE": "offline"}),
            patch("rc_car.experiments.tracking._load_wandb") as loader,
        ):
            init = loader.return_value.init
            run = init.return_value
            run.id, run.url, run.entity, run.project = "test", None, "team", "project"
            run.name = "sdk-generated-name"
            start_tracking(self.config, tmp)
            self.assertEqual(init.call_args.kwargs["mode"], "offline")
            identity = json.loads((Path(tmp) / "wandb_run.json").read_text())
            self.assertEqual(identity["mode"], "offline")

    def test_empty_video_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.mp4"
            path.touch()
            with self.assertRaises(ValueError):
                log_video(None, path)


if __name__ == "__main__":
    unittest.main()

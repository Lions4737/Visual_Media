from __future__ import annotations

import unittest

from scripts import make_self_similarity_figure as figures


def detection(score: float = 0.8) -> dict[str, float]:
    return {
        "score": score,
        "cx": 0.5,
        "cy": 0.5,
        "w": 0.2,
        "h": 0.2,
    }


def example() -> dict[str, object]:
    return {
        "relative_path": "two/two_backpacks/example.png",
        "class_name": "backpack",
        "requested_count": 2,
        "predicted_count": 2,
    }


def record() -> dict[str, object]:
    return {
        "relative_path": "two/two_backpacks/example.png",
        "class_name": "backpack",
        "prompt": "backpack",
        "requested_count": 2,
        "predicted_count": 2,
        "min_saved_threshold": 0.4,
        "seed": figures.FROZEN_SEED,
        "checkpoint_sha256": figures.EXPECTED_CHECKPOINT_SHA256,
        "detections": [detection(), detection(0.7)],
    }


class BoundingBoxArtifactValidationTests(unittest.TestCase):
    def test_exact_frozen_record_is_accepted(self) -> None:
        selected = figures.validate_record(example(), record())

        self.assertEqual(len(selected), 2)

    def test_count_drift_is_rejected(self) -> None:
        changed = record()
        changed["detections"] = [detection()]

        with self.assertRaisesRegex(RuntimeError, "Detection count changed"):
            figures.validate_record(example(), changed)

    def test_detection_at_threshold_is_rejected(self) -> None:
        changed = record()
        changed["detections"] = [detection(), detection(0.4)]

        with self.assertRaisesRegex(RuntimeError, "at/below 0.4"):
            figures.validate_record(example(), changed)

    def test_saved_threshold_must_equal_frozen_threshold(self) -> None:
        changed = record()
        changed["min_saved_threshold"] = 0.3

        with self.assertRaisesRegex(RuntimeError, "Expected saved threshold"):
            figures.validate_record(example(), changed)

    def test_seed_drift_is_rejected(self) -> None:
        changed = record()
        changed["seed"] = 7

        with self.assertRaisesRegex(RuntimeError, "seed"):
            figures.validate_record(example(), changed)

    def test_checkpoint_drift_is_rejected(self) -> None:
        changed = record()
        changed["checkpoint_sha256"] = "0" * 64

        with self.assertRaisesRegex(RuntimeError, "checkpoint_sha256"):
            figures.validate_record(example(), changed)


if __name__ == "__main__":
    unittest.main()

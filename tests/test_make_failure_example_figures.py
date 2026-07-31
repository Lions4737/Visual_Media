from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import make_failure_example_figures as figures


def raw_row(
    *,
    prompt_id: str = "canonical",
    prompt: str = "clock",
    min_saved_threshold: float = 0.2,
    gt_count: int = 2,
) -> dict[str, object]:
    return {
        "relative_path": "two/two_clocks/example.png",
        "prompt_id": prompt_id,
        "prompt": prompt,
        "min_saved_threshold": min_saved_threshold,
        "gt_count": gt_count,
        "detections": [],
    }


def load_single(row: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "raw.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return figures.load_records([path])


class RawInputValidationTests(unittest.TestCase):
    def test_prompt_text_must_match_frozen_prompt_id(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Prompt text does not match"):
            load_single(raw_row(prompt="digital clock"))

    def test_saved_threshold_must_not_exceed_figure_threshold(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "truncated above"):
            load_single(raw_row(min_saved_threshold=0.400001))


class FrozenCountValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.example = {
            "relative_path": "two/two_clocks/example.png",
            "requested_count": 2,
            "before_count": 0,
            "digital_count": 0,
            "after_count": 0,
        }
        self.canonical = raw_row()
        self.digital = raw_row(prompt_id="digital", prompt="digital clock")

    def test_prompt_rows_must_agree_on_requested_count(self) -> None:
        self.digital["gt_count"] = 3
        with self.assertRaisesRegex(RuntimeError, "disagree on requested count"):
            figures.validate_counts(
                self.example,
                [],
                [],
                [],
                self.canonical,
                self.digital,
            )

    def test_frozen_count_drift_is_rejected(self) -> None:
        self.example["before_count"] = 1
        with self.assertRaisesRegex(RuntimeError, "Frozen count changed"):
            figures.validate_counts(
                self.example,
                [],
                [],
                [],
                self.canonical,
                self.digital,
            )


if __name__ == "__main__":
    unittest.main()

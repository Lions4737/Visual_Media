from __future__ import annotations

import unittest

from scripts import analyze_prompt_union as prompt_union


def detection(
    *,
    score: float = 0.9,
    cx: float = 0.5,
    cy: float = 0.5,
    width: float = 0.2,
    height: float = 0.2,
) -> dict[str, float]:
    return {
        "score": score,
        "cx": cx,
        "cy": cy,
        "w": width,
        "h": height,
    }


class SelectionTests(unittest.TestCase):
    def test_detection_at_threshold_is_excluded(self) -> None:
        row = {"detections": [detection(score=0.4)]}

        self.assertEqual(prompt_union.selected(row), [])

    def test_detection_above_threshold_is_included(self) -> None:
        item = detection(score=0.400001)
        row = {"detections": [item]}

        self.assertEqual(prompt_union.selected(row), [item])


class SpatialUnionTests(unittest.TestCase):
    def test_identical_cross_prompt_boxes_merge(self) -> None:
        merged = prompt_union.spatial_union(
            {
                "canonical": [detection(score=0.8)],
                "digital": [detection(score=0.9)],
            },
            prompt_union.IOU_THRESHOLD,
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["member_count"], 2)
        self.assertEqual(merged[0]["prompts"], ["clock", "digital clock"])
        self.assertEqual(merged[0]["score"], 0.9)

    def test_identical_same_prompt_boxes_do_not_directly_merge(self) -> None:
        merged = prompt_union.spatial_union(
            {
                "canonical": [detection(score=0.8), detection(score=0.9)],
                "digital": [],
            },
            prompt_union.IOU_THRESHOLD,
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(
            sorted(component["member_count"] for component in merged),
            [1, 1],
        )

    def test_cross_prompt_links_merge_same_prompt_boxes_transitively(self) -> None:
        merged = prompt_union.spatial_union(
            {
                "canonical": [
                    detection(score=0.7, cx=0.48),
                    detection(score=0.8, cx=0.52),
                ],
                "digital": [detection(score=0.9, cx=0.50)],
            },
            prompt_union.IOU_THRESHOLD,
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["member_count"], 3)
        self.assertEqual(merged[0]["prompts"], ["clock", "digital clock"])


if __name__ == "__main__":
    unittest.main()

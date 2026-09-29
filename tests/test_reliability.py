"""Unit tests for the Reliability Engine. Run: python -m unittest tests.test_reliability -v"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.reliability import ReliabilityEngine, find_contiguous_segments  # noqa: E402


class TestFindContiguousSegments(unittest.TestCase):
    def test_extraction(self):
        seq = [0, 1, 1, 0, 0, 1, 0, 1, 1, 1]
        self.assertEqual(find_contiguous_segments(seq), [(1, 2), (5, 5), (7, 9)])

    def test_edges_and_numpy_input(self):
        self.assertEqual(find_contiguous_segments(np.array([1, 1, 0, 1])), [(0, 1), (3, 3)])
        self.assertEqual(find_contiguous_segments([1, 1, 1]), [(0, 2)])

    def test_empty_and_all_zero(self):
        self.assertEqual(find_contiguous_segments([]), [])
        self.assertEqual(find_contiguous_segments(np.array([])), [])
        self.assertEqual(find_contiguous_segments([0, 0, 0]), [])


class TestReliabilityEngine(unittest.TestCase):
    @staticmethod
    def _run(y_true, y_pred):
        return ReliabilityEngine(y_true, y_pred).evaluate()

    def test_perfect_detection(self):
        y = [0, 1, 1, 0, 0, 1, 1, 1, 0]
        r = self._run(y, y)
        self.assertEqual((r["true_segments"], r["detected_segments"], r["missed_segments"]), (2, 2, 0))
        self.assertEqual(r["detection_coverage"], 1.0)
        self.assertEqual(r["detection_delays"], [0, 0])
        self.assertEqual(r["false_alarm_segments"], 0)
        self.assertEqual(r["false_alarm_segment_rate"], 0.0)

    def test_detection_delay(self):
        y_true = [0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
        y_pred = [0, 0, 0, 1, 1, 0, 0, 0, 1, 0]  # first hit 2 late, then 1 late
        r = self._run(y_true, y_pred)
        self.assertEqual(r["detected_segments"], 2)
        self.assertEqual(r["detection_delays"], [2, 1])
        self.assertAlmostEqual(r["mean_detection_delay"], 1.5)

    def test_early_prediction_has_zero_delay(self):
        # predicted segment starts before the true one; first overlapping sample is the true start
        r = self._run([0, 0, 1, 1, 0], [0, 1, 1, 0, 0])
        self.assertEqual(r["detection_delays"], [0])
        self.assertEqual(r["false_alarm_segments"], 0)

    def test_missed_segment(self):
        y_true = [0, 1, 1, 0, 0, 1, 1, 0]
        y_pred = [0, 1, 0, 0, 0, 0, 0, 0]
        r = self._run(y_true, y_pred)
        self.assertEqual((r["detected_segments"], r["missed_segments"]), (1, 1))
        self.assertEqual(r["detection_coverage"], 0.5)
        self.assertEqual(r["missed_segment_list"], [(5, 6)])

    def test_false_alarm_segment(self):
        y_true = [0, 1, 1, 0, 0, 0, 0, 0]
        y_pred = [0, 1, 1, 0, 1, 0, 1, 1]  # 3 predicted segments, 2 in normal region
        r = self._run(y_true, y_pred)
        self.assertEqual(r["predicted_segments"], 3)
        self.assertEqual(r["false_alarm_segments"], 2)
        self.assertAlmostEqual(r["false_alarm_segment_rate"], 2 / 3)
        self.assertEqual(r["false_alarm_segment_list"], [(4, 4), (6, 7)])

    def test_segment_rate_is_not_sample_fpr(self):
        # one long false-alarm segment: segment rate = 1/1, sample FPR = 6/8
        y_true = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        y_pred = [1, 1, 0, 1, 1, 1, 1, 1, 1, 0]
        r = self._run(y_true, y_pred)
        self.assertAlmostEqual(r["false_alarm_segment_rate"], 0.5)  # 1 of 2 predicted segments
        fpr = 6 / 8
        self.assertNotAlmostEqual(r["false_alarm_segment_rate"], fpr)

    def test_no_anomaly_edge_case(self):
        r = self._run([0, 0, 0, 0], [0, 1, 0, 0])
        self.assertEqual((r["true_segments"], r["detected_segments"], r["missed_segments"]), (0, 0, 0))
        self.assertIsNone(r["detection_coverage"])
        self.assertEqual(r["detection_delays"], [])
        self.assertIsNone(r["mean_detection_delay"])
        self.assertEqual((r["predicted_segments"], r["false_alarm_segments"]), (1, 1))
        self.assertEqual(r["false_alarm_segment_rate"], 1.0)

    def test_no_prediction_edge_case(self):
        r = self._run([0, 1, 1, 0, 1], [0, 0, 0, 0, 0])
        self.assertEqual((r["true_segments"], r["detected_segments"], r["missed_segments"]), (2, 0, 2))
        self.assertEqual(r["detection_coverage"], 0.0)
        self.assertEqual(r["predicted_segments"], 0)
        self.assertEqual(r["false_alarm_segments"], 0)
        self.assertIsNone(r["false_alarm_segment_rate"])

    def test_empty_inputs(self):
        r = self._run([], [])
        self.assertEqual((r["true_segments"], r["predicted_segments"]), (0, 0))
        self.assertIsNone(r["detection_coverage"])
        self.assertIsNone(r["false_alarm_segment_rate"])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ReliabilityEngine([0, 1, 0], [0, 1])

    def test_non_binary_raises(self):
        with self.assertRaises(ValueError):
            ReliabilityEngine([0, 2], [0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)

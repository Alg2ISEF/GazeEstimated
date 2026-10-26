import unittest

import numpy as np

from code import LiveDemo, gaze_to_screen_space


class CalibrationSampleFilterTest(unittest.TestCase):
    def test_filters_outlier_hits_and_keeps_grid_dense(self):
        samples = np.array([
            [100.0, 100.0],
            [101.0, 99.0],
            [102.0, 101.0],
            [99.0, 100.0],
            [900.0, 900.0],
            [920.0, 910.0],
        ], dtype=float)

        kept = LiveDemo._filter_bad_samples(samples, max_distance_px=60.0)
        self.assertEqual(len(kept), 4)
        self.assertTrue(np.allclose(kept, samples[:4]))

        pts = LiveDemo._calibration_points(5)
        self.assertEqual(len(pts), 25)
        self.assertIn((0.1, 0.1), pts)
        self.assertIn((0.5, 0.5), pts)
        self.assertIn((0.9, 0.9), pts)

    def test_gaze_to_screen_space_returns_screen_coordinates(self):
        res = (1920, 1080)
        geometry = __import__("screen_mapping", fromlist=["ScreenGeometry"]).ScreenGeometry.laptop(res, (350.0, 200.0), cam_to_top_mm=10.0)
        center = np.array([0.0, 0.0, 600.0], dtype=float)
        gaze = np.array([0.0, 0.0, -1.0], dtype=float)

        coords = gaze_to_screen_space(center, gaze, res=res, geometry=geometry)
        self.assertIsNotNone(coords)
        self.assertEqual(len(coords), 2)
        self.assertTrue(np.all(np.isfinite(coords)))


if __name__ == "__main__":
    unittest.main()

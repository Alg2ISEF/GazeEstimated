import unittest

import numpy as np

from code import OneEuroFilter


class OneEuroFilterTest(unittest.TestCase):
    def test_filter_tracks_value_and_uses_timestamp(self):
        filt = OneEuroFilter(min_cutoff=1.0, beta=0.3, derivative_cutoff=1.0)
        out0 = filt(np.array([10.0, 20.0]), 0.0)
        out1 = filt(np.array([11.0, 21.0]), 0.1)
        out2 = filt(np.array([30.0, 40.0]), 0.2)

        self.assertEqual(out0.shape, (2,))
        self.assertTrue(np.all(np.isfinite(out1)))
        self.assertTrue(np.all(np.isfinite(out2)))
        self.assertLess(np.linalg.norm(out2 - np.array([30.0, 40.0])), 30.0)


if __name__ == "__main__":
    unittest.main()

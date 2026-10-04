"""
Smoothing for a noisy gaze-on-screen signal.

GazeSmoother(p, t) takes a screen position in pixels and a timestamp in
seconds and returns the smoothed position. Three stages:

  1. Median of the last `median` samples. Removes the single-frame spikes
     (blinks, head-pose glitches) that otherwise open the One Euro filter wide.
  2. One Euro filter with a SMALL beta. Beta multiplies speed in the signal's
     own units; for pixels, noise alone produces hundreds of px/s of apparent
     speed, so a beta like 0.08 keeps the filter nearly transparent.
  3. Fixation hold. While the filtered point stays within `hold_px` of the
     shown point, the shown point does not move. When it leaves that radius,
     the shown point follows, trailing by hold_px / 2.

Tuning: lower beta / bigger median / bigger hold_px = steadier but laggier.
"""
from collections import deque

import numpy as np


class OneEuro:
    def __init__(self, min_cutoff=0.3, beta=0.005, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self._t = None
        self._x = None
        self._dx = None

    @staticmethod
    def _alpha(dt, fc):
        tau = 1.0 / (2.0 * np.pi * fc)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        if self._t is None:
            self._t, self._x, self._dx = t, x, np.zeros_like(x)
            return x
        dt = max(t - self._t, 1e-6)
        dx = (x - self._x) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        fc = self.min_cutoff + self.beta * np.linalg.norm(self._dx)
        a = self._alpha(dt, fc)
        self._x = a * x + (1.0 - a) * self._x
        self._t = t
        return self._x


class GazeSmoother:
    def __init__(self, median=5, min_cutoff=0.3, beta=0.005, d_cutoff=1.0,
                 hold_px=30.0, max_gap_s=0.5):
        self.median = median
        self.hold_px = hold_px
        self.max_gap_s = max_gap_s
        self._euro = OneEuro(min_cutoff, beta, d_cutoff)
        self.reset()

    def reset(self):
        self._hist = deque(maxlen=self.median)
        self._euro.reset()
        self._shown = None
        self._last_t = None

    def __call__(self, p, t):
        if self._last_t is not None and t - self._last_t > self.max_gap_s:
            self.reset()                      # face was lost for a while: start fresh
        self._last_t = t
        self._hist.append(np.asarray(p, dtype=np.float64))
        y = self._euro(np.median(np.asarray(self._hist), axis=0), t)
        if self.hold_px <= 0:
            self._shown = y
        elif self._shown is None:
            self._shown = y.copy()
        else:
            d = float(np.linalg.norm(y - self._shown))
            if d > self.hold_px:
                self._shown = y - (y - self._shown) / d * (self.hold_px * 0.5)
        return self._shown
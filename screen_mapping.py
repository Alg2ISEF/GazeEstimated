"""
Camera space -> screen space, done geometrically.

Frames
  camera: origin at the camera, x right, y down, z forward (OpenCV / ptgaze), mm.
  screen: origin at the top-left corner of the screen, u right, v down, in
          pixels; the screen is the plane z_screen = 0.

The screen pose is a rigid transform  X_cam = R @ S + t  where S = (u*sx, v*sy, 0)
is a screen point in mm (sx, sy = mm per pixel).

A gaze sample is a ray: origin o (eye/face centre in the camera frame) and a
unit direction d (camera frame). The pixel being looked at is the ray-plane
intersection, expressed back in screen pixels:

    n   = R[:, 2]                         # screen normal in the camera frame
    lam = n . (t - o) / (n . d)
    hit = o + lam * d
    S   = R^T (hit - t)                   # back into the screen frame
    u, v = S.x / sx, S.y / sy

Where the pose comes from
  * Laptop: webcam is in the lid, so R is about identity and t = (-W/2, gap, 0),
    where gap is the camera-to-top-edge distance. ScreenGeometry.laptop().
  * Anything else (external webcam, tilted camera): fit R and t from the
    calibration dots with fit_screen_pose().
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# ---- gaze direction <-> (pitch, yaw), ptgaze convention: d = -[cp*sy, sp, cp*cy] ----
def vector_to_angles(d):
    d = np.asarray(d, dtype=np.float64)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return np.arcsin(np.clip(-d[..., 1], -1, 1)), np.arctan2(-d[..., 0], -d[..., 2])


def angles_to_vector(pitch, yaw):
    return -np.stack([np.cos(pitch) * np.sin(yaw), np.sin(pitch),
                      np.cos(pitch) * np.cos(yaw)], axis=-1)


class ScreenGeometry:
    def __init__(self, rotvec, t_mm, res_wh, size_mm_wh):
        self.rotvec = np.asarray(rotvec, dtype=np.float64)
        self.t = np.asarray(t_mm, dtype=np.float64)
        self.res = tuple(res_wh)
        self.sx = size_mm_wh[0] / res_wh[0]
        self.sy = size_mm_wh[1] / res_wh[1]
        self.R = Rotation.from_rotvec(self.rotvec).as_matrix()

    @classmethod
    def laptop(cls, res_wh, size_mm_wh, cam_to_top_mm=10.0):
        """Webcam centred above the top edge of the screen, in the screen plane."""
        return cls([0, 0, 0], [-size_mm_wh[0] / 2.0, cam_to_top_mm, 0.0], res_wh, size_mm_wh)

    def pixel_to_cam(self, px):
        """3D position (mm, camera frame) of screen pixel(s) px = (u, v)."""
        px = np.atleast_2d(np.asarray(px, dtype=np.float64))
        S = np.column_stack([px[:, 0] * self.sx, px[:, 1] * self.sy, np.zeros(len(px))])
        return S @ self.R.T + self.t

    def ray_to_pixel(self, origin_mm, direction):
        """Pixel (u, v) the ray hits, or None if it points away from the screen."""
        o = np.asarray(origin_mm, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        n = self.R[:, 2]
        denom = float(n @ d)
        if abs(denom) < 1e-6:
            return None
        lam = float(n @ (self.t - o)) / denom
        if lam <= 0:
            return None
        S = self.R.T @ (o + lam * d - self.t)
        return np.array([S[0] / self.sx, S[1] / self.sy])


def fit_screen_pose(origins_mm, directions, targets_px, res_wh, size_mm_wh,
                    init_geometry=None, fit_bias=False,
                    prior_rot_rad=0.35, prior_t_mm=120.0):
    """Fit the screen pose (and optionally a gaze angle gain/offset) to calibration dots.

    origins_mm, directions: (N, 3) per-dot gaze origin and direction (use the
        per-dot median); targets_px: (N, 2) where the user was looking.
    The cost is the angle (degrees) between each measured gaze direction and
    the direction from the eye to the true dot position, plus a weak prior that
    keeps the pose near init_geometry (default: laptop). With fit_bias=True it
    also fits a yaw/pitch gain and offset, which absorbs the way gaze networks
    tend to under-estimate large angles.

    Returns (ScreenGeometry, bias, mean_residual_deg) where bias is
    (gain_pitch, gain_yaw, off_pitch, off_yaw) or None.
    """
    o = np.asarray(origins_mm, dtype=np.float64)
    d = np.asarray(directions, dtype=np.float64)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    px = np.asarray(targets_px, dtype=np.float64)
    g0 = init_geometry or ScreenGeometry.laptop(res_wh, size_mm_wh)
    x0 = np.concatenate([g0.rotvec, g0.t])
    scale0 = np.concatenate([np.full(3, prior_rot_rad), np.full(3, prior_t_mm)])
    if fit_bias:
        x0 = np.concatenate([x0, [1.0, 1.0, 0.0, 0.0]])
        scale0 = np.concatenate([scale0, [0.3, 0.3, np.deg2rad(4), np.deg2rad(4)]])
    pitch_m, yaw_m = vector_to_angles(d)

    def unpack(x):
        g = ScreenGeometry(x[:3], x[3:6], res_wh, size_mm_wh)
        if not fit_bias:
            return g, d
        gp, gy, op, oy = x[6:10]
        return g, angles_to_vector(gp * pitch_m + op, gy * yaw_m + oy)

    def residuals(x):
        g, dirs = unpack(x)
        want = g.pixel_to_cam(px) - o
        want /= np.linalg.norm(want, axis=1, keepdims=True)
        ang = np.degrees(np.arccos(np.clip(np.sum(want * dirs, axis=1), -1, 1)))
        return np.concatenate([ang, 0.5 * (x - x0) / scale0])   # data (deg) + weak prior

    sol = least_squares(residuals, x0, method="trf")
    g, dirs = unpack(sol.x)
    want = g.pixel_to_cam(px) - o
    want /= np.linalg.norm(want, axis=1, keepdims=True)
    mean_deg = float(np.mean(np.degrees(np.arccos(np.clip(np.sum(want * dirs, axis=1), -1, 1)))))
    bias = tuple(sol.x[6:10]) if fit_bias else None
    return g, bias, mean_deg


def apply_bias(direction, bias):
    """Apply a fitted (gain_pitch, gain_yaw, off_pitch, off_yaw) to a gaze direction."""
    if bias is None:
        return np.asarray(direction, dtype=np.float64)
    p, y = vector_to_angles(direction)
    return angles_to_vector(bias[0] * p + bias[2], bias[1] * y + bias[3])
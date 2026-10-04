"""
Map a gaze direction to a point on the screen. Needs only numpy.

Two ways to do it:

A) Geometric: intersect the gaze ray with the screen plane.
   Needs: 3D eye/face position in camera coordinates (mm), gaze vector in
   CAMERA coordinates, and where the screen sits relative to the webcam.

B) Calibration: show a few dots, record what the model outputs while the
   user looks at each one, and fit a small regression to screen pixels.
   This absorbs camera offset, face-model scale errors and per-person bias.

Camera coordinate convention (OpenCV): x right, y down, z away from the camera
into the scene. The screen is assumed to lie in the plane z = 0.
"""
import numpy as np


def pitch_yaw_to_vector(pitch, yaw):
    """Pitch/yaw (radians) -> unit gaze vector.

    WARNING: sign conventions differ between codebases. Verify this against the
    code you use (e.g. look at where the gaze arrow points on a test image).
    Also note the raw model output is in the *normalized* head space; it must be
    rotated back to camera space (ptgaze does this for you) before use here.
    """
    return np.array([
        -np.cos(pitch) * np.sin(yaw),
        -np.sin(pitch),
        -np.cos(pitch) * np.cos(yaw),
    ])


def gaze_to_screen_mm(origin_mm, gaze_vec):
    """Intersect the gaze ray with the plane z = 0.

    origin_mm: (x, y, z) of the eyes/face centre in camera coordinates, mm.
    gaze_vec:  gaze direction in camera coordinates (does not need to be unit).
    Returns (x, y) on the screen plane in mm relative to the camera, or None
    if the user is looking away from the screen plane.
    """
    o = np.asarray(origin_mm, dtype=float)
    d = np.asarray(gaze_vec, dtype=float)
    d = d / np.linalg.norm(d)
    if d[2] >= -1e-6:          # not pointing back toward the camera plane
        return None
    t = -o[2] / d[2]
    hit = o + t * d
    return float(hit[0]), float(hit[1])


class ScreenGeometry:
    """Where the screen is relative to the webcam, and its size."""

    def __init__(self, width_mm, height_mm, res_w, res_h,
                 cam_to_left_mm=None, cam_to_top_mm=15.0):
        # Default: webcam centred above the top edge of the screen.
        self.w_mm, self.h_mm = width_mm, height_mm
        self.res_w, self.res_h = res_w, res_h
        self.x_left = -(cam_to_left_mm if cam_to_left_mm is not None else width_mm / 2)
        self.y_top = cam_to_top_mm   # screen top edge is this far BELOW the camera (y down)

    def mm_to_px(self, x_mm, y_mm):
        px = (x_mm - self.x_left) / self.w_mm * self.res_w
        py = (y_mm - self.y_top) / self.h_mm * self.res_h
        return px, py

    def px_to_mm(self, px, py):
        x = self.x_left + px / self.res_w * self.w_mm
        y = self.y_top + py / self.res_h * self.h_mm
        return x, y


class GazeCalibrator:
    """Fit pixels = poly2(features) with ridge regression.

    features can be (yaw, pitch) or the (x, y) mm hit point from
    gaze_to_screen_mm. Collect ~9-16 samples (several frames per dot, take the
    median) spread across the screen.
    """

    def __init__(self, ridge=1e-8):
        self.ridge = ridge
        self.w = None

    @staticmethod
    def _design(f):
        f = np.asarray(f, dtype=float)
        a, b = f[:, 0], f[:, 1]
        return np.stack([np.ones_like(a), a, b, a * a, a * b, b * b], axis=1)

    def fit(self, features, pixels):
        X = self._design(features)
        Y = np.asarray(pixels, dtype=float)
        A = X.T @ X + self.ridge * np.eye(X.shape[1])
        self.w = np.linalg.solve(A, X.T @ Y)

    def predict(self, features):
        return self._design(np.atleast_2d(features)) @ self.w


if __name__ == "__main__":
    # Synthetic self-test of the geometry (no camera or model needed).
    geo = ScreenGeometry(width_mm=344, height_mm=215, res_w=1920, res_h=1200)
    face = np.array([20.0, -30.0, 600.0])          # face 60 cm in front of the camera
    target_px = (1500, 300)                        # pretend the user looks here
    tx, ty = geo.px_to_mm(*target_px)
    gaze = np.array([tx, ty, 0.0]) - face          # ray from face to that screen point
    hit = gaze_to_screen_mm(face, gaze)
    print("recovered px:", geo.mm_to_px(*hit), "expected:", target_px)

    # Calibration demo: model output is biased/scaled; the fit should undo that.
    rng = np.random.default_rng(0)
    px = rng.uniform([0, 0], [1920, 1200], size=(16, 2))
    feats = np.stack([px[:, 0] / 4000 - 0.2, px[:, 1] / 3000 + 0.1], axis=1)
    cal = GazeCalibrator()
    cal.fit(feats, px)
    err = np.abs(cal.predict(feats) - px).max()
    print("max calibration error on synthetic data (px):", round(float(err), 3))
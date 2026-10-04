"""
Live gaze-on-screen using ptgaze (ETH-XGaze mode).

Put this file in the same folder as gaze_to_screen.py and gaze_smoother.py, then run:
    python gaze_live.py --device cuda
    python gaze_live.py --device cpu --res 1920x1080 --invert-gaze

Keys: q or Esc = quit, c = recalibrate.

How it works: ptgaze's Demo class builds the camera, face detector and model
from its own config. We reuse it, replace run() with our own loop, and use
face.center and face.gaze_vector (camera coordinates) to intersect the gaze
ray with the screen plane. A 9-dot calibration then maps that hit point to
screen pixels. The on-screen position is smoothed by GazeSmoother
(median -> One Euro -> fixation hold).
"""
import argparse
import copy
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
from omegaconf import OmegaConf
from ptgaze.demo import Demo

from gaze_smoother import GazeSmoother, OneEuro
from gaze_to_screen import GazeCalibrator, gaze_to_screen_mm
from screen_mapping import ScreenGeometry

class OneEuroFilter(OneEuro):
    def __init__(self, min_cutoff=0.5, beta=0.005, derivative_cutoff=1.0,
                 d_cutoff=None, **kwargs):
        if d_cutoff is None:
            d_cutoff = derivative_cutoff
        super().__init__(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

WIN = "gaze"
SET = {"res": (1920, 1080), "invert": False, "show_background": True, "calib_only": False}


def detect_screen_size(default):
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return size
    except Exception:
        return default


def configure_face_detector(config):
    """ptgaze freezes config after building it, so unlock only for this edit."""
    was_readonly = OmegaConf.is_readonly(config)
    if was_readonly:
        OmegaConf.set_readonly(config, False)
    try:
        fd = config.face_detector
        # Video mode lets mediapipe track the face between frames instead of
        # re-detecting from scratch on every frame; one face is all we need.
        fd.mediapipe_static_image_mode = False
        fd.mediapipe_max_num_faces = 1
    finally:
        if was_readonly:
            OmegaConf.set_readonly(config, True)


class LiveDemo(Demo):
    def __init__(self, config):
        fd = config.face_detector
        configure_face_detector(config)
        super().__init__(config)
        cam = self.gaze_estimator.camera
        self._skip_undistort = bool(np.allclose(cam.dist_coefficients, 0))
        self._latest_frame = None
        self._latest_crop = None
        self.screen_geo = ScreenGeometry.laptop(SET["res"], (350.0, 200.0), cam_to_top_mm=10.0)
        self.gaze_bias = None
        self.times = defaultdict(float)
        self.frames = 0
        print(f"face detector: {fd.mode}, static_image_mode={fd.mediapipe_static_image_mode}, "
              f"max faces={fd.mediapipe_max_num_faces}, undistort skipped: {self._skip_undistort}")

    @staticmethod
    def _filter_bad_samples(samples, max_distance_px=80.0, min_samples=4):
        arr = np.asarray(samples, dtype=np.float64)
        if arr.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 2:
            return np.empty((0, 2), dtype=np.float64)

        center = np.median(arr, axis=0)
        dist = np.linalg.norm(arr - center, axis=1)
        keep = dist <= max_distance_px
        filtered = arr[keep]
        if len(filtered) >= min_samples:
            return filtered
        return arr if len(arr) >= min_samples else filtered

    @staticmethod
    def _calibration_points(grid=5):
        if grid == 5:
            anchors = np.array([0.10, 0.30, 0.50, 0.70, 0.90], dtype=float)
        else:
            anchors = np.linspace(0.10, 0.90, grid, dtype=float)
        pts = []
        for y in anchors:
            for x in anchors:
                pts.append((float(x), float(y)))
        return pts

    def _tick(self, **stage_seconds):
        for name, sec in stage_seconds.items():
            self.times[name] += sec
        self.frames += 1

    def _reset_timers(self):
        self.times.clear()
        self.frames = 0

    def _report(self):
        if self.frames and self.frames % 60 == 0:
            parts = ", ".join(f"{k} {1000 * v / self.frames:.1f}" for k, v in self.times.items())
            print(f"avg ms/frame over {self.frames} frames: {parts}")

    def run(self):
        try:
            self._live()
        finally:
            self.gaze_estimator.close()
            self.cap.release()
            cv2.destroyAllWindows()

    # ---- one measurement: camera frame -> (hit_x_mm, hit_y_mm) --------------
    def _center_crop(self, frame, face, crop_size=320, crop_enabled=False):
        # If cropping is disabled, return the original frame and face unchanged
        if not crop_enabled:
            return frame, face

        h, w = frame.shape[:2]
        screen_ratio = SET["res"][0] / max(1, SET["res"][1])
        center = np.asarray(face.center, dtype=np.float32).reshape(-1)
        if center.size >= 2:
            cx, cy = float(center[0]), float(center[1])
        else:
            cx, cy = float(w) / 2.0, float(h) / 2.0

        crop_w = min(w, max(160, int(crop_size)))
        crop_h = int(crop_w / screen_ratio)
        if crop_h > h:
            crop_h = h
            crop_w = int(crop_h * screen_ratio)
        if crop_w > w:
            crop_w = w
            crop_h = int(crop_w / screen_ratio)

        half_w = crop_w / 2.0
        half_h = crop_h / 2.0
        x0 = int(np.clip(cx - half_w, 0, max(0, w - crop_w)))
        y0 = int(np.clip(cy - half_h, 0, max(0, h - crop_h)))
        x1 = min(x0 + crop_w, w)
        y1 = min(y0 + crop_h, h)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return frame, face

        cropped_face = copy.copy(face)
        if hasattr(cropped_face, "center"):
            cropped_center = np.asarray(cropped_face.center, dtype=np.float32).reshape(-1)
            if cropped_center.size >= 2:
                cropped_center = cropped_center - np.array([x0, y0], dtype=np.float32)
            else:
                cropped_center = np.array([float(crop_w) / 2.0, float(crop_h) / 2.0], dtype=np.float32)
            cropped_face.center = cropped_center[:2]
        if hasattr(cropped_face, "bbox"):
            bbox = np.asarray(cropped_face.bbox, dtype=np.float32).copy()
            bbox[..., 0] -= x0
            bbox[..., 1] -= y0
            cropped_face.bbox = bbox
        if hasattr(cropped_face, "landmarks"):
            landmarks = np.asarray(cropped_face.landmarks, dtype=np.float32).copy()
            if landmarks.ndim >= 2:
                landmarks[:, 0] -= x0
                landmarks[:, 1] -= y0
            cropped_face.landmarks = landmarks
        return crop, cropped_face


    def _measure(self):
        t0 = time.perf_counter()
        ok, frame = self.cap.read()
        t1 = time.perf_counter()
        if not ok:
            return None, "camera read failed"
        cam = self.gaze_estimator.camera
        if self._skip_undistort:
            und = frame
        else:
            und = cv2.undistort(frame, cam.camera_matrix, cam.dist_coefficients)
        self._latest_frame = und.copy()
        self._latest_crop = None
        t2 = time.perf_counter()
        faces = self.gaze_estimator.detect_faces(und)
        t3 = time.perf_counter()
        if not faces:
            self._tick(read=t1 - t0, undistort=t2 - t1, detect=t3 - t2)
            return None, "no face detected"
        face = faces[0]
        crop, cropped_face = self._center_crop(und, face, crop_size=1280, crop_enabled=False)
        self._latest_crop = None
        self.gaze_estimator.estimate_gaze(und, face)
        gaze_vec = np.asarray(getattr(face, "gaze_vector", np.zeros(3)), dtype=float).reshape(-1)
        if gaze_vec.size < 3:
            gaze_vec = np.pad(gaze_vec, (0, max(0, 3 - gaze_vec.size)))
        face.gaze_vector = gaze_vec[:3]
        t4 = time.perf_counter()
        self._tick(read=t1 - t0, undistort=t2 - t1, detect=t3 - t2, gaze=t4 - t3)
        center = np.asarray(getattr(face, "center", np.zeros(3)), dtype=float).reshape(-1)
        if center.size < 3:
            center = np.pad(center, (0, max(0, 3 - center.size)))
        center = center[:3]
        gaze = np.asarray(face.gaze_vector, dtype=float).reshape(-1)
        if gaze.size < 3:
            gaze = np.pad(gaze, (0, max(0, 3 - gaze.size)))
        gaze = gaze[:3]
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(gaze)):
            return None, "invalid gaze measurement"
        if np.linalg.norm(center) < 10:      # looks like metres -> convert to mm
            center = center * 1000.0
        if SET["invert"]:
            gaze = -gaze

        # Prefer the geometric camera-to-screen ray intersection when available.
        geo_px = self.screen_geo.ray_to_pixel(center, gaze)
        if geo_px is not None and np.all(np.isfinite(geo_px)):
            px = np.asarray(geo_px, dtype=float)
            px = np.clip(px, [0, 0], [SET["res"][0] - 1, SET["res"][1] - 1])
            return tuple(float(v) for v in px), ""

        hit = gaze_to_screen_mm(center, gaze)
        if hit is None:
            return None, "gaze not pointing at screen plane (try --invert-gaze)"
        if not np.all(np.isfinite(hit)):
            return None, "gaze hit is not finite"
        return hit, ""

    # ---- calibration ---------------------------------------------------------
    def _calibrate(self):
        W, H = SET["res"]
        points = self._calibration_points(grid=5)
        feats, pix = [], []
        for x, y in points:
            tx, ty = int(x * W), int(y * H)
            t0, samples, msg = time.time(), [], ""
            while time.time() - t0 < 2.0:
                el = time.time() - t0
                if SET.get("show_background", True) and self._latest_frame is not None:
                    base_frame = self._latest_frame.copy()
                else:
                    base_frame = np.zeros((H, W, 3), np.uint8)
                canvas = cv2.resize(base_frame, (W, H)) if base_frame.shape[:2] != (H, W) else base_frame.copy()
                r = int(30 * (1 - el / 2.0)) + 8
                cv2.circle(canvas, (tx, ty), r, (0, 255, 255), -1)
                cv2.circle(canvas, (tx, ty), 3, (0, 0, 0), -1)
                hit, msg = self._measure()
                if hit is not None and np.all(np.isfinite(hit)) and el > 0.8:
                    samples.append(np.asarray(hit, dtype=float))
                    if SET.get("calib_only", False):
                        print(f"target=({tx}, {ty}) -> screen=({hit[0]:.1f}, {hit[1]:.1f})")
                elif msg:
                    cv2.putText(canvas, msg, (40, H - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(WIN, canvas)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    return None

            if len(samples) >= 4:
                good = self._filter_bad_samples(samples, max_distance_px=80.0, min_samples=4)
                if len(good) >= 4:
                    feats.append(np.median(good, axis=0))
                    pix.append((tx, ty))

        if len(feats) < 10:
            print("Calibration failed: too few good samples. Keep your face visible.")
            return None
        feats = np.asarray(feats, dtype=float)
        if feats.size == 0 or not np.all(np.isfinite(feats)):
            print("Calibration failed: no valid gaze samples.")
            return None
        mu, sd = feats.mean(axis=0), feats.std(axis=0) + 1e-6
        cal = GazeCalibrator(ridge=1e-3)
        cal.fit((feats - mu) / sd, np.array(pix, dtype=float))
        err = np.linalg.norm(cal.predict((feats - mu) / sd) - np.array(pix), axis=1)
        if not np.all(np.isfinite(err)):
            print("Calibration failed: non-finite regression error.")
            return None
        print(f"Calibrated on {len(feats)} dots, mean fit error {err.mean():.0f}px")
        self.screen_geo = ScreenGeometry.laptop((W, H), (350.0, 200.0), cam_to_top_mm=10.0)
        self.gaze_bias = None
        return cal, mu, sd

    # ---- main loop -------------------------------------------------------------
    def _live(self):
        W, H = SET["res"]
        cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        calib = self._calibrate()
        if calib is None:
            return
        if SET.get("calib_only", False):
            print("Calibration-only mode finished.")
            return
        cal, mu, sd = calib
        smooth = None
        fps, t_prev = 0.0, time.time()
        # median(5) -> One Euro (beta sized for pixels) -> 30 px fixation hold
        smoother = GazeSmoother(median=5, min_cutoff=0.3, beta=0.005,
                                d_cutoff=1.0, hold_px=30.0)
        base = np.zeros((H, W, 3), np.uint8)
        self._reset_timers()
        while True:
            hit, msg = self._measure()
            if SET.get("show_background", True) and self._latest_frame is not None:
                canvas = cv2.resize(self._latest_frame, (W, H))
            else:
                canvas = base.copy()
            if hit is not None:
                f = (np.asarray(hit) - mu) / sd
                p = cal.predict(f[None, :])[0]
                if not np.all(np.isfinite(p)):
                    continue
                p = np.clip(p, [0, 0], [W - 1, H - 1])
                smooth = smoother(p, time.perf_counter())
                if not np.all(np.isfinite(smooth)):
                    smooth = None
                else:
                    cv2.circle(canvas, (int(smooth[0]), int(smooth[1])), 25, (0, 255, 0), 3)
                    cv2.circle(canvas, (int(smooth[0]), int(smooth[1])), 4, (0, 255, 0), -1)
            elif msg:
                cv2.putText(canvas, msg, (40, H - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            cv2.putText(canvas, f"{fps:.1f} FPS   q: quit   c: recalibrate", (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            t_d = time.perf_counter()
            cv2.imshow(WIN, canvas)
            key = cv2.waitKey(1) & 0xFF
            self.times["display"] += time.perf_counter() - t_d
            self._report()
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                new = self._calibrate()
                if new is not None:
                    cal, mu, sd = new
                    smoother.reset()
                    smooth = None
                self._reset_timers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--res", default=None, help="screen resolution, e.g. 1920x1080")
    ap.add_argument("--invert-gaze", action="store_true",
                    help="flip the gaze vector if every sample is rejected")
    ap.add_argument("--no-bg", "--hide-bg", dest="show_background", action="store_false",
                    help="hide the camera background from the display (show only calibration / gaze overlay)")
    ap.add_argument("--calib-only", action="store_true",
                    help="open only the calibration window and print screen coordinates without running live tracking")
    args = ap.parse_args()

    if args.res:
        w, h = args.res.lower().split("x")
        SET["res"] = (int(w), int(h))
    else:
        SET["res"] = detect_screen_size(SET["res"])
    SET["invert"] = args.invert_gaze
    SET["show_background"] = args.show_background
    SET["calib_only"] = args.calib_only

    # Let ptgaze build its own config exactly as the CLI does, but run our Demo.
    sys.argv = ["ptgaze", "--mode", "eth-xgaze", "--device", args.device]
    import ptgaze.main as ptmain
    if not hasattr(ptmain, "Demo"):
        sys.exit("ptgaze.main does not expose 'Demo'. Run: python -c \"import ptgaze.main, "
                 "inspect; print(inspect.getsource(ptgaze.main))\" and send me the output.")
    ptmain.Demo = LiveDemo
    ptmain.main()


if __name__ == "__main__":
    main()
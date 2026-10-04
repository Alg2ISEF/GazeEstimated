"""
Live gaze-on-screen using ptgaze (ETH-XGaze mode).

Put this file in the same folder as gaze_to_screen.py, then run:
    python gaze_live.py --device cuda
    python gaze_live.py --device cpu --res 1920x1080 --invert-gaze

Keys: q or Esc = quit, c = recalibrate.

How it works: ptgaze's Demo class builds the camera, face detector and model
from its own config. We reuse it, replace run() with our own loop, and use
face.center and face.gaze_vector (camera coordinates) to intersect the gaze
ray with the screen plane. A 9-dot calibration then maps that hit point to
screen pixels.
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

from caldefd_filter import OneEuroFilter
from gaze_to_screen import GazeCalibrator, gaze_to_screen_mm

WIN = "gaze"
SET = {"res": (1920, 1080), "invert": False}


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
        self.times = defaultdict(float)
        self.frames = 0
        print(f"face detector: {fd.mode}, static_image_mode={fd.mediapipe_static_image_mode}, "
              f"max faces={fd.mediapipe_max_num_faces}, undistort skipped: {self._skip_undistort}")

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
    def _center_crop(self, frame, face, crop_size=640):
        h, w = frame.shape[:2]
        center = np.asarray(face.center, dtype=np.float32).reshape(-1)
        if center.size >= 2:
            cx, cy = float(center[0]), float(center[1])
        else:
            cx, cy = float(w) / 2.0, float(h) / 2.0
        crop_size = int(min(crop_size, max(320, min(w, h))))
        half = crop_size / 2.0
        x0 = int(np.clip(cx - half, 0, max(0, w - crop_size)))
        y0 = int(np.clip(cy - half, 0, max(0, h - crop_size)))
        x1 = min(x0 + crop_size, w)
        y1 = min(y0 + crop_size, h)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return frame, face
        cropped_face = copy.copy(face)
        if hasattr(cropped_face, "center"):
            cropped_center = np.asarray(cropped_face.center, dtype=np.float32).reshape(-1)
            if cropped_center.size >= 2:
                cropped_center = cropped_center - np.array([x0, y0], dtype=np.float32)
            else:
                cropped_center = np.array([float(crop_size) / 2.0, float(crop_size) / 2.0], dtype=np.float32)
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
        t2 = time.perf_counter()
        faces = self.gaze_estimator.detect_faces(und)
        t3 = time.perf_counter()
        if not faces:
            self._tick(read=t1 - t0, undistort=t2 - t1, detect=t3 - t2)
            return None, "no face detected"
        face = faces[0]
        crop, cropped_face = self._center_crop(und, face)
        self.gaze_estimator.estimate_gaze(crop, cropped_face)
        face.gaze_vector = np.asarray(cropped_face.gaze_vector, dtype=float)
        t4 = time.perf_counter()
        self._tick(read=t1 - t0, undistort=t2 - t1, detect=t3 - t2, gaze=t4 - t3)
        center = np.asarray(face.center, dtype=float)
        gaze = np.asarray(face.gaze_vector, dtype=float)
        if np.linalg.norm(center) < 10:      # looks like metres -> convert to mm
            center = center * 1000.0
        if SET["invert"]:
            gaze = -gaze
        hit = gaze_to_screen_mm(center, gaze)
        if hit is None:
            return None, "gaze not pointing at screen plane (try --invert-gaze)"
        return hit, ""

    # ---- calibration ---------------------------------------------------------
    def _calibrate(self):
        W, H = SET["res"]
        m = 0.1
        xs, ys = [m, 0.5, 1 - m], [m, 0.5, 1 - m]
        feats, pix = [], []
        for y in ys:
            for x in xs:
                tx, ty = int(x * W), int(y * H)
                t0, samples, msg = time.time(), [], ""
                while time.time() - t0 < 2.0:
                    el = time.time() - t0
                    canvas = np.zeros((H, W, 3), np.uint8)
                    r = int(30 * (1 - el / 2.0)) + 8
                    cv2.circle(canvas, (tx, ty), r, (0, 255, 255), -1)
                    cv2.circle(canvas, (tx, ty), 3, (0, 0, 0), -1)
                    hit, msg = self._measure()
                    if hit is not None and el > 0.8:   # skip the settling time
                        samples.append(hit)
                    if msg:
                        cv2.putText(canvas, msg, (40, H - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(WIN, canvas)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        return None
                if len(samples) >= 5:
                    feats.append(np.median(samples, axis=0))
                    pix.append((tx, ty))
        if len(feats) < 6:
            print("Calibration failed: too few good samples. Keep your face visible.")
            return None
        feats = np.array(feats)
        mu, sd = feats.mean(axis=0), feats.std(axis=0) + 1e-6
        cal = GazeCalibrator(ridge=1e-3)
        cal.fit((feats - mu) / sd, np.array(pix, dtype=float))
        err = np.linalg.norm(cal.predict((feats - mu) / sd) - np.array(pix), axis=1)
        print(f"Calibrated on {len(feats)} dots, mean fit error {err.mean():.0f}px")
        return cal, mu, sd

    # ---- main loop -------------------------------------------------------------
    def _live(self):
        W, H = SET["res"]
        cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        calib = self._calibrate()
        if calib is None:
            return
        cal, mu, sd = calib
        smooth = None
        fps, t_prev = 0.0, time.time()
        euro = OneEuroFilter(min_cutoff=0.5, beta=0.2, derivative_cutoff=0.5)
        base = np.zeros((H, W, 3), np.uint8)
        self._reset_timers()
        while True:
            hit, msg = self._measure()
            if self._latest_frame is not None:
                canvas = cv2.resize(self._latest_frame, (W, H))
            else:
                canvas = base.copy()
            if hit is not None:
                f = (np.asarray(hit) - mu) / sd
                p = cal.predict(f[None, :])[0]
                p = np.clip(p, [0, 0], [W - 1, H - 1])
                smooth = euro(np.asarray(p, dtype=np.float64), time.perf_counter())
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
                    euro.reset()
                    smooth = None
                self._reset_timers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--res", default=None, help="screen resolution, e.g. 1920x1080")
    ap.add_argument("--invert-gaze", action="store_true",
                    help="flip the gaze vector if every sample is rejected")
    args = ap.parse_args()

    if args.res:
        w, h = args.res.lower().split("x")
        SET["res"] = (int(w), int(h))
    else:
        SET["res"] = detect_screen_size(SET["res"])
    SET["invert"] = args.invert_gaze

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
import pickle
import socket
import sys
import threading
import time

import cv2
import numpy as np
from omegaconf import OmegaConf
from ptgaze.demo import Demo

from gaze_smoother import GazeSmoother
from gaze_to_screen import GazeCalibrator, gaze_to_screen_mm
from screen_mapping import ScreenGeometry

WIN = "gaze_streamer"
SET = {"res": (1920, 1080), "invert": False, "show_background": True}


# --------------------------------------------------------------------------
# config: let ptgaze build it exactly like its CLI does, then capture it
# --------------------------------------------------------------------------
def make_config(device="cuda"):
    import ptgaze.main as ptmain
    if not hasattr(ptmain, "Demo"):
        raise RuntimeError("ptgaze.main does not expose 'Demo' in this version")

    captured = {}

    class _Capture:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            pass

    old_argv, old_demo = sys.argv, ptmain.Demo
    sys.argv = ["ptgaze", "--mode", "eth-xgaze", "--device", device]
    ptmain.Demo = _Capture
    try:
        ptmain.main()
    finally:
        sys.argv, ptmain.Demo = old_argv, old_demo
    return captured["config"]


def configure_face_detector(config):
    was_readonly = OmegaConf.is_readonly(config)
    if was_readonly:
        OmegaConf.set_readonly(config, False)
    try:
        fd = config.face_detector
        fd.mediapipe_static_image_mode = False
        fd.mediapipe_max_num_faces = 1
    finally:
        if was_readonly:
            OmegaConf.set_readonly(config, True)


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


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def gaze_to_screen_space(center_mm, gaze_vec, res=None, geometry=None):
    if res is None:
        res = SET["res"]
    if geometry is None:
        geometry = ScreenGeometry.laptop(res, (350.0, 200.0), cam_to_top_mm=10.0)

    center = np.asarray(center_mm, dtype=np.float64).reshape(-1)
    gaze = np.asarray(gaze_vec, dtype=np.float64).reshape(-1)
    if center.size < 3:
        center = np.pad(center, (0, 3 - center.size))
    if gaze.size < 3:
        gaze = np.pad(gaze, (0, 3 - gaze.size))
    center, gaze = center[:3], gaze[:3]

    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(gaze)):
        return None
    if np.linalg.norm(center) < 10.0:
        center = center * 1000.0

    lo = np.array([0.0, 0.0])
    hi = np.array([res[0] - 1, res[1] - 1], dtype=np.float64)

    geo_px = geometry.ray_to_pixel(center, gaze)
    if geo_px is not None and np.all(np.isfinite(geo_px)):
        return tuple(float(v) for v in np.clip(np.asarray(geo_px, dtype=np.float64), lo, hi))

    hit = gaze_to_screen_mm(center, gaze)
    if hit is None or not np.all(np.isfinite(hit)):
        return None
    return tuple(float(v) for v in np.clip(np.asarray(hit, dtype=np.float64), lo, hi))


# --------------------------------------------------------------------------
# camera reader that always returns the newest frame (no V4L2 queue lag)
# --------------------------------------------------------------------------
class LatestFrame:
    def __init__(self, cap):
        self.cap, self.frame, self.ok = cap, None, False
        self.lock, self.running = threading.Lock(), True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.005)
            with self.lock:
                self.ok, self.frame = ok, f

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------
class CalibrationDemo(Demo):
    def __init__(self, config, camera=None, res=None, invert=False,
                 show_background=True, cam_size=None, cam_fps=None):
        SET["res"] = tuple(res) if res else detect_screen_size(SET["res"])
        SET["invert"] = invert
        SET["show_background"] = show_background

        configure_face_detector(config)
        super().__init__(config)

        if camera is not None:
            self._open_camera(camera, cam_size, cam_fps)
        else:
            self.cap = LatestFrame(self.cap)

        cam = self.gaze_estimator.camera
        self._skip_undistort = bool(np.allclose(cam.dist_coefficients, 0))
        self._latest_frame = None
        self.screen_geo = ScreenGeometry.laptop(SET["res"], (350.0, 200.0), cam_to_top_mm=10.0)

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _filter_bad_samples(samples, max_distance_px=80.0, min_samples=4):
        arr = np.asarray(samples, dtype=np.float64)
        if arr.size == 0:
            return np.empty((0, 2))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 2:
            return np.empty((0, 2))
        center = np.median(arr, axis=0)
        dist = np.linalg.norm(arr - center, axis=1)
        filtered = arr[dist <= max_distance_px]
        if len(filtered) >= min_samples:
            return filtered
        return arr if len(arr) >= min_samples else filtered

    @staticmethod
    def _calibration_points(grid=5):
        anchors = (np.array([0.10, 0.30, 0.50, 0.70, 0.90]) if grid == 5
                   else np.linspace(0.10, 0.90, grid))
        return [(float(x), float(y)) for y in anchors for x in anchors]

    def _open_camera(self, camera, cam_size=None, cam_fps=None):
        """camera: int index (2) or device path ("/dev/video2" or "/dev/v4l/by-id/...")."""
        if self.cap is not None:
            self.cap.release()

        cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
        if not cap.isOpened():                      # fall back to any backend
            cap.release()
            cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video device: {camera!r}")

        cam = self.gaze_estimator.camera
        w, h = cam_size if cam_size else (cam.width, cam.height)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if cam_fps:
            cap.set(cv2.CAP_PROP_FPS, cam_fps)

        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"camera {camera!r} opened at {aw}x{ah}")
        if (aw, ah) != (cam.width, cam.height):
            print(f"warning: camera params assume {cam.width}x{cam.height}; "
                  f"gaze geometry will be off if these differ")
        self.cap = LatestFrame(cap)

    @staticmethod
    def _open_window():
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    @staticmethod
    def _close_window():
        """Close the OpenCV window for real.

        On Linux the GUI toolkit only processes the close while waitKey() runs,
        so we leave fullscreen, destroy, then pump events for a moment.
        """
        try:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        except cv2.error:
            pass
        try:
            cv2.destroyWindow(WIN)
        except cv2.error:
            pass
        cv2.destroyAllWindows()
        for _ in range(10):
            cv2.waitKey(10)

    # ---- one measurement --------------------------------------------------
    def _measure(self):
        """Returns (screen_px or None, pitch, yaw, message)."""
        ok, frame = self.cap.read()
        if not ok:
            return None, None, None, "camera read failed"
        cam = self.gaze_estimator.camera
        und = frame if self._skip_undistort else cv2.undistort(
            frame, cam.camera_matrix, cam.dist_coefficients)
        self._latest_frame = und.copy()

        faces = self.gaze_estimator.detect_faces(und)
        if not faces:
            return None, None, None, "no face detected"
        face = faces[0]
        self.gaze_estimator.estimate_gaze(und, face)

        gaze = np.asarray(getattr(face, "gaze_vector", np.zeros(3)), dtype=float).reshape(-1)
        if gaze.size < 3:
            gaze = np.pad(gaze, (0, 3 - gaze.size))
        gaze = gaze[:3]
        center = np.asarray(getattr(face, "center", np.zeros(3)), dtype=float).reshape(-1)
        if center.size < 3:
            center = np.pad(center, (0, 3 - center.size))
        center = center[:3]

        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(gaze)):
            return None, None, None, "invalid gaze measurement"
        if np.linalg.norm(center) < 10:      # metres -> mm
            center = center * 1000.0
        if SET["invert"]:
            gaze = -gaze

        yaw = float(np.arctan2(gaze[0], gaze[2]))
        pitch = float(np.arcsin(np.clip(-gaze[1], -1.0, 1.0)))

        px = gaze_to_screen_space(center, gaze, res=SET["res"], geometry=self.screen_geo)
        if px is not None:
            return px, pitch, yaw, ""
        return None, pitch, yaw, "gaze not pointing at screen plane (try invert=True)"

    # ---- function 1: calibration (the only place a window is shown) ----------
    def calibrate_gaze(self):
        """Fullscreen dot calibration. The window is always closed on exit."""
        try:
            return self._calibrate()
        finally:
            self._close_window()

    def _calibrate(self):
        W, H = SET["res"]
        self._open_window()
        feats, pix = [], []

        for x, y in self._calibration_points(grid=5):
            tx, ty = int(x * W), int(y * H)
            t0, samples = time.time(), []
            while time.time() - t0 < 2.0:
                el = time.time() - t0
                if SET["show_background"] and self._latest_frame is not None:
                    canvas = cv2.resize(self._latest_frame, (W, H))
                else:
                    canvas = np.zeros((H, W, 3), np.uint8)
                r = int(30 * (1 - el / 2.0)) + 8
                cv2.circle(canvas, (tx, ty), r, (0, 255, 255), -1)
                cv2.circle(canvas, (tx, ty), 3, (0, 0, 0), -1)

                hit, _, _, msg = self._measure()
                if hit is not None and np.all(np.isfinite(hit)) and el > 0.8:
                    samples.append(np.asarray(hit, dtype=float))
                elif msg:
                    cv2.putText(canvas, msg, (40, H - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow(WIN, canvas)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    return None

            if len(samples) >= 4:
                good = self._filter_bad_samples(samples)
                if len(good) >= 4:
                    feats.append(np.median(good, axis=0))
                    pix.append((tx, ty))

        if len(feats) < 10:
            print("Calibration failed: too few good samples. Keep your face visible.")
            return None

        feats = np.asarray(feats, dtype=float)
        mu, sd = feats.mean(axis=0), feats.std(axis=0) + 1e-6
        cal = GazeCalibrator(ridge=1e-3)
        cal.fit((feats - mu) / sd, np.array(pix, dtype=float))
        err = np.linalg.norm(cal.predict((feats - mu) / sd) - np.array(pix), axis=1)
        if not np.all(np.isfinite(err)):
            print("Calibration failed: non-finite regression error.")
            return None
        print(f"Calibrated on {len(feats)} dots, mean fit error {err.mean():.0f}px")
        return cal, mu, sd

    # ---- function 2: streaming (headless unless show_window=True) -----------
    def stream_gaze_loop(self, calib_tuple, udp_ip="127.0.0.1", udp_port=5006,
                         show_window=False, median=3, min_cutoff=1.0, beta=0.01):
        W, H = SET["res"]
        cal, mu, sd = calib_tuple
        smoother = GazeSmoother(median=median, min_cutoff=min_cutoff, beta=beta,
                                d_cutoff=1.0, hold_px=30.0)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if show_window:
            self._open_window()
        print(f"Streaming gaze data to UDP {udp_ip}:{udp_port} (Ctrl+C to stop)")

        try:
            while True:
                hit, pitch, yaw, msg = self._measure()

                coords = None
                if hit is not None:
                    f = (np.asarray(hit) - mu) / sd
                    p = cal.predict(f[None, :])[0]
                    if np.all(np.isfinite(p)):
                        p = np.clip(p, [0, 0], [W - 1, H - 1])
                        smooth = smoother(p, time.perf_counter())
                        if smooth is not None and np.all(np.isfinite(smooth)):
                            coords = (float(smooth[0]), float(smooth[1]))

                payload = {"timestamp": time.time(), "pitch": pitch,
                           "yaw": yaw, "coords": coords}
                sock.sendto(pickle.dumps(payload), (udp_ip, udp_port))

                if show_window:
                    if SET["show_background"] and self._latest_frame is not None:
                        canvas = cv2.resize(self._latest_frame, (W, H))
                    else:
                        canvas = np.zeros((H, W, 3), np.uint8)
                    if coords is not None:
                        c = (int(coords[0]), int(coords[1]))
                        cv2.circle(canvas, c, 25, (0, 255, 0), 3)
                        cv2.circle(canvas, c, 4, (0, 255, 0), -1)
                    elif msg:
                        cv2.putText(canvas, msg, (40, H - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(canvas, "q: quit   c: recalibrate", (40, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow(WIN, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    if key == ord("c"):
                        new = self.calibrate_gaze()      # closes its own window
                        if new is not None:
                            cal, mu, sd = new
                            smoother.reset()
                        self._open_window()
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            sock.close()
            getattr(self.gaze_estimator, "close", lambda: None)()
            self.cap.release()
            self._close_window()
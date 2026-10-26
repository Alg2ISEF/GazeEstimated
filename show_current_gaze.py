import argparse
import sys
import time

import code
from ptgaze.main import load_mode_config
from ptgaze.utils import download_ethxgaze_model, download_mediapipe_face_landmarker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--res", default=None, help="screen resolution, e.g. 1920x1080")
    args = ap.parse_args()

    if args.res:
        w, h = args.res.lower().split("x")
        code.SET["res"] = (int(w), int(h))

    sys.argv = ["ptgaze", "--mode", "eth-xgaze", "--device", args.device]

    cfg = load_mode_config(type("Args", (), {
        "mode": "eth-xgaze",
        "face_detector": "mediapipe",
        "device": args.device,
        "image": None,
        "video": None,
        "camera": None,
        "output_dir": None,
        "ext": None,
        "no_screen": False,
    })())

    # ptgaze expects the mediapipe landmarker and gaze checkpoint to exist in
    # ~/.ptgaze before Demo() is instantiated.
    download_mediapipe_face_landmarker()
    download_ethxgaze_model()

    try:
        demo = code.LiveDemo(cfg)
    except FileNotFoundError as exc:
        print("Missing ptgaze model files.")
        print("Run: python -m ptgaze --mode eth-xgaze --device cpu")
        print(f"Details: {exc}")
        raise SystemExit(1)
    print("Press Ctrl+C to stop")

    try:
        while True:
            point = demo.get_screen_gaze()
            if point is not None:
                print(f"gaze=({point[0]:.1f}, {point[1]:.1f})")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping gaze printout")
    finally:
        demo.gaze_estimator.close()
        demo.cap.release()


if __name__ == "__main__":
    main()

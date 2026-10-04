import importlib.util
import pathlib
import unittest
from types import SimpleNamespace

from omegaconf import OmegaConf

module_path = pathlib.Path(__file__).resolve().parents[1] / "code.py"
spec = importlib.util.spec_from_file_location("gaze_code", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
configure_face_detector = module.configure_face_detector


class FaceDetectorConfigTest(unittest.TestCase):
    def test_config_can_be_updated_when_readonly(self):
        args = SimpleNamespace(
            mode="eth-xgaze",
            face_detector="mediapipe",
            device="cpu",
            image=None,
            video=None,
            camera=None,
            output_dir=None,
            ext=None,
            no_screen=False,
        )

        from ptgaze.main import load_mode_config

        config = load_mode_config(args)
        OmegaConf.set_readonly(config, True)

        configure_face_detector(config)

        self.assertIs(config.face_detector.mediapipe_static_image_mode, False)
        self.assertEqual(config.face_detector.mediapipe_max_num_faces, 1)


if __name__ == "__main__":
    unittest.main()

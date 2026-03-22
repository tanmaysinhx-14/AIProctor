from typing import Optional

from hardware.hardware_detector import HardwareMode
from vision.face_detector import FaceDetector


def create_face_detector(mode: Optional[HardwareMode] = None, max_faces: int = 5) -> FaceDetector:
  if mode == HardwareMode.CPU_MODE:
    return FaceDetector(max_faces=max(1, min(2, int(max_faces))))
  return FaceDetector(max_faces=max(1, int(max_faces)))

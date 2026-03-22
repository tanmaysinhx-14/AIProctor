from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from config import settings

try:
  import mediapipe as mp
except ImportError:  # pragma: no cover - optional dependency
  mp = None


@dataclass(frozen=True)
class PersonPoseValidation:
  valid: bool = False
  pose_keypoints_detected: bool = False
  skeleton_confidence: float = 0.0
  shoulder_ratio: float = 0.0
  left_shoulder: Tuple[float, float] = (0.0, 0.0)
  right_shoulder: Tuple[float, float] = (0.0, 0.0)
  nose: Tuple[float, float] = (0.0, 0.0)
  eye_center: Tuple[float, float] = (0.0, 0.0)


class PersonPoseValidator:
  def __init__(self) -> None:
    self._pose = None
    self._available = False

    if mp is None:
      return

    try:
      self._pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
      )
      self._available = True
    except Exception:
      self._pose = None
      self._available = False

  @property
  def available(self) -> bool:
    return self._available and self._pose is not None

  def close(self) -> None:
    if self._pose is not None:
      try:
        self._pose.close()
      except Exception:
        pass
    self._pose = None
    self._available = False

  def validate(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> PersonPoseValidation:
    if not self.available or frame_bgr is None:
      return PersonPoseValidation()

    frame_h, frame_w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = self._expand_bbox(
      bbox=bbox,
      frame_w=frame_w,
      frame_h=frame_h,
      x_ratio=0.10,
      y_ratio=0.16,
    )
    if x2 <= x1 or y2 <= y1:
      return PersonPoseValidation()

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 48 or crop.shape[1] < 48:
      return PersonPoseValidation()

    try:
      result = self._pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    except Exception:
      return PersonPoseValidation()

    landmarks = getattr(result, "pose_landmarks", None)
    if landmarks is None or not getattr(landmarks, "landmark", None):
      return PersonPoseValidation()

    points = landmarks.landmark
    required = (0, 2, 5, 11, 12)  # nose, left eye, right eye, left shoulder, right shoulder
    vis_gate = float(settings.person_pose_min_visibility)
    confidences = []

    for idx in required:
      if idx >= len(points):
        return PersonPoseValidation()
      landmark = points[idx]
      visibility = float(getattr(landmark, "visibility", 0.0))
      presence = float(getattr(landmark, "presence", visibility))
      if visibility < vis_gate or presence < vis_gate:
        return PersonPoseValidation()
      if not (0.0 <= float(landmark.x) <= 1.0 and 0.0 <= float(landmark.y) <= 1.0):
        return PersonPoseValidation()
      confidences.append(min(visibility, presence))

    left_shoulder = points[11]
    right_shoulder = points[12]
    shoulder_px = abs(float(right_shoulder.x) - float(left_shoulder.x)) * max(float(x2 - x1), 1.0)
    shoulder_ratio = shoulder_px / max(float(bbox[2] - bbox[0]), 1.0)
    nose = points[0]
    left_eye = points[2]
    right_eye = points[5]
    left_shoulder_abs = (
      float(x1) + (float(left_shoulder.x) * max(float(x2 - x1), 1.0)),
      float(y1) + (float(left_shoulder.y) * max(float(y2 - y1), 1.0)),
    )
    right_shoulder_abs = (
      float(x1) + (float(right_shoulder.x) * max(float(x2 - x1), 1.0)),
      float(y1) + (float(right_shoulder.y) * max(float(y2 - y1), 1.0)),
    )
    nose_abs = (
      float(x1) + (float(nose.x) * max(float(x2 - x1), 1.0)),
      float(y1) + (float(nose.y) * max(float(y2 - y1), 1.0)),
    )
    eye_center_abs = (
      float(x1) + (((float(left_eye.x) + float(right_eye.x)) * 0.5) * max(float(x2 - x1), 1.0)),
      float(y1) + (((float(left_eye.y) + float(right_eye.y)) * 0.5) * max(float(y2 - y1), 1.0)),
    )
    pose_keypoints_detected = True
    skeleton_confidence = float(np.mean(confidences)) if confidences else 0.0
    valid = pose_keypoints_detected and shoulder_ratio >= float(settings.person_shoulder_width_ratio_min)
    return PersonPoseValidation(
      valid=bool(valid),
      pose_keypoints_detected=pose_keypoints_detected,
      skeleton_confidence=skeleton_confidence,
      shoulder_ratio=shoulder_ratio,
      left_shoulder=left_shoulder_abs,
      right_shoulder=right_shoulder_abs,
      nose=nose_abs,
      eye_center=eye_center_abs,
    )

  @staticmethod
  def _expand_bbox(
    bbox: Tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    x_ratio: float,
    y_ratio: float,
  ) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    pad_x = int(round(bw * max(0.0, x_ratio)))
    pad_y = int(round(bh * max(0.0, y_ratio)))
    return (
      max(0, x1 - pad_x),
      max(0, y1 - pad_y),
      min(frame_w, x2 + pad_x),
      min(frame_h, y2 + pad_y),
    )

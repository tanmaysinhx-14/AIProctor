from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings

import cv2
import numpy as np

from config import settings
from vision.face_tracker import TrackedFace

try:
  import mediapipe as mp
except ImportError:  # pragma: no cover - optional dependency resolution
  mp = None


@dataclass(frozen=True)
class HandBBox:
  x1: int
  y1: int
  x2: int
  y2: int


class HandFaceInteractionDetector:
  def __init__(self) -> None:
    self._hands = None
    self._available = False
    self._initialize_hands()

  @property
  def is_available(self) -> bool:
    return self._available

  def _initialize_hands(self) -> None:
    try:
      if mp is None:
        raise RuntimeError("mediapipe is not installed")

      hands_api = getattr(getattr(mp, "solutions", None), "hands", None)
      if hands_api is None:
        raise RuntimeError("mediapipe Hands API is unavailable in the installed build")

      self._hands = hands_api.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
      )
      self._available = True
    except Exception as exc:  # pragma: no cover - runtime environment specific
      self._hands = None
      self._available = False
      warnings.warn(f"Hand detector disabled: {exc}", RuntimeWarning)

  def detect(self, frame_bgr: np.ndarray, tracked_faces: List[TrackedFace]) -> Dict[int, bool]:
    if self._hands is None or not tracked_faces:
      return {face.id: False for face in tracked_faces}

    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = self._hands.process(frame_rgb)

    hand_boxes: List[HandBBox] = []
    if result.multi_hand_landmarks:
      for hand_landmarks in result.multi_hand_landmarks:
        hand_boxes.append(self._to_bbox(hand_landmarks, width=w, height=h))

    if not hand_boxes:
      return {face.id: False for face in tracked_faces}

    overlap_threshold = max(0.0, min(settings.hand_on_face_overlap_threshold, 1.0))
    expand_ratio = max(0.0, settings.hand_on_face_bbox_expand_ratio)

    face_to_hand: Dict[int, bool] = {}
    for face in tracked_faces:
      face_box = self._expand_bbox(face.bbox, frame_width=w, frame_height=h, ratio=expand_ratio)
      hit = any(self._overlap_ratio(face_box, hand_box) >= overlap_threshold for hand_box in hand_boxes)
      face_to_hand[face.id] = bool(hit)

    return face_to_hand

  def close(self) -> None:
    if self._hands is not None:
      self._hands.close()

  def _to_bbox(self, hand_landmarks, width: int, height: int) -> HandBBox:
    xs = []
    ys = []
    for landmark in hand_landmarks.landmark:
      xs.append(int(np.clip(landmark.x * width, 0, width - 1)))
      ys.append(int(np.clip(landmark.y * height, 0, height - 1)))

    return HandBBox(
      x1=int(min(xs)),
      y1=int(min(ys)),
      x2=int(max(xs)),
      y2=int(max(ys)),
    )

  def _expand_bbox(
    self,
    bbox: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    ratio: float,
  ) -> HandBBox:
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(round(bw * ratio))
    pad_y = int(round(bh * ratio))
    return HandBBox(
      x1=int(np.clip(x1 - pad_x, 0, frame_width - 1)),
      y1=int(np.clip(y1 - pad_y, 0, frame_height - 1)),
      x2=int(np.clip(x2 + pad_x, 0, frame_width - 1)),
      y2=int(np.clip(y2 + pad_y, 0, frame_height - 1)),
    )

  def _overlap_ratio(self, a: HandBBox, b: HandBBox) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = float(inter_w * inter_h)

    a_area = float(max(1, (a.x2 - a.x1) * (a.y2 - a.y1)))
    return inter_area / a_area

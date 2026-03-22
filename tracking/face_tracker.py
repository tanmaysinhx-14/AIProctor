from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from tracking.face_embedding import FaceEmbeddingExtractor
from vision.face_detector import FaceObservation
from vision.face_tracker import CentroidFaceTracker, TrackedFace

BBox = Tuple[int, int, int, int]


@dataclass
class _TrackedState:
  id: int
  bbox: BBox
  centroid: Tuple[float, float]
  landmarks: np.ndarray
  tracker: Optional[object]
  embedding: Optional[np.ndarray] = None
  velocity: Tuple[float, float] = (0.0, 0.0)
  missed: int = 0
  persistence_frames: int = 1


class AdaptiveFaceTracker:
  """
  Periodic detector + lightweight tracker.
  Detection runs every N frames, intermediate frames update bboxes via OpenCV trackers.
  """

  def __init__(
    self,
    max_disappeared: int = 12,
    distance_threshold: float = 90.0,
    tracker_priority: Tuple[str, ...] = ("KCF", "CSRT", "MIL"),
  ) -> None:
    self._id_tracker = CentroidFaceTracker(
      max_disappeared=max_disappeared,
      distance_threshold=distance_threshold,
    )
    self._states: Dict[int, _TrackedState] = {}
    self._tracker_priority = tracker_priority
    self._max_missed = max(2, int(max_disappeared))
    self._tracking_backend_available = True
    self._detection_dedupe_iou = 0.62
    self._state_overlap_iou = 0.45
    self._bbox_smoothing_alpha = 0.62
    self._predictive_hold_frames = 3
    self._max_predictive_shift_ratio = 0.34
    self._embedding_extractor = FaceEmbeddingExtractor()

  def active_count(self) -> int:
    return len(self._states)

  def reset(self) -> None:
    self._states.clear()

  def update(
    self,
    frame: np.ndarray,
    detections: Optional[List[FaceObservation]] = None,
    run_detection: bool = False,
  ) -> List[TrackedFace]:
    if frame is None:
      return []

    if run_detection and detections is not None:
      return self._update_from_detection(frame, detections)
    return self._update_from_trackers(frame)

  def _update_from_detection(self, frame: np.ndarray, detections: List[FaceObservation]) -> List[TrackedFace]:
    detections = self._attach_embeddings(frame, self._dedupe_detections(detections))
    tracked = self._id_tracker.update(detections)
    seen_ids = set()
    new_states: Dict[int, _TrackedState] = {}

    for face in tracked:
      seen_ids.add(face.id)
      previous = self._states.get(face.id)
      tracker = self._create_tracker()
      if tracker is not None:
        self._init_tracker(tracker, frame, face.bbox)

      landmarks = face.landmarks
      if landmarks is None or len(landmarks) == 0:
        if previous is not None:
          landmarks = previous.landmarks
        else:
          landmarks = np.zeros((0, 2), dtype=np.float32)

      velocity = (0.0, 0.0)
      if previous is not None:
        vx = float(face.centroid[0] - previous.centroid[0])
        vy = float(face.centroid[1] - previous.centroid[1])
        pvx, pvy = previous.velocity
        velocity = ((0.65 * pvx) + (0.35 * vx), (0.65 * pvy) + (0.35 * vy))

      new_states[face.id] = _TrackedState(
        id=face.id,
        bbox=face.bbox,
        centroid=face.centroid,
        landmarks=landmarks,
        tracker=tracker,
        embedding=face.embedding,
        velocity=velocity,
        missed=0,
        persistence_frames=int(getattr(face, "persistence_frames", max(getattr(previous, "persistence_frames", 0), 0) + 1)),
      )

    new_bboxes = [state.bbox for state in new_states.values()]
    # Keep unmatched states briefly, but never keep overlapping ghosts that duplicate new detections.
    for track_id, state in list(self._states.items()):
      if track_id in seen_ids:
        continue
      state.missed += 1
      if state.missed <= self._max_missed and not self._overlaps_any(state.bbox, new_bboxes, self._state_overlap_iou):
        new_states[track_id] = state

    self._states = new_states
    return self._to_tracked_faces(include_missed=(len(detections) == 0))

  def _update_from_trackers(self, frame: np.ndarray) -> List[TrackedFace]:
    if not self._states:
      return []

    updated: Dict[int, _TrackedState] = {}
    for track_id, state in list(self._states.items()):
      next_bbox: Optional[BBox] = None
      if self._tracking_backend_available and state.tracker is not None:
        ok, raw_bbox = self._tracker_update(state.tracker, frame)
        if ok and raw_bbox is not None:
          next_bbox = self._sanitize_bbox(raw_bbox, frame.shape)

      if next_bbox is None:
        state.missed += 1
        if state.missed > self._max_missed:
          continue
        if state.missed <= self._predictive_hold_frames:
          predicted_bbox = self._shift_bbox(
            bbox=state.bbox,
            velocity=state.velocity,
            frame_shape=frame.shape,
          )
          if predicted_bbox is not None:
            cx = (predicted_bbox[0] + predicted_bbox[2]) / 2.0
            cy = (predicted_bbox[1] + predicted_bbox[3]) / 2.0
            updated[track_id] = _TrackedState(
              id=state.id,
              bbox=predicted_bbox,
              centroid=(cx, cy),
              landmarks=state.landmarks,
              tracker=state.tracker,
              embedding=state.embedding,
              velocity=(state.velocity[0] * 0.84, state.velocity[1] * 0.84),
              missed=state.missed,
              persistence_frames=state.persistence_frames,
            )
            continue
        # Fallback: keep previous bbox for a short time if tracker transiently fails.
        updated[track_id] = state
        continue

      next_bbox = self._blend_bbox(state.bbox, next_bbox)
      x1, y1, x2, y2 = state.bbox
      nx1, ny1, nx2, ny2 = next_bbox
      dx = float(nx1 - x1)
      dy = float(ny1 - y1)

      shifted_landmarks = state.landmarks
      if shifted_landmarks.size > 0:
        shifted_landmarks = shifted_landmarks.copy()
        shifted_landmarks[:, 0] += dx
        shifted_landmarks[:, 1] += dy
        shifted_landmarks[:, 0] = np.clip(shifted_landmarks[:, 0], 0, frame.shape[1] - 1)
        shifted_landmarks[:, 1] = np.clip(shifted_landmarks[:, 1], 0, frame.shape[0] - 1)

      centroid = ((nx1 + nx2) / 2.0, (ny1 + ny2) / 2.0)
      vx = float(centroid[0] - state.centroid[0])
      vy = float(centroid[1] - state.centroid[1])
      pvx, pvy = state.velocity
      velocity = ((0.65 * pvx) + (0.35 * vx), (0.65 * pvy) + (0.35 * vy))
      updated[track_id] = _TrackedState(
        id=track_id,
        bbox=next_bbox,
        centroid=centroid,
        landmarks=shifted_landmarks,
        tracker=state.tracker,
        embedding=state.embedding,
        velocity=velocity,
        missed=0,
        persistence_frames=state.persistence_frames + 1,
      )

    self._states = updated
    return self._to_tracked_faces(include_missed=False)

  def _to_tracked_faces(self, include_missed: bool = False) -> List[TrackedFace]:
    faces = [
      TrackedFace(
        id=state.id,
        bbox=state.bbox,
        centroid=state.centroid,
        landmarks=state.landmarks,
        embedding=state.embedding,
        persistence_frames=state.persistence_frames,
      )
      for state in self._states.values()
      if include_missed or state.missed == 0
    ]
    faces = self._dedupe_tracked_faces(faces)
    faces.sort(key=lambda item: int(item.id))
    return faces

  def _dedupe_detections(self, detections: List[FaceObservation]) -> List[FaceObservation]:
    if len(detections) <= 1:
      return list(detections)

    def area(det: FaceObservation) -> int:
      x1, y1, x2, y2 = det.bbox
      return max(1, x2 - x1) * max(1, y2 - y1)

    ordered = sorted(detections, key=area, reverse=True)
    kept: List[FaceObservation] = []
    kept_boxes: List[BBox] = []

    for det in ordered:
      if self._overlaps_any(det.bbox, kept_boxes, self._detection_dedupe_iou):
        continue
      kept.append(det)
      kept_boxes.append(det.bbox)
    return kept

  def _attach_embeddings(self, frame: np.ndarray, detections: List[FaceObservation]) -> List[FaceObservation]:
    enriched: List[FaceObservation] = []
    for detection in detections:
      embedding = self._embedding_extractor.extract(frame, detection.bbox)
      enriched.append(
        FaceObservation(
          bbox=detection.bbox,
          centroid=detection.centroid,
          landmarks=detection.landmarks,
          embedding=embedding,
        )
      )
    return enriched

  def _dedupe_tracked_faces(self, faces: List[TrackedFace]) -> List[TrackedFace]:
    if len(faces) <= 1:
      return faces

    def area(face: TrackedFace) -> int:
      x1, y1, x2, y2 = face.bbox
      return max(1, x2 - x1) * max(1, y2 - y1)

    ordered = sorted(faces, key=area, reverse=True)
    kept: List[TrackedFace] = []
    kept_boxes: List[BBox] = []

    for face in ordered:
      if self._overlaps_any(face.bbox, kept_boxes, self._detection_dedupe_iou):
        continue
      kept.append(face)
      kept_boxes.append(face.bbox)
    return kept

  def _create_tracker(self) -> Optional[object]:
    for name in self._tracker_priority:
      tracker = self._create_tracker_by_name(name)
      if tracker is not None:
        return tracker
    self._tracking_backend_available = False
    return None

  def _create_tracker_by_name(self, name: str) -> Optional[object]:
    creator_names = [
      f"Tracker{name}_create",
      f"Tracker{name.upper()}_create",
      f"Tracker{name.lower()}_create",
    ]

    for creator in creator_names:
      fn = getattr(cv2, creator, None)
      if callable(fn):
        try:
          return fn()
        except Exception:
          continue

      legacy = getattr(cv2, "legacy", None)
      if legacy is not None:
        fn = getattr(legacy, creator, None)
        if callable(fn):
          try:
            return fn()
          except Exception:
            continue
    return None

  @staticmethod
  def _init_tracker(tracker: object, frame: np.ndarray, bbox: BBox) -> None:
    x1, y1, x2, y2 = bbox
    width = max(1, int(x2 - x1))
    height = max(1, int(y2 - y1))
    try:
      tracker.init(frame, (int(x1), int(y1), width, height))
    except Exception:
      pass

  @staticmethod
  def _tracker_update(tracker: object, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[float, float, float, float]]]:
    try:
      ok, bbox = tracker.update(frame)
      if not ok:
        return False, None
      if bbox is None or len(bbox) < 4:
        return False, None
      return True, (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except Exception:
      return False, None

  @staticmethod
  def _sanitize_bbox(raw_bbox: Tuple[float, float, float, float], frame_shape: Tuple[int, int, int]) -> BBox:
    x, y, w, h = raw_bbox
    frame_h, frame_w = frame_shape[:2]
    x1 = max(0, min(int(round(x)), frame_w - 1))
    y1 = max(0, min(int(round(y)), frame_h - 1))
    x2 = max(0, min(int(round(x + w)), frame_w - 1))
    y2 = max(0, min(int(round(y + h)), frame_h - 1))
    if x2 <= x1:
      x2 = min(frame_w - 1, x1 + 1)
    if y2 <= y1:
      y2 = min(frame_h - 1, y1 + 1)
    return (x1, y1, x2, y2)

  def _blend_bbox(self, previous: BBox, current: BBox) -> BBox:
    px1, py1, px2, py2 = previous
    cx1, cy1, cx2, cy2 = current
    prev_w = float(max(px2 - px1, 1))
    prev_h = float(max(py2 - py1, 1))
    shift = max(abs(cx1 - px1), abs(cy1 - py1))
    shift_ratio = shift / max(prev_w, prev_h, 1.0)
    alpha = 0.82 if shift_ratio >= 0.28 else self._bbox_smoothing_alpha

    bx1 = int(round((1.0 - alpha) * px1 + alpha * cx1))
    by1 = int(round((1.0 - alpha) * py1 + alpha * cy1))
    bx2 = int(round((1.0 - alpha) * px2 + alpha * cx2))
    by2 = int(round((1.0 - alpha) * py2 + alpha * cy2))
    if bx2 <= bx1:
      bx2 = bx1 + 1
    if by2 <= by1:
      by2 = by1 + 1
    return (bx1, by1, bx2, by2)

  def _shift_bbox(
    self,
    bbox: BBox,
    velocity: Tuple[float, float],
    frame_shape: Tuple[int, int, int],
  ) -> Optional[BBox]:
    x1, y1, x2, y2 = bbox
    w = float(max(x2 - x1, 1))
    h = float(max(y2 - y1, 1))
    max_dx = w * self._max_predictive_shift_ratio
    max_dy = h * self._max_predictive_shift_ratio
    vx = float(np.clip(velocity[0], -max_dx, max_dx))
    vy = float(np.clip(velocity[1], -max_dy, max_dy))
    if abs(vx) < 0.25 and abs(vy) < 0.25:
      return bbox
    return self._sanitize_bbox(
      (float(x1 + vx), float(y1 + vy), float(x2 - x1), float(y2 - y1)),
      frame_shape,
    )

  @classmethod
  def _overlaps_any(cls, bbox: BBox, candidates: List[BBox], iou_threshold: float) -> bool:
    return any(cls._bbox_iou(bbox, other) >= iou_threshold for other in candidates)

  @staticmethod
  def _bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = float(inter_w * inter_h)
    if inter_area <= 0.0:
      return 0.0
    area_a = float(max((ax2 - ax1) * (ay2 - ay1), 1))
    area_b = float(max((bx2 - bx1) * (by2 - by1), 1))
    return inter_area / max(area_a + area_b - inter_area, 1.0)

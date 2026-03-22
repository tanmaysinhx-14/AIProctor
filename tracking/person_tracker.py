from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from vision.object_detector import DetectedObject

BBox = Tuple[int, int, int, int]


@dataclass
class TrackedPerson:
  id: int
  bbox: BBox
  centroid: Tuple[float, float]
  confidence: float
  disappeared: int = 0
  visible_frames: int = 1


@dataclass
class _TrackState:
  bbox: BBox
  centroid: Tuple[float, float]
  confidence: float
  disappeared: int = 0
  visible_frames: int = 1


class CentroidPersonTracker:
  def __init__(self, max_disappeared: int = 10, distance_threshold: float = 160.0) -> None:
    self._max_disappeared = max(1, int(max_disappeared))
    self._distance_threshold = max(1.0, float(distance_threshold))
    self._next_id = 1
    self._tracks: Dict[int, _TrackState] = {}

  def active_count(self) -> int:
    return len(self._tracks)

  def reset(self) -> None:
    self._tracks.clear()
    self._next_id = 1

  def update(self, detections: List[DetectedObject]) -> List[TrackedPerson]:
    person_detections = [det for det in detections if det.label == "person" and det.bbox is not None]
    if not person_detections:
      return self._advance_without_detections()

    person_detections = self._dedupe(person_detections)
    if not self._tracks:
      return [self._register(det) for det in person_detections]

    track_ids = list(self._tracks.keys())
    track_centroids = np.array([self._tracks[track_id].centroid for track_id in track_ids], dtype=np.float32)
    det_centroids = np.array([self._centroid(det.bbox) for det in person_detections], dtype=np.float32)
    distance_matrix = cdist(track_centroids, det_centroids)
    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    used_track_rows = set()
    used_det_cols = set()
    result: List[TrackedPerson] = []

    for row_idx, col_idx in zip(row_indices, col_indices):
      if row_idx in used_track_rows or col_idx in used_det_cols:
        continue

      track_id = track_ids[row_idx]
      state = self._tracks[track_id]
      detection = person_detections[col_idx]
      assert detection.bbox is not None

      distance = float(distance_matrix[row_idx, col_idx])
      adaptive_distance = max(
        self._distance_threshold,
        self._bbox_diag(state.bbox) * 0.75,
        self._bbox_diag(detection.bbox) * 0.68,
      )
      if distance > adaptive_distance and self._bbox_iou(state.bbox, detection.bbox) < 0.16:
        continue

      centroid = self._centroid(detection.bbox)
      self._tracks[track_id] = _TrackState(
        bbox=detection.bbox,
        centroid=centroid,
        confidence=float(detection.confidence),
        disappeared=0,
        visible_frames=state.visible_frames + 1,
      )
      result.append(
        TrackedPerson(
          id=track_id,
          bbox=detection.bbox,
          centroid=centroid,
          confidence=float(detection.confidence),
          disappeared=0,
          visible_frames=state.visible_frames + 1,
        )
      )
      used_track_rows.add(row_idx)
      used_det_cols.add(col_idx)

    for row_idx, track_id in enumerate(track_ids):
      if row_idx in used_track_rows:
        continue
      state = self._tracks[track_id]
      state.disappeared += 1
      if state.disappeared > self._max_disappeared:
        del self._tracks[track_id]
        continue
      result.append(
        TrackedPerson(
          id=track_id,
          bbox=state.bbox,
          centroid=state.centroid,
          confidence=state.confidence,
          disappeared=state.disappeared,
          visible_frames=state.visible_frames,
        )
      )

    for col_idx, detection in enumerate(person_detections):
      if col_idx in used_det_cols:
        continue
      result.append(self._register(detection))

    result.sort(key=lambda item: int(item.id))
    return result

  def _advance_without_detections(self) -> List[TrackedPerson]:
    result: List[TrackedPerson] = []
    for track_id in list(self._tracks.keys()):
      state = self._tracks[track_id]
      state.disappeared += 1
      if state.disappeared > self._max_disappeared:
        del self._tracks[track_id]
        continue
      result.append(
        TrackedPerson(
          id=track_id,
          bbox=state.bbox,
          centroid=state.centroid,
          confidence=state.confidence,
          disappeared=state.disappeared,
          visible_frames=state.visible_frames,
        )
      )
    result.sort(key=lambda item: int(item.id))
    return result

  def _register(self, detection: DetectedObject) -> TrackedPerson:
    assert detection.bbox is not None
    track_id = self._next_id
    self._next_id += 1
    centroid = self._centroid(detection.bbox)
    confidence = float(detection.confidence)
    self._tracks[track_id] = _TrackState(
      bbox=detection.bbox,
      centroid=centroid,
      confidence=confidence,
      disappeared=0,
      visible_frames=1,
    )
    return TrackedPerson(
      id=track_id,
      bbox=detection.bbox,
      centroid=centroid,
      confidence=confidence,
      disappeared=0,
      visible_frames=1,
    )

  def _dedupe(self, detections: List[DetectedObject]) -> List[DetectedObject]:
    if len(detections) <= 1:
      return list(detections)

    ordered = sorted(
      detections,
      key=lambda item: float(item.confidence) * float(self._bbox_area(item.bbox or (0, 0, 1, 1))),
      reverse=True,
    )
    kept: List[DetectedObject] = []
    kept_boxes: List[BBox] = []
    for detection in ordered:
      bbox = detection.bbox
      if bbox is None:
        continue
      if any(self._bbox_iou(bbox, other) >= 0.62 for other in kept_boxes):
        continue
      kept.append(detection)
      kept_boxes.append(bbox)
    return kept

  @staticmethod
  def _centroid(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

  @staticmethod
  def _bbox_area(bbox: BBox) -> int:
    x1, y1, x2, y2 = bbox
    return max(1, x2 - x1) * max(1, y2 - y1)

  @staticmethod
  def _bbox_diag(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    w = float(max(x2 - x1, 1))
    h = float(max(y2 - y1, 1))
    return float(np.hypot(w, h))

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

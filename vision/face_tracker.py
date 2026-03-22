from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from config import settings
from tracking.face_embedding import FaceEmbeddingExtractor
from vision.face_detector import FaceObservation


@dataclass
class TrackedFace:
  id: int
  bbox: Tuple[int, int, int, int]
  centroid: Tuple[float, float]
  landmarks: np.ndarray
  embedding: Optional[np.ndarray] = None
  persistence_frames: int = 1


@dataclass
class _TrackState:
  centroid: Tuple[float, float]
  bbox: Tuple[int, int, int, int]
  landmarks: np.ndarray
  embedding: Optional[np.ndarray] = None
  velocity: Tuple[float, float] = (0.0, 0.0)
  disappeared: int = 0
  persistence_frames: int = 1


class CentroidFaceTracker:
  def __init__(self, max_disappeared: int = 12, distance_threshold: float = 90.0) -> None:
    self.max_disappeared = max_disappeared
    self.distance_threshold = distance_threshold
    self._merge_iou = float(settings.tracker_merge_iou)
    self._merge_similarity = float(settings.tracker_embedding_similarity_threshold)
    self._next_id = 1
    self._tracks: Dict[int, _TrackState] = {}

  def update(self, detections: List[FaceObservation]) -> List[TrackedFace]:
    if not detections:
      for track_id in list(self._tracks.keys()):
        self._tracks[track_id].disappeared += 1
        if self._tracks[track_id].disappeared > self.max_disappeared:
          del self._tracks[track_id]
      return []

    if not self._tracks:
      return [self._register(det) for det in detections]

    track_ids = list(self._tracks.keys())
    track_centroids = np.array([self._predict_centroid(self._tracks[t]) for t in track_ids], dtype=np.float32)
    detection_centroids = np.array([d.centroid for d in detections], dtype=np.float32)
    distance_matrix = cdist(track_centroids, detection_centroids)
    cost_matrix = np.full(distance_matrix.shape, 1e6, dtype=np.float32)
    match_meta: Dict[Tuple[int, int], Tuple[float, float, float]] = {}

    for track_idx, track_id in enumerate(track_ids):
      track_state = self._tracks[track_id]
      adaptive_distance = max(
        float(self.distance_threshold),
        self._bbox_diag(track_state.bbox) * 0.58,
      )
      for detection_idx, detection in enumerate(detections):
        distance = float(distance_matrix[track_idx, detection_idx])
        iou = self._bbox_iou(track_state.bbox, detection.bbox)
        similarity = FaceEmbeddingExtractor.similarity(track_state.embedding, detection.embedding)
        if not self._is_match_candidate(
          distance=distance,
          adaptive_distance=adaptive_distance,
          iou=iou,
          similarity=similarity,
        ):
          continue
        cost_matrix[track_idx, detection_idx] = self._match_cost(
          distance=distance,
          adaptive_distance=adaptive_distance,
          iou=iou,
          similarity=similarity,
        )
        match_meta[(track_idx, detection_idx)] = (distance, iou, similarity)

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    used_tracks = set()
    used_detections = set()
    result: List[TrackedFace] = []

    for track_idx, detection_idx in zip(row_indices, col_indices):
      if track_idx in used_tracks or detection_idx in used_detections:
        continue

      distance, iou, similarity = match_meta.get(
        (track_idx, detection_idx),
        (float(distance_matrix[track_idx, detection_idx]), 0.0, 0.0),
      )
      track_id = track_ids[track_idx]
      track_state = self._tracks[track_id]
      detection = detections[detection_idx]
      adaptive_distance = max(float(self.distance_threshold), self._bbox_diag(track_state.bbox) * 0.58)
      if not self._is_match_candidate(
        distance=distance,
        adaptive_distance=adaptive_distance,
        iou=iou,
        similarity=similarity,
      ):
        continue

      vx = float(detection.centroid[0] - track_state.centroid[0])
      vy = float(detection.centroid[1] - track_state.centroid[1])
      prev_vx, prev_vy = track_state.velocity
      fused_velocity = (
        (0.65 * prev_vx) + (0.35 * vx),
        (0.65 * prev_vy) + (0.35 * vy),
      )
      self._tracks[track_id] = _TrackState(
        centroid=detection.centroid,
        bbox=detection.bbox,
        landmarks=detection.landmarks,
        embedding=FaceEmbeddingExtractor.blend(track_state.embedding, detection.embedding),
        velocity=fused_velocity,
        disappeared=0,
        persistence_frames=track_state.persistence_frames + 1,
      )
      result.append(
        TrackedFace(
          id=track_id,
          bbox=detection.bbox,
          centroid=detection.centroid,
          landmarks=detection.landmarks,
          embedding=self._tracks[track_id].embedding,
          persistence_frames=self._tracks[track_id].persistence_frames,
        )
      )
      used_tracks.add(track_idx)
      used_detections.add(detection_idx)

    for track_idx, track_id in enumerate(track_ids):
      if track_idx not in used_tracks:
        self._tracks[track_id].disappeared += 1
        if self._tracks[track_id].disappeared > self.max_disappeared:
          del self._tracks[track_id]

    for detection_idx, detection in enumerate(detections):
      if detection_idx not in used_detections:
        merged_face = self._merge_detection(track_ids, used_tracks, detection)
        if merged_face is not None:
          result.append(merged_face)
          continue
        result.append(self._register(detection))

    result.sort(key=lambda face: face.id)
    return result

  def _register(self, detection: FaceObservation) -> TrackedFace:
    track_id = self._next_id
    self._next_id += 1
    self._tracks[track_id] = _TrackState(
      centroid=detection.centroid,
      bbox=detection.bbox,
      landmarks=detection.landmarks,
      embedding=detection.embedding,
      velocity=(0.0, 0.0),
      disappeared=0,
      persistence_frames=1,
    )
    return TrackedFace(
      id=track_id,
      bbox=detection.bbox,
      centroid=detection.centroid,
      landmarks=detection.landmarks,
      embedding=detection.embedding,
      persistence_frames=1,
    )

  def _merge_detection(
    self,
    track_ids: List[int],
    used_tracks: set[int],
    detection: FaceObservation,
  ) -> TrackedFace | None:
    best_track_idx: int | None = None
    best_track_id: int | None = None
    best_score = float("-inf")

    for track_idx, track_id in enumerate(track_ids):
      if track_idx in used_tracks or track_id not in self._tracks:
        continue
      state = self._tracks[track_id]
      iou = self._bbox_iou(state.bbox, detection.bbox)
      similarity = FaceEmbeddingExtractor.similarity(state.embedding, detection.embedding)
      if iou < self._merge_iou and similarity < self._merge_similarity:
        continue
      score = max(iou, similarity)
      if score <= best_score:
        continue
      best_track_idx = track_idx
      best_track_id = track_id
      best_score = score

    if best_track_idx is None or best_track_id is None:
      return None

    state = self._tracks[best_track_id]
    vx = float(detection.centroid[0] - state.centroid[0])
    vy = float(detection.centroid[1] - state.centroid[1])
    prev_vx, prev_vy = state.velocity
    fused_velocity = (
      (0.65 * prev_vx) + (0.35 * vx),
      (0.65 * prev_vy) + (0.35 * vy),
    )
    self._tracks[best_track_id] = _TrackState(
      centroid=detection.centroid,
      bbox=detection.bbox,
      landmarks=detection.landmarks,
      embedding=FaceEmbeddingExtractor.blend(state.embedding, detection.embedding),
      velocity=fused_velocity,
      disappeared=0,
      persistence_frames=state.persistence_frames + 1,
    )
    used_tracks.add(best_track_idx)
    return TrackedFace(
      id=best_track_id,
      bbox=detection.bbox,
      centroid=detection.centroid,
      landmarks=detection.landmarks,
      embedding=self._tracks[best_track_id].embedding,
      persistence_frames=self._tracks[best_track_id].persistence_frames,
    )

  def _is_match_candidate(
    self,
    distance: float,
    adaptive_distance: float,
    iou: float,
    similarity: float,
  ) -> bool:
    return (
      distance <= adaptive_distance
      or iou >= self._merge_iou
      or similarity >= self._merge_similarity
      or (iou >= 0.22 and distance <= adaptive_distance * 1.25)
      or (similarity >= self._merge_similarity * 0.82 and distance <= adaptive_distance * 1.20)
    )

  @staticmethod
  def _match_cost(distance: float, adaptive_distance: float, iou: float, similarity: float) -> float:
    norm_distance = min(distance / max(adaptive_distance, 1e-6), 3.0)
    return float(norm_distance - (0.85 * iou) - (0.75 * max(similarity, 0.0)))

  @staticmethod
  def _predict_centroid(state: _TrackState) -> Tuple[float, float]:
    vx, vy = state.velocity
    cx, cy = state.centroid
    lead = 1.0 + min(max(float(state.disappeared), 0.0), 2.0) * 0.35
    return (cx + (vx * lead), cy + (vy * lead))

  @staticmethod
  def _bbox_diag(bbox: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    w = float(max(x2 - x1, 1))
    h = float(max(y2 - y1, 1))
    return float(np.hypot(w, h))

  @staticmethod
  def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
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

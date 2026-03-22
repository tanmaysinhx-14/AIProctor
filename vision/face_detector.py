from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import Deque, Dict, List, Optional, Tuple
import warnings

import cv2
import numpy as np

try:
  import mediapipe as mp
except ImportError:  # pragma: no cover - optional dependency resolution
  mp = None


@dataclass
class FaceObservation:
  bbox: Tuple[int, int, int, int]
  centroid: Tuple[float, float]
  landmarks: np.ndarray
  embedding: Optional[np.ndarray] = None


@dataclass
class _PendingFace:
  bbox: Tuple[int, int, int, int]
  centroid: Tuple[float, float]
  landmarks: np.ndarray
  hits: int
  last_seen: float


class FaceDetector:
  def __init__(self, max_faces: int = 5) -> None:
    self._mesh = None
    self._available = False
    self._max_faces = max(1, int(max_faces))

    # Temporal suppression for transient false positives.
    self._pending: Dict[int, _PendingFace] = {}
    self._next_pending_id = 1
    self._confirmed_faces: List[FaceObservation] = []
    self._confirmed_miss_count = 0

    # Tuned for proctoring-style near-camera face capture.
    self._new_face_confirm_hits = 2
    self._pending_ttl_sec = 1.4
    self._confirmed_retention_cycles = 2
    self._match_iou = 0.24
    self._match_dist_px = 95.0
    self._dedupe_iou = 0.62

    self._min_area_ratio = 0.0035
    self._max_area_ratio = 0.78
    self._min_aspect = 0.45
    self._max_aspect = 2.1
    self._min_relative_new_face_area = 0.32

    # Horizontal-motion-aware hardening:
    # when primary face moves rapidly, require stronger confirmation for new faces.
    self._primary_motion_history: Deque[Tuple[float, float]] = deque(maxlen=10)
    self._last_primary_signature: Optional[Tuple[float, float]] = None
    self._horizontal_boost_cycles = 0
    self._horizontal_boost_window = 7
    self._horizontal_shift_ratio_trigger = 0.10
    self._background_motion_zones: List[Tuple[float, float, float, float]] = []
    self._background_zone_iou_reject = 0.20
    self._background_zone_center_margin = 0.08

    self._initialize_mesh(max_faces=max_faces)

  @property
  def is_available(self) -> bool:
    return self._available

  def _initialize_mesh(self, max_faces: int) -> None:
    try:
      if mp is None:
        raise RuntimeError("mediapipe is not installed")

      # Prefer legacy Face Mesh API when available.
      face_mesh_api = getattr(getattr(mp, "solutions", None), "face_mesh", None)
      if face_mesh_api is None:
        raise RuntimeError("mediapipe FaceMesh API is unavailable in the installed build")

      self._mesh = face_mesh_api.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_faces,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
      )
      self._available = True
    except Exception as exc:  # pragma: no cover - runtime environment specific
      self._mesh = None
      self._available = False
      warnings.warn(f"Face detector disabled: {exc}", RuntimeWarning)

  def detect(
    self,
    frame_bgr: np.ndarray,
    roi_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
  ) -> List[FaceObservation]:
    if self._mesh is None:
      return []
    candidates: List[FaceObservation] = []
    prepared_rois = self._prepare_roi_boxes(roi_boxes or [], frame_shape=frame_bgr.shape)
    for roi_bbox in prepared_rois:
      candidates.extend(self._detect_in_region(frame_bgr, roi_bbox))

    if not candidates:
      candidates = self._detect_in_region(frame_bgr, None)
      if not candidates:
        self._advance_without_detection()
        return []

    candidates = self._dedupe_overlaps(candidates)
    self._update_primary_motion(candidates)
    accepted = self._temporal_filter(candidates, frame_shape=frame_bgr.shape)
    accepted = self._sort_and_limit(accepted)

    if accepted:
      self._confirmed_faces = accepted
      self._confirmed_miss_count = 0
    else:
      self._advance_without_detection()

    return accepted

  def _detect_in_region(
    self,
    frame_bgr: np.ndarray,
    roi_bbox: Optional[Tuple[int, int, int, int]],
  ) -> List[FaceObservation]:
    frame_h, frame_w = frame_bgr.shape[:2]
    if roi_bbox is None:
      region = frame_bgr
      origin_x = 0
      origin_y = 0
    else:
      x1, y1, x2, y2 = roi_bbox
      x1 = max(0, min(x1, frame_w - 1))
      y1 = max(0, min(y1, frame_h - 1))
      x2 = max(x1 + 2, min(x2, frame_w))
      y2 = max(y1 + 2, min(y2, frame_h))
      region = frame_bgr[y1:y2, x1:x2]
      origin_x = x1
      origin_y = y1

    if region.size == 0:
      return []

    region_h, region_w = region.shape[:2]
    if region_w < 32 or region_h < 32:
      return []

    region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    result = self._mesh.process(region_rgb)
    if not result.multi_face_landmarks:
      return []

    candidates: List[FaceObservation] = []
    for face_landmarks in result.multi_face_landmarks:
      points = []
      for landmark in face_landmarks.landmark:
        x = int(np.clip(landmark.x * region_w, 0, region_w - 1)) + origin_x
        y = int(np.clip(landmark.y * region_h, 0, region_h - 1)) + origin_y
        points.append((x, y))

      landmarks = np.asarray(points, dtype=np.float32)
      if landmarks.size == 0:
        continue

      min_xy = landmarks.min(axis=0).astype(int)
      max_xy = landmarks.max(axis=0).astype(int)
      x1, y1 = int(min_xy[0]), int(min_xy[1])
      x2, y2 = int(max_xy[0]), int(max_xy[1])
      centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
      face = FaceObservation(
        bbox=(x1, y1, x2, y2),
        centroid=centroid,
        landmarks=landmarks,
      )
      if self._passes_geometry(face, frame_shape=frame_bgr.shape):
        candidates.append(face)
    return candidates

  def _prepare_roi_boxes(
    self,
    roi_boxes: List[Tuple[int, int, int, int]],
    frame_shape: Tuple[int, int, int],
  ) -> List[Tuple[int, int, int, int]]:
    if not roi_boxes:
      return []

    frame_h, frame_w = frame_shape[:2]
    ordered = sorted(roi_boxes, key=self._bbox_area, reverse=True)
    prepared: List[Tuple[int, int, int, int]] = []
    for bbox in ordered[: max(self._max_faces + 2, 3)]:
      expanded = self._expand_roi_bbox(bbox, frame_w=frame_w, frame_h=frame_h)
      if any(self._bbox_iou(expanded, other) >= self._dedupe_iou for other in prepared):
        continue
      prepared.append(expanded)
    return prepared

  @staticmethod
  def _expand_roi_bbox(
    bbox: Tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
  ) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    pad_x = int(round(bw * 0.18))
    pad_y_top = int(round(bh * 0.20))
    pad_y_bottom = int(round(bh * 0.08))
    return (
      max(0, x1 - pad_x),
      max(0, y1 - pad_y_top),
      min(frame_w, x2 + pad_x),
      min(frame_h, y2 + pad_y_bottom),
    )

  def _passes_geometry(self, face: FaceObservation, frame_shape: Tuple[int, int, int]) -> bool:
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = face.bbox
    width = float(max(x2 - x1, 1))
    height = float(max(y2 - y1, 1))
    area = width * height
    area_ratio = area / max(float(frame_w * frame_h), 1.0)
    aspect = width / max(height, 1.0)

    if area_ratio < self._min_area_ratio or area_ratio > self._max_area_ratio:
      return False
    if aspect < self._min_aspect or aspect > self._max_aspect:
      return False
    if width < 28.0 or height < 28.0:
      return False

    landmarks = face.landmarks
    required_indices = [1, 10, 33, 61, 152, 263, 291]
    if landmarks.shape[0] <= max(required_indices):
      return False

    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose = landmarks[1]
    mouth_l = landmarks[61]
    mouth_r = landmarks[291]
    chin = landmarks[152]
    forehead = landmarks[10]

    eye_dist = float(np.linalg.norm(left_eye - right_eye))
    mouth_width = float(np.linalg.norm(mouth_l - mouth_r))
    eye_y = float((left_eye[1] + right_eye[1]) * 0.5)
    mouth_y = float((mouth_l[1] + mouth_r[1]) * 0.5)
    nose_y = float(nose[1])
    chin_y = float(chin[1])
    forehead_y = float(forehead[1])
    eye_mid_x = float((left_eye[0] + right_eye[0]) * 0.5)

    if eye_dist < (0.14 * width) or eye_dist > (0.92 * width):
      return False
    if mouth_width < (0.10 * width) or mouth_width > (0.98 * width):
      return False

    if eye_y > (mouth_y - 0.05 * height):
      return False
    if mouth_y > (chin_y + 0.20 * height):
      return False
    if forehead_y > (eye_y + 0.35 * height):
      return False

    if nose_y < (eye_y - 0.25 * height) or nose_y > (mouth_y + 0.35 * height):
      return False

    # Reject highly asymmetric structures common in moving background artifacts.
    if abs(eye_mid_x - float(nose[0])) > (0.42 * width):
      return False

    hull = cv2.convexHull(landmarks.astype(np.float32))
    hull_area = float(cv2.contourArea(hull))
    if hull_area / max(area, 1.0) < 0.16:
      return False

    return True

  def _temporal_filter(self, faces: List[FaceObservation], frame_shape: Tuple[int, int, int]) -> List[FaceObservation]:
    now = monotonic()
    self._prune_pending(now)

    if not faces:
      return []

    # Bootstrap quickly when detector first locks onto a face.
    if not self._confirmed_faces and not self._pending:
      ordered = sorted(faces, key=lambda item: self._bbox_area(item.bbox), reverse=True)
      for face in ordered:
        if not self._is_in_background_motion_zone(face.bbox, frame_shape):
          return [face]
      return [ordered[0]]

    primary_confirmed_area = max(
      (self._bbox_area(face.bbox) for face in self._confirmed_faces),
      default=0,
    )
    confirm_hits_required = self._new_face_confirm_hits + (1 if self._horizontal_boost_cycles > 0 else 0)

    accepted: List[FaceObservation] = []
    for face in faces:
      if self._matches_confirmed(face):
        accepted.append(face)
        continue

      if primary_confirmed_area > 0:
        candidate_area = self._bbox_area(face.bbox)
        # Reject tiny secondary candidates relative to current confirmed face.
        if candidate_area < int(primary_confirmed_area * self._min_relative_new_face_area):
          continue

        # During rapid horizontal motion, suppress secondary tiny candidates in the primary trail.
        if (
          self._horizontal_boost_cycles > 0
          and candidate_area < int(primary_confirmed_area * 0.56)
          and self._is_in_primary_motion_trail(face)
        ):
          continue

        # Suppress transient background artifacts from calibration-known motion regions.
        if (
          self._is_in_background_motion_zone(face.bbox, frame_shape)
          and candidate_area < int(primary_confirmed_area * 1.1)
        ):
          continue
      elif self._is_in_background_motion_zone(face.bbox, frame_shape):
        continue

      pending_id = self._find_pending_match(face)
      if pending_id is None:
        pending_id = self._next_pending_id
        self._next_pending_id += 1
        self._pending[pending_id] = _PendingFace(
          bbox=face.bbox,
          centroid=face.centroid,
          landmarks=face.landmarks,
          hits=1,
          last_seen=now,
        )
      else:
        pending = self._pending[pending_id]
        pending.bbox = face.bbox
        pending.centroid = face.centroid
        pending.landmarks = face.landmarks
        pending.hits += 1
        pending.last_seen = now

      pending = self._pending.get(pending_id)
      if pending is not None and pending.hits >= confirm_hits_required:
        accepted.append(
          FaceObservation(
            bbox=pending.bbox,
            centroid=pending.centroid,
            landmarks=pending.landmarks,
          )
        )
        del self._pending[pending_id]

    return accepted

  def _matches_confirmed(self, face: FaceObservation) -> bool:
    if not self._confirmed_faces:
      return False

    for confirmed in self._confirmed_faces:
      if self._bbox_iou(face.bbox, confirmed.bbox) >= self._match_iou:
        return True
      if self._centroid_distance(face.centroid, confirmed.centroid) <= self._match_dist_px:
        return True
    return False

  def _find_pending_match(self, face: FaceObservation) -> Optional[int]:
    best_id: Optional[int] = None
    best_dist = float("inf")

    for pending_id, pending in self._pending.items():
      iou = self._bbox_iou(face.bbox, pending.bbox)
      dist = self._centroid_distance(face.centroid, pending.centroid)
      if iou >= self._match_iou or dist <= self._match_dist_px:
        if dist < best_dist:
          best_dist = dist
          best_id = pending_id

    return best_id

  def _advance_without_detection(self) -> None:
    self._confirmed_miss_count += 1
    if self._horizontal_boost_cycles > 0:
      self._horizontal_boost_cycles -= 1
    if self._confirmed_miss_count > self._confirmed_retention_cycles:
      self._confirmed_faces = []
    self._prune_pending(monotonic())

  def _prune_pending(self, now: float) -> None:
    stale_ids = [
      pending_id
      for pending_id, pending in self._pending.items()
      if (now - pending.last_seen) > self._pending_ttl_sec
    ]
    for pending_id in stale_ids:
      del self._pending[pending_id]

  def _dedupe_overlaps(self, faces: List[FaceObservation]) -> List[FaceObservation]:
    if len(faces) <= 1:
      return faces

    ordered = sorted(faces, key=lambda item: self._bbox_area(item.bbox), reverse=True)
    kept: List[FaceObservation] = []
    for face in ordered:
      if any(self._bbox_iou(face.bbox, other.bbox) >= self._dedupe_iou for other in kept):
        continue
      kept.append(face)
    return kept

  def _sort_and_limit(self, faces: List[FaceObservation]) -> List[FaceObservation]:
    ordered = sorted(faces, key=lambda item: self._bbox_area(item.bbox), reverse=True)
    return ordered[: self._max_faces]

  def set_background_motion_zones(self, zones: List[Dict[str, float]]) -> None:
    parsed: List[Tuple[float, float, float, float]] = []
    for item in zones or []:
      if not isinstance(item, dict):
        continue
      try:
        x = float(item.get("x", 0.0))
        y = float(item.get("y", 0.0))
        w = float(item.get("w", 0.0))
        h = float(item.get("h", 0.0))
      except (TypeError, ValueError):
        continue
      if w <= 0.0 or h <= 0.0:
        continue
      x = float(np.clip(x, 0.0, 1.0))
      y = float(np.clip(y, 0.0, 1.0))
      w = float(np.clip(w, 0.0, 1.0 - x))
      h = float(np.clip(h, 0.0, 1.0 - y))
      if w <= 0.0 or h <= 0.0:
        continue
      parsed.append((x, y, w, h))
    self._background_motion_zones = parsed[:12]

  def background_motion_zone_count(self) -> int:
    return len(self._background_motion_zones)

  def _update_primary_motion(self, faces: List[FaceObservation]) -> None:
    if not faces:
      if self._horizontal_boost_cycles > 0:
        self._horizontal_boost_cycles -= 1
      self._last_primary_signature = None
      return

    primary = max(faces, key=lambda item: self._bbox_area(item.bbox))
    px, py = primary.centroid
    pbox_w = float(max(primary.bbox[2] - primary.bbox[0], 1))
    self._primary_motion_history.append((px, py))

    if self._last_primary_signature is not None:
      prev_x, prev_w = self._last_primary_signature
      shift_ratio = abs(px - prev_x) / max(prev_w, pbox_w, 1.0)
      if shift_ratio >= self._horizontal_shift_ratio_trigger:
        self._horizontal_boost_cycles = self._horizontal_boost_window
      elif self._horizontal_boost_cycles > 0:
        self._horizontal_boost_cycles -= 1

    self._last_primary_signature = (px, pbox_w)

  def _is_in_background_motion_zone(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> bool:
    if not self._background_motion_zones:
      return False
    frame_h, frame_w = frame_shape[:2]
    cx = float((bbox[0] + bbox[2]) * 0.5)
    cy = float((bbox[1] + bbox[3]) * 0.5)
    for zx, zy, zw, zh in self._background_motion_zones:
      z_bbox = self._zone_to_bbox((zx, zy, zw, zh), frame_w, frame_h)
      if self._bbox_iou(bbox, z_bbox) >= self._background_zone_iou_reject:
        return True
      if self._point_in_bbox(
        point=(cx, cy),
        bbox=z_bbox,
        margin=self._background_zone_center_margin,
        frame_w=frame_w,
        frame_h=frame_h,
      ):
        return True
    return False

  @staticmethod
  def _zone_to_bbox(zone: Tuple[float, float, float, float], frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    x, y, w, h = zone
    x1 = int(round(x * max(frame_w - 1, 1)))
    y1 = int(round(y * max(frame_h - 1, 1)))
    x2 = int(round((x + w) * max(frame_w - 1, 1)))
    y2 = int(round((y + h) * max(frame_h - 1, 1)))
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(0, min(max(x2, x1 + 1), frame_w - 1))
    y2 = max(0, min(max(y2, y1 + 1), frame_h - 1))
    return (x1, y1, x2, y2)

  @staticmethod
  def _point_in_bbox(
    point: Tuple[float, float],
    bbox: Tuple[int, int, int, int],
    margin: float,
    frame_w: int,
    frame_h: int,
  ) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    margin_x = float(max(frame_w, 1)) * max(margin, 0.0)
    margin_y = float(max(frame_h, 1)) * max(margin, 0.0)
    return (x1 - margin_x) <= x <= (x2 + margin_x) and (y1 - margin_y) <= y <= (y2 + margin_y)

  def _is_in_primary_motion_trail(self, face: FaceObservation) -> bool:
    if not self._primary_motion_history:
      return False

    cx, cy = face.centroid
    x1, y1, x2, y2 = face.bbox
    width = float(max(x2 - x1, 1))
    height = float(max(y2 - y1, 1))
    radius = max(48.0, max(width, height) * 1.4)
    radius_sq = radius * radius

    for hx, hy in self._primary_motion_history:
      dx = float(cx - hx)
      dy = float(cy - hy)
      if (dx * dx + dy * dy) <= radius_sq:
        return True
    return False

  @staticmethod
  def _bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(1, x2 - x1) * max(1, y2 - y1)

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

  @staticmethod
  def _centroid_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    return float(np.hypot(ax - bx, ay - by))

  def close(self) -> None:
    if self._mesh is not None:
      self._mesh.close()

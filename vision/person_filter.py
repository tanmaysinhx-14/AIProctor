from collections import deque
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import settings
from tracking.person_tracker import TrackedPerson
from vision.person_pose_validator import PersonPoseValidation, PersonPoseValidator

BBox = Tuple[int, int, int, int]


@dataclass
class _ValidatorState:
  face_match_frames: int = 0
  pose_match_frames: int = 0
  recent_face_frames: int = 0
  recent_pose_frames: int = 0
  recent_anchor_frames: int = 0
  confirmed_hold_frames: int = 0
  false_zone_seed_frames: int = 0
  centroid_history: Deque[Tuple[float, float, float]] = field(
    default_factory=lambda: deque(maxlen=max(4, int(settings.person_centroid_history_frames)))
  )
  shoulder_history: Deque[Tuple[float, float, float, float]] = field(
    default_factory=lambda: deque(maxlen=max(3, int(settings.person_skeleton_stability_window)))
  )


class PersonValidator:
  def __init__(self) -> None:
    self._states: Dict[int, _ValidatorState] = {}
    self._background_motion_zones: List[Tuple[float, float, float, float]] = []
    self._runtime_false_zones: List[Tuple[float, float, float, float]] = []
    self._prev_gray: Optional[np.ndarray] = None
    self._face_track_frames: Dict[int, int] = {}
    self._pose_validator = PersonPoseValidator()

  def reset(self) -> None:
    self._states.clear()
    self._runtime_false_zones.clear()
    self._prev_gray = None
    self._face_track_frames.clear()

  def close(self) -> None:
    self._pose_validator.close()

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
      x = max(0.0, min(x, 1.0))
      y = max(0.0, min(y, 1.0))
      w = max(0.0, min(w, 1.0 - x))
      h = max(0.0, min(h, 1.0 - y))
      if w <= 0.0 or h <= 0.0:
        continue
      parsed.append((x, y, w, h))
    self._background_motion_zones = parsed[:12]

  def runtime_false_zone_count(self) -> int:
    return len(self._runtime_false_zones)

  def validate(
    self,
    tracked_persons: List[TrackedPerson],
    tracked_faces: List[Any],
    frame: np.ndarray,
    scene_analyzer: Any,
  ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    frame_h, frame_w = frame.shape[:2]
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(curr_gray)) if curr_gray.size else 0.0
    bright_scene = brightness >= float(settings.person_bright_scene_mean_threshold)
    person_confidence_floor = (
      float(settings.person_bright_scene_confirmation_confidence)
      if bright_scene
      else float(settings.person_normal_scene_confirmation_confidence)
    )
    prev_gray = self._prev_gray if self._prev_gray is not None and self._prev_gray.shape == curr_gray.shape else None

    active_ids = {int(person.id) for person in tracked_persons}
    stable_faces = self._update_stable_faces(tracked_faces)
    face_matches = self._match_faces(
      tracked_persons,
      tracked_faces,
      top_ratio_limit=float(settings.person_face_top_ratio_max),
    )
    anchor_matches = self._match_faces(
      tracked_persons,
      stable_faces,
      top_ratio_limit=float(settings.person_face_anchor_top_ratio_max),
    )
    debug = {
      "rawTrackedCount": float(len(tracked_persons)),
      "confirmedCount": 0.0,
      "suppressedCount": 0.0,
      "runtimeSuppressionZones": float(len(self._runtime_false_zones)),
      "calibrationSuppressionZones": float(len(self._background_motion_zones)),
      "stableFaceCount": float(len(stable_faces)),
      "flowRejectedCount": 0.0,
      "faceRejectedCount": 0.0,
      "anchorRejectedCount": 0.0,
      "oscillationRejectedCount": 0.0,
      "periodicRejectedCount": 0.0,
      "sizeRejectedCount": 0.0,
      "backgroundRejectedCount": 0.0,
      "edgeRejectedCount": 0.0,
      "poseRejectedCount": 0.0,
      "skeletonStaticRejectedCount": 0.0,
      "driftRejectedCount": 0.0,
      "skeleton_valid": 0.0,
      "brightness": round(brightness, 3),
      "brightScene": 1.0 if bright_scene else 0.0,
      "personConfidenceFloor": round(person_confidence_floor, 3),
    }

    confirmed_payload: List[Dict[str, Any]] = []
    for person in tracked_persons:
      person_id = int(person.id)
      state = self._states.setdefault(person_id, _ValidatorState())
      match = face_matches.get(person_id)
      anchor_match = anchor_matches.get(person_id)
      if match is not None:
        state.face_match_frames = min(
          state.face_match_frames + 1,
          int(settings.person_recent_face_ttl_frames) + int(settings.person_face_confirmation_frames) + 6,
        )
        state.recent_face_frames = max(1, int(settings.person_recent_face_ttl_frames))
      else:
        state.face_match_frames = max(0, state.face_match_frames - 1)
        state.recent_face_frames = max(0, state.recent_face_frames - 1)
      if anchor_match is not None:
        state.recent_anchor_frames = max(1, int(settings.person_face_anchor_ttl_frames))
      else:
        state.recent_anchor_frames = max(0, state.recent_anchor_frames - 1)

      bbox = person.bbox
      self._append_centroid(state, bbox)
      pose_validation = self._pose_validator.validate(frame, bbox)
      if pose_validation.pose_keypoints_detected:
        self._append_skeleton(state, pose_validation)
      if pose_validation.valid:
        state.pose_match_frames = min(
          state.pose_match_frames + 1,
          int(settings.person_recent_face_ttl_frames) + 6,
        )
        state.recent_pose_frames = max(1, int(settings.person_recent_face_ttl_frames))
      else:
        state.pose_match_frames = max(0, state.pose_match_frames - 1)
        state.recent_pose_frames = max(0, state.recent_pose_frames - 1)
      motion_ratio = float(scene_analyzer.region_motion_ratio(bbox, expand_ratio=0.04))
      flow_metrics = self._compute_flow_metrics(prev_gray, curr_gray, bbox)
      centroid_metrics = self._compute_centroid_metrics(state)
      skeleton_metrics = self._compute_skeleton_metrics(state)
      active_background_overlap = float(scene_analyzer.active_background_overlap_ratio(bbox))
      edge_density = self._edge_density(curr_gray, bbox)
      calibration_zone_overlap = self._zone_overlap_ratio(
        bbox,
        frame_w=frame_w,
        frame_h=frame_h,
        zones=self._background_motion_zones,
      )
      runtime_zone_overlap = self._zone_overlap_ratio(
        bbox,
        frame_w=frame_w,
        frame_h=frame_h,
        zones=self._runtime_false_zones,
      )
      frame_height_ratio = self._frame_height_ratio(bbox, frame_h)
      confidence_ok = self._passes_confidence(float(person.confidence), brightness)
      bbox_ratio_ok = self._passes_bbox_ratio(bbox)
      size_ok = self._passes_min_size(frame_height_ratio)
      persistence_ok = int(person.visible_frames) >= int(settings.person_confirmation_frames)
      drift_ok = float(centroid_metrics.get("range_px", 0.0)) >= float(settings.person_centroid_drift_min_px)
      face_persistent = state.face_match_frames >= int(settings.person_face_confirmation_frames)
      face_recent = state.recent_face_frames > 0
      pose_persistent = state.pose_match_frames > 0
      pose_recent = state.recent_pose_frames > 0
      motion_ok = motion_ratio >= float(settings.person_min_motion_ratio)
      flow_ok = self._passes_flow_semantics(flow_metrics, motion_ok)
      oscillating_motion = self._passes_centroid_oscillation_gate(centroid_metrics, motion_ratio)
      periodic_motion = self._passes_periodic_motion_gate(centroid_metrics, motion_ratio)
      cloth_like_motion = motion_ok and not flow_ok
      skeleton_static = self._fails_skeleton_stability(skeleton_metrics, motion_ratio, pose_validation.valid)
      skeleton_valid = bool(pose_validation.valid) and not skeleton_static
      texture_ok = edge_density >= float(settings.person_min_edge_density) or face_persistent
      anchor_required = int(person.visible_frames) > int(settings.person_face_anchor_grace_frames)
      anchor_valid = (anchor_match is not None) or (not anchor_required) or (state.recent_anchor_frames > 0)
      face_skeleton_alignment_ok = self._passes_face_skeleton_alignment(match, pose_validation, bbox)
      in_suppression_zone = runtime_zone_overlap >= float(settings.person_false_zone_overlap_threshold)
      in_calibration_zone = calibration_zone_overlap >= float(settings.person_false_zone_overlap_threshold)
      background_dominant = (
        active_background_overlap >= float(settings.person_background_overlap_threshold)
        and not face_persistent
      )

      if not size_ok:
        debug["sizeRejectedCount"] += 1.0
      if cloth_like_motion:
        debug["flowRejectedCount"] += 1.0
      if oscillating_motion:
        debug["oscillationRejectedCount"] += 1.0
      if periodic_motion:
        debug["periodicRejectedCount"] += 1.0
      if not texture_ok:
        debug["edgeRejectedCount"] += 1.0
      if background_dominant:
        debug["backgroundRejectedCount"] += 1.0
      if not pose_persistent:
        debug["poseRejectedCount"] += 1.0
      if skeleton_static:
        debug["skeletonStaticRejectedCount"] += 1.0
      if not face_persistent:
        debug["faceRejectedCount"] += 1.0
      if not drift_ok:
        debug["driftRejectedCount"] += 1.0
      if skeleton_valid:
        debug["skeleton_valid"] += 1.0
      if anchor_required and not anchor_valid:
        debug["anchorRejectedCount"] += 1.0

      confirmed = (
        int(person.disappeared) <= 1
        and confidence_ok
        and bbox_ratio_ok
        and size_ok
        and persistence_ok
        and drift_ok
        and pose_persistent
        and skeleton_valid
        and face_persistent
        and anchor_valid
        and face_skeleton_alignment_ok
        and texture_ok
        and not background_dominant
        and not cloth_like_motion
        and not oscillating_motion
        and not periodic_motion
        and not in_suppression_zone
      )
      if confirmed:
        state.confirmed_hold_frames = max(1, int(settings.person_confirmation_hold_frames))
      else:
        hold_eligible = (
          state.confirmed_hold_frames > 0
          and int(person.disappeared) <= 1
          and confidence_ok
          and bbox_ratio_ok
          and size_ok
          and pose_recent
          and skeleton_valid
          and face_recent
          and anchor_valid
          and face_skeleton_alignment_ok
          and texture_ok
          and not background_dominant
          and not cloth_like_motion
          and not oscillating_motion
          and not periodic_motion
          and not in_suppression_zone
        )
        if hold_eligible:
          state.confirmed_hold_frames = max(0, state.confirmed_hold_frames - 1)
          confirmed = True
        else:
          state.confirmed_hold_frames = 0

      should_seed_false_zone = (
        int(person.disappeared) == 0
        and confidence_ok
        and bbox_ratio_ok
        and size_ok
        and int(person.visible_frames) >= int(settings.person_confirmation_frames)
        and (state.face_match_frames == 0 or (anchor_required and not anchor_valid))
        and (
          cloth_like_motion
          or oscillating_motion
          or periodic_motion
          or background_dominant
          or not texture_ok
          or not pose_recent
          or not face_skeleton_alignment_ok
          or skeleton_static
          or motion_ratio < max(float(settings.person_min_motion_ratio) * 0.75, 0.012)
          or in_calibration_zone
        )
        and self._bbox_area(bbox) / max(float(frame_w * frame_h), 1.0)
        >= float(settings.person_false_zone_min_area_ratio)
      )
      if should_seed_false_zone:
        state.false_zone_seed_frames += 1
      else:
        state.false_zone_seed_frames = max(0, state.false_zone_seed_frames - 1)

      if (
        state.false_zone_seed_frames >= int(settings.person_false_zone_learning_frames)
        and runtime_zone_overlap < 0.12
      ):
        self._register_runtime_false_zone(bbox, frame_w=frame_w, frame_h=frame_h)
        debug["runtimeSuppressionZones"] = float(len(self._runtime_false_zones))
        state.false_zone_seed_frames = 0

      if not confirmed:
        continue

      confirmed_payload.append(
        {
          "id": person_id,
          "bbox": self._bbox_payload(person.bbox),
          "confidence": round(float(person.confidence), 3),
          "disappeared": int(person.disappeared),
          "visibleFrames": int(person.visible_frames),
          "poseKeypointsDetected": bool(pose_validation.pose_keypoints_detected),
          "skeletonConfidence": round(float(pose_validation.skeleton_confidence), 4),
          "skeletonValid": bool(skeleton_valid),
          "skeletonStability": round(float(skeleton_metrics["shoulder_std_px"]), 4),
          "shoulderWidthRatio": round(float(pose_validation.shoulder_ratio), 4),
          "faceFrames": int(state.face_match_frames),
          "motionRatio": round(motion_ratio, 4),
          "motionFrequency": round(float(centroid_metrics["motion_frequency_hz"]), 4),
          "flowVariance": round(float(flow_metrics["direction_std"]), 4),
          "frameHeightRatio": round(frame_height_ratio, 4),
          "anchorValidated": bool(anchor_valid),
          "anchorRequired": bool(anchor_required),
          "centroidRangePx": round(float(centroid_metrics["range_px"]), 3),
          "directionFlips": int(centroid_metrics["flip_count"]),
          "faceValidated": bool(face_persistent),
          "activeBackgroundOverlap": round(active_background_overlap, 4),
          "edgeDensity": round(edge_density, 5),
          "brightness": round(brightness, 3),
          "confidenceFloor": round(person_confidence_floor, 3),
          "centroidDriftSatisfied": bool(drift_ok),
          "faceTopRatio": (
            round(float(match["top_ratio"]), 4)
            if isinstance(match, dict) and "top_ratio" in match
            else None
          ),
          "calibrationZoneOverlap": round(calibration_zone_overlap, 4),
          "suppressionZoneOverlap": round(runtime_zone_overlap, 4),
          "confirmed": True,
        }
      )

    for person_id in list(self._states.keys()):
      if person_id not in active_ids:
        del self._states[person_id]

    debug["confirmedCount"] = float(len(confirmed_payload))
    debug["suppressedCount"] = float(max(len(tracked_persons) - len(confirmed_payload), 0))
    self._prev_gray = curr_gray
    return confirmed_payload, debug

  def filter(
    self,
    tracked_persons: List[TrackedPerson],
    tracked_faces: List[Any],
    frame: np.ndarray,
    scene_analyzer: Any,
  ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    return self.validate(
      tracked_persons=tracked_persons,
      tracked_faces=tracked_faces,
      frame=frame,
      scene_analyzer=scene_analyzer,
    )

  def _update_stable_faces(self, tracked_faces: List[Any]) -> List[Any]:
    seen_ids = set()
    stable_faces: List[Any] = []
    max_face_frames = int(settings.person_face_confirmation_frames) + int(settings.person_recent_face_ttl_frames) + 8
    for face in tracked_faces:
      face_id = int(getattr(face, "id", 0) or 0)
      if face_id <= 0 or not self._passes_face_landmark_sanity(face):
        continue
      seen_ids.add(face_id)
      self._face_track_frames[face_id] = min(self._face_track_frames.get(face_id, 0) + 1, max_face_frames)
      if self._face_track_frames[face_id] >= int(settings.person_face_confirmation_frames):
        stable_faces.append(face)

    for face_id in list(self._face_track_frames.keys()):
      if face_id in seen_ids:
        continue
      self._face_track_frames[face_id] = max(0, self._face_track_frames[face_id] - 1)
      if self._face_track_frames[face_id] <= 0:
        del self._face_track_frames[face_id]

    return stable_faces

  def _match_faces(
    self,
    tracked_persons: List[TrackedPerson],
    tracked_faces: List[Any],
    top_ratio_limit: float,
  ) -> Dict[int, Dict[str, float]]:
    matches: Dict[int, Dict[str, float]] = {}
    for person in tracked_persons:
      px1, py1, px2, py2 = person.bbox
      width = float(max(px2 - px1, 1))
      height = float(max(py2 - py1, 1))
      best_match: Optional[Dict[str, float]] = None
      for face in tracked_faces:
        bbox = getattr(face, "bbox", None)
        if bbox is None or not self._passes_face_landmark_sanity(face):
          continue
        fx1, fy1, fx2, fy2 = bbox
        fcx = (fx1 + fx2) * 0.5
        fcy = (fy1 + fy2) * 0.5
        if not (px1 <= fcx <= px2 and py1 <= fcy <= py2):
          continue

        top_ratio = (fcy - py1) / height
        lateral_offset = abs(fcx - ((px1 + px2) * 0.5)) / width
        if top_ratio > float(top_ratio_limit):
          continue
        if lateral_offset > float(settings.person_face_center_offset_ratio_max):
          continue

        face_area_ratio = self._bbox_area(bbox) / max(float(self._bbox_area(person.bbox)), 1.0)
        if face_area_ratio < 0.015 or face_area_ratio > 0.45:
          continue

        score = float(face_area_ratio) - (top_ratio * 0.35) - (lateral_offset * 0.20)
        if best_match is None or score > float(best_match["score"]):
          best_match = {
            "score": score,
            "top_ratio": top_ratio,
            "lateral_offset": lateral_offset,
            "face_area_ratio": face_area_ratio,
            "center_x": fcx,
            "center_y": fcy,
          }

      if best_match is not None:
        matches[int(person.id)] = best_match
    return matches

  def _append_centroid(self, state: _ValidatorState, bbox: BBox) -> None:
    cx, cy = self._center(bbox)
    state.centroid_history.append((monotonic(), cx, cy))

  def _append_skeleton(self, state: _ValidatorState, validation: PersonPoseValidation) -> None:
    state.shoulder_history.append(
      (
        float(validation.left_shoulder[0]),
        float(validation.left_shoulder[1]),
        float(validation.right_shoulder[0]),
        float(validation.right_shoulder[1]),
      )
    )

  def _compute_centroid_metrics(self, state: _ValidatorState) -> Dict[str, float]:
    history = list(state.centroid_history)
    if len(history) < 2:
      return {
        "range_px": 0.0,
        "flip_count": 0.0,
        "history_len": float(len(history)),
        "motion_frequency_hz": 0.0,
        "periodic_ratio": 0.0,
      }

    xs = [point[1] for point in history]
    ys = [point[2] for point in history]
    range_px = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))

    vectors: List[np.ndarray] = []
    magnitudes: List[float] = []
    timestamps: List[float] = []
    min_step = float(settings.person_centroid_direction_min_step_px)
    for idx in range(1, len(history)):
      dx = float(history[idx][1] - history[idx - 1][1])
      dy = float(history[idx][2] - history[idx - 1][2])
      mag = float(np.hypot(dx, dy))
      magnitudes.append(mag)
      timestamps.append(float(history[idx][0]))
      if mag < min_step:
        continue
      vectors.append(np.asarray([dx, dy], dtype=np.float32))

    flip_count = 0
    for idx in range(1, len(vectors)):
      prev_vec = vectors[idx - 1]
      curr_vec = vectors[idx]
      prev_mag = float(np.linalg.norm(prev_vec))
      curr_mag = float(np.linalg.norm(curr_vec))
      if prev_mag <= 1e-6 or curr_mag <= 1e-6:
        continue
      cosine = float(np.dot(prev_vec, curr_vec) / max(prev_mag * curr_mag, 1e-6))
      if cosine <= -0.35:
        flip_count += 1

    motion_frequency_hz, periodic_ratio = self._motion_frequency(magnitudes, timestamps)
    return {
      "range_px": range_px,
      "flip_count": float(flip_count),
      "history_len": float(len(history)),
      "motion_frequency_hz": motion_frequency_hz,
      "periodic_ratio": periodic_ratio,
    }

  def _compute_skeleton_metrics(self, state: _ValidatorState) -> Dict[str, float]:
    history = list(state.shoulder_history)
    min_len = max(3, int(settings.person_skeleton_stability_window))
    if len(history) < min_len:
      return {
        "history_len": float(len(history)),
        "shoulder_std_px": 0.0,
      }
    coords = np.asarray(history, dtype=np.float32)
    shoulder_std = float(np.mean(np.std(coords, axis=0)))
    return {
      "history_len": float(len(history)),
      "shoulder_std_px": shoulder_std,
    }

  def _passes_centroid_oscillation_gate(self, centroid_metrics: Dict[str, float], motion_ratio: float) -> bool:
    if float(centroid_metrics.get("history_len", 0.0)) < 5.0:
      return False
    if motion_ratio < float(settings.person_centroid_oscillation_motion_threshold):
      return False
    if float(centroid_metrics.get("range_px", 0.0)) >= float(settings.person_centroid_range_min_px):
      return False
    return float(centroid_metrics.get("flip_count", 0.0)) >= float(settings.person_centroid_direction_flip_threshold)

  def _passes_periodic_motion_gate(self, centroid_metrics: Dict[str, float], motion_ratio: float) -> bool:
    if motion_ratio < float(settings.person_centroid_oscillation_motion_threshold):
      return False
    return (
      float(centroid_metrics.get("motion_frequency_hz", 0.0)) > float(settings.person_periodic_motion_hz_threshold)
      and float(centroid_metrics.get("periodic_ratio", 0.0)) >= float(settings.person_periodic_motion_power_ratio_threshold)
    )

  def _fails_skeleton_stability(
    self,
    skeleton_metrics: Dict[str, float],
    motion_ratio: float,
    pose_valid: bool,
  ) -> bool:
    if not pose_valid:
      return False
    if float(skeleton_metrics.get("history_len", 0.0)) < max(3.0, float(settings.person_skeleton_stability_window)):
      return False
    if motion_ratio < float(settings.person_skeleton_stability_motion_threshold):
      return False
    return float(skeleton_metrics.get("shoulder_std_px", 0.0)) <= float(settings.person_skeleton_stability_min_std_px)

  def _passes_face_skeleton_alignment(
    self,
    match: Optional[Dict[str, float]],
    validation: PersonPoseValidation,
    bbox: BBox,
  ) -> bool:
    if match is None or not validation.pose_keypoints_detected:
      return False
    person_top = float(bbox[1])
    person_height = max(float(bbox[3] - bbox[1]), 1.0)
    face_center_y = float(match.get("center_y", person_top))
    face_relative_y = (face_center_y - person_top) / person_height
    shoulder_floor = min(float(validation.left_shoulder[1]), float(validation.right_shoulder[1]))
    return face_relative_y <= 0.35 and shoulder_floor > face_center_y

  def _passes_face_landmark_sanity(self, face: Any) -> bool:
    landmarks = getattr(face, "landmarks", None)
    if not isinstance(landmarks, np.ndarray) or landmarks.ndim != 2 or landmarks.shape[0] <= 291:
      return False

    try:
      left_eye = landmarks[33]
      right_eye = landmarks[263]
      nose = landmarks[1]
      mouth_l = landmarks[61]
      mouth_r = landmarks[291]
    except Exception:
      return False

    eye_distance = float(np.linalg.norm(left_eye - right_eye))
    if eye_distance < 6.0:
      return False

    eye_mid = (left_eye + right_eye) * 0.5
    nose_eye_ratio = float(np.linalg.norm(nose - eye_mid)) / eye_distance
    mouth_width = float(np.linalg.norm(mouth_r - mouth_l)) / eye_distance
    eye_level_delta = abs(float(left_eye[1] - right_eye[1])) / eye_distance
    nose_center_offset = abs(float(nose[0] - eye_mid[0])) / eye_distance
    return (
      float(settings.person_face_nose_eye_ratio_min)
      <= nose_eye_ratio
      <= float(settings.person_face_nose_eye_ratio_max)
      and 0.45 <= mouth_width <= 1.65
      and eye_level_delta <= 0.32
      and nose_center_offset <= 0.34
    )

  def _passes_confidence(self, confidence: float, brightness: float) -> bool:
    threshold = (
      float(settings.person_bright_scene_confirmation_confidence)
      if brightness >= float(settings.person_bright_scene_mean_threshold)
      else float(settings.person_normal_scene_confirmation_confidence)
    )
    return confidence >= threshold

  def _passes_bbox_ratio(self, bbox: BBox) -> bool:
    x1, y1, x2, y2 = bbox
    width = float(max(x2 - x1, 1))
    height = float(max(y2 - y1, 1))
    ratio = height / width
    return (
      width >= 24.0
      and height >= 40.0
      and float(settings.person_min_height_width_ratio) <= ratio <= float(settings.person_max_height_width_ratio)
    )

  def _frame_height_ratio(self, bbox: BBox, frame_h: int) -> float:
    return max(float(bbox[3] - bbox[1]), 1.0) / max(float(frame_h), 1.0)

  def _passes_min_size(self, frame_height_ratio: float) -> bool:
    return frame_height_ratio >= float(settings.person_min_frame_height_ratio)

  def _passes_flow_semantics(self, flow_metrics: Dict[str, float], motion_ok: bool) -> bool:
    if not motion_ok:
      return True
    if float(flow_metrics.get("active_ratio", 0.0)) < 0.015:
      return True
    if not bool(flow_metrics.get("valid", False)):
      return True
    return float(flow_metrics.get("direction_std", 0.0)) <= float(settings.person_motion_variance_max)

  def _compute_flow_metrics(
    self,
    prev_gray: Optional[np.ndarray],
    curr_gray: np.ndarray,
    bbox: BBox,
  ) -> Dict[str, float]:
    if prev_gray is None or prev_gray.shape != curr_gray.shape:
      return {"valid": False, "direction_std": 0.0, "active_ratio": 0.0}

    frame_h, frame_w = curr_gray.shape[:2]
    x1, y1, x2, y2 = self._expand_bbox(bbox, frame_w=frame_w, frame_h=frame_h, ratio=0.03)
    if x2 <= x1 or y2 <= y1:
      return {"valid": False, "direction_std": 0.0, "active_ratio": 0.0}

    prev_roi = prev_gray[y1:y2, x1:x2]
    curr_roi = curr_gray[y1:y2, x1:x2]
    if prev_roi.size == 0 or curr_roi.size == 0:
      return {"valid": False, "direction_std": 0.0, "active_ratio": 0.0}

    roi_h, roi_w = curr_roi.shape[:2]
    if roi_w < 24 or roi_h < 24:
      return {"valid": False, "direction_std": 0.0, "active_ratio": 0.0}

    max_dim = max(32, int(settings.person_optical_flow_max_dim))
    scale = min(1.0, max_dim / max(float(roi_w), float(roi_h)))
    if scale < 1.0:
      resized_w = max(24, int(round(roi_w * scale)))
      resized_h = max(24, int(round(roi_h * scale)))
      prev_roi = cv2.resize(prev_roi, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
      curr_roi = cv2.resize(curr_roi, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    flow = cv2.calcOpticalFlowFarneback(
      prev_roi,
      curr_roi,
      None,
      0.5,
      2,
      11,
      2,
      5,
      1.1,
      0,
    )
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    active_mask = magnitude >= float(settings.person_optical_flow_min_magnitude)
    active_ratio = float(np.count_nonzero(active_mask)) / max(float(magnitude.size), 1.0)
    if np.count_nonzero(active_mask) < 18:
      return {"valid": False, "direction_std": 0.0, "active_ratio": active_ratio}

    weights = magnitude[active_mask].astype(np.float64)
    angles = angle[active_mask].astype(np.float64)
    resultant = np.abs(np.sum(weights * np.exp(1j * angles)) / max(np.sum(weights), 1e-6))
    resultant = min(max(float(resultant), 1e-6), 0.999999)
    direction_std = float(np.sqrt(max(-2.0 * np.log(resultant), 0.0)))
    return {
      "valid": True,
      "direction_std": direction_std,
      "active_ratio": active_ratio,
    }

  def _motion_frequency(self, magnitudes: List[float], timestamps: List[float]) -> Tuple[float, float]:
    min_samples = max(4, int(settings.person_motion_fft_min_samples))
    if len(magnitudes) < min_samples or len(timestamps) != len(magnitudes):
      return 0.0, 0.0

    dt = np.diff(np.asarray(timestamps, dtype=np.float64))
    if dt.size == 0:
      return 0.0, 0.0
    median_dt = float(np.median(dt))
    if median_dt <= 1e-4:
      return 0.0, 0.0

    signal = np.asarray(magnitudes, dtype=np.float32)
    signal = signal - float(np.mean(signal))
    if float(np.max(np.abs(signal))) <= 1e-4:
      return 0.0, 0.0

    spectrum = np.fft.rfft(signal)
    power = np.abs(spectrum) ** 2
    if power.size <= 1:
      return 0.0, 0.0

    freqs = np.fft.rfftfreq(signal.size, d=median_dt)
    power[0] = 0.0
    peak_idx = int(np.argmax(power))
    peak_power = float(power[peak_idx])
    total_power = float(np.sum(power))
    if peak_idx <= 0 or peak_power <= 1e-6 or total_power <= 1e-6:
      return 0.0, 0.0
    return float(freqs[peak_idx]), float(peak_power / total_power)

  def _edge_density(self, gray_frame: np.ndarray, bbox: BBox) -> float:
    frame_h, frame_w = gray_frame.shape[:2]
    x1, y1, x2, y2 = self._expand_bbox(bbox, frame_w=frame_w, frame_h=frame_h, ratio=0.02)
    if x2 <= x1 or y2 <= y1:
      return 0.0
    roi = gray_frame[y1:y2, x1:x2]
    if roi.size == 0:
      return 0.0
    edges = cv2.Canny(roi, 50, 150)
    return float(np.count_nonzero(edges)) / max(float(edges.size), 1.0)

  def _register_runtime_false_zone(self, bbox: BBox, frame_w: int, frame_h: int) -> None:
    x1, y1, x2, y2 = bbox
    zone = (
      round(x1 / max(float(frame_w), 1.0), 5),
      round(y1 / max(float(frame_h), 1.0), 5),
      round(max(x2 - x1, 1) / max(float(frame_w), 1.0), 5),
      round(max(y2 - y1, 1) / max(float(frame_h), 1.0), 5),
    )
    zone_bbox = self._zone_to_bbox(zone, frame_w=frame_w, frame_h=frame_h)
    if any(self._bbox_iou(zone_bbox, self._zone_to_bbox(other, frame_w, frame_h)) >= 0.55 for other in self._runtime_false_zones):
      return
    self._runtime_false_zones.append(zone)
    if len(self._runtime_false_zones) > int(settings.person_false_zone_max_count):
      self._runtime_false_zones = self._runtime_false_zones[-int(settings.person_false_zone_max_count):]

  def _zone_overlap_ratio(
    self,
    bbox: BBox,
    frame_w: int,
    frame_h: int,
    zones: List[Tuple[float, float, float, float]],
  ) -> float:
    if not zones:
      return 0.0
    bbox_area = float(max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1))
    overlap_area = 0.0
    for zone in zones:
      overlap_area += self._intersection_area(bbox, self._zone_to_bbox(zone, frame_w, frame_h))
    return overlap_area / bbox_area

  @staticmethod
  def _center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

  @staticmethod
  def _expand_bbox(bbox: BBox, frame_w: int, frame_h: int, ratio: float) -> BBox:
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    pad_x = int(round(bw * max(0.0, ratio)))
    pad_y = int(round(bh * max(0.0, ratio)))
    return (
      max(0, x1 - pad_x),
      max(0, y1 - pad_y),
      min(frame_w, x2 + pad_x),
      min(frame_h, y2 + pad_y),
    )

  @staticmethod
  def _bbox_payload(bbox: BBox) -> Dict[str, int]:
    return {
      "x": int(bbox[0]),
      "y": int(bbox[1]),
      "w": int(max(bbox[2] - bbox[0], 0)),
      "h": int(max(bbox[3] - bbox[1], 0)),
    }

  @staticmethod
  def _bbox_area(bbox: BBox) -> int:
    x1, y1, x2, y2 = bbox
    return max(1, x2 - x1) * max(1, y2 - y1)

  @staticmethod
  def _bbox_iou(a: BBox, b: BBox) -> float:
    inter = PersonValidator._intersection_area(a, b)
    if inter <= 0.0:
      return 0.0
    area_a = float(PersonValidator._bbox_area(a))
    area_b = float(PersonValidator._bbox_area(b))
    return inter / max(area_a + area_b - inter, 1.0)

  @staticmethod
  def _intersection_area(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    return float(inter_w * inter_h)

  @staticmethod
  def _zone_to_bbox(zone: Tuple[float, float, float, float], frame_w: int, frame_h: int) -> BBox:
    x, y, w, h = zone
    x1 = int(round(x * max(float(frame_w), 1.0)))
    y1 = int(round(y * max(float(frame_h), 1.0)))
    x2 = int(round((x + w) * max(float(frame_w), 1.0)))
    y2 = int(round((y + h) * max(float(frame_h), 1.0)))
    x1 = max(0, min(x1, frame_w))
    y1 = max(0, min(y1, frame_h))
    x2 = max(x1 + 1, min(x2, frame_w))
    y2 = max(y1 + 1, min(y2, frame_h))
    return (x1, y1, x2, y2)


ConfirmedPersonFilter = PersonValidator

from dataclasses import dataclass
from time import monotonic
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import settings
from vision.object_detector import DetectedObject


@dataclass(frozen=True)
class CalibrationStage:
  code: str
  label: str
  instruction: str
  min_duration_sec: float
  max_duration_sec: float


class CalibrationManager:
  def __init__(self) -> None:
    self._gaze_target_codes = (
      "gaze_center",
      "gaze_left",
      "gaze_right",
      "gaze_top",
      "gaze_bottom",
    )
    self._stages = [
      CalibrationStage(
        code="neutral",
        label="Neutral Baseline",
        instruction="Look at the screen and stay still. Keep hands away from face.",
        min_duration_sec=settings.calibration_stage_neutral_min_sec,
        max_duration_sec=settings.calibration_stage_neutral_max_sec,
      ),
      CalibrationStage(
        code="gaze_center",
        label="Gaze Center",
        instruction="Look at the center of the screen and keep your head still.",
        min_duration_sec=settings.calibration_stage_gaze_target_min_sec,
        max_duration_sec=settings.calibration_stage_gaze_target_max_sec,
      ),
      CalibrationStage(
        code="gaze_left",
        label="Gaze Left Edge",
        instruction="Look at the left edge of the screen without turning your head.",
        min_duration_sec=settings.calibration_stage_gaze_target_min_sec,
        max_duration_sec=settings.calibration_stage_gaze_target_max_sec,
      ),
      CalibrationStage(
        code="gaze_right",
        label="Gaze Right Edge",
        instruction="Look at the right edge of the screen without turning your head.",
        min_duration_sec=settings.calibration_stage_gaze_target_min_sec,
        max_duration_sec=settings.calibration_stage_gaze_target_max_sec,
      ),
      CalibrationStage(
        code="gaze_top",
        label="Gaze Top Edge",
        instruction="Look at the top edge of the screen without lifting your head.",
        min_duration_sec=settings.calibration_stage_gaze_target_min_sec,
        max_duration_sec=settings.calibration_stage_gaze_target_max_sec,
      ),
      CalibrationStage(
        code="gaze_bottom",
        label="Gaze Bottom Edge",
        instruction="Look at the bottom edge of the screen without lowering your head too much.",
        min_duration_sec=settings.calibration_stage_gaze_target_min_sec,
        max_duration_sec=settings.calibration_stage_gaze_target_max_sec,
      ),
      CalibrationStage(
        code="head_vertical",
        label="Head Vertical",
        instruction="Move your head UP and DOWN naturally for a few cycles while facing the camera.",
        min_duration_sec=settings.calibration_stage_head_min_sec,
        max_duration_sec=settings.calibration_stage_head_max_sec,
      ),
      CalibrationStage(
        code="hand_face",
        label="Hand + Face",
        instruction="Bring your hand near your face briefly, then move it away.",
        min_duration_sec=settings.calibration_stage_hand_min_sec,
        max_duration_sec=settings.calibration_stage_hand_max_sec,
      ),
    ]

    self._active = False
    self._completed = False
    self._started_ts = 0.0
    self._completed_ts = 0.0
    self._stage_started_ts = 0.0
    self._stage_index = 0
    self._profile: Optional[Dict[str, Any]] = None
    self._neutral_pitch_ref: Optional[float] = None
    self._prev_pitch: Optional[float] = None
    self._neutral_yaw_ref: Optional[float] = None
    self._prev_yaw: Optional[float] = None
    self._bg_prev_gray: Optional[np.ndarray] = None
    self._bg_motion_accum: Optional[np.ndarray] = None
    self._bg_motion_frames = 0

    self._samples: Dict[str, List[float]] = {}
    self._gaze_stage_samples: Dict[str, Dict[str, List[float]]] = {}
    self._gaze_counts: Dict[str, int] = {}
    self._stage_outcomes: Dict[str, Dict[str, Any]] = {}
    self._stage_face_frames: Dict[str, int] = {}
    self._stage_missing_frames: Dict[str, int] = {}
    self._stage_feedback = ""
    self._reset_samples()

  def _reset_samples(self) -> None:
    self._samples = {
      "neutral_pitch": [],
      "neutral_yaw": [],
      "neutral_pitch_ratio": [],
      "neutral_frontalness": [],
      "neutral_movement": [],
      "neutral_pitch_velocity": [],
      "neutral_yaw_velocity": [],
      "neutral_gaze_h_offset": [],
      "neutral_gaze_v_offset": [],
      "neutral_gaze_x": [],
      "neutral_gaze_y": [],
      "neutral_eye_visible_flag": [],
      "neutral_eye_visibility": [],
      "neutral_face_area_ratio": [],
      "neutral_display_diag_px": [],
      "neutral_phone_conf": [],
      "neutral_phone_area_ratio": [],
      "gaze_lr_signal": [],
      "gaze_horizontal_offset": [],
      "gaze_vertical_offset": [],
      "gaze_confidence": [],
      "gaze_eye_visible_flag": [],
      "head_vertical_pitch_ratio": [],
      "head_vertical_delta": [],
      "head_vertical_signed_delta": [],
      "head_vertical_movement": [],
      "hand_face_flag": [],
    }
    self._gaze_stage_samples = {
      code: {"x": [], "y": [], "visible": [], "confidence": []}
      for code in self._gaze_target_codes
    }
    self._gaze_counts = {}
    self._stage_outcomes = {}
    self._stage_feedback = ""
    self._stage_face_frames = {stage.code: 0 for stage in self._stages}
    self._stage_missing_frames = {stage.code: 0 for stage in self._stages}

  @property
  def active(self) -> bool:
    return self._active

  @property
  def completed(self) -> bool:
    return self._completed

  def start(self) -> None:
    self._active = True
    self._completed = False
    self._started_ts = monotonic()
    self._completed_ts = 0.0
    self._stage_started_ts = self._started_ts
    self._stage_index = 0
    self._profile = None
    self._neutral_pitch_ref = None
    self._prev_pitch = None
    self._neutral_yaw_ref = None
    self._prev_yaw = None
    self._bg_prev_gray = None
    self._bg_motion_accum = None
    self._bg_motion_frames = 0
    self._reset_samples()

  def cancel(self) -> None:
    self._active = False

  def status_payload(self) -> Dict[str, Any]:
    stage = self._current_stage()
    now = monotonic()

    stage_elapsed = 0.0
    stage_min = 0.0
    stage_max = 0.0
    stage_ready = False
    stage_feedback = self._stage_feedback
    completed_ago = 0.0

    if self._active and stage is not None:
      stage_elapsed = max(0.0, now - self._stage_started_ts)
      stage_min = float(stage.min_duration_sec)
      stage_max = float(stage.max_duration_sec)
      stage_ready, stage_feedback = self._stage_ready(stage.code)
      stage_progress = min(stage_elapsed / max(stage_max, 1e-6), 1.0)
      overall_progress = self._overall_progress(now)
      remaining = self._remaining_sec(now)
      mode = "RUNNING"
      stage_code = stage.code
      stage_label = stage.label
      instruction = self._format_instruction(stage, stage_ready, stage_feedback)
    elif self._completed:
      mode = "COMPLETED"
      stage_progress = 1.0
      overall_progress = 1.0
      remaining = 0.0
      stage_code = "completed"
      stage_label = "Calibration Complete"
      if self._completed_ts > 0:
        completed_ago = max(0.0, now - self._completed_ts)
      if self._profile is not None and not bool(self._profile.get("ready", False)):
        instruction = "Calibration finished with partial confidence; conservative defaults are still applied."
      else:
        instruction = "Calibration profile is active."
      stage_ready = True
      stage_feedback = "Calibration complete."
    else:
      mode = "IDLE"
      stage_progress = 0.0
      overall_progress = 0.0
      remaining = 0.0
      stage_code = "idle"
      stage_label = "Calibration Idle"
      instruction = "Press Run Calibration to build personalized thresholds."

    profile_thresholds = self._profile.get("thresholds", {}) if self._profile else {}
    return {
      "mode": mode,
      "active": self._active,
      "completed": self._completed,
      "stageCode": stage_code,
      "stageLabel": stage_label,
      "instruction": instruction,
      "stageProgress": round(float(stage_progress), 3),
      "overallProgress": round(float(overall_progress), 3),
      "remainingSec": round(float(remaining), 2),
      "stageElapsedSec": round(float(stage_elapsed), 2),
      "stageMinSec": round(float(stage_min), 2),
      "stageMaxSec": round(float(stage_max), 2),
      "stageReady": bool(stage_ready),
      "stageFeedback": stage_feedback,
      "completedAgoSec": round(float(completed_ago), 2),
      "completedHoldSec": round(float(settings.calibration_completed_hold_sec), 2),
      "stageOutcomes": dict(self._stage_outcomes),
      "profile": self._profile,
      "profileThresholds": profile_thresholds,
    }

  def update(
    self,
    tracked_faces: List[dict],
    objects: List[DetectedObject],
    frame_shape: Tuple[int, int, int],
    frame: Optional[np.ndarray] = None,
    client_context: Optional[Dict[str, Any]] = None,
  ) -> Tuple[Dict[str, Any], Optional[Dict[str, float]]]:
    if not self._active:
      return self.status_payload(), None

    now = monotonic()
    stage = self._current_stage()
    if stage is None:
      return self.status_payload(), None

    self._collect_stage_samples(
      stage_code=stage.code,
      tracked_faces=tracked_faces,
      objects=objects,
      frame_shape=frame_shape,
      frame=frame,
      client_context=client_context,
    )
    stage_elapsed = max(0.0, now - self._stage_started_ts)
    total_elapsed = max(0.0, now - self._started_ts)

    stage_ready, stage_feedback = self._stage_ready(stage.code)
    self._stage_feedback = stage_feedback

    should_advance = False
    reason = "ready"

    if stage_elapsed >= stage.min_duration_sec and stage_ready:
      should_advance = True
      reason = "ready"
    elif stage_elapsed >= stage.max_duration_sec:
      should_advance = True
      reason = "stage_timeout"
    elif total_elapsed >= settings.calibration_total_max_sec:
      should_advance = True
      reason = "global_timeout"

    if should_advance:
      self._record_stage_outcome(stage.code, stage_ready, stage_elapsed, reason)
      self._stage_index += 1
      self._stage_started_ts = now
      self._stage_feedback = ""

      if self._stage_index >= len(self._stages):
        self._active = False
        self._completed = True
        self._completed_ts = now
        self._profile = self._build_profile()
        return self.status_payload(), dict(self._profile.get("thresholds", {}))

    return self.status_payload(), None

  def _current_stage(self) -> Optional[CalibrationStage]:
    if not self._active:
      return None
    if self._stage_index < 0 or self._stage_index >= len(self._stages):
      return None
    return self._stages[self._stage_index]

  def _remaining_sec(self, now: float) -> float:
    if not self._active:
      return 0.0

    remaining = 0.0
    for idx, stage in enumerate(self._stages):
      if idx < self._stage_index:
        continue
      if idx == self._stage_index:
        elapsed = max(0.0, now - self._stage_started_ts)
        remaining += max(stage.max_duration_sec - elapsed, 0.0)
      else:
        remaining += stage.max_duration_sec

    if settings.calibration_total_max_sec > 0 and self._started_ts > 0:
      total_elapsed = max(0.0, now - self._started_ts)
      global_remaining = max(settings.calibration_total_max_sec - total_elapsed, 0.0)
      remaining = min(remaining, global_remaining)

    return remaining

  def _overall_progress(self, now: float) -> float:
    stage = self._current_stage()
    if stage is None:
      return 1.0 if self._completed else 0.0

    total_budget = float(sum(s.max_duration_sec for s in self._stages))
    if settings.calibration_total_max_sec > 0:
      total_budget = min(total_budget, settings.calibration_total_max_sec)
    total_budget = max(total_budget, 1e-6)

    elapsed = 0.0
    for idx, stg in enumerate(self._stages):
      if idx < self._stage_index:
        elapsed += stg.max_duration_sec
      elif idx == self._stage_index:
        elapsed += min(max(0.0, now - self._stage_started_ts), stg.max_duration_sec)
        break

    if settings.calibration_total_max_sec > 0 and self._started_ts > 0:
      elapsed = min(elapsed, settings.calibration_total_max_sec)

    return self._clamp(elapsed / total_budget, 0.0, 1.0)

  def _format_instruction(self, stage: CalibrationStage, ready: bool, feedback: str) -> str:
    if ready:
      return f"{stage.instruction} Good capture detected for this stage."
    if feedback:
      return f"{stage.instruction} {feedback}"
    return stage.instruction

  def _record_stage_outcome(self, stage_code: str, ready: bool, elapsed_sec: float, reason: str) -> None:
    self._stage_outcomes[stage_code] = {
      "ready": bool(ready),
      "elapsedSec": round(float(elapsed_sec), 2),
      "reason": reason,
      "faceFrames": int(self._stage_face_frames.get(stage_code, 0)),
      "missingFrames": int(self._stage_missing_frames.get(stage_code, 0)),
    }

  def _stage_ready(self, stage_code: str) -> Tuple[bool, str]:
    if stage_code == "neutral":
      face_frames = int(self._stage_face_frames.get(stage_code, 0))
      target = max(int(settings.calibration_min_face_frames), int(settings.calibration_neutral_target_frames))
      if face_frames < target:
        return False, f"Keep your face centered and steady ({face_frames}/{target} baseline frames)."

      jitter_p90 = self._pct(self._samples["neutral_movement"], 90, 0.0)
      if jitter_p90 > 0.11:
        return False, "Reduce motion slightly so baseline noise stays low."

      eye_visible_ratio = float(np.mean(self._samples["neutral_eye_visible_flag"])) if self._samples["neutral_eye_visible_flag"] else 0.0
      if eye_visible_ratio < 0.55:
        return False, "Keep eyes visible and avoid heavy squinting during baseline."

      return True, "Neutral baseline quality is sufficient."

    if stage_code in self._gaze_target_codes:
      stage_samples = self._gaze_stage_samples.get(stage_code, {})
      gaze_x = stage_samples.get("x", [])
      gaze_y = stage_samples.get("y", [])
      visible = stage_samples.get("visible", [])
      target_frames = int(settings.calibration_gaze_target_frames)
      if len(gaze_x) < target_frames:
        return False, f"Hold the target a little longer ({len(gaze_x)}/{target_frames} frames)."

      visible_ratio = float(np.mean(visible)) if visible else 0.0
      if visible_ratio < float(settings.calibration_gaze_target_visible_ratio):
        return False, "Keep eyes open and visible while holding the target."

      center_x = self._median(self._gaze_stage_samples.get("gaze_center", {}).get("x", []))
      center_y = self._median(self._gaze_stage_samples.get("gaze_center", {}).get("y", []))
      median_x = self._median(gaze_x)
      median_y = self._median(gaze_y)
      sign_threshold = float(settings.calibration_gaze_signal_sign_threshold)

      if stage_code == "gaze_center":
        spread = max(
          self._pct([abs(value - median_x) for value in gaze_x], 90, 0.0),
          self._pct([abs(value - median_y) for value in gaze_y], 90, 0.0),
        )
        if spread > 0.08:
          return False, "Hold the center target more steadily."
        return True, "Center gaze lock looks stable."

      if stage_code == "gaze_left":
        if median_x > (center_x - sign_threshold):
          return False, "Look farther toward the left edge."
        return True, "Left gaze boundary captured."

      if stage_code == "gaze_right":
        if median_x < (center_x + sign_threshold):
          return False, "Look farther toward the right edge."
        return True, "Right gaze boundary captured."

      if stage_code == "gaze_top":
        if median_y > (center_y - sign_threshold):
          return False, "Look farther toward the top edge."
        return True, "Top gaze boundary captured."

      if stage_code == "gaze_bottom":
        if median_y < (center_y + sign_threshold):
          return False, "Look farther toward the bottom edge."
        return True, "Bottom gaze boundary captured."

    if stage_code == "head_vertical":
      pitch_ratios = self._samples["head_vertical_pitch_ratio"]
      min_samples = int(settings.calibration_head_min_samples)
      if len(pitch_ratios) < min_samples:
        return False, f"Continue natural up-down movement ({len(pitch_ratios)}/{min_samples} samples)."

      neutral_ratio = self._median(self._samples["neutral_pitch_ratio"])
      min_dir = int(settings.calibration_head_min_direction_frames)
      ratio_margin = max(float(settings.calibration_pitch_ratio_min_range) * 0.35, 0.008)
      up_frames = sum(1 for value in pitch_ratios if value <= (neutral_ratio - ratio_margin))
      down_frames = sum(1 for value in pitch_ratios if value >= (neutral_ratio + ratio_margin))
      if up_frames < min_dir or down_frames < min_dir:
        return False, f"Need clearer up/down ratio change (up:{up_frames}, down:{down_frames})."

      ratio_range = self._pct(pitch_ratios, 95, neutral_ratio) - self._pct(pitch_ratios, 5, neutral_ratio)
      if ratio_range < float(settings.calibration_pitch_ratio_min_range):
        return False, f"Increase vertical head range a bit ({ratio_range:.3f} ratio captured)."

      return True, "Vertical chin-nose ratio profile is sufficient."

    if stage_code == "hand_face":
      hand_samples = self._samples["hand_face_flag"]
      min_samples = int(settings.calibration_hand_min_samples)
      if len(hand_samples) < min_samples:
        return False, f"Hold this step briefly ({len(hand_samples)}/{min_samples} frames)."

      positive = int(sum(hand_samples))
      min_positive = int(settings.calibration_hand_min_positive_frames)
      if positive < min_positive:
        return False, f"Bring hand near face briefly ({positive}/{min_positive} detections)."

      return True, "Hand-to-face behavior captured."

    return True, ""

  def _collect_stage_samples(
    self,
    stage_code: str,
    tracked_faces: List[dict],
    objects: List[DetectedObject],
    frame_shape: Tuple[int, int, int],
    frame: Optional[np.ndarray],
    client_context: Optional[Dict[str, Any]] = None,
  ) -> None:
    if frame is not None and settings.calibration_background_motion_enabled:
      self._collect_background_motion(frame=frame, tracked_faces=tracked_faces)

    face = self._primary_face(tracked_faces)
    if face is None:
      self._stage_missing_frames[stage_code] = self._stage_missing_frames.get(stage_code, 0) + 1
      self._prev_pitch = None
      self._prev_yaw = None
      return

    self._stage_face_frames[stage_code] = self._stage_face_frames.get(stage_code, 0) + 1

    head_pose = face.get("headPose", {}) or {}
    pitch = float(head_pose.get("pitch", 0.0))
    yaw = float(head_pose.get("yaw", 0.0))
    movement = float(face.get("movement", 0.0))
    hand_on_face = bool(face.get("handOnFace", False))
    gaze = str(face.get("gaze", "CENTER")).upper()
    gaze_offset_x = float(face.get("gazeOffsetX", 0.0))
    gaze_offset_y = float(face.get("gazeOffsetY", 0.0))
    gaze_x = float(face.get("gazeX", gaze_offset_x))
    gaze_y = float(face.get("gazeY", gaze_offset_y))
    gaze_confidence = float(face.get("gazeConfidence", 0.0))
    eye_visible = bool(face.get("eyeVisible", False))
    eye_visibility = float(face.get("eyeVisibility", 0.0))
    pitch_ratio = float(face.get("pitchRatio", 0.5))
    frontalness = float(face.get("frontalness", face.get("faceFrontalness", 1.0)))

    pitch_velocity = 0.0
    if self._prev_pitch is not None:
      pitch_velocity = abs(pitch - self._prev_pitch)
    self._prev_pitch = pitch

    yaw_velocity = 0.0
    if self._prev_yaw is not None:
      yaw_velocity = abs(yaw - self._prev_yaw)
    self._prev_yaw = yaw

    if stage_code == "neutral":
      self._samples["neutral_pitch"].append(pitch)
      self._samples["neutral_yaw"].append(yaw)
      self._samples["neutral_movement"].append(movement)
      self._samples["neutral_pitch_velocity"].append(pitch_velocity)
      self._samples["neutral_yaw_velocity"].append(yaw_velocity)
      self._samples["neutral_gaze_h_offset"].append(gaze_offset_x)
      self._samples["neutral_gaze_v_offset"].append(gaze_offset_y)
      self._samples["neutral_gaze_x"].append(gaze_x)
      self._samples["neutral_gaze_y"].append(gaze_y)
      self._samples["neutral_eye_visible_flag"].append(1.0 if eye_visible else 0.0)
      self._samples["neutral_eye_visibility"].append(eye_visibility)
      self._samples["neutral_pitch_ratio"].append(pitch_ratio)
      self._samples["neutral_frontalness"].append(frontalness)
      face_area_ratio = self._face_bbox_area_ratio(face=face, frame_shape=frame_shape)
      if face_area_ratio > 0.0:
        self._samples["neutral_face_area_ratio"].append(face_area_ratio)
      display_diag = self._display_diag_from_context(client_context)
      if display_diag > 0.0:
        self._samples["neutral_display_diag_px"].append(display_diag)
      if self._neutral_pitch_ref is None:
        self._neutral_pitch_ref = pitch
      else:
        self._neutral_pitch_ref = 0.92 * self._neutral_pitch_ref + 0.08 * pitch
      if self._neutral_yaw_ref is None:
        self._neutral_yaw_ref = yaw
      else:
        self._neutral_yaw_ref = 0.92 * self._neutral_yaw_ref + 0.08 * yaw

      for obj in objects:
        if obj.label != "phone" or obj.bbox is None:
          continue
        self._samples["neutral_phone_conf"].append(float(obj.confidence))
        ratio = self._bbox_area_ratio(obj.bbox, frame_shape)
        self._samples["neutral_phone_area_ratio"].append(ratio)

    elif stage_code in self._gaze_target_codes:
      self._gaze_counts[gaze] = self._gaze_counts.get(gaze, 0) + 1
      self._samples["gaze_horizontal_offset"].append(gaze_offset_x)
      self._samples["gaze_vertical_offset"].append(gaze_offset_y)
      self._samples["gaze_confidence"].append(gaze_confidence)
      self._samples["gaze_eye_visible_flag"].append(1.0 if eye_visible else 0.0)
      stage_samples = self._gaze_stage_samples.setdefault(stage_code, {"x": [], "y": [], "visible": [], "confidence": []})
      stage_samples["x"].append(gaze_x)
      stage_samples["y"].append(gaze_y)
      stage_samples["visible"].append(1.0 if eye_visible else 0.0)
      stage_samples["confidence"].append(gaze_confidence)
      if stage_code == "gaze_left":
        self._samples["gaze_lr_signal"].append(-1.0)
      elif stage_code == "gaze_right":
        self._samples["gaze_lr_signal"].append(1.0)

    elif stage_code == "head_vertical":
      neutral_ref = self._neutral_pitch_ref if self._neutral_pitch_ref is not None else pitch
      signed_delta = pitch - neutral_ref
      self._samples["head_vertical_signed_delta"].append(signed_delta)
      self._samples["head_vertical_delta"].append(abs(signed_delta))
      self._samples["head_vertical_movement"].append(movement)
      self._samples["head_vertical_pitch_ratio"].append(pitch_ratio)

    elif stage_code == "hand_face":
      self._samples["hand_face_flag"].append(1.0 if hand_on_face else 0.0)

  def _collect_background_motion(self, frame: np.ndarray, tracked_faces: List[dict]) -> None:
    if frame.ndim != 3 or frame.shape[2] < 3:
      return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    grid_w = max(16, int(settings.calibration_background_motion_grid_w))
    grid_h = max(9, int(settings.calibration_background_motion_grid_h))

    if self._bg_prev_gray is None:
      self._bg_prev_gray = gray
      self._bg_motion_accum = np.zeros((grid_h, grid_w), dtype=np.float32)
      self._bg_motion_frames = 0
      return

    diff = cv2.absdiff(gray, self._bg_prev_gray)
    self._bg_prev_gray = gray

    threshold = float(settings.calibration_background_motion_diff_threshold)
    _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.medianBlur(motion_mask, 3)

    # Ignore face regions so only persistent background movers are profiled.
    face_exclude_ratio = float(settings.calibration_background_motion_face_exclude_ratio)
    for face in tracked_faces:
      bbox = self._face_payload_bbox(face)
      if bbox is None:
        continue
      x1, y1, x2, y2 = bbox
      bw = max(x2 - x1, 1)
      bh = max(y2 - y1, 1)
      pad_x = int(round(bw * face_exclude_ratio))
      pad_y = int(round(bh * face_exclude_ratio))
      x1 = max(0, x1 - pad_x)
      y1 = max(0, y1 - pad_y)
      x2 = min(frame.shape[1] - 1, x2 + pad_x)
      y2 = min(frame.shape[0] - 1, y2 + pad_y)
      motion_mask[y1:y2, x1:x2] = 0

    global_motion_ratio = float(np.count_nonzero(motion_mask)) / max(float(motion_mask.size), 1.0)
    if global_motion_ratio > float(settings.calibration_background_motion_max_global_ratio):
      return

    if self._bg_motion_accum is None:
      self._bg_motion_accum = np.zeros((grid_h, grid_w), dtype=np.float32)

    small = cv2.resize(motion_mask, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    self._bg_motion_accum += (small > 0).astype(np.float32)
    self._bg_motion_frames += 1

  def _background_motion_zones(self) -> List[Dict[str, float]]:
    if self._bg_motion_accum is None or self._bg_motion_frames <= 0:
      return []

    ratio_map = self._bg_motion_accum / max(float(self._bg_motion_frames), 1.0)
    threshold = float(settings.calibration_background_motion_cell_activity_threshold)
    binary = (ratio_map >= threshold).astype(np.uint8)
    if int(np.count_nonzero(binary)) == 0:
      return []

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
      return []

    grid_h, grid_w = binary.shape
    zones: List[Dict[str, float]] = []
    min_cells = max(1, int(settings.calibration_background_motion_min_zone_cells))
    for label_idx in range(1, num_labels):
      area = int(stats[label_idx, cv2.CC_STAT_AREA])
      if area < min_cells:
        continue
      x = int(stats[label_idx, cv2.CC_STAT_LEFT])
      y = int(stats[label_idx, cv2.CC_STAT_TOP])
      w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
      h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
      component_mask = labels == label_idx
      activity = float(np.mean(ratio_map[component_mask])) if np.any(component_mask) else 0.0
      zones.append(
        {
          "x": round(float(x / max(grid_w, 1)), 5),
          "y": round(float(y / max(grid_h, 1)), 5),
          "w": round(float(w / max(grid_w, 1)), 5),
          "h": round(float(h / max(grid_h, 1)), 5),
          "activity": round(activity, 5),
        }
      )

    zones.sort(key=lambda item: (float(item.get("activity", 0.0)), float(item.get("w", 0.0)) * float(item.get("h", 0.0))), reverse=True)
    max_zones = max(0, int(settings.calibration_background_motion_max_zones))
    if max_zones > 0:
      zones = zones[:max_zones]
    return zones

  def _face_payload_bbox(self, face: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    bbox = face.get("bbox") or {}
    if isinstance(bbox, dict):
      x = int(float(bbox.get("x", 0.0)))
      y = int(float(bbox.get("y", 0.0)))
      w = int(max(float(bbox.get("w", 0.0)), 0.0))
      h = int(max(float(bbox.get("h", 0.0)), 0.0))
      if w <= 0 or h <= 0:
        return None
      return (x, y, x + w, y + h)
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
      x1 = int(float(bbox[0]))
      y1 = int(float(bbox[1]))
      x2 = int(float(bbox[2]))
      y2 = int(float(bbox[3]))
      if x2 <= x1 or y2 <= y1:
        return None
      return (x1, y1, x2, y2)
    return None

  def _face_bbox_area_ratio(self, face: Dict[str, Any], frame_shape: Tuple[int, int, int]) -> float:
    bbox = self._face_payload_bbox(face)
    if bbox is None:
      return 0.0
    frame_h, frame_w = frame_shape[:2]
    frame_area = float(max(frame_w * frame_h, 1))
    x1, y1, x2, y2 = bbox
    bw = float(max(x2 - x1, 1))
    bh = float(max(y2 - y1, 1))
    return (bw * bh) / frame_area

  def _display_diag_from_context(self, context: Optional[Dict[str, Any]]) -> float:
    if not context:
      return 0.0
    try:
      dpr = float(context.get("devicePixelRatio", 1.0) or 1.0)
      screen_w = float(context.get("screenWidth", 0.0) or 0.0) * max(dpr, 1.0)
      screen_h = float(context.get("screenHeight", 0.0) or 0.0) * max(dpr, 1.0)
      if screen_w <= 0.0 or screen_h <= 0.0:
        viewport_w = float(context.get("viewportWidth", 0.0) or 0.0) * max(dpr, 1.0)
        viewport_h = float(context.get("viewportHeight", 0.0) or 0.0) * max(dpr, 1.0)
        screen_w, screen_h = viewport_w, viewport_h
      if screen_w <= 0.0 or screen_h <= 0.0:
        return 0.0
      return float((screen_w ** 2 + screen_h ** 2) ** 0.5)
    except (TypeError, ValueError):
      return 0.0

  def _primary_face(self, tracked_faces: List[dict]) -> Optional[dict]:
    if not tracked_faces:
      return None
    best = None
    best_area = -1.0
    for face in tracked_faces:
      bbox = face.get("bbox") or {}
      w = float(max(int(bbox.get("w", 0)), 0))
      h = float(max(int(bbox.get("h", 0)), 0))
      area = w * h
      if area > best_area:
        best_area = area
        best = face
    return best

  def _bbox_area_ratio(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> float:
    h, w = frame_shape[:2]
    frame_area = float(max(w * h, 1))
    x1, y1, x2, y2 = bbox
    bw = float(max(x2 - x1, 1))
    bh = float(max(y2 - y1, 1))
    return (bw * bh) / frame_area

  def _count_direction_changes(self, signal: List[float]) -> int:
    if not signal:
      return 0
    changes = 0
    prev = int(np.sign(signal[0]))
    for value in signal[1:]:
      current = int(np.sign(value))
      if current == 0:
        continue
      if prev != 0 and current != prev:
        changes += 1
      prev = current
    return changes

  def _build_profile(self) -> Dict[str, Any]:
    neutral_movement = self._samples["neutral_movement"]
    neutral_pitch_velocity = self._samples["neutral_pitch_velocity"]
    neutral_yaw_velocity = self._samples["neutral_yaw_velocity"]
    neutral_pitch = self._samples["neutral_pitch"]
    neutral_yaw = self._samples["neutral_yaw"]
    neutral_pitch_ratio = self._samples["neutral_pitch_ratio"]
    neutral_frontalness = self._samples["neutral_frontalness"]
    neutral_gaze_h = self._samples["neutral_gaze_h_offset"]
    neutral_gaze_v = self._samples["neutral_gaze_v_offset"]
    neutral_gaze_x = self._samples["neutral_gaze_x"]
    neutral_gaze_y = self._samples["neutral_gaze_y"]
    neutral_eye_visible = self._samples["neutral_eye_visible_flag"]
    neutral_eye_visibility = self._samples["neutral_eye_visibility"]
    neutral_face_area_ratio = self._samples["neutral_face_area_ratio"]
    neutral_display_diag_px = self._samples["neutral_display_diag_px"]
    gaze_h = self._samples["gaze_horizontal_offset"]
    gaze_v = self._samples["gaze_vertical_offset"]
    gaze_signal = self._samples["gaze_lr_signal"]
    gaze_conf = self._samples["gaze_confidence"]
    gaze_eye_visible = self._samples["gaze_eye_visible_flag"]
    gaze_center = self._gaze_stage_samples.get("gaze_center", {})
    gaze_left = self._gaze_stage_samples.get("gaze_left", {})
    gaze_right = self._gaze_stage_samples.get("gaze_right", {})
    gaze_top = self._gaze_stage_samples.get("gaze_top", {})
    gaze_bottom = self._gaze_stage_samples.get("gaze_bottom", {})
    gaze_center_x = gaze_center.get("x", [])
    gaze_center_y = gaze_center.get("y", [])
    gaze_left_x = gaze_left.get("x", [])
    gaze_right_x = gaze_right.get("x", [])
    gaze_top_y = gaze_top.get("y", [])
    gaze_bottom_y = gaze_bottom.get("y", [])
    gaze_target_visible = [
      *(gaze_center.get("visible", [])),
      *(gaze_left.get("visible", [])),
      *(gaze_right.get("visible", [])),
      *(gaze_top.get("visible", [])),
      *(gaze_bottom.get("visible", [])),
    ]
    vertical_delta = self._samples["head_vertical_delta"]
    vertical_signed_delta = self._samples["head_vertical_signed_delta"]
    vertical_movement = self._samples["head_vertical_movement"]
    vertical_pitch_ratio = self._samples["head_vertical_pitch_ratio"]
    phone_conf = self._samples["neutral_phone_conf"]
    phone_area = self._samples["neutral_phone_area_ratio"]
    hand_samples = self._samples["hand_face_flag"]

    neutral_move_p95 = self._pct(neutral_movement, 95, settings.vertical_head_baseline_movement_threshold)
    neutral_pitch_vel_p95 = self._pct(neutral_pitch_velocity, 95, settings.vertical_head_pitch_velocity_gate * 0.6)
    neutral_yaw_vel_p95 = self._pct(neutral_yaw_velocity, 95, settings.horizontal_head_yaw_velocity_gate * 0.6)
    neutral_pitch_mean = float(np.mean(neutral_pitch)) if neutral_pitch else 0.0
    neutral_yaw_mean = float(np.mean(neutral_yaw)) if neutral_yaw else 0.0
    neutral_pitch_median = self._median(neutral_pitch)
    neutral_yaw_median = self._median(neutral_yaw)
    neutral_pitch_std = self._std(neutral_pitch)
    neutral_yaw_std = self._std(neutral_yaw)
    neutral_pitch_noise = self._mad_noise(neutral_pitch, neutral_pitch_median)
    neutral_yaw_noise = self._mad_noise(neutral_yaw, neutral_yaw_median)
    neutral_pitch_noise_p95 = self._pct(
      [abs(value - neutral_pitch_median) for value in neutral_pitch],
      95,
      settings.face_off_camera_pitch_gate * 0.45,
    )
    neutral_yaw_noise_p95 = self._pct(
      [abs(value - neutral_yaw_median) for value in neutral_yaw],
      95,
      settings.face_off_camera_yaw_gate * 0.42,
    )
    neutral_yaw_abs_p95 = self._pct(
      [abs(value - neutral_yaw_median) for value in neutral_yaw],
      95,
      settings.horizontal_head_yaw_gate * 0.42,
    )
    vertical_delta_p75 = self._pct(vertical_delta, 75, settings.vertical_head_pitch_enter_threshold)
    vertical_move_p60 = self._pct(vertical_movement, 60, settings.vertical_head_activation_movement_threshold)

    neutral_gaze_h_mean = float(np.mean(neutral_gaze_h)) if neutral_gaze_h else 0.0
    neutral_gaze_v_mean = float(np.mean(neutral_gaze_v)) if neutral_gaze_v else 0.0
    neutral_gaze_h_median = self._median(neutral_gaze_h)
    neutral_gaze_v_median = self._median(neutral_gaze_v)
    neutral_gaze_h_noise = self._mad_noise(neutral_gaze_h, neutral_gaze_h_median)
    neutral_gaze_v_noise = self._mad_noise(neutral_gaze_v, neutral_gaze_v_median)
    neutral_gaze_noise = max(neutral_gaze_h_noise, neutral_gaze_v_noise)
    neutral_h_noise_p90 = self._pct([abs(value) for value in neutral_gaze_h], 90, 0.016)
    neutral_v_noise_p90 = self._pct([abs(value) for value in neutral_gaze_v], 90, 0.018)
    neutral_gaze_h_noise_p95 = self._pct(
      [abs(value - neutral_gaze_h_median) for value in neutral_gaze_h],
      95,
      0.018,
    )
    neutral_gaze_v_noise_p95 = self._pct(
      [abs(value - neutral_gaze_v_median) for value in neutral_gaze_v],
      95,
      0.02,
    )
    max_natural_deviation = {
      "yaw": round(float(neutral_yaw_noise_p95), 4),
      "pitch": round(float(neutral_pitch_noise_p95), 4),
      "gaze": round(float(max(neutral_gaze_h_noise_p95, neutral_gaze_v_noise_p95)), 5),
    }
    gaze_h_amp_p90 = self._pct([abs(value) for value in gaze_h], 90, settings.calibration_gaze_min_signal_amplitude)
    gaze_v_amp_p85 = self._pct([abs(value) for value in gaze_v], 85, 0.08)
    gaze_conf_avg = float(np.mean(gaze_conf)) if gaze_conf else 0.0
    gaze_face_area_ref_ratio = self._clamp(
      self._pct(
        neutral_face_area_ratio,
        50,
        settings.gaze_distance_reference_face_area_ratio,
      ),
      0.02,
      0.42,
    )
    gaze_display_ref_diag_px = self._clamp(
      self._pct(
        neutral_display_diag_px,
        50,
        settings.gaze_display_reference_diag_px,
      ),
      800.0,
      7200.0,
    )
    neutral_pitch_ratio_median = self._median(neutral_pitch_ratio) if neutral_pitch_ratio else 0.5
    neutral_frontalness_mean = float(np.mean(neutral_frontalness)) if neutral_frontalness else 1.0
    gaze_center_x_median = self._median(gaze_center_x) if gaze_center_x else self._median(neutral_gaze_x)
    gaze_center_y_median = self._median(gaze_center_y) if gaze_center_y else self._median(neutral_gaze_y)
    gaze_left_x_median = self._median(gaze_left_x) if gaze_left_x else (gaze_center_x_median - settings.gaze_extreme_horizontal_threshold)
    gaze_right_x_median = self._median(gaze_right_x) if gaze_right_x else (gaze_center_x_median + settings.gaze_extreme_horizontal_threshold)
    gaze_top_y_median = self._median(gaze_top_y) if gaze_top_y else (gaze_center_y_median - settings.gaze_extreme_vertical_threshold)
    gaze_bottom_y_median = self._median(gaze_bottom_y) if gaze_bottom_y else (gaze_center_y_median + settings.gaze_extreme_vertical_threshold)
    gaze_boundary_scale = float(settings.calibration_gaze_boundary_scale)
    gaze_left_boundary = gaze_center_x_median + ((gaze_left_x_median - gaze_center_x_median) * gaze_boundary_scale)
    gaze_right_boundary = gaze_center_x_median + ((gaze_right_x_median - gaze_center_x_median) * gaze_boundary_scale)
    gaze_top_boundary = gaze_center_y_median + ((gaze_top_y_median - gaze_center_y_median) * gaze_boundary_scale)
    gaze_bottom_boundary = gaze_center_y_median + ((gaze_bottom_y_median - gaze_center_y_median) * gaze_boundary_scale)
    pitch_ratio_p10 = self._pct(vertical_pitch_ratio, 10, neutral_pitch_ratio_median - settings.calibration_pitch_ratio_min_range)
    pitch_ratio_p90 = self._pct(vertical_pitch_ratio, 90, neutral_pitch_ratio_median + settings.calibration_pitch_ratio_min_range)
    pitch_ratio_scale = float(settings.calibration_pitch_ratio_boundary_scale)
    pitch_ratio_up_threshold = neutral_pitch_ratio_median + ((pitch_ratio_p10 - neutral_pitch_ratio_median) * pitch_ratio_scale)
    pitch_ratio_down_threshold = neutral_pitch_ratio_median + ((pitch_ratio_p90 - neutral_pitch_ratio_median) * pitch_ratio_scale)

    # Vertical head motion thresholds.
    enter_th = max(
      settings.vertical_head_pitch_enter_threshold,
      vertical_delta_p75 * 0.78,
      neutral_pitch_std * 5.0 + 6.5,
    )
    enter_th = self._clamp(enter_th, settings.vertical_head_pitch_enter_threshold, 32.0)
    exit_th = self._clamp(
      max(settings.vertical_head_pitch_exit_threshold, enter_th * 0.58),
      settings.vertical_head_pitch_exit_threshold,
      24.0,
    )
    vertical_activation_frames = max(int(round(settings.vertical_head_activation_frames)), 5)
    vertical_activation_movement_threshold = self._clamp(
      max(settings.vertical_head_activation_movement_threshold, neutral_move_p95 * 1.8),
      settings.vertical_head_activation_movement_threshold,
      0.06,
    )
    vertical_velocity_gate = self._clamp(
      max(settings.vertical_head_pitch_velocity_gate, neutral_pitch_vel_p95 * 2.1),
      settings.vertical_head_pitch_velocity_gate,
      4.0,
    )
    vertical_span_gate = self._clamp(
      max(settings.vertical_head_pitch_span_gate, neutral_pitch_std * 2.4 + 2.8),
      settings.vertical_head_pitch_span_gate,
      14.0,
    )
    vertical_delta_velocity_gate = self._clamp(
      max(settings.vertical_head_delta_velocity_gate, neutral_pitch_vel_p95 * 0.92),
      settings.vertical_head_delta_velocity_gate,
      2.4,
    )

    # Horizontal head motion thresholds.
    horizontal_activation_frames = max(int(round(settings.horizontal_head_activation_frames)), 4)
    horizontal_yaw_gate = self._clamp(
      max(
        18.0,
        neutral_yaw_noise * 2.5,
        neutral_yaw_abs_p95 * 1.25,
      ),
      18.0,
      34.0,
    )
    horizontal_yaw_velocity_gate = self._clamp(
      max(settings.horizontal_head_yaw_velocity_gate, neutral_yaw_vel_p95 * 2.2),
      settings.horizontal_head_yaw_velocity_gate,
      4.2,
    )
    horizontal_x_velocity_gate = self._clamp(
      max(settings.horizontal_head_x_velocity_gate, neutral_move_p95 * 2.4),
      settings.horizontal_head_x_velocity_gate,
      0.12,
    )
    horizontal_yaw_span_gate = self._clamp(
      max(settings.horizontal_head_yaw_span_gate, neutral_yaw_noise * 3.2 + 3.2),
      settings.horizontal_head_yaw_span_gate,
      20.0,
    )
    horizontal_x_span_gate = self._clamp(
      max(settings.horizontal_head_x_span_gate, neutral_move_p95 * 8.0),
      settings.horizontal_head_x_span_gate,
      0.24,
    )
    horizontal_exit_velocity_gate = self._clamp(
      max(settings.horizontal_head_exit_velocity_gate, horizontal_yaw_velocity_gate * 0.45),
      settings.horizontal_head_exit_velocity_gate,
      2.2,
    )
    vertical_y_velocity_gate = self._clamp(
      max(settings.vertical_head_y_velocity_gate, neutral_move_p95 * 6.4),
      settings.vertical_head_y_velocity_gate,
      0.14,
    )
    vertical_y_span_gate = self._clamp(
      max(settings.vertical_head_y_span_gate, neutral_move_p95 * 12.0),
      settings.vertical_head_y_span_gate,
      0.28,
    )
    vertical_large_movement_gate = self._clamp(
      max(settings.vertical_head_large_movement_gate, neutral_move_p95 * 1.9),
      settings.vertical_head_large_movement_gate,
      0.05,
    )
    horizontal_large_movement_gate = self._clamp(
      max(settings.horizontal_head_large_movement_gate, neutral_move_p95 * 2.2),
      settings.horizontal_head_large_movement_gate,
      0.06,
    )
    horizontal_large_x_span_gate = self._clamp(
      max(settings.horizontal_head_large_x_span_gate, neutral_move_p95 * 10.0),
      settings.horizontal_head_large_x_span_gate,
      0.34,
    )
    face_off_camera_yaw_gate = self._clamp(
      max(
        18.0,
        neutral_yaw_noise * 2.5,
        neutral_yaw_abs_p95 * 1.45,
      ),
      18.0,
      40.0,
    )
    face_off_camera_pitch_gate = self._clamp(
      max(
        16.0,
        neutral_pitch_noise * 2.5,
        neutral_pitch_noise_p95 * 1.35,
      ),
      16.0,
      30.0,
    )
    face_off_camera_yaw_exit_gate = self._clamp(
      max(settings.face_off_camera_yaw_exit_gate, face_off_camera_yaw_gate * 0.62),
      settings.face_off_camera_yaw_exit_gate,
      26.0,
    )
    face_off_camera_pitch_exit_gate = self._clamp(
      max(settings.face_off_camera_pitch_exit_gate, face_off_camera_pitch_gate * 0.64),
      settings.face_off_camera_pitch_exit_gate,
      24.0,
    )
    face_off_camera_baseline_movement_gate = self._clamp(
      max(settings.face_off_camera_baseline_movement_gate, neutral_move_p95 * 1.35),
      settings.face_off_camera_baseline_movement_gate,
      0.06,
    )

    high_movement_threshold = self._clamp(
      max(settings.high_movement_threshold, neutral_move_p95 * 2.25, vertical_move_p60 * 0.92),
      settings.high_movement_threshold,
      0.12,
    )

    # Eye tracker thresholds from personalized gaze dynamics.
    eye_horizontal_deadzone = self._clamp(
      max(0.035, neutral_h_noise_p90 * 2.0, gaze_h_amp_p90 * 0.45),
      0.035,
      0.11,
    )
    eye_vertical_up_threshold = self._clamp(
      max(0.07, neutral_v_noise_p90 * 1.9, gaze_v_amp_p85 * 0.50),
      0.06,
      0.18,
    )
    eye_vertical_down_threshold = self._clamp(
      max(0.07, neutral_v_noise_p90 * 1.9, gaze_v_amp_p85 * 0.50),
      0.06,
      0.18,
    )
    neutral_eye_visible_ratio = float(np.mean(neutral_eye_visible)) if neutral_eye_visible else 0.0
    eye_visibility_threshold = self._clamp(
      0.11 - (neutral_eye_visible_ratio * 0.03),
      0.05,
      0.11,
    )
    eye_baseline_alpha = self._clamp(
      0.05 + (neutral_h_noise_p90 * 1.4),
      0.05,
      0.14,
    )
    gaze_extreme_horizontal_threshold = self._clamp(
      max(
        settings.gaze_extreme_horizontal_threshold,
        eye_horizontal_deadzone * 1.18,
        neutral_gaze_h_noise * 2.2,
        neutral_h_noise_p90 * 2.0,
        gaze_h_amp_p90 * 0.50,
      ),
      0.048,
      0.20,
    )
    gaze_extreme_vertical_threshold = self._clamp(
      max(
        settings.gaze_extreme_vertical_threshold,
        ((eye_vertical_up_threshold + eye_vertical_down_threshold) * 0.5) * 1.12,
        neutral_gaze_v_noise * 2.2,
        neutral_v_noise_p90 * 2.1,
        gaze_v_amp_p85 * 0.52,
      ),
      0.058,
      0.22,
    )
    gaze_extreme_activation_frames = max(
      2,
      min(
        5,
        int(round(2.0 + (1.0 - min(1.0, gaze_h_amp_p90 / max(settings.calibration_gaze_min_signal_amplitude, 1e-6))) * 2.0)),
      ),
    )
    gaze_extreme_min_confidence = self._clamp(
      max(settings.gaze_extreme_min_confidence, 0.17 - (gaze_conf_avg * 0.10)),
      0.05,
      0.18,
    )
    gaze_distance_scale = self._clamp(
      max(settings.gaze_distance_scale, neutral_move_p95 * 4.8),
      0.16,
      0.62,
    )
    gaze_display_scale = self._clamp(
      max(settings.gaze_display_scale, 0.14 + neutral_h_noise_p90 * 1.8),
      0.12,
      0.42,
    )
    gaze_vertical_display_scale = self._clamp(
      max(settings.gaze_vertical_display_scale, 0.18 + neutral_v_noise_p90 * 1.8),
      0.14,
      0.48,
    )
    gaze_face_yaw_gate = self._clamp(
      max(settings.gaze_face_yaw_gate, face_off_camera_yaw_gate * 0.86),
      10.0,
      32.0,
    )
    gaze_face_pitch_gate = self._clamp(
      max(settings.gaze_face_pitch_gate, face_off_camera_pitch_gate * 0.86),
      8.0,
      28.0,
    )
    gaze_extreme_horizontal_threshold = self._clamp(
      max(
        gaze_extreme_horizontal_threshold,
        abs(gaze_left_boundary - gaze_center_x_median),
        abs(gaze_right_boundary - gaze_center_x_median),
      ),
      0.04,
      0.24,
    )
    gaze_extreme_vertical_threshold = self._clamp(
      max(
        gaze_extreme_vertical_threshold,
        abs(gaze_top_boundary - gaze_center_y_median),
        abs(gaze_bottom_boundary - gaze_center_y_median),
      ),
      0.05,
      0.26,
    )

    # Phone thresholds.
    phone_min_confidence = settings.phone_min_confidence
    phone_min_area_ratio = settings.phone_min_area_ratio
    phone_confirm_frames = settings.phone_confirm_frames
    if phone_conf:
      conf_p85 = self._pct(phone_conf, 85, settings.phone_min_confidence)
      neutral_phone_hits = len(phone_conf)
      if neutral_phone_hits >= 8:
        phone_min_confidence = self._clamp(
          max(phone_min_confidence, conf_p85 + 0.03),
          settings.phone_min_confidence,
          0.58,
        )
      elif neutral_phone_hits >= 4:
        phone_min_confidence = self._clamp(
          max(phone_min_confidence, conf_p85 + 0.02),
          settings.phone_min_confidence,
          0.54,
        )
    if phone_area:
      area_p85 = self._pct(phone_area, 85, settings.phone_min_area_ratio)
      phone_min_area_ratio = self._clamp(
        max(phone_min_area_ratio, area_p85 * 1.05),
        settings.phone_min_area_ratio,
        0.008,
      )
    if len(phone_conf) >= 12:
      phone_confirm_frames = min(settings.phone_confirm_frames + 1, 2)

    background_motion_zones = self._background_motion_zones()
    background_motion_coverage = 0.0
    if background_motion_zones:
      background_motion_coverage = float(
        sum(float(zone.get("w", 0.0)) * float(zone.get("h", 0.0)) for zone in background_motion_zones)
      )

    hand_ratio = float(np.mean(hand_samples)) if hand_samples else 0.0

    thresholds = {
      "high_movement_threshold": round(float(high_movement_threshold), 4),
      "vertical_head_pitch_enter_threshold": round(float(enter_th), 3),
      "vertical_head_pitch_exit_threshold": round(float(exit_th), 3),
      "vertical_pitch_ratio_up_threshold": round(float(pitch_ratio_up_threshold), 4),
      "vertical_pitch_ratio_down_threshold": round(float(pitch_ratio_down_threshold), 4),
      "vertical_head_activation_frames": float(vertical_activation_frames),
      "vertical_head_activation_movement_threshold": round(float(vertical_activation_movement_threshold), 4),
      "vertical_head_pitch_velocity_gate": round(float(vertical_velocity_gate), 3),
      "vertical_head_pitch_span_gate": round(float(vertical_span_gate), 3),
      "vertical_head_delta_velocity_gate": round(float(vertical_delta_velocity_gate), 3),
      "vertical_head_y_velocity_gate": round(float(vertical_y_velocity_gate), 4),
      "vertical_head_y_span_gate": round(float(vertical_y_span_gate), 4),
      "vertical_head_large_movement_gate": round(float(vertical_large_movement_gate), 4),
      "horizontal_head_activation_frames": float(horizontal_activation_frames),
      "horizontal_head_yaw_velocity_gate": round(float(horizontal_yaw_velocity_gate), 3),
      "horizontal_head_x_velocity_gate": round(float(horizontal_x_velocity_gate), 4),
      "horizontal_head_yaw_gate": round(float(horizontal_yaw_gate), 3),
      "horizontal_head_exit_velocity_gate": round(float(horizontal_exit_velocity_gate), 3),
      "horizontal_head_yaw_span_gate": round(float(horizontal_yaw_span_gate), 3),
      "horizontal_head_x_span_gate": round(float(horizontal_x_span_gate), 4),
      "horizontal_head_large_movement_gate": round(float(horizontal_large_movement_gate), 4),
      "horizontal_head_large_x_span_gate": round(float(horizontal_large_x_span_gate), 4),
      "face_off_camera_yaw_gate": round(float(face_off_camera_yaw_gate), 3),
      "face_off_camera_pitch_gate": round(float(face_off_camera_pitch_gate), 3),
      "face_off_camera_yaw_exit_gate": round(float(face_off_camera_yaw_exit_gate), 3),
      "face_off_camera_pitch_exit_gate": round(float(face_off_camera_pitch_exit_gate), 3),
      "face_off_camera_baseline_movement_gate": round(float(face_off_camera_baseline_movement_gate), 4),
      "gaze_extreme_horizontal_threshold": round(float(gaze_extreme_horizontal_threshold), 4),
      "gaze_extreme_vertical_threshold": round(float(gaze_extreme_vertical_threshold), 4),
      "gaze_extreme_activation_frames": float(gaze_extreme_activation_frames),
      "gaze_extreme_min_confidence": round(float(gaze_extreme_min_confidence), 4),
      "gaze_distance_reference_face_area_ratio": round(float(gaze_face_area_ref_ratio), 5),
      "gaze_distance_scale": round(float(gaze_distance_scale), 4),
      "gaze_display_reference_diag_px": round(float(gaze_display_ref_diag_px), 2),
      "gaze_display_scale": round(float(gaze_display_scale), 4),
      "gaze_vertical_display_scale": round(float(gaze_vertical_display_scale), 4),
      "gaze_face_yaw_gate": round(float(gaze_face_yaw_gate), 3),
      "gaze_face_pitch_gate": round(float(gaze_face_pitch_gate), 3),
      "eye_horizontal_deadzone": round(float(eye_horizontal_deadzone), 4),
      "eye_vertical_up_threshold": round(float(eye_vertical_up_threshold), 4),
      "eye_vertical_down_threshold": round(float(eye_vertical_down_threshold), 4),
      "eye_visibility_threshold": round(float(eye_visibility_threshold), 4),
      "eye_baseline_alpha": round(float(eye_baseline_alpha), 4),
      "phone_min_confidence": round(float(phone_min_confidence), 3),
      "phone_min_area_ratio": round(float(phone_min_area_ratio), 5),
      "phone_confirm_frames": float(phone_confirm_frames),
    }

    attention_model = {
      "yawMean": round(float(neutral_yaw_mean), 4),
      "yawMedian": round(float(neutral_yaw_median), 4),
      "yawStd": round(float(neutral_yaw_std), 4),
      "yawNoise": round(float(neutral_yaw_noise), 4),
      "yawNoiseFloor": round(float(neutral_yaw_noise_p95), 4),
      "pitchMean": round(float(neutral_pitch_mean), 4),
      "pitchMedian": round(float(neutral_pitch_median), 4),
      "pitchStd": round(float(neutral_pitch_std), 4),
      "pitchNoise": round(float(neutral_pitch_noise), 4),
      "pitchNoiseFloor": round(float(neutral_pitch_noise_p95), 4),
      "gazeMeanX": round(float(neutral_gaze_h_mean), 5),
      "gazeMeanY": round(float(neutral_gaze_v_mean), 5),
      "gazeMedianX": round(float(neutral_gaze_h_median), 5),
      "gazeMedianY": round(float(neutral_gaze_v_median), 5),
      "gazeNoise": round(float(neutral_gaze_noise), 5),
      "gazeNoiseX": round(float(neutral_gaze_h_noise), 5),
      "gazeNoiseY": round(float(neutral_gaze_v_noise), 5),
      "gazeNoiseFloorX": round(float(neutral_gaze_h_noise_p95), 5),
      "gazeNoiseFloorY": round(float(neutral_gaze_v_noise_p95), 5),
      "gazeCenterX": round(float(gaze_center_x_median), 5),
      "gazeCenterY": round(float(gaze_center_y_median), 5),
      "gazeLeftBoundary": round(float(gaze_left_boundary), 5),
      "gazeRightBoundary": round(float(gaze_right_boundary), 5),
      "gazeTopBoundary": round(float(gaze_top_boundary), 5),
      "gazeBottomBoundary": round(float(gaze_bottom_boundary), 5),
      "pitchRatioCenter": round(float(neutral_pitch_ratio_median), 5),
      "pitchRatioUpThreshold": round(float(pitch_ratio_up_threshold), 5),
      "pitchRatioDownThreshold": round(float(pitch_ratio_down_threshold), 5),
      "frontalnessMean": round(float(neutral_frontalness_mean), 4),
      "maxNaturalDeviation": dict(max_natural_deviation),
      "baselineFrames": float(len(neutral_pitch)),
    }

    neutral_ready, _ = self._stage_ready("neutral")
    gaze_stage_flags = {code: self._stage_ready(code)[0] for code in self._gaze_target_codes}
    gaze_ready = all(gaze_stage_flags.values())
    head_ready, _ = self._stage_ready("head_vertical")
    hand_ready, _ = self._stage_ready("hand_face")

    neutral_target = max(float(settings.calibration_neutral_target_frames), 1.0)
    neutral_score_frames = min(1.0, len(neutral_pitch) / neutral_target)
    neutral_stability = 1.0 if neutral_move_p95 <= 0.11 else max(0.2, 0.11 / max(neutral_move_p95, 1e-6))
    neutral_score = 0.65 * neutral_score_frames + 0.35 * neutral_stability

    sign_threshold = float(settings.calibration_gaze_signal_sign_threshold)
    gaze_left = max(
      sum(1 for value in gaze_signal if value < 0),
      sum(1 for value in gaze_h if value <= -sign_threshold),
    )
    gaze_right = max(
      sum(1 for value in gaze_signal if value > 0),
      sum(1 for value in gaze_h if value >= sign_threshold),
    )
    gaze_offset_signal = [
      -1.0 if value <= -sign_threshold else 1.0
      for value in gaze_h
      if abs(value) >= sign_threshold
    ]
    gaze_transitions = max(
      self._count_direction_changes(gaze_signal),
      self._count_direction_changes(gaze_offset_signal),
    )
    gaze_score_samples = min(1.0, len(gaze_signal) / max(float(settings.calibration_gaze_min_samples), 1.0))
    gaze_score_balance = min(1.0, min(gaze_left, gaze_right) / max(float(settings.calibration_gaze_min_side_frames), 1.0))
    gaze_score_transitions = min(1.0, gaze_transitions / max(float(settings.calibration_gaze_min_transitions), 1.0))
    gaze_visible_ratio = float(np.mean(gaze_eye_visible)) if gaze_eye_visible else 0.0
    gaze_score_visibility = min(1.0, gaze_visible_ratio / max(float(settings.calibration_gaze_min_eye_visible_ratio), 1e-6))
    gaze_score_signal = min(1.0, gaze_h_amp_p90 / max(float(settings.calibration_gaze_min_signal_amplitude), 1e-6))
    gaze_score = (
      0.30 * gaze_score_samples
      + 0.25 * gaze_score_balance
      + 0.20 * gaze_score_transitions
      + 0.15 * gaze_score_signal
      + 0.10 * gaze_score_visibility
    )
    gaze_target_score = float(sum(1.0 for ready_flag in gaze_stage_flags.values() if ready_flag)) / max(float(len(gaze_stage_flags)), 1.0)
    gaze_boundary_span_x = abs(gaze_right_boundary - gaze_left_boundary)
    gaze_boundary_span_y = abs(gaze_bottom_boundary - gaze_top_boundary)
    gaze_target_visible_ratio = float(np.mean(gaze_target_visible)) if gaze_target_visible else 0.0
    gaze_score = (
      0.35 * gaze_target_score
      + 0.25 * min(1.0, gaze_boundary_span_x / 0.12)
      + 0.20 * min(1.0, gaze_boundary_span_y / 0.10)
      + 0.10 * gaze_score_signal
      + 0.10 * min(1.0, gaze_target_visible_ratio / max(float(settings.calibration_gaze_target_visible_ratio), 1e-6))
    )

    ratio_margin = max(float(settings.calibration_pitch_ratio_min_range) * 0.35, 0.008)
    head_up_frames = sum(1 for value in vertical_pitch_ratio if value <= (neutral_pitch_ratio_median - ratio_margin))
    head_down_frames = sum(1 for value in vertical_pitch_ratio if value >= (neutral_pitch_ratio_median + ratio_margin))
    head_range = pitch_ratio_p90 - pitch_ratio_p10
    head_score_samples = min(1.0, len(vertical_pitch_ratio) / max(float(settings.calibration_head_min_samples), 1.0))
    head_score_direction = min(
      1.0,
      min(head_up_frames, head_down_frames) / max(float(settings.calibration_head_min_direction_frames), 1.0),
    )
    head_score_range = min(1.0, head_range / max(float(settings.calibration_pitch_ratio_min_range), 1.0e-6))
    head_score = 0.35 * head_score_samples + 0.35 * head_score_direction + 0.30 * head_score_range

    hand_positive = int(sum(hand_samples))
    hand_score_samples = min(1.0, len(hand_samples) / max(float(settings.calibration_hand_min_samples), 1.0))
    hand_score_positive = min(1.0, hand_positive / max(float(settings.calibration_hand_min_positive_frames), 1.0))
    hand_score = 0.45 * hand_score_samples + 0.55 * hand_score_positive

    quality_score = (
      0.33 * neutral_score
      + 0.24 * gaze_score
      + 0.30 * head_score
      + 0.13 * hand_score
    )

    stats = {
      "neutralFaceFrames": len(neutral_pitch),
      "neutralMovementP95": round(float(neutral_move_p95), 5),
      "neutralYawMean": round(float(neutral_yaw_mean), 4),
      "neutralYawMedian": round(float(neutral_yaw_median), 4),
      "neutralPitchStd": round(float(neutral_pitch_std), 4),
      "neutralPitchMean": round(float(neutral_pitch_mean), 4),
      "neutralPitchMedian": round(float(neutral_pitch_median), 4),
      "neutralYawStd": round(float(neutral_yaw_std), 4),
      "neutralYawNoise": round(float(neutral_yaw_noise), 4),
      "neutralYawNoiseP95": round(float(neutral_yaw_noise_p95), 4),
      "neutralPitchNoise": round(float(neutral_pitch_noise), 4),
      "neutralPitchNoiseP95": round(float(neutral_pitch_noise_p95), 4),
      "neutralPitchRatioMedian": round(float(neutral_pitch_ratio_median), 5),
      "neutralFrontalnessMean": round(float(neutral_frontalness_mean), 4),
      "neutralPitchVelP95": round(float(neutral_pitch_vel_p95), 4),
      "neutralYawVelP95": round(float(neutral_yaw_vel_p95), 4),
      "neutralGazeNoiseP90": round(float(neutral_h_noise_p90), 5),
      "neutralGazeMeanX": round(float(neutral_gaze_h_mean), 5),
      "neutralGazeMeanY": round(float(neutral_gaze_v_mean), 5),
      "neutralGazeMedianX": round(float(neutral_gaze_h_median), 5),
      "neutralGazeMedianY": round(float(neutral_gaze_v_median), 5),
      "neutralGazeNoise": round(float(neutral_gaze_noise), 5),
      "neutralGazeNoiseXP95": round(float(neutral_gaze_h_noise_p95), 5),
      "neutralGazeNoiseYP95": round(float(neutral_gaze_v_noise_p95), 5),
      "maxNaturalDeviation": dict(max_natural_deviation),
      "gazeSignalAmpP90": round(float(gaze_h_amp_p90), 5),
      "gazeVerticalSignalAmpP85": round(float(gaze_v_amp_p85), 5),
      "gazeVisibleRatio": round(float(gaze_visible_ratio), 3),
      "gazeTargetVisibleRatio": round(float(gaze_target_visible_ratio), 3),
      "gazeConfidenceAvg": round(float(gaze_conf_avg), 3),
      "gazeFaceAreaRefRatio": round(float(gaze_face_area_ref_ratio), 5),
      "gazeDisplayRefDiagPx": round(float(gaze_display_ref_diag_px), 2),
      "gazeBoundaries": {
        "centerX": round(float(gaze_center_x_median), 5),
        "centerY": round(float(gaze_center_y_median), 5),
        "left": round(float(gaze_left_boundary), 5),
        "right": round(float(gaze_right_boundary), 5),
        "top": round(float(gaze_top_boundary), 5),
        "bottom": round(float(gaze_bottom_boundary), 5),
      },
      "verticalDeltaP75": round(float(vertical_delta_p75), 3),
      "verticalMovementP60": round(float(vertical_move_p60), 5),
      "pitchRatioRange": round(float(head_range), 5),
      "pitchRatioThresholds": {
        "up": round(float(pitch_ratio_up_threshold), 5),
        "down": round(float(pitch_ratio_down_threshold), 5),
      },
      "neutralPhoneDetections": len(phone_conf),
      "handNearFaceRatio": round(float(hand_ratio), 3),
      "gazeCounts": dict(self._gaze_counts),
      "gazeTransitions": int(gaze_transitions),
      "headRangeDeg": round(float(self._pct(vertical_signed_delta, 95, 0.0) - self._pct(vertical_signed_delta, 5, 0.0)), 3),
      "headDirectionFrames": {
        "up": int(head_up_frames),
        "down": int(head_down_frames),
      },
      "handPositiveFrames": int(hand_positive),
      "backgroundMotionFrames": int(self._bg_motion_frames),
      "backgroundMotionZonesCount": int(len(background_motion_zones)),
      "backgroundMotionCoverage": round(float(background_motion_coverage), 4),
      "qualityComponents": {
        "neutral": round(float(neutral_score), 3),
        "gaze": round(float(gaze_score), 3),
        "headVertical": round(float(head_score), 3),
        "handFace": round(float(hand_score), 3),
      },
      "qualityScore": round(float(quality_score), 3),
      "qualityFlags": {
        "neutral": bool(neutral_ready),
        "gaze": bool(gaze_ready),
        "gazeTargets": dict(gaze_stage_flags),
        "headVertical": bool(head_ready),
        "handFace": bool(hand_ready),
      },
      "stageOutcomes": dict(self._stage_outcomes),
    }

    ready = bool(neutral_ready and gaze_ready and head_ready and quality_score >= 0.62)

    return {
      "ready": ready,
      "thresholds": thresholds,
      "attentionModel": attention_model,
      "backgroundMotionZones": background_motion_zones,
      "stats": stats,
    }

  def _pct(self, values: List[float], q: float, default: float) -> float:
    if not values:
      return float(default)
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))

  def _median(self, values: List[float]) -> float:
    if not values:
      return 0.0
    return float(np.median(np.asarray(values, dtype=np.float32)))

  def _mad_noise(self, values: List[float], median_value: Optional[float] = None) -> float:
    if not values:
      return 0.0
    median = float(median_value) if median_value is not None else self._median(values)
    deviations = np.abs(np.asarray(values, dtype=np.float32) - median)
    return float(1.4826 * np.median(deviations))

  def _std(self, values: List[float]) -> float:
    if len(values) < 2:
      return 0.0
    return float(np.std(np.asarray(values, dtype=np.float32)))

  def _clamp(self, value: float, low: float, high: float) -> float:
    return float(max(low, min(value, high)))




from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import hypot
from time import monotonic
from typing import Any, Deque, Dict, List, Optional, Tuple

from config import settings
from utils.smoothing import MovingAverage
from vision.attention_model import AttentionModel, AttentionSnapshot, AttentionState, AttentionThresholds
from vision.object_detector import DetectedObject


@dataclass
class _VerticalPoseState:
  smooth_pitch_ratio: float = 0.5
  baseline_pitch_ratio: float = 0.5
  prev_pitch_ratio: float = 0.5
  low_visibility_since: Optional[float] = None
  evidence: Deque[float] = field(
    default_factory=lambda: deque(maxlen=max(3, int(settings.vertical_head_evidence_frames)))
  )
  smooth_pitch: float = 0.0
  baseline_pitch: float = 0.0
  prev_smooth_pitch: float = 0.0
  prev_delta: float = 0.0
  pitch_velocity_ema: float = 0.0
  delta_velocity_ema: float = 0.0
  prev_center_y: float = 0.0
  y_velocity_ema: float = 0.0
  center_y_window: Deque[float] = field(default_factory=deque)
  pitch_window: Deque[float] = field(default_factory=deque)
  active_counter: int = 0
  active: bool = False
  initialized: bool = False


@dataclass
class _HorizontalMotionState:
  smooth_yaw: float = 0.0
  prev_smooth_yaw: float = 0.0
  yaw_velocity_ema: float = 0.0
  prev_center_x: float = 0.0
  x_velocity_ema: float = 0.0
  yaw_window: Deque[float] = field(default_factory=deque)
  center_window: Deque[float] = field(default_factory=deque)
  active_counter: int = 0
  active: bool = False
  initialized: bool = False


@dataclass
class _FacePresenceState:
  missing_frames: int = 0
  hidden_intent_frames: int = 0
  off_camera_counter: int = 0
  smooth_yaw: float = 0.0
  smooth_pitch: float = 0.0
  baseline_yaw: float = 0.0
  baseline_pitch: float = 0.0
  initialized: bool = False


@dataclass
class _GazeRiskState:
  counter: int = 0
  active: bool = False
  baseline_yaw: float = 0.0
  baseline_pitch: float = 0.0
  initialized: bool = False


@dataclass
class _TemporalRuleState:
  active_since: Optional[float] = None
  inactive_since: Optional[float] = None
  latched: bool = False


class RiskEngine:
  def __init__(self) -> None:
    self._smoother = MovingAverage(settings.smoothing_window)
    self._streaks: Dict[str, int] = defaultdict(int)
    self._ema_score: Optional[float] = None
    self._vertical_states: Dict[int, _VerticalPoseState] = {}
    self._horizontal_states: Dict[int, _HorizontalMotionState] = {}
    self._presence_state = _FacePresenceState()
    self._gaze_states: Dict[int, _GazeRiskState] = {}
    self._temporal_states: Dict[str, _TemporalRuleState] = {}
    self._adaptive_thresholds: Dict[str, float] = {}
    self._attention_model = AttentionModel()
    self._last_attention_debug: Dict[str, Any] = {}

    self._severity = {
      "multiple_persons": 3.6,
      "multiple_faces": 3.0,
      "phone_detected": 5.0,
      "wearing_audio_device": 4.2,
      "face_missing": 2.0,
      "face_not_facing_camera": 1.9,
      "high_movement": 2.0,
      "extreme_gaze": 2.0,
      "vertical_head_motion": 2.0,
      "horizontal_head_motion": 2.2,
      "hand_on_face": 2.0,
    }
    self._labels = {
      "multiple_persons": "Multiple persons",
      "multiple_faces": "Multiple faces",
      "phone_detected": "Phone detected",
      "wearing_audio_device": "Audio device detected",
      "face_missing": "Face missing",
      "face_not_facing_camera": "Face not facing camera",
      "high_movement": "High movement",
      "extreme_gaze": "Extreme gaze",
      "vertical_head_motion": "Vertical head movement",
      "horizontal_head_motion": "Horizontal head movement",
      "hand_on_face": "Hand on face",
    }
    self._last_breakdown = self._empty_breakdown()

  def reset(self) -> None:
    self._smoother = MovingAverage(settings.smoothing_window)
    self._streaks.clear()
    self._ema_score = None
    self._vertical_states.clear()
    self._horizontal_states.clear()
    self._presence_state = _FacePresenceState()
    self._gaze_states.clear()
    self._temporal_states.clear()
    self._attention_model.reset()
    self._last_attention_debug = {}
    self._last_breakdown = self._empty_breakdown()

  def set_adaptive_thresholds(self, overrides: Dict[str, float]) -> None:
    if not overrides:
      return
    allowed = {
      "high_movement_threshold",
      "gaze_extreme_horizontal_threshold",
      "gaze_extreme_vertical_threshold",
      "gaze_extreme_activation_frames",
      "gaze_extreme_min_confidence",
      "gaze_distance_reference_face_area_ratio",
      "gaze_distance_scale",
      "gaze_display_reference_diag_px",
      "gaze_display_scale",
      "gaze_vertical_display_scale",
      "gaze_face_yaw_gate",
      "gaze_face_pitch_gate",
      "vertical_head_pitch_enter_threshold",
      "vertical_head_pitch_exit_threshold",
      "vertical_pitch_ratio_up_threshold",
      "vertical_pitch_ratio_down_threshold",
      "vertical_head_activation_frames",
      "vertical_head_pitch_ema_alpha",
      "vertical_head_pitch_baseline_alpha",
      "vertical_head_pitch_active_baseline_alpha",
      "vertical_head_pitch_velocity_gate",
      "vertical_head_baseline_movement_threshold",
      "vertical_head_activation_movement_threshold",
      "vertical_head_baseline_yaw_threshold",
      "vertical_head_span_window_frames",
      "vertical_head_pitch_span_gate",
      "vertical_head_delta_velocity_gate",
      "vertical_head_y_velocity_gate",
      "vertical_head_y_span_gate",
      "vertical_head_large_movement_gate",
      "horizontal_head_activation_frames",
      "horizontal_head_yaw_ema_alpha",
      "horizontal_head_yaw_velocity_gate",
      "horizontal_head_x_velocity_gate",
      "horizontal_head_movement_gate",
      "horizontal_head_yaw_gate",
      "horizontal_head_exit_velocity_gate",
      "horizontal_head_span_window_frames",
      "horizontal_head_yaw_span_gate",
      "horizontal_head_x_span_gate",
      "horizontal_head_large_movement_gate",
      "horizontal_head_large_x_span_gate",
      "face_missing_activation_frames",
      "face_missing_evasive_activation_frames",
      "face_hide_motion_gate",
      "face_hide_yaw_gate",
      "face_hide_pitch_gate",
      "face_orientation_ema_alpha",
      "face_orientation_baseline_alpha",
      "face_orientation_active_baseline_alpha",
      "face_off_camera_yaw_gate",
      "face_off_camera_pitch_gate",
      "face_off_camera_yaw_exit_gate",
      "face_off_camera_pitch_exit_gate",
      "face_off_camera_baseline_movement_gate",
      "face_off_camera_activation_frames",
      "face_off_camera_memory_frames",
    }
    for key, value in overrides.items():
      if key not in allowed:
        continue
      try:
        self._adaptive_thresholds[key] = float(value)
      except (TypeError, ValueError):
        continue

  def get_adaptive_thresholds(self) -> Dict[str, float]:
    return dict(self._adaptive_thresholds)

  def set_calibration_profile(self, profile: Dict[str, Any]) -> None:
    self._attention_model.set_calibration_profile(profile)

  def _threshold(self, key: str, default_value: float) -> float:
    value = self._adaptive_thresholds.get(key)
    if value is None:
      return float(default_value)
    return float(value)

  def compute(
    self,
    face_count: int,
    tracked_faces: List[dict],
    objects: List[DetectedObject],
    face_signal_available: bool = True,
    context: Optional[Dict[str, Any]] = None,
    person_count: Optional[int] = None,
  ) -> Tuple[float, str, Dict[str, Any]]:
    risk_context = context or {}
    frame_w, frame_h, _, _ = self._context_dimensions(risk_context, tracked_faces=tracked_faces)
    high_movement_threshold = self._threshold("high_movement_threshold", settings.high_movement_threshold)
    attention_snapshot = self._attention_model.update(
      tracked_faces=tracked_faces,
      frame_width=frame_w,
      frame_height=frame_h,
      thresholds=self._attention_thresholds(),
    )

    attention_blocked = attention_snapshot.signal_confidence < 0.55
    self._last_attention_debug = attention_snapshot.as_dict()
    self._annotate_attention(tracked_faces=tracked_faces, snapshot=attention_snapshot)

    detected_persons = sum(1 for obj in objects if obj.label == "person")
    effective_person_count = max(int(person_count or 0), int(detected_persons))
    has_phone = any(obj.label == "phone" for obj in objects)
    has_audio_device = any(obj.label == "audio_device" for obj in objects)
    has_high_movement = (
      not attention_blocked
      and any(face["movement"] >= high_movement_threshold for face in tracked_faces)
    )
    has_vertical_head_motion = self._detect_vertical_head_motion(tracked_faces, frame_height=frame_h)
    has_horizontal_head_motion = (
      not attention_blocked
      and attention_snapshot.horizontal_state == AttentionState.LOOKING_AWAY
    )
    has_face_missing = self._detect_face_presence_risks(
      face_count=face_count,
      tracked_faces=tracked_faces,
      has_vertical_head_motion=has_vertical_head_motion,
      has_horizontal_head_motion=has_horizontal_head_motion,
      attention_snapshot=attention_snapshot,
    )
    has_face_not_facing_camera = (
      not attention_blocked
      and attention_snapshot.orientation_state == AttentionState.LOOKING_AWAY
    )

    has_extreme_gaze = (
      not attention_blocked
      and attention_snapshot.gaze_state == AttentionState.LOOKING_AWAY
    )

    skip_attention = attention_snapshot.deviation_score < 0.08

    has_hand_on_face = any(bool(face.get("handOnFace", False)) for face in tracked_faces)
    raw_violations = {
      "multiple_persons": effective_person_count > 1,
      "multiple_faces": face_signal_available and face_count > 1 and effective_person_count <= 1,
      "phone_detected": has_phone,
      "wearing_audio_device": has_audio_device,
      "face_missing": face_signal_available and has_face_missing,
      "face_not_facing_camera": (
        settings.enable_face_orientation_risk
        and (not skip_attention)
        and has_face_not_facing_camera
      ),
      "high_movement": has_high_movement,
      "extreme_gaze": (not skip_attention) and has_extreme_gaze,
      "vertical_head_motion": settings.enable_vertical_head_risk and has_vertical_head_motion,
      "horizontal_head_motion": (
        settings.enable_horizontal_head_risk
        and (not skip_attention)
        and has_horizontal_head_motion
      ),
      "hand_on_face": settings.enable_hand_on_face_risk and has_hand_on_face,
    }
    violations = self._apply_temporal_gates(raw_violations)

    raw_score = 0.0
    rule_states: List[Dict[str, Any]] = []
    for name, active in violations.items():
      base_severity = float(self._severity[name])
      streak = 0
      repetition_factor = 0.0
      multiplier = 1.0
      points = 0.0
      if active:
        self._streaks[name] += 1
        repetition_frames = max(
          self._streaks[name] - settings.repeated_violation_start_frames,
          0,
        )
        repetition_factor = min(
          repetition_frames * settings.repeated_violation_gain,
          settings.repeated_violation_cap,
        )
        multiplier = 1.0 + repetition_factor
        if name in ["extreme_gaze", "horizontal_head_motion", "face_not_facing_camera"]:
          confidence_scale = max(0.3, float(attention_snapshot.attention_confidence))
        else:
          confidence_scale = 1.0

        points = base_severity * multiplier * confidence_scale
        raw_score += points
        streak = self._streaks[name]
      else:
        self._streaks[name] = 0
        streak = 0

      rule_states.append(
        {
          "code": name,
          "label": self._labels.get(name, name),
          "active": bool(active),
          "baseSeverity": round(base_severity, 3),
          "streak": int(streak),
          "repetitionFactor": round(float(repetition_factor), 3),
          "multiplier": round(float(multiplier), 3),
          "points": round(float(points), 3),
        }
      )

    alpha = settings.exponential_smoothing_alpha
    if self._ema_score is None:
      self._ema_score = raw_score
    else:
      self._ema_score = alpha * raw_score + (1.0 - alpha) * self._ema_score

    smoothed_score = self._smoother.add(self._ema_score)
    score = round(smoothed_score, 3)
    level = self._level_for_score(smoothed_score)
    breakdown = self._build_breakdown(
      rule_states=rule_states,
      raw_score=raw_score,
      ema_score=self._ema_score,
      smoothed_score=score,
      level=level,
    )
    self._last_breakdown = breakdown
    return score, level, breakdown

  def _apply_temporal_gates(self, raw_violations: Dict[str, bool]) -> Dict[str, bool]:
    now = monotonic()
    filtered: Dict[str, bool] = {}
    for code, active in raw_violations.items():
      activate_sec, release_sec = self._temporal_thresholds_for(code)
      if activate_sec <= 0.0 and release_sec <= 0.0:
        filtered[code] = bool(active)
        continue

      state = self._temporal_states.setdefault(code, _TemporalRuleState())
      if active:
        state.inactive_since = None
        if state.active_since is None:
          state.active_since = now
        if activate_sec <= 0.0 or state.latched or (now - state.active_since) >= activate_sec:
          state.latched = True
        filtered[code] = bool(state.latched)
        continue

      state.active_since = None
      if state.inactive_since is None:
        state.inactive_since = now
      if state.latched:
        if release_sec <= 0.0 or (now - state.inactive_since) >= release_sec:
          state.latched = False
      filtered[code] = bool(state.latched)
    return filtered

  @staticmethod
  def _temporal_thresholds_for(code: str) -> Tuple[float, float]:
    if code == "multiple_persons":
      return float(settings.multiple_person_activation_sec), float(settings.multiple_person_release_sec)
    if code == "multiple_faces":
      return float(settings.multiple_faces_activation_sec), float(settings.multiple_faces_release_sec)
    if code == "face_missing":
      return float(settings.face_missing_activation_sec), float(settings.face_missing_release_sec)
    if code == "face_not_facing_camera":
      return 0.0, 0.0
    if code == "extreme_gaze":
      return 0.0, 0.0
    if code == "vertical_head_motion":
      return float(settings.vertical_head_motion_activation_sec), float(settings.vertical_head_motion_release_sec)
    if code == "horizontal_head_motion":
      return 0.0, 0.0
    if code == "phone_detected":
      return float(settings.phone_detected_activation_sec), float(settings.phone_detected_release_sec)
    if code == "wearing_audio_device":
      return float(settings.phone_detected_activation_sec), float(settings.phone_detected_release_sec)
    return 0.0, 0.0

  def _level_for_score(self, score: float) -> str:
    low_to_medium, medium_to_high, high_to_critical = settings.risk_level_thresholds
    if score < low_to_medium:
      return "LOW"
    if score < medium_to_high:
      return "MEDIUM"
    if score < high_to_critical:
      return "HIGH"
    return "CRITICAL"

  def current(self) -> Tuple[float, str, Dict[str, Any]]:
    score = self._smoother.current
    level = self._level_for_score(score)
    breakdown = dict(self._last_breakdown)
    breakdown["smoothedScore"] = round(score, 3)
    breakdown["riskLevel"] = level
    breakdown["adaptiveThresholds"] = self.get_adaptive_thresholds()
    breakdown["attentionModel"] = dict(self._last_attention_debug)
    return round(score, 3), level, breakdown

  def _detect_extreme_gaze(self, tracked_faces: List[dict], context: Dict[str, Any]) -> bool:
    if not tracked_faces:
      self._gaze_states.clear()
      return False

    base_h = self._threshold("gaze_extreme_horizontal_threshold", settings.gaze_extreme_horizontal_threshold)
    base_v = self._threshold("gaze_extreme_vertical_threshold", settings.gaze_extreme_vertical_threshold)
    activation_frames = max(
      1,
      int(round(self._threshold("gaze_extreme_activation_frames", settings.gaze_extreme_activation_frames))),
    )
    min_conf = self._threshold("gaze_extreme_min_confidence", settings.gaze_extreme_min_confidence)
    distance_ref_ratio = self._threshold(
      "gaze_distance_reference_face_area_ratio",
      settings.gaze_distance_reference_face_area_ratio,
    )
    distance_scale = self._threshold("gaze_distance_scale", settings.gaze_distance_scale)
    display_ref_diag = self._threshold("gaze_display_reference_diag_px", settings.gaze_display_reference_diag_px)
    display_scale = self._threshold("gaze_display_scale", settings.gaze_display_scale)
    vertical_display_scale = self._threshold("gaze_vertical_display_scale", settings.gaze_vertical_display_scale)
    yaw_gate = self._threshold("gaze_face_yaw_gate", settings.gaze_face_yaw_gate)
    pitch_gate = self._threshold("gaze_face_pitch_gate", settings.gaze_face_pitch_gate)
    baseline_motion_gate = self._threshold(
      "face_off_camera_baseline_movement_gate",
      settings.face_off_camera_baseline_movement_gate,
    )

    capture_w, capture_h, display_w, display_h = self._context_dimensions(context, tracked_faces=tracked_faces)
    display_diag = hypot(display_w, display_h)
    display_delta = (display_diag - display_ref_diag) / max(display_ref_diag, 1.0)
    display_factor = 1.0 + display_scale * max(-0.45, min(display_delta, 0.85))
    display_factor = max(0.72, min(display_factor, 1.55))
    display_aspect = display_h / max(display_w, 1.0)
    vertical_display_factor = display_factor * (
      1.0 + vertical_display_scale * max(0.0, min(display_aspect - 0.56, 0.9))
    )

    active_ids = set()
    any_active = False

    for face in tracked_faces:
      face_id = int(face.get("id", -1))
      if face_id < 0:
        continue
      active_ids.add(face_id)

      state = self._gaze_states.setdefault(face_id, _GazeRiskState())
      if not bool(face.get("eyeVisible", False)):
        state.counter = max(state.counter - 1, 0)
        if state.counter == 0:
          state.active = False
        continue

      pose = face.get("headPose", {}) or {}
      yaw = float(pose.get("yaw", 0.0))
      pitch = float(pose.get("pitch", 0.0))
      if not state.initialized:
        state.baseline_yaw = yaw
        state.baseline_pitch = pitch
        state.initialized = True

      h_offset = abs(float(face.get("gazeOffsetX", 0.0)))
      v_offset = abs(float(face.get("gazeOffsetY", 0.0)))
      conf = float(face.get("gazeConfidence", 0.0))
      movement = float(face.get("movement", 0.0))
      bbox = face.get("bbox", {}) or {}
      face_w = max(float(bbox.get("w", 0.0)), 1.0)
      face_h = max(float(bbox.get("h", 0.0)), 1.0)
      area_ratio = (face_w * face_h) / max(capture_w * capture_h, 1.0)
      distance_delta = (distance_ref_ratio - area_ratio) / max(distance_ref_ratio, 1e-6)
      distance_factor = 1.0 + distance_scale * max(-0.45, min(distance_delta, 1.2))
      distance_factor = max(0.72, min(distance_factor, 1.95))

      h_factor = max(0.62, min(display_factor * distance_factor, 1.68))
      v_factor = max(0.62, min(vertical_display_factor * distance_factor, 1.72))
      h_threshold = base_h * h_factor
      v_threshold = base_v * v_factor

      yaw_abs = abs(yaw)
      pitch_abs = abs(pitch)
      yaw_delta = abs(yaw - state.baseline_yaw)
      pitch_delta = abs(pitch - state.baseline_pitch)
      yaw_abs_gate = max(yaw_gate + 7.0, yaw_gate * 1.45)
      pitch_abs_gate = max(pitch_gate + 6.0, pitch_gate * 1.45)
      orientation_block = (
        (yaw_abs >= yaw_abs_gate and yaw_delta >= yaw_gate * 0.74)
        or (pitch_abs >= pitch_abs_gate and pitch_delta >= pitch_gate * 0.74)
      )

      baseline_update_allowed = (
        not state.active
        and not orientation_block
        and movement <= (baseline_motion_gate * 1.35)
        and h_offset <= (h_threshold * 0.74)
        and (
          not settings.enable_vertical_gaze_risk
          or v_offset <= (v_threshold * 0.74)
        )
      )
      if baseline_update_allowed:
        baseline_alpha = 0.04 if conf >= min_conf else 0.025
        state.baseline_yaw = (1.0 - baseline_alpha) * state.baseline_yaw + baseline_alpha * yaw
        state.baseline_pitch = (1.0 - baseline_alpha) * state.baseline_pitch + baseline_alpha * pitch
        yaw_delta = abs(yaw - state.baseline_yaw)
        pitch_delta = abs(pitch - state.baseline_pitch)
        orientation_block = (
          (yaw_abs >= yaw_abs_gate and yaw_delta >= yaw_gate * 0.74)
          or (pitch_abs >= pitch_abs_gate and pitch_delta >= pitch_gate * 0.74)
        )

      horizontal_signal = h_offset >= h_threshold
      vertical_signal = settings.enable_vertical_gaze_risk and v_offset >= v_threshold
      signal_strength = max(
        h_offset / max(h_threshold, 1e-6),
        v_offset / max(v_threshold, 1e-6),
      )
      confidence_ready = conf >= min_conf or signal_strength >= 1.26
      signal = (horizontal_signal or vertical_signal) and confidence_ready and not orientation_block

      release_signal = (
        (orientation_block and signal_strength < 1.15)
        or conf < (min_conf * 0.58)
        or (
          h_offset <= (h_threshold * 0.68)
          and (
            not settings.enable_vertical_gaze_risk
            or v_offset <= (v_threshold * 0.68)
          )
        )
      )

      if state.active:
        if signal:
          state.counter = min(state.counter + 1, activation_frames + 3)
        elif release_signal:
          state.counter = max(state.counter - 2, 0)
        else:
          state.counter = max(state.counter - 1, 0)
        if state.counter == 0:
          state.active = False
      else:
        if signal:
          state.counter = min(state.counter + 1, activation_frames)
        else:
          state.counter = max(state.counter - 1, 0)
        if state.counter >= activation_frames:
          state.active = True

      if state.active:
        any_active = True

    stale_ids = [face_id for face_id in self._gaze_states if face_id not in active_ids]
    for face_id in stale_ids:
      del self._gaze_states[face_id]

    return any_active

  @staticmethod
  def _context_dimensions(
    context: Dict[str, Any],
    tracked_faces: Optional[List[dict]] = None,
  ) -> Tuple[float, float, float, float]:
    frame_w = float(context.get("frameWidth", 0.0) or 0.0)
    frame_h = float(context.get("frameHeight", 0.0) or 0.0)
    if (frame_w <= 1.0 or frame_h <= 1.0) and tracked_faces:
      inferred_w = 1.0
      inferred_h = 1.0
      for face in tracked_faces:
        bbox = face.get("bbox", {}) or {}
        x = float(bbox.get("x", 0.0) or 0.0)
        y = float(bbox.get("y", 0.0) or 0.0)
        w = float(bbox.get("w", 0.0) or 0.0)
        h = float(bbox.get("h", 0.0) or 0.0)
        inferred_w = max(inferred_w, x + max(w, 1.0))
        inferred_h = max(inferred_h, y + max(h, 1.0))
      frame_w = max(frame_w, inferred_w)
      frame_h = max(frame_h, inferred_h)

    frame_w = max(frame_w, 1.0)
    frame_h = max(frame_h, 1.0)

    display_w = float(context.get("displayWidth", 0.0) or 0.0)
    display_h = float(context.get("displayHeight", 0.0) or 0.0)
    if display_w <= 0.0 or display_h <= 0.0:
      viewport_w = float(context.get("viewportWidth", 0.0) or 0.0)
      viewport_h = float(context.get("viewportHeight", 0.0) or 0.0)
      if viewport_w > 0.0 and viewport_h > 0.0:
        display_w, display_h = viewport_w, viewport_h
      else:
        display_w, display_h = frame_w, frame_h

    return frame_w, frame_h, max(display_w, 1.0), max(display_h, 1.0)

  def _detect_vertical_head_motion(self, tracked_faces: List[dict], frame_height: float) -> bool:
    if not tracked_faces:
      self._vertical_states.clear()
      return False

    any_active = False
    now = monotonic()
    active_ids = set()
    ema_alpha = self._threshold("vertical_head_pitch_ema_alpha", settings.vertical_head_pitch_ema_alpha)
    baseline_alpha = self._threshold("vertical_head_pitch_baseline_alpha", settings.vertical_head_pitch_baseline_alpha)
    active_baseline_alpha = self._threshold("vertical_head_pitch_active_baseline_alpha", settings.vertical_head_pitch_active_baseline_alpha)
    baseline_movement_th = self._threshold("vertical_head_baseline_movement_threshold", settings.vertical_head_baseline_movement_threshold)
    baseline_yaw_th = self._threshold("vertical_head_baseline_yaw_threshold", settings.vertical_head_baseline_yaw_threshold)
    up_threshold = self._threshold("vertical_pitch_ratio_up_threshold", settings.vertical_pitch_ratio_up_threshold)
    down_threshold = self._threshold("vertical_pitch_ratio_down_threshold", settings.vertical_pitch_ratio_down_threshold)
    eye_visibility_threshold = self._threshold("eye_visibility_threshold", 0.075)
    pitch_ratio_dead_zone = self._threshold("vertical_head_pitch_ratio_dead_zone", settings.vertical_head_pitch_ratio_dead_zone)
    low_visibility_persist_sec = self._threshold(
      "vertical_head_low_visibility_persist_sec",
      settings.vertical_head_low_visibility_persist_sec,
    )
    evidence_ratio = float(settings.vertical_head_evidence_ratio)
    min_evidence_frames = max(6, int(settings.vertical_head_evidence_frames // 2))

    for face in tracked_faces:
      face_id = int(face.get("id", -1))
      if face_id < 0:
        continue
      active_ids.add(face_id)

      pose = face.get("headPose", {}) or {}
      yaw = float(pose.get("yaw", 0.0))
      pitch_ratio = float(face.get("pitchRatio", 0.5))
      eye_visible = bool(face.get("eyeVisible", False))
      eye_visibility = float(face.get("eyeVisibility", 0.0))
      movement = float(face.get("movement", 0.0))
      frontalness = float(face.get("frontalness", face.get("faceFrontalness", 1.0)) or 0.0)
      landmark_stability = float(face.get("landmarkStability", 1.0) or 0.0)
      face_confidence = float(face.get("faceConfidence", 1.0) or 0.0)

      state = self._vertical_states.setdefault(face_id, _VerticalPoseState())
      if not state.initialized:
        state.smooth_pitch_ratio = pitch_ratio
        state.baseline_pitch_ratio = pitch_ratio
        state.prev_pitch_ratio = pitch_ratio
        state.initialized = True
      else:
        state.smooth_pitch_ratio = (1.0 - ema_alpha) * state.smooth_pitch_ratio + ema_alpha * pitch_ratio

      baseline_adapt_allowed = (
        movement <= baseline_movement_th
        and abs(yaw) <= baseline_yaw_th
        and eye_visible
        and eye_visibility >= eye_visibility_threshold
      )
      if baseline_adapt_allowed:
        alpha = active_baseline_alpha if state.active else baseline_alpha
        state.baseline_pitch_ratio = (1.0 - alpha) * state.baseline_pitch_ratio + alpha * state.smooth_pitch_ratio

      pitch_ratio_delta = state.smooth_pitch_ratio - state.baseline_pitch_ratio
      in_dead_zone = abs(pitch_ratio_delta) < pitch_ratio_dead_zone
      signal_confidence = float(
        max(
          0.0,
          min(
            1.0,
            (0.35 * landmark_stability)
            + (0.25 * max(0.0, min(eye_visibility, 1.0)))
            + (0.20 * max(0.0, min(frontalness, 1.0)))
            + (0.20 * max(0.0, min(face_confidence, 1.0))),
          ),
        )
      )
      upward_signal = (not in_dead_zone) and state.smooth_pitch_ratio <= up_threshold
      downward_signal = (not in_dead_zone) and state.smooth_pitch_ratio >= down_threshold
      low_visibility = (not eye_visible) or eye_visibility < eye_visibility_threshold
      if low_visibility and downward_signal:
        if state.low_visibility_since is None:
          state.low_visibility_since = now
      else:
        state.low_visibility_since = None
      downward_fallback = (
        low_visibility
        and downward_signal
        and state.low_visibility_since is not None
        and (now - state.low_visibility_since) >= low_visibility_persist_sec
      )
      deviation_strength = 0.0
      if upward_signal:
        deviation_strength = max(
          deviation_strength,
          abs(state.smooth_pitch_ratio - up_threshold) / max(abs(state.baseline_pitch_ratio - up_threshold), 1e-6),
        )
      if downward_signal:
        deviation_strength = max(
          deviation_strength,
          abs(state.smooth_pitch_ratio - down_threshold) / max(abs(down_threshold - state.baseline_pitch_ratio), 1e-6),
        )
      attention_confidence = deviation_strength * signal_confidence
      evidence = 1.0 if (
        signal_confidence >= float(settings.attention_signal_confidence_min)
        and attention_confidence >= float(settings.attention_warning_confidence_min)
        and (upward_signal or downward_fallback or (downward_signal and not low_visibility))
      ) else 0.0
      state.evidence.append(evidence)
      state.prev_pitch_ratio = state.smooth_pitch_ratio
      evidence_score = float(sum(state.evidence) / max(float(len(state.evidence)), 1.0))
      state.active = len(state.evidence) >= min_evidence_frames and evidence_score >= evidence_ratio

      if state.active:
        any_active = True

    stale_ids = [face_id for face_id in self._vertical_states if face_id not in active_ids]
    for face_id in stale_ids:
      del self._vertical_states[face_id]

    return any_active

  def _detect_horizontal_head_motion(self, tracked_faces: List[dict], frame_width: float) -> bool:
    if not tracked_faces:
      self._horizontal_states.clear()
      return False

    any_active = False
    active_ids = set()

    activation_frames = max(
      1,
      int(round(self._threshold("horizontal_head_activation_frames", settings.horizontal_head_activation_frames))),
    )
    yaw_alpha = self._threshold("horizontal_head_yaw_ema_alpha", settings.horizontal_head_yaw_ema_alpha)
    yaw_vel_gate = self._threshold("horizontal_head_yaw_velocity_gate", settings.horizontal_head_yaw_velocity_gate)
    x_vel_gate = self._threshold("horizontal_head_x_velocity_gate", settings.horizontal_head_x_velocity_gate)
    movement_gate = self._threshold("horizontal_head_movement_gate", settings.horizontal_head_movement_gate)
    yaw_gate = self._threshold("horizontal_head_yaw_gate", settings.horizontal_head_yaw_gate)
    exit_vel_gate = self._threshold("horizontal_head_exit_velocity_gate", settings.horizontal_head_exit_velocity_gate)
    span_window_frames = max(
      3,
      int(round(self._threshold("horizontal_head_span_window_frames", settings.horizontal_head_span_window_frames))),
    )
    yaw_span_gate = self._threshold("horizontal_head_yaw_span_gate", settings.horizontal_head_yaw_span_gate)
    x_span_gate = self._threshold("horizontal_head_x_span_gate", settings.horizontal_head_x_span_gate)
    large_movement_gate = self._threshold("horizontal_head_large_movement_gate", settings.horizontal_head_large_movement_gate)
    large_x_span_gate = self._threshold("horizontal_head_large_x_span_gate", settings.horizontal_head_large_x_span_gate)

    for face in tracked_faces:
      face_id = int(face.get("id", -1))
      if face_id < 0:
        continue
      active_ids.add(face_id)

      pose = face.get("headPose", {}) or {}
      yaw = float(pose.get("yaw", 0.0))
      movement = float(face.get("movement", 0.0))
      bbox = face.get("bbox", {}) or {}
      x = float(bbox.get("x", 0.0))
      w = float(max(float(bbox.get("w", 1.0)), 1.0))
      center_x = x + (w * 0.5)
      center_x_norm = center_x / max(frame_width, 1.0)

      state = self._horizontal_states.setdefault(face_id, _HorizontalMotionState())
      if not state.initialized:
        state.smooth_yaw = yaw
        state.prev_smooth_yaw = yaw
        state.yaw_velocity_ema = 0.0
        state.prev_center_x = center_x
        state.x_velocity_ema = 0.0
        state.yaw_window.clear()
        state.yaw_window.append(yaw)
        state.center_window.clear()
        state.center_window.append(center_x_norm)
        state.active_counter = 0
        state.active = False
        state.initialized = True
      else:
        state.smooth_yaw = (1.0 - yaw_alpha) * state.smooth_yaw + yaw_alpha * yaw
        yaw_vel = abs(state.smooth_yaw - state.prev_smooth_yaw)
        state.yaw_velocity_ema = 0.75 * state.yaw_velocity_ema + 0.25 * yaw_vel
        state.prev_smooth_yaw = state.smooth_yaw

        x_vel = abs(center_x - state.prev_center_x) / max(frame_width, 1.0)
        state.x_velocity_ema = 0.75 * state.x_velocity_ema + 0.25 * x_vel
        state.prev_center_x = center_x
        state.yaw_window.append(state.smooth_yaw)
        state.center_window.append(center_x_norm)
        while len(state.yaw_window) > span_window_frames:
          state.yaw_window.popleft()
        while len(state.center_window) > span_window_frames:
          state.center_window.popleft()

      yaw_span = 0.0
      if len(state.yaw_window) >= 2:
        yaw_span = max(state.yaw_window) - min(state.yaw_window)

      x_span = 0.0
      if len(state.center_window) >= 2:
        x_span = max(state.center_window) - min(state.center_window)
      lateral_displacement = 0.0
      if state.center_window:
        lateral_displacement = abs(center_x_norm - state.center_window[0])

      yaw_signal = (
        yaw_span >= yaw_span_gate
        and (
          state.yaw_velocity_ema >= yaw_vel_gate
          or abs(state.smooth_yaw) >= (yaw_gate * 1.12)
        )
        and abs(state.smooth_yaw) >= (yaw_gate * 0.72)
      )
      x_signal = (
        x_span >= x_span_gate
        and movement >= (movement_gate * 0.78)
        and (
          state.x_velocity_ema >= x_vel_gate
          or lateral_displacement >= (large_x_span_gate * 0.72)
          or movement >= large_movement_gate
        )
      )
      activation_signal = (
        yaw_signal
        or x_signal
        or (
          movement >= large_movement_gate
          and x_span >= (x_span_gate * 0.92)
        )
      )
      deactivation_signal = (
        state.yaw_velocity_ema <= exit_vel_gate
        and state.x_velocity_ema <= (exit_vel_gate * 0.65)
        and yaw_span <= yaw_span_gate * 0.45
        and x_span <= x_span_gate * 0.45
      )

      if state.active:
        if deactivation_signal:
          state.active_counter = max(state.active_counter - 1, 0)
        else:
          state.active_counter = min(state.active_counter + 1, activation_frames)
        if state.active_counter == 0:
          state.active = False
      else:
        if activation_signal:
          state.active_counter = min(state.active_counter + 1, activation_frames)
        else:
          state.active_counter = max(state.active_counter - 1, 0)
        if state.active_counter >= activation_frames:
          state.active = True

      if state.active:
        any_active = True

    stale_ids = [face_id for face_id in self._horizontal_states if face_id not in active_ids]
    for face_id in stale_ids:
      del self._horizontal_states[face_id]

    return any_active

  def _detect_face_presence_risks(
    self,
    face_count: int,
    tracked_faces: List[dict],
    has_vertical_head_motion: bool,
    has_horizontal_head_motion: bool,
    attention_snapshot: Optional[AttentionSnapshot] = None,
  ) -> bool:
    state = self._presence_state
    missing_frames_th = max(
      1,
      int(round(self._threshold("face_missing_activation_frames", settings.face_missing_activation_frames))),
    )
    missing_evasive_th = max(
      1,
      int(round(self._threshold("face_missing_evasive_activation_frames", settings.face_missing_evasive_activation_frames))),
    )
    hide_motion_gate = self._threshold("face_hide_motion_gate", settings.face_hide_motion_gate)
    off_camera_activation = max(
      1,
      int(round(self._threshold("face_off_camera_activation_frames", settings.face_off_camera_activation_frames))),
    )
    off_camera_memory = max(
      1,
      int(round(self._threshold("face_off_camera_memory_frames", settings.face_off_camera_memory_frames))),
    )

    if face_count <= 0 or not tracked_faces:
      state.missing_frames += 1
      evasive_context = (
        state.hidden_intent_frames > 0
        or (
          attention_snapshot is not None
          and (
            attention_snapshot.orientation_state != AttentionState.FOCUSED
            or attention_snapshot.horizontal_state != AttentionState.FOCUSED
          )
        )
      )
      has_missing = state.missing_frames >= missing_frames_th
      has_evasive_missing = evasive_context and state.missing_frames >= missing_evasive_th
      state.hidden_intent_frames = max(state.hidden_intent_frames - 1, 0)
      state.off_camera_counter = max(state.off_camera_counter - 1, 0)
      return has_missing or has_evasive_missing

    state.missing_frames = 0
    primary = self._primary_face(tracked_faces)
    if primary is None:
      return False

    movement = float(primary.get("movement", 0.0))
    evasive_movement_now = (
      movement >= hide_motion_gate
      and (
        has_vertical_head_motion
        or has_horizontal_head_motion
        or (
          attention_snapshot is not None
          and (
            attention_snapshot.orientation_state != AttentionState.FOCUSED
            or attention_snapshot.horizontal_state != AttentionState.FOCUSED
          )
        )
      )
    )
    if evasive_movement_now:
      state.hidden_intent_frames = max(state.hidden_intent_frames, off_camera_memory)
    else:
      state.hidden_intent_frames = max(state.hidden_intent_frames - 1, 0)

    if attention_snapshot is not None and attention_snapshot.orientation_state == AttentionState.LOOKING_AWAY:
      state.hidden_intent_frames = max(state.hidden_intent_frames, max(2, off_camera_activation // 2))

    return False

  def _attention_thresholds(self) -> AttentionThresholds:
    horizontal_enter_yaw = self._threshold("horizontal_head_yaw_gate", settings.horizontal_head_yaw_gate)
    horizontal_enter_x = self._threshold("horizontal_head_x_span_gate", settings.horizontal_head_x_span_gate)
    orientation_enter_yaw = self._threshold("face_off_camera_yaw_gate", settings.face_off_camera_yaw_gate)
    orientation_enter_pitch = self._threshold("face_off_camera_pitch_gate", settings.face_off_camera_pitch_gate)
    gaze_enter_h = self._threshold("gaze_extreme_horizontal_threshold", settings.gaze_extreme_horizontal_threshold)
    gaze_enter_v = self._threshold("gaze_extreme_vertical_threshold", settings.gaze_extreme_vertical_threshold)

    return AttentionThresholds(
      horizontal_enter_yaw=float(horizontal_enter_yaw),
      horizontal_exit_yaw=max(
        self._threshold("horizontal_head_yaw_gate", settings.horizontal_head_yaw_gate) * 0.66,
        self._threshold("horizontal_head_yaw_gate", settings.horizontal_head_yaw_gate) - 4.0,
      ),
      horizontal_enter_x=float(horizontal_enter_x),
      horizontal_exit_x=max(float(horizontal_enter_x) * 0.68, float(horizontal_enter_x) - 0.025),
      orientation_enter_yaw=float(orientation_enter_yaw),
      orientation_exit_yaw=float(
        self._threshold("face_off_camera_yaw_exit_gate", settings.face_off_camera_yaw_exit_gate)
      ),
      orientation_enter_pitch=float(orientation_enter_pitch),
      orientation_exit_pitch=float(
        self._threshold("face_off_camera_pitch_exit_gate", settings.face_off_camera_pitch_exit_gate)
      ),
      gaze_enter_h=float(gaze_enter_h),
      gaze_exit_h=max(float(gaze_enter_h) * 0.72, float(gaze_enter_h) - 0.018),
      gaze_enter_v=float(gaze_enter_v),
      gaze_exit_v=max(float(gaze_enter_v) * 0.72, float(gaze_enter_v) - 0.02),
      gaze_head_yaw=float(self._threshold("gaze_face_yaw_gate", settings.gaze_face_yaw_gate)),
      gaze_head_pitch=float(self._threshold("gaze_face_pitch_gate", settings.gaze_face_pitch_gate)),
      gaze_min_confidence=float(self._threshold("gaze_extreme_min_confidence", settings.gaze_extreme_min_confidence)),
    )

  def _annotate_attention(self, tracked_faces: List[dict], snapshot: AttentionSnapshot) -> None:
    if not tracked_faces:
      return
    for face in tracked_faces:
      face["attentionState"] = AttentionState.FOCUSED
      face["signalConfidence"] = 0.0
      face["attentionConfidence"] = 0.0
      face["attentionScore"] = 0.0
      face["baselineYaw"] = None
      face["baselinePitch"] = None
      face["deviationScore"] = 0.0
      face["fastDeviationTriggered"] = False
      face["yawNoise"] = 0.0
      face["pitchNoise"] = 0.0
      face["gazeNoise"] = 0.0
      face["landmarkStability"] = round(float(face.get("landmarkStability", 1.0)), 4)
      face["horizontalAttentionState"] = AttentionState.FOCUSED
      face["gazeAttentionState"] = AttentionState.FOCUSED
      face["orientationAttentionState"] = AttentionState.FOCUSED

    for face in tracked_faces:
      if int(face.get("id", -1)) != int(snapshot.primary_face_id):
        continue
      face["attentionState"] = snapshot.state
      face["signalConfidence"] = round(float(snapshot.signal_confidence), 4)
      face["attentionConfidence"] = round(float(snapshot.attention_confidence), 4)
      face["attentionScore"] = round(float(snapshot.attention_score), 4)
      face["baselineYaw"] = round(float(snapshot.baseline_yaw), 3)
      face["baselinePitch"] = round(float(snapshot.baseline_pitch), 3)
      face["deviationScore"] = round(float(snapshot.deviation_score), 4)
      face["fastDeviationTriggered"] = bool(snapshot.fast_deviation_triggered)
      face["yawNoise"] = round(float(snapshot.yaw_noise), 4)
      face["pitchNoise"] = round(float(snapshot.pitch_noise), 4)
      face["gazeNoise"] = round(float(snapshot.gaze_noise), 5)
      face["landmarkStability"] = round(float(snapshot.landmark_stability), 4)
      face["horizontalAttentionState"] = snapshot.horizontal_state
      face["gazeAttentionState"] = snapshot.gaze_state
      face["orientationAttentionState"] = snapshot.orientation_state
      break

  @staticmethod
  def _primary_face(tracked_faces: List[dict]) -> Optional[dict]:
    if not tracked_faces:
      return None
    return max(
      tracked_faces,
      key=lambda face: float(max(float((face.get("bbox", {}) or {}).get("w", 0.0)), 0.0))
      * float(max(float((face.get("bbox", {}) or {}).get("h", 0.0)), 0.0)),
    )

  def _empty_breakdown(self) -> Dict[str, Any]:
    empty_states = [
      {
        "code": code,
        "label": self._labels.get(code, code),
        "active": False,
        "baseSeverity": round(float(severity), 3),
        "streak": 0,
        "repetitionFactor": 0.0,
        "multiplier": 1.0,
        "points": 0.0,
      }
      for code, severity in self._severity.items()
    ]
    return {
      "rawScore": 0.0,
      "emaScore": 0.0,
      "smoothedScore": 0.0,
      "riskLevel": "LOW",
      "activeRules": [],
      "ruleStates": empty_states,
      "dominantRule": None,
      "adaptiveThresholds": {},
      "attentionModel": {},
    }

  def _build_breakdown(
    self,
    rule_states: List[Dict[str, Any]],
    raw_score: float,
    ema_score: float,
    smoothed_score: float,
    level: str,
  ) -> Dict[str, Any]:
    active_rules = [state for state in rule_states if bool(state["active"])]
    active_rules.sort(key=lambda item: float(item.get("points", 0.0)), reverse=True)
    dominant_rule = active_rules[0]["code"] if active_rules else None
    return {
      "rawScore": round(float(raw_score), 3),
      "emaScore": round(float(ema_score), 3),
      "smoothedScore": round(float(smoothed_score), 3),
      "riskLevel": level,
      "activeRules": active_rules,
      "ruleStates": rule_states,
      "dominantRule": dominant_rule,
      "adaptiveThresholds": self.get_adaptive_thresholds(),
      "attentionModel": dict(self._last_attention_debug),
    }

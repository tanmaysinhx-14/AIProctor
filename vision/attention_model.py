from collections import deque
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from config import settings


class AttentionState:
  FOCUSED = "FOCUSED"
  TRANSITIONING = "TRANSITIONING"
  LOOKING_AWAY = "LOOKING_AWAY"


@dataclass(frozen=True)
class AttentionThresholds:
  horizontal_enter_yaw: float
  horizontal_exit_yaw: float
  horizontal_enter_x: float
  horizontal_exit_x: float
  orientation_enter_yaw: float
  orientation_exit_yaw: float
  orientation_enter_pitch: float
  orientation_exit_pitch: float
  gaze_enter_h: float
  gaze_exit_h: float
  gaze_enter_v: float
  gaze_exit_v: float
  gaze_head_yaw: float
  gaze_head_pitch: float
  gaze_min_confidence: float


@dataclass
class AttentionSnapshot:
  state: str = AttentionState.FOCUSED
  signal_confidence: float = 1.0
  attention_confidence: float = 0.0
  face_confidence: float = 1.0
  attention_score: float = 0.0
  deviation_score: float = 0.0
  fast_deviation_triggered: bool = False
  baseline_yaw: float = 0.0
  baseline_pitch: float = 0.0
  baseline_gaze_x: float = 0.0
  baseline_gaze_y: float = 0.0
  gaze_left_boundary: float = 0.0
  gaze_right_boundary: float = 0.0
  gaze_top_boundary: float = 0.0
  gaze_bottom_boundary: float = 0.0
  pitch_ratio_up_threshold: float = 0.0
  pitch_ratio_down_threshold: float = 0.0
  yaw_noise: float = 0.0
  pitch_noise: float = 0.0
  gaze_noise: float = 0.0
  landmark_stability: float = 1.0
  horizontal_state: str = AttentionState.FOCUSED
  horizontal_score: float = 0.0
  horizontal_deviation: float = 0.0
  gaze_state: str = AttentionState.FOCUSED
  gaze_score: float = 0.0
  gaze_deviation: float = 0.0
  orientation_state: str = AttentionState.FOCUSED
  orientation_score: float = 0.0
  orientation_deviation: float = 0.0
  yaw: float = 0.0
  pitch: float = 0.0
  roll: float = 0.0
  pitch_ratio: float = 0.5
  gaze_x: float = 0.0
  gaze_y: float = 0.0
  eye_visibility: float = 0.0
  frontalness: float = 1.0
  yaw_velocity: float = 0.0
  pitch_velocity: float = 0.0
  fused_attention_x: float = 0.0
  fused_attention_y: float = 0.0
  primary_face_id: int = -1
  deviation_frames: int = 0
  window_frames: int = 0

  def as_dict(self) -> Dict[str, Any]:
    return {
      "state": self.state,
      "confidence": round(float(self.signal_confidence), 4),
      "signalConfidence": round(float(self.signal_confidence), 4),
      "attention_confidence": round(float(self.attention_confidence), 4),
      "attentionConfidence": round(float(self.attention_confidence), 4),
      "faceConfidence": round(float(self.face_confidence), 4),
      "attentionScore": round(float(self.attention_score), 4),
      "deviationScore": round(float(self.deviation_score), 4),
      "fastDeviationTriggered": bool(self.fast_deviation_triggered),
      "baselineYaw": round(float(self.baseline_yaw), 3),
      "baselinePitch": round(float(self.baseline_pitch), 3),
      "baselineGazeX": round(float(self.baseline_gaze_x), 5),
      "baselineGazeY": round(float(self.baseline_gaze_y), 5),
      "gazeLeftBoundary": round(float(self.gaze_left_boundary), 5),
      "gazeRightBoundary": round(float(self.gaze_right_boundary), 5),
      "gazeTopBoundary": round(float(self.gaze_top_boundary), 5),
      "gazeBottomBoundary": round(float(self.gaze_bottom_boundary), 5),
      "pitchRatioUpThreshold": round(float(self.pitch_ratio_up_threshold), 5),
      "pitchRatioDownThreshold": round(float(self.pitch_ratio_down_threshold), 5),
      "yawNoise": round(float(self.yaw_noise), 4),
      "pitchNoise": round(float(self.pitch_noise), 4),
      "gazeNoise": round(float(self.gaze_noise), 5),
      "landmarkStability": round(float(self.landmark_stability), 4),
      "horizontalState": self.horizontal_state,
      "horizontalScore": round(float(self.horizontal_score), 4),
      "horizontalDeviation": round(float(self.horizontal_deviation), 4),
      "gazeState": self.gaze_state,
      "gazeScore": round(float(self.gaze_score), 4),
      "gazeDeviation": round(float(self.gaze_deviation), 4),
      "orientationState": self.orientation_state,
      "orientationScore": round(float(self.orientation_score), 4),
      "orientationDeviation": round(float(self.orientation_deviation), 4),
      "yaw": round(float(self.yaw), 3),
      "pitch": round(float(self.pitch), 3),
      "roll": round(float(self.roll), 3),
      "pitchRatio": round(float(self.pitch_ratio), 4),
      "gazeX": round(float(self.gaze_x), 4),
      "gazeY": round(float(self.gaze_y), 4),
      "eyeVisibility": round(float(self.eye_visibility), 4),
      "frontalness": round(float(self.frontalness), 4),
      "yawVelocity": round(float(self.yaw_velocity), 4),
      "pitchVelocity": round(float(self.pitch_velocity), 4),
      "fusedAttentionX": round(float(self.fused_attention_x), 4),
      "fusedAttentionY": round(float(self.fused_attention_y), 4),
      "primaryFaceId": int(self.primary_face_id),
      "deviationFrames": int(self.deviation_frames),
      "windowFrames": int(self.window_frames),
    }


@dataclass
class _WindowSample:
  ts: float
  horizontal_flag: float
  gaze_flag: float
  orientation_flag: float
  horizontal_deviation: float
  gaze_deviation: float
  orientation_deviation: float


class _ScalarEma:
  def __init__(self, alpha: float = 0.3) -> None:
    self._alpha = float(np.clip(alpha, 0.05, 0.95))
    self._initialized = False
    self._value = 0.0

  def reset(self) -> None:
    self._initialized = False
    self._value = 0.0

  def update(self, value: float) -> float:
    if not self._initialized:
      self._value = float(value)
      self._initialized = True
      return float(value)
    self._value = ((1.0 - self._alpha) * self._value) + (self._alpha * float(value))
    return float(self._value)


@dataclass
class _RuleState:
  state: str = AttentionState.FOCUSED


@dataclass
class _FaceState:
  yaw_filter: _ScalarEma = field(default_factory=_ScalarEma)
  pitch_filter: _ScalarEma = field(default_factory=_ScalarEma)
  roll_filter: _ScalarEma = field(default_factory=_ScalarEma)
  samples: Deque[_WindowSample] = field(default_factory=deque)
  horizontal_evidence: Deque[float] = field(
    default_factory=lambda: deque(maxlen=max(3, int(settings.attention_evidence_frames)))
  )
  gaze_evidence: Deque[float] = field(
    default_factory=lambda: deque(maxlen=max(3, int(settings.attention_evidence_frames)))
  )
  orientation_evidence: Deque[float] = field(
    default_factory=lambda: deque(maxlen=max(3, int(settings.attention_evidence_frames)))
  )
  horizontal_rule: _RuleState = field(default_factory=_RuleState)
  gaze_rule: _RuleState = field(default_factory=_RuleState)
  orientation_rule: _RuleState = field(default_factory=_RuleState)
  baseline_yaw: float = 0.0
  baseline_pitch: float = 0.0
  baseline_gaze_x: float = 0.0
  baseline_gaze_y: float = 0.0
  baseline_center_x: float = 0.5
  initialized: bool = False
  prev_yaw: float = 0.0
  prev_pitch: float = 0.0
  prev_roll: float = 0.0
  prev_raw_yaw: float = 0.0
  prev_raw_pitch: float = 0.0
  last_landmark_stability: float = 1.0
  last_ts: float = 0.0


class AttentionModel:
  def __init__(self) -> None:
    self._states: Dict[int, _FaceState] = {}
    self._profile: Dict[str, float] = {}
    self._last_snapshot = AttentionSnapshot()

  def reset(self) -> None:
    self._states.clear()
    self._profile = {}
    self._last_snapshot = AttentionSnapshot()

  def set_calibration_profile(self, profile: Dict[str, Any]) -> None:
    attention_profile = profile.get("attentionModel") or {}
    if not isinstance(attention_profile, dict):
      self._profile = {}
      return
    parsed: Dict[str, float] = {}
    for key, value in attention_profile.items():
      try:
        parsed[key] = float(value)
      except (TypeError, ValueError):
        continue
    self._profile = parsed

  def current(self) -> AttentionSnapshot:
    return self._last_snapshot

  def update(
    self,
    tracked_faces: List[Dict[str, Any]],
    frame_width: float,
    frame_height: float,
    thresholds: AttentionThresholds,
  ) -> AttentionSnapshot:
    if not tracked_faces:
      self._states.clear()
      self._last_snapshot = AttentionSnapshot()
      return self._last_snapshot

    primary = self._primary_face(tracked_faces)
    if primary is None:
      self._last_snapshot = AttentionSnapshot()
      return self._last_snapshot

    face_id = int(primary.get("id", -1))
    if face_id < 0:
      self._last_snapshot = AttentionSnapshot()
      return self._last_snapshot

    state = self._states.setdefault(face_id, _FaceState())
    now = monotonic()
    pose = primary.get("headPose", {}) or {}
    yaw_raw = float(pose.get("yaw", 0.0))
    pitch_raw = float(pose.get("pitch", 0.0))
    roll_raw = float(pose.get("roll", 0.0))
    yaw = state.yaw_filter.update(yaw_raw)
    pitch = state.pitch_filter.update(pitch_raw)
    roll = state.roll_filter.update(roll_raw)

    bbox = primary.get("bbox", {}) or {}
    center_x = (float(bbox.get("x", 0.0)) + (float(bbox.get("w", 0.0)) * 0.5)) / max(frame_width, 1.0)
    gaze_x = float(primary.get("gazeX", primary.get("gazeOffsetX", 0.0)))
    gaze_y = float(primary.get("gazeY", primary.get("gazeOffsetY", 0.0)))
    gaze_conf = float(primary.get("gazeConfidence", 0.0))
    eye_visible = bool(primary.get("eyeVisible", False))
    eye_visibility = float(np.clip(float(primary.get("eyeVisibility", 0.0)), 0.0, 1.0))
    movement = float(primary.get("movement", 0.0))
    landmark_stability = float(np.clip(float(primary.get("landmarkStability", 1.0)), 0.0, 1.0))
    eye_symmetry = float(np.clip(float(primary.get("eyeSymmetry", 0.0)), 0.0, 2.0))
    pitch_ratio = float(np.clip(float(primary.get("pitchRatio", 0.5)), 0.0, 1.5))
    PITCH_DEADZONE = 0.04
    if abs(pitch_ratio - state.baseline_pitch) < PITCH_DEADZONE:
      pitch_ratio = state.baseline_pitch
    frontalness = float(np.clip(float(primary.get("frontalness", primary.get("faceFrontalness", 1.0))), 0.0, 1.0))
    track_persistence = max(1.0, float(primary.get("trackPersistenceFrames", 1.0)))
    face_confidence = float(
      np.clip(
        float(primary.get("faceConfidence", min(1.0, 0.35 + (0.13 * track_persistence)))),
        0.0,
        1.0,
      )
    )
    signal_confidence = float(
      np.clip(
        (0.35 * landmark_stability)
        + (0.25 * eye_visibility)
        + (0.20 * frontalness)
        + (0.20 * face_confidence),
        0.0,
        1.0,
      )
    )

    if not state.initialized:
      state.baseline_yaw = self._profile_value("yawMedian", self._profile_value("yawMean", yaw))
      state.baseline_pitch = self._profile_value("pitchMedian", self._profile_value("pitchMean", pitch))
      state.baseline_gaze_x = self._profile_value("gazeCenterX", self._profile_value("gazeMedianX", self._profile_value("gazeMeanX", gaze_x)))
      state.baseline_gaze_y = self._profile_value("gazeCenterY", self._profile_value("gazeMedianY", self._profile_value("gazeMeanY", gaze_y)))
      state.baseline_center_x = center_x
      state.prev_yaw = yaw
      state.prev_pitch = pitch
      state.prev_roll = roll
      state.prev_raw_yaw = yaw_raw
      state.prev_raw_pitch = pitch_raw
      state.last_landmark_stability = landmark_stability
      state.last_ts = now
      state.initialized = True

    effective_thresholds = self._effective_thresholds(thresholds)
    yaw_noise = max(self._profile_value("yawNoise", 0.0), self._profile_value("yawNoiseFloor", 0.0) * 0.6)
    pitch_noise = max(self._profile_value("pitchNoise", 0.0), self._profile_value("pitchNoiseFloor", 0.0) * 0.6)
    gaze_noise = max(
      self._profile_value("gazeNoise", 0.0),
      self._profile_value("gazeNoiseX", 0.0),
      self._profile_value("gazeNoiseY", 0.0),
    )

    yaw_velocity = abs(yaw - state.prev_yaw)
    pitch_velocity = abs(pitch - state.prev_pitch)
    fast_yaw_delta = abs(yaw_raw - state.prev_raw_yaw)
    fast_pitch_delta = abs(pitch_raw - state.prev_raw_pitch)
    state.prev_yaw = yaw
    state.prev_pitch = pitch
    state.prev_roll = roll
    state.prev_raw_yaw = yaw_raw
    state.prev_raw_pitch = pitch_raw
    state.last_landmark_stability = landmark_stability
    state.last_ts = now

    if self._allow_baseline_update(
      state=state,
      yaw=yaw,
      pitch=pitch,
      gaze_x=gaze_x,
      gaze_y=gaze_y,
      center_x=center_x,
      movement=movement,
      eye_visible=eye_visible,
      landmark_stability=landmark_stability,
      eye_symmetry=eye_symmetry,
      yaw_velocity=yaw_velocity,
      pitch_velocity=pitch_velocity,
      thresholds=effective_thresholds,
    ):
      alpha = float(settings.attention_baseline_alpha)
      state.baseline_yaw = (1.0 - alpha) * state.baseline_yaw + alpha * yaw
      state.baseline_pitch = (1.0 - alpha) * state.baseline_pitch + alpha * pitch
      if "gazeCenterX" in self._profile or "gazeLeftBoundary" in self._profile:
        state.baseline_gaze_x = self._profile_value("gazeCenterX", state.baseline_gaze_x)
      else:
        state.baseline_gaze_x = (1.0 - alpha) * state.baseline_gaze_x + alpha * gaze_x
      if "gazeCenterY" in self._profile or "gazeTopBoundary" in self._profile:
        state.baseline_gaze_y = self._profile_value("gazeCenterY", state.baseline_gaze_y)
      else:
        state.baseline_gaze_y = (1.0 - alpha) * state.baseline_gaze_y + alpha * gaze_y
      state.baseline_center_x = (1.0 - alpha) * state.baseline_center_x + alpha * center_x

    yaw_delta = yaw - state.baseline_yaw
    pitch_delta = pitch - state.baseline_pitch
    center_x_delta = center_x - state.baseline_center_x
    gaze_x_delta = gaze_x - state.baseline_gaze_x
    gaze_y_delta = gaze_y - state.baseline_gaze_y
    gaze_left_boundary = self._profile_value("gazeLeftBoundary", state.baseline_gaze_x - float(effective_thresholds.gaze_enter_h))
    gaze_right_boundary = self._profile_value("gazeRightBoundary", state.baseline_gaze_x + float(effective_thresholds.gaze_enter_h))
    gaze_top_boundary = self._profile_value("gazeTopBoundary", state.baseline_gaze_y - float(effective_thresholds.gaze_enter_v))
    gaze_bottom_boundary = self._profile_value("gazeBottomBoundary", state.baseline_gaze_y + float(effective_thresholds.gaze_enter_v))

    yaw_dev = abs(yaw_delta)
    x_dev = abs(center_x_delta)

    if yaw_dev < 3.0:
      yaw_dev = 0.0
    if x_dev < 0.015:
      x_dev = 0.0

    horizontal_deviation = max(
      yaw_dev / max(effective_thresholds.horizontal_enter_yaw, 1e-6),
      x_dev / max(effective_thresholds.horizontal_enter_x, 1e-6),
    )
    orientation_deviation = max(
      abs(yaw) / max(effective_thresholds.orientation_enter_yaw, 1e-6),
      eye_symmetry / max(float(settings.face_orientation_eye_symmetry_threshold), 1e-6),
    )
    head_x = yaw_delta / max(effective_thresholds.gaze_head_yaw, 1e-6)
    head_y = pitch_delta / max(effective_thresholds.gaze_head_pitch, 1e-6)
    eye_x = self._boundary_deviation(gaze_x, state.baseline_gaze_x, gaze_left_boundary, gaze_right_boundary)
    eye_y = self._boundary_deviation(gaze_y, state.baseline_gaze_y, gaze_top_boundary, gaze_bottom_boundary)
    fused_x = (settings.attention_head_weight * head_x) + (settings.attention_eye_weight * eye_x)
    fused_y = (settings.attention_head_weight * head_y) + (settings.attention_eye_weight * eye_y)

    if abs(eye_x) < 0.12:
      eye_x = 0.0
    if abs(eye_y) < 0.12:
      eye_y = 0.0

    gaze_deviation = max(abs(eye_x), abs(eye_y) if settings.enable_vertical_gaze_risk else 0.0)
    confidence_allowed = signal_confidence >= float(settings.attention_signal_confidence_min)
    confidence_gate = float(settings.attention_warning_confidence_min)
    horizontal_attention_confidence = horizontal_deviation * signal_confidence
    gaze_attention_confidence = gaze_deviation * signal_confidence
    orientation_attention_confidence = orientation_deviation * signal_confidence

    pose_micro_movement = max(yaw_velocity, pitch_velocity) < float(settings.attention_micro_velocity_deg)
    confidence_ready = eye_visible and (gaze_conf >= effective_thresholds.gaze_min_confidence or gaze_deviation >= 1.12)
    fast_horizontal_trigger = landmark_stability >= float(settings.pose_landmark_stability_min) and (
      fast_yaw_delta >= float(settings.attention_fast_yaw_delta_deg)
    )
    fast_orientation_trigger = landmark_stability >= float(settings.pose_landmark_stability_min) and (
      fast_yaw_delta >= float(settings.attention_fast_yaw_delta_deg)
      or fast_pitch_delta >= float(settings.attention_fast_pitch_delta_deg)
    )
    hard_horizontal_trigger = abs(yaw_delta) >= float(settings.attention_hard_yaw_deg) or abs(yaw) >= float(settings.attention_hard_yaw_deg)
    hard_orientation_trigger = (
      abs(yaw) >= float(settings.attention_hard_yaw_deg)
      or eye_symmetry >= float(settings.face_orientation_eye_symmetry_threshold) * 2.2
    )
    hard_gaze_trigger = max(
      abs(eye_x),
      abs(eye_y) if settings.enable_vertical_gaze_risk else 0.0,
    ) >= float(settings.attention_hard_gaze_offset * 1.4)
    horizontal_flag = (
      1.0
      if confidence_allowed
      and horizontal_attention_confidence >= (confidence_gate * 1.25)
      and (fast_horizontal_trigger or hard_horizontal_trigger or self._deviation_flag(horizontal_deviation, pose_micro_movement) > 0.0)
      else 0.0
    )
    orientation_flag = (
      1.0
      if confidence_allowed
      and orientation_attention_confidence >= confidence_gate
      and (fast_orientation_trigger or hard_orientation_trigger or self._deviation_flag(orientation_deviation, False) > 0.0)
      else 0.0
    )
    gaze_flag = (
      1.0
      if confidence_allowed
      and confidence_ready
      and gaze_attention_confidence >= (confidence_gate * 1.3)
      and (hard_gaze_trigger or self._deviation_flag(gaze_deviation, False) > 0.0)
      else 0.0
    )
    state.horizontal_evidence.append(horizontal_flag)
    state.gaze_evidence.append(gaze_flag)
    state.orientation_evidence.append(orientation_flag)

    state.samples.append(
      _WindowSample(
        ts=now,
        horizontal_flag=horizontal_flag,
        gaze_flag=gaze_flag,
        orientation_flag=orientation_flag,
        horizontal_deviation=horizontal_deviation,
        gaze_deviation=gaze_deviation,
        orientation_deviation=orientation_deviation,
      )
    )
    self._prune_samples(state.samples, now)

    horizontal_score = max(self._window_score(state.samples, "horizontal_flag"), self._evidence_score(state.horizontal_evidence))
    gaze_score = self._window_score(
      state.samples,
      "gaze_flag",
      window_sec=float(settings.attention_gaze_window_sec),
      now=now,
    )
    gaze_score = max(gaze_score, self._evidence_score(state.gaze_evidence))
    orientation_score = max(self._window_score(state.samples, "orientation_flag"), self._evidence_score(state.orientation_evidence))
    horizontal_peak = self._window_peak(state.samples, "horizontal_deviation")
    gaze_peak = self._window_peak(
      state.samples,
      "gaze_deviation",
      window_sec=float(settings.attention_gaze_window_sec),
      now=now,
    )
    orientation_peak = self._window_peak(state.samples, "orientation_deviation")
    gaze_sample_count = self._window_count(
      state.samples,
      window_sec=float(settings.attention_gaze_window_sec),
      now=now,
    )

    if not confidence_allowed:
      state.samples.clear()
      state.horizontal_evidence.clear()
      state.gaze_evidence.clear()
      state.orientation_evidence.clear()
      horizontal_score = 0.0
      gaze_score = 0.0
      orientation_score = 0.0
      horizontal_peak = 0.0
      gaze_peak = 0.0
      orientation_peak = 0.0
      state.horizontal_rule.state = AttentionState.FOCUSED
      state.gaze_rule.state = AttentionState.FOCUSED
      state.orientation_rule.state = AttentionState.FOCUSED

    state.horizontal_rule.state = self._advance_rule_state(state.horizontal_rule.state, horizontal_score, len(state.horizontal_evidence))
    state.gaze_rule.state = self._advance_rule_state(state.gaze_rule.state, gaze_score, len(state.gaze_evidence))
    state.orientation_rule.state = self._advance_rule_state(state.orientation_rule.state, orientation_score, len(state.orientation_evidence))
    if horizontal_flag > 0.0 and (fast_horizontal_trigger or hard_horizontal_trigger):
      state.horizontal_rule.state = AttentionState.LOOKING_AWAY
    if orientation_flag > 0.0 and (fast_orientation_trigger or hard_orientation_trigger):
      state.orientation_rule.state = AttentionState.LOOKING_AWAY
    if gaze_flag > 0.0 and hard_gaze_trigger:
      state.gaze_rule.state = AttentionState.LOOKING_AWAY

    overall_state = self._combine_states(
      state.horizontal_rule.state,
      state.gaze_rule.state,
      state.orientation_rule.state,
    )
    deviation_frames = sum(
      1
      for sample in state.samples
      if sample.horizontal_flag > 0.0 or sample.gaze_flag > 0.0 or sample.orientation_flag > 0.0
    )

    self._prune_states(active_ids={face_id})
    fast_deviation_triggered = bool(
      horizontal_flag > 0.0 and (fast_horizontal_trigger or hard_horizontal_trigger)
      or orientation_flag > 0.0 and (fast_orientation_trigger or hard_orientation_trigger)
      or gaze_flag > 0.0 and hard_gaze_trigger
    )
    deviation_score = max(horizontal_peak, gaze_peak, orientation_peak)
    attention_confidence = deviation_score * signal_confidence
    snapshot = AttentionSnapshot(
      state=overall_state,
      signal_confidence=signal_confidence,
      attention_confidence=float(np.clip(attention_confidence, 0.0, 2.0)),
      face_confidence=face_confidence,
      attention_score=float(
        np.clip(((0.5 * horizontal_score) + (0.3 * gaze_score) + (0.2 * (1.0 - frontalness))) * signal_confidence, 0.0, 1.5)
      ),
      deviation_score=deviation_score,
      fast_deviation_triggered=fast_deviation_triggered,
      baseline_yaw=state.baseline_yaw,
      baseline_pitch=state.baseline_pitch,
      baseline_gaze_x=state.baseline_gaze_x,
      baseline_gaze_y=state.baseline_gaze_y,
      gaze_left_boundary=gaze_left_boundary,
      gaze_right_boundary=gaze_right_boundary,
      gaze_top_boundary=gaze_top_boundary,
      gaze_bottom_boundary=gaze_bottom_boundary,
      pitch_ratio_up_threshold=self._profile_value("pitchRatioUpThreshold", float(settings.vertical_pitch_ratio_up_threshold)),
      pitch_ratio_down_threshold=self._profile_value("pitchRatioDownThreshold", float(settings.vertical_pitch_ratio_down_threshold)),
      yaw_noise=yaw_noise,
      pitch_noise=pitch_noise,
      gaze_noise=gaze_noise,
      landmark_stability=landmark_stability,
      horizontal_state=state.horizontal_rule.state,
      horizontal_score=horizontal_score,
      horizontal_deviation=horizontal_peak,
      gaze_state=state.gaze_rule.state,
      gaze_score=gaze_score,
      gaze_deviation=gaze_peak,
      orientation_state=state.orientation_rule.state,
      orientation_score=orientation_score,
      orientation_deviation=orientation_peak,
      yaw=yaw,
      pitch=pitch,
      roll=roll,
      pitch_ratio=pitch_ratio,
      gaze_x=gaze_x,
      gaze_y=gaze_y,
      eye_visibility=eye_visibility,
      frontalness=frontalness,
      yaw_velocity=yaw_velocity,
      pitch_velocity=pitch_velocity,
      fused_attention_x=fused_x,
      fused_attention_y=fused_y,
      primary_face_id=face_id,
      deviation_frames=deviation_frames,
      window_frames=len(state.samples),
    )
    self._last_snapshot = snapshot

    signal_confidence = float(np.clip(signal_confidence, 0.0, 1.0))
    snapshot.signal_confidence = signal_confidence

    if signal_confidence < 0.55:
      state.samples.clear()
      state.horizontal_evidence.clear()
      state.gaze_evidence.clear()
      state.orientation_evidence.clear()

      snapshot.state = AttentionState.FOCUSED
      snapshot.attention_confidence = 0.0
      snapshot.attention_score = 0.0
      snapshot.deviation_score = 0.0
      snapshot.horizontal_state = AttentionState.FOCUSED
      snapshot.gaze_state = AttentionState.FOCUSED
      snapshot.orientation_state = AttentionState.FOCUSED

      self._last_snapshot = snapshot
      return snapshot

    attention_confidence = float(snapshot.deviation_score * signal_confidence)
    snapshot.attention_confidence = attention_confidence

    if attention_confidence < 0.35:
      snapshot.state = AttentionState.FOCUSED
    return snapshot

  def _profile_value(self, key: str, default: float) -> float:
    value = self._profile.get(key)
    if value is None:
      return float(default)
    return float(value)

  def _allow_baseline_update(
    self,
    state: _FaceState,
    yaw: float,
    pitch: float,
    gaze_x: float,
    gaze_y: float,
    center_x: float,
    movement: float,
    eye_visible: bool,
    landmark_stability: float,
    eye_symmetry: float,
    yaw_velocity: float,
    pitch_velocity: float,
    thresholds: AttentionThresholds,
  ) -> bool:
    if state.horizontal_rule.state != AttentionState.FOCUSED:
      return False
    if state.gaze_rule.state == AttentionState.LOOKING_AWAY or state.orientation_rule.state == AttentionState.LOOKING_AWAY:
      return False
    if landmark_stability < float(settings.pose_landmark_stability_min):
      return False
    if eye_symmetry > float(settings.face_orientation_eye_symmetry_threshold) * 0.78:
      return False

    yaw_delta = abs(yaw - state.baseline_yaw)
    pitch_delta = abs(pitch - state.baseline_pitch)
    gaze_x_delta = abs(gaze_x - state.baseline_gaze_x)
    gaze_y_delta = abs(gaze_y - state.baseline_gaze_y)
    center_x_delta = abs(center_x - state.baseline_center_x)

    return (
      movement <= float(settings.face_off_camera_baseline_movement_gate) * 1.5
      and yaw_velocity <= float(settings.attention_micro_velocity_deg)
      and pitch_velocity <= float(settings.attention_micro_velocity_deg)
      and abs(yaw) <= thresholds.orientation_exit_yaw * 0.82
      and pitch_delta <= thresholds.orientation_exit_pitch * 0.82
      and center_x_delta <= thresholds.horizontal_exit_x * 0.82
      and gaze_x_delta <= thresholds.gaze_exit_h * 0.88
      and (not settings.enable_vertical_gaze_risk or gaze_y_delta <= thresholds.gaze_exit_v * 0.88)
      and (eye_visible or gaze_x_delta <= thresholds.gaze_exit_h * 0.55)
    )

  def _effective_thresholds(self, thresholds: AttentionThresholds) -> AttentionThresholds:
    noise_mult = float(settings.attention_noise_multiplier)
    gaze_noise_mult = float(settings.attention_gaze_noise_multiplier)
    yaw_noise = max(self._profile_value("yawNoise", 0.0), self._profile_value("yawNoiseFloor", 0.0) / max(noise_mult, 1e-6))
    pitch_noise = max(self._profile_value("pitchNoise", 0.0), self._profile_value("pitchNoiseFloor", 0.0) / max(noise_mult, 1e-6))
    gaze_noise_x = max(
      self._profile_value("gazeNoiseX", self._profile_value("gazeNoise", 0.0)),
      self._profile_value("gazeNoiseFloorX", 0.0) / max(gaze_noise_mult, 1e-6),
    )
    gaze_noise_y = max(
      self._profile_value("gazeNoiseY", self._profile_value("gazeNoise", 0.0)),
      self._profile_value("gazeNoiseFloorY", 0.0) / max(gaze_noise_mult, 1e-6),
    )

    yaw_floor = max(18.0, yaw_noise * 2.5)
    pitch_floor = max(16.0, pitch_noise * 2.5)
    gaze_h_floor = max(float(thresholds.gaze_enter_h), gaze_noise_x * gaze_noise_mult)
    gaze_v_floor = max(float(thresholds.gaze_enter_v), gaze_noise_y * gaze_noise_mult)

    horizontal_enter_yaw = max(float(thresholds.horizontal_enter_yaw), yaw_floor)
    horizontal_exit_yaw = max(float(thresholds.horizontal_exit_yaw), horizontal_enter_yaw * 0.62)
    orientation_enter_yaw = max(float(thresholds.orientation_enter_yaw), yaw_floor)
    orientation_exit_yaw = max(float(thresholds.orientation_exit_yaw), orientation_enter_yaw * 0.62)
    orientation_enter_pitch = max(float(thresholds.orientation_enter_pitch), pitch_floor)
    orientation_exit_pitch = max(float(thresholds.orientation_exit_pitch), orientation_enter_pitch * 0.62)
    gaze_enter_h = max(float(thresholds.gaze_enter_h), gaze_h_floor)
    gaze_exit_h = max(float(thresholds.gaze_exit_h), gaze_enter_h * 0.72)
    gaze_enter_v = max(float(thresholds.gaze_enter_v), gaze_v_floor)
    gaze_exit_v = max(float(thresholds.gaze_exit_v), gaze_enter_v * 0.72)
    gaze_head_yaw = max(float(thresholds.gaze_head_yaw), yaw_floor * 0.92)
    gaze_head_pitch = max(float(thresholds.gaze_head_pitch), pitch_floor * 0.92)

    return AttentionThresholds(
      horizontal_enter_yaw=horizontal_enter_yaw,
      horizontal_exit_yaw=horizontal_exit_yaw,
      horizontal_enter_x=float(thresholds.horizontal_enter_x),
      horizontal_exit_x=float(thresholds.horizontal_exit_x),
      orientation_enter_yaw=orientation_enter_yaw,
      orientation_exit_yaw=orientation_exit_yaw,
      orientation_enter_pitch=orientation_enter_pitch,
      orientation_exit_pitch=orientation_exit_pitch,
      gaze_enter_h=gaze_enter_h,
      gaze_exit_h=gaze_exit_h,
      gaze_enter_v=gaze_enter_v,
      gaze_exit_v=gaze_exit_v,
      gaze_head_yaw=gaze_head_yaw,
      gaze_head_pitch=gaze_head_pitch,
      gaze_min_confidence=float(thresholds.gaze_min_confidence),
    )

  @staticmethod
  def _deviation_flag(deviation: float, micro_movement: bool) -> float:
    if deviation < 1.0:
      return 0.0
    return 1.0

  @staticmethod
  def _boundary_deviation(value: float, center: float, lower: float, upper: float) -> float:
    if value <= center:
      span = max(abs(center - lower), 1e-6)
    else:
      span = max(abs(upper - center), 1e-6)
    return float(abs(value - center) / span)

  @staticmethod
  def _evidence_score(evidence: Deque[float]) -> float:
    if not evidence:
      return 0.0
    return float(sum(float(item) for item in evidence) / max(float(len(evidence)), 1.0))

  @staticmethod
  def _recent_samples(
    samples: Deque[_WindowSample],
    window_sec: Optional[float] = None,
    now: Optional[float] = None,
  ) -> List[_WindowSample]:
    if not samples:
      return []
    if window_sec is None or now is None:
      return list(samples)
    return [sample for sample in samples if (now - sample.ts) <= window_sec]

  @classmethod
  def _window_score(
    cls,
    samples: Deque[_WindowSample],
    field_name: str,
    window_sec: Optional[float] = None,
    now: Optional[float] = None,
  ) -> float:
    scoped = cls._recent_samples(samples, window_sec=window_sec, now=now)
    if not scoped:
      return 0.0
    values = [float(getattr(sample, field_name)) for sample in scoped]
    return float(sum(values) / max(len(values), 1))

  @classmethod
  def _window_peak(
    cls,
    samples: Deque[_WindowSample],
    field_name: str,
    window_sec: Optional[float] = None,
    now: Optional[float] = None,
  ) -> float:
    scoped = cls._recent_samples(samples, window_sec=window_sec, now=now)
    if not scoped:
      return 0.0
    return float(max(float(getattr(sample, field_name)) for sample in scoped))

  @classmethod
  def _window_count(
    cls,
    samples: Deque[_WindowSample],
    window_sec: Optional[float] = None,
    now: Optional[float] = None,
  ) -> int:
    return len(cls._recent_samples(samples, window_sec=window_sec, now=now))

  @staticmethod
  def _combine_states(*states: str) -> str:
    if any(state == AttentionState.LOOKING_AWAY for state in states):
      return AttentionState.LOOKING_AWAY
    if any(state == AttentionState.TRANSITIONING for state in states):
      return AttentionState.TRANSITIONING
    return AttentionState.FOCUSED

  @staticmethod
  def _primary_face(tracked_faces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not tracked_faces:
      return None
    return max(
      tracked_faces,
      key=lambda face: float(max(float((face.get("bbox", {}) or {}).get("w", 0.0)), 0.0))
      * float(max(float((face.get("bbox", {}) or {}).get("h", 0.0)), 0.0)),
    )

  @staticmethod
  def _advance_rule_state(current_state: str, score: float, sample_count: int) -> str:
    if sample_count < int(settings.attention_min_window_frames):
      return AttentionState.TRANSITIONING if score >= float(settings.attention_transition_ratio) else AttentionState.FOCUSED

    if current_state == AttentionState.FOCUSED:
      if score >= float(settings.attention_enter_ratio):
        return AttentionState.LOOKING_AWAY
      if score >= float(settings.attention_transition_ratio):
        return AttentionState.TRANSITIONING
      return AttentionState.FOCUSED

    if current_state == AttentionState.TRANSITIONING:
      if score >= float(settings.attention_enter_ratio):
        return AttentionState.LOOKING_AWAY
      if score <= float(settings.attention_exit_ratio):
        return AttentionState.FOCUSED
      return AttentionState.TRANSITIONING

    if score <= float(settings.attention_exit_ratio):
      return AttentionState.FOCUSED
    if score <= float(settings.attention_transition_ratio):
      return AttentionState.TRANSITIONING
    return AttentionState.LOOKING_AWAY

  @staticmethod
  def _prune_samples(samples: Deque[_WindowSample], now: float) -> None:
    window_sec = float(settings.attention_window_sec)
    while samples and (now - samples[0].ts) > window_sec:
      samples.popleft()

  def _prune_states(self, active_ids: set[int]) -> None:
    stale_ids = [face_id for face_id in self._states if face_id not in active_ids]
    for face_id in stale_ids:
      del self._states[face_id]

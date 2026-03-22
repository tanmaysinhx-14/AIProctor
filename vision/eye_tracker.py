from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Set, Tuple

import numpy as np


@dataclass
class EyeGazeResult:
  direction: str
  eye_visible: bool
  eye_visibility: float = 0.0
  horizontal_offset: float = 0.0
  vertical_offset: float = 0.0
  normalized_x: float = 0.0
  normalized_y: float = 0.0
  horizontal_ratio: float = 0.5
  vertical_ratio: float = 0.5
  baseline_h: float = 0.5
  baseline_v: float = 0.5
  confidence: float = 0.0


@dataclass
class _EyeState:
  smooth_h: float = 0.5
  smooth_v: float = 0.5
  baseline_h: float = 0.5
  baseline_v: float = 0.5
  history: Deque[str] = field(default_factory=lambda: deque(maxlen=5))


class EyeTracker:
  # Landmark indices for eye corners/lids and iris centers in MediaPipe Face Mesh.
  _left_corners = (33, 133)
  _right_corners = (362, 263)
  _left_lids = (159, 145)
  _right_lids = (386, 374)
  _left_iris = 468
  _right_iris = 473

  def __init__(self) -> None:
    self._states: Dict[int, _EyeState] = {}
    self._ema_alpha = 0.4
    self._baseline_alpha = 0.08
    self._default_eye_visible_threshold = 0.075
    self._default_horizontal_deadzone = 0.085
    self._default_vertical_down_threshold = 0.105
    self._default_vertical_up_threshold = 0.105
    self._eye_visible_threshold = self._default_eye_visible_threshold
    self._horizontal_deadzone = self._default_horizontal_deadzone
    self._vertical_down_threshold = self._default_vertical_down_threshold
    self._vertical_up_threshold = self._default_vertical_up_threshold

  def estimate(
    self,
    landmarks: np.ndarray,
    face_id: Optional[int] = None,
    head_pose: Optional[Tuple[float, float]] = None,
  ) -> EyeGazeResult:
    if landmarks.ndim != 2 or landmarks.shape[1] < 2:
      return EyeGazeResult(direction="CENTER", eye_visible=False)

    max_index = max(
      *self._left_corners,
      *self._right_corners,
      *self._left_lids,
      *self._right_lids,
      self._left_iris,
      self._right_iris,
    )
    if landmarks.shape[0] <= max_index:
      return EyeGazeResult(direction="CENTER", eye_visible=False)
    if not np.isfinite(landmarks).all():
      return EyeGazeResult(direction="CENTER", eye_visible=False)

    left_h = self._horizontal_ratio(landmarks, self._left_corners, self._left_iris)
    right_h = self._horizontal_ratio(landmarks, self._right_corners, self._right_iris)
    horizontal_ratio = (left_h + right_h) / 2.0

    left_v = self._vertical_ratio(landmarks, self._left_lids, self._left_iris)
    right_v = self._vertical_ratio(landmarks, self._right_lids, self._right_iris)
    vertical_ratio = (left_v + right_v) / 2.0

    yaw = float(head_pose[0]) if head_pose else 0.0
    pitch = float(head_pose[1]) if head_pose else 0.0

    ear = self._eye_aspect_ratio(landmarks)
    eye_visible = ear > self._eye_visible_threshold
    normalized_x = float(horizontal_ratio - 0.5)
    normalized_y = float(vertical_ratio - 0.5)

    if face_id is None:
      horizontal_offset, vertical_offset = self._offsets(
        horizontal_ratio=horizontal_ratio,
        vertical_ratio=vertical_ratio,
        baseline_h=0.5,
        baseline_v=0.5,
        yaw=yaw,
        pitch=pitch,
      )
      direction = self._classify(
        horizontal_ratio=horizontal_ratio,
        vertical_ratio=vertical_ratio,
        baseline_h=0.5,
        baseline_v=0.5,
        eye_visible=eye_visible,
        yaw=yaw,
        pitch=pitch,
      )
      return EyeGazeResult(
        direction=direction,
        eye_visible=eye_visible,
        eye_visibility=ear,
        horizontal_offset=horizontal_offset,
        vertical_offset=vertical_offset,
        normalized_x=normalized_x,
        normalized_y=normalized_y,
        horizontal_ratio=horizontal_ratio,
        vertical_ratio=vertical_ratio,
        baseline_h=0.5,
        baseline_v=0.5,
        confidence=self._confidence(horizontal_offset, vertical_offset, eye_visible),
      )

    state = self._states.setdefault(face_id, _EyeState())
    state.smooth_h = (1.0 - self._ema_alpha) * state.smooth_h + self._ema_alpha * horizontal_ratio
    state.smooth_v = (1.0 - self._ema_alpha) * state.smooth_v + self._ema_alpha * vertical_ratio

    horizontal_delta = state.smooth_h - state.baseline_h
    vertical_delta = state.smooth_v - state.baseline_v
    near_neutral = (
      abs(horizontal_delta) <= (self._horizontal_deadzone * 0.82)
      and abs(vertical_delta) <= (max(self._vertical_up_threshold, self._vertical_down_threshold) * 0.82)
    )

    # Update neutral eye center only when gaze is close to center; prevents baseline drift during gaze sweeps.
    if eye_visible and abs(yaw) < 14.0 and abs(pitch) < 12.0 and near_neutral:
      state.baseline_h = (1.0 - self._baseline_alpha) * state.baseline_h + self._baseline_alpha * state.smooth_h
      state.baseline_v = (1.0 - self._baseline_alpha) * state.baseline_v + self._baseline_alpha * state.smooth_v

    direction = self._classify(
      horizontal_ratio=state.smooth_h,
      vertical_ratio=state.smooth_v,
      baseline_h=state.baseline_h,
      baseline_v=state.baseline_v,
      eye_visible=eye_visible,
      yaw=yaw,
      pitch=pitch,
    )

    if eye_visible:
      state.history.append(direction)
      stable_direction = self._majority_vote(state.history)
    else:
      state.history.clear()
      stable_direction = "CENTER"

    horizontal_offset, vertical_offset = self._offsets(
      horizontal_ratio=state.smooth_h,
      vertical_ratio=state.smooth_v,
      baseline_h=state.baseline_h,
      baseline_v=state.baseline_v,
      yaw=yaw,
      pitch=pitch,
    )
    return EyeGazeResult(
      direction=stable_direction,
      eye_visible=eye_visible,
      eye_visibility=ear,
      horizontal_offset=horizontal_offset,
      vertical_offset=vertical_offset,
      normalized_x=float(state.smooth_h - 0.5),
      normalized_y=float(state.smooth_v - 0.5),
      horizontal_ratio=state.smooth_h,
      vertical_ratio=state.smooth_v,
      baseline_h=state.baseline_h,
      baseline_v=state.baseline_v,
      confidence=self._confidence(horizontal_offset, vertical_offset, eye_visible),
    )

  def prune(self, active_face_ids: Set[int]) -> None:
    stale_ids = [face_id for face_id in self._states if face_id not in active_face_ids]
    for face_id in stale_ids:
      del self._states[face_id]

  def reset_overrides(self) -> None:
    self._eye_visible_threshold = self._default_eye_visible_threshold
    self._horizontal_deadzone = self._default_horizontal_deadzone
    self._vertical_down_threshold = self._default_vertical_down_threshold
    self._vertical_up_threshold = self._default_vertical_up_threshold

  def set_overrides(self, overrides: Dict[str, float]) -> None:
    if not overrides:
      return

    if "eye_visibility_threshold" in overrides:
      self._eye_visible_threshold = float(
        np.clip(float(overrides["eye_visibility_threshold"]), 0.045, 0.16)
      )

    if "eye_horizontal_deadzone" in overrides:
      self._horizontal_deadzone = float(
        np.clip(float(overrides["eye_horizontal_deadzone"]), 0.03, 0.14)
      )

    if "eye_vertical_up_threshold" in overrides:
      self._vertical_up_threshold = float(
        np.clip(float(overrides["eye_vertical_up_threshold"]), 0.045, 0.2)
      )

    if "eye_vertical_down_threshold" in overrides:
      self._vertical_down_threshold = float(
        np.clip(float(overrides["eye_vertical_down_threshold"]), 0.045, 0.2)
      )

    if "eye_baseline_alpha" in overrides:
      self._baseline_alpha = float(np.clip(float(overrides["eye_baseline_alpha"]), 0.03, 0.22))

  def _classify(
    self,
    horizontal_ratio: float,
    vertical_ratio: float,
    baseline_h: float,
    baseline_v: float,
    eye_visible: bool,
    yaw: float,
    pitch: float,
  ) -> str:
    if not eye_visible:
      return "CENTER"

    horizontal_offset, vertical_offset = self._offsets(
      horizontal_ratio=horizontal_ratio,
      vertical_ratio=vertical_ratio,
      baseline_h=baseline_h,
      baseline_v=baseline_v,
      yaw=yaw,
      pitch=pitch,
    )

    if vertical_offset < -self._vertical_up_threshold:
      return "UP"
    if vertical_offset > self._vertical_down_threshold:
      return "DOWN"
    if horizontal_offset < -self._horizontal_deadzone:
      return "LEFT"
    if horizontal_offset > self._horizontal_deadzone:
      return "RIGHT"
    return "CENTER"

  def _offsets(
    self,
    horizontal_ratio: float,
    vertical_ratio: float,
    baseline_h: float,
    baseline_v: float,
    yaw: float,
    pitch: float,
  ) -> Tuple[float, float]:
    # Compensate a small portion of head pose to reduce false gaze shifts.
    yaw_comp = float(np.clip(yaw / 90.0, -0.35, 0.35) * -0.045)
    pitch_comp = float(np.clip(pitch / 90.0, -0.35, 0.35) * -0.04)
    horizontal_offset = (horizontal_ratio - baseline_h) + yaw_comp
    vertical_offset = (vertical_ratio - baseline_v) + pitch_comp
    return float(horizontal_offset), float(vertical_offset)

  def _confidence(self, horizontal_offset: float, vertical_offset: float, eye_visible: bool) -> float:
    if not eye_visible:
      return 0.0
    h_norm = abs(horizontal_offset) / max(self._horizontal_deadzone, 1e-6)
    v_norm = max(
      abs(vertical_offset) / max(self._vertical_up_threshold, 1e-6),
      abs(vertical_offset) / max(self._vertical_down_threshold, 1e-6),
    )
    return float(np.clip(max(h_norm, v_norm) / 2.2, 0.0, 1.0))

  def _majority_vote(self, history: Deque[str]) -> str:
    if not history:
      return "CENTER"
    counts: Dict[str, int] = {}
    for item in history:
      counts[item] = counts.get(item, 0) + 1
    max_count = max(counts.values())
    candidates = [item for item, count in counts.items() if count == max_count]
    last_item = history[-1]
    if last_item in candidates:
      return last_item
    return candidates[0]

  def _horizontal_ratio(self, landmarks: np.ndarray, corners: Tuple[int, int], iris_idx: int) -> float:
    corner_a = landmarks[corners[0]]
    corner_b = landmarks[corners[1]]
    iris = landmarks[iris_idx]

    left_x = float(min(corner_a[0], corner_b[0]))
    right_x = float(max(corner_a[0], corner_b[0]))
    width = max(right_x - left_x, 1e-6)
    ratio = (float(iris[0]) - left_x) / width
    return float(np.clip(ratio, 0.0, 1.0))

  def _vertical_ratio(self, landmarks: np.ndarray, lids: Tuple[int, int], iris_idx: int) -> float:
    top = landmarks[lids[0]]
    bottom = landmarks[lids[1]]
    iris = landmarks[iris_idx]

    top_y = float(min(top[1], bottom[1]))
    bottom_y = float(max(top[1], bottom[1]))
    height = max(bottom_y - top_y, 1e-6)
    ratio = (float(iris[1]) - top_y) / height
    return float(np.clip(ratio, 0.0, 1.0))

  def _eye_aspect_ratio(self, landmarks: np.ndarray) -> float:
    left_width = np.linalg.norm(landmarks[self._left_corners[0]] - landmarks[self._left_corners[1]])
    right_width = np.linalg.norm(landmarks[self._right_corners[0]] - landmarks[self._right_corners[1]])
    left_height = np.linalg.norm(landmarks[self._left_lids[0]] - landmarks[self._left_lids[1]])
    right_height = np.linalg.norm(landmarks[self._right_lids[0]] - landmarks[self._right_lids[1]])

    left_ear = float(left_height / max(left_width, 1e-6))
    right_ear = float(right_height / max(right_width, 1e-6))
    return (left_ear + right_ear) / 2.0

from dataclasses import dataclass
from typing import Dict, Tuple

from hardware.hardware_detector import HardwareMode, HardwareProfile


@dataclass
class SchedulerState:
  mode: HardwareMode
  frame_stride: int
  face_detect_interval: int
  target_height: int
  target_yolo_fps: float
  dynamic_stride_boost: int = 0

  @property
  def effective_stride(self) -> int:
    return max(1, int(self.frame_stride + self.dynamic_stride_boost))


class AdaptiveFrameScheduler:
  def __init__(self, profile: HardwareProfile) -> None:
    self._state = SchedulerState(
      mode=profile.mode,
      frame_stride=max(1, int(profile.frame_stride)),
      face_detect_interval=max(1, int(profile.face_detect_interval)),
      target_height=max(240, int(profile.target_height)),
      target_yolo_fps=max(1.0, float(profile.target_yolo_fps)),
      dynamic_stride_boost=0,
    )
    self._overloaded_count = 0
    self._recovered_count = 0

  def mode(self) -> HardwareMode:
    return self._state.mode

  def as_dict(self) -> Dict[str, float]:
    return {
      "mode": self._state.mode.value,
      "frameStride": float(self._state.frame_stride),
      "faceDetectInterval": float(self._state.face_detect_interval),
      "targetHeight": float(self._state.target_height),
      "targetYoloFps": float(self._state.target_yolo_fps),
      "effectiveStride": float(self._state.effective_stride),
    }

  def should_process_frame(self, frame_index: int) -> bool:
    stride = self._state.effective_stride
    return (int(frame_index) % stride) == 0

  def should_run_face_detection(self, frame_index: int, track_count: int) -> bool:
    interval = max(1, int(self._state.face_detect_interval))
    if track_count <= 0:
      return True
    return (int(frame_index) % interval) == 0

  def max_width_for_frame(self, frame_shape: Tuple[int, int, int]) -> int:
    h, w = frame_shape[:2]
    target_h = max(120, int(self._state.target_height))
    if h <= target_h:
      return int(w)
    ratio = target_h / float(max(h, 1))
    return max(160, int(round(w * ratio)))

  def target_yolo_fps(self) -> float:
    return float(self._state.target_yolo_fps)

  def update_feedback(self, cpu_percent: float, frame_latency_ms: float) -> None:
    overloaded = cpu_percent >= 85.0 or frame_latency_ms >= 260.0
    recovered = cpu_percent <= 62.0 and frame_latency_ms <= 180.0

    if overloaded:
      self._overloaded_count += 1
      self._recovered_count = 0
    elif recovered:
      self._recovered_count += 1
      self._overloaded_count = 0
    else:
      self._overloaded_count = max(0, self._overloaded_count - 1)
      self._recovered_count = max(0, self._recovered_count - 1)

    if self._overloaded_count >= 8:
      self._state.dynamic_stride_boost = min(2, self._state.dynamic_stride_boost + 1)
      self._overloaded_count = 0

    if self._recovered_count >= 25:
      self._state.dynamic_stride_boost = max(0, self._state.dynamic_stride_boost - 1)
      self._recovered_count = 0

  def set_profile(self, profile: HardwareProfile) -> None:
    self._state.mode = profile.mode
    self._state.frame_stride = max(1, int(profile.frame_stride))
    self._state.face_detect_interval = max(1, int(profile.face_detect_interval))
    self._state.target_height = max(240, int(profile.target_height))
    self._state.target_yolo_fps = max(1.0, float(profile.target_yolo_fps))
    self._state.dynamic_stride_boost = 0
    self._overloaded_count = 0
    self._recovered_count = 0

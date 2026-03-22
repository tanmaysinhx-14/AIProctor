from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import psutil
import torch


class HardwareMode(str, Enum):
  CPU_MODE = "CPU_MODE"
  MID_GPU_MODE = "MID_GPU_MODE"
  HIGH_GPU_MODE = "HIGH_GPU_MODE"


@dataclass(frozen=True)
class HardwareProfile:
  mode: HardwareMode
  cpu_cores: int
  ram_gb: float
  gpu_available: bool
  gpu_name: str
  gpu_total_gb: float
  frame_stride: int
  face_detect_interval: int
  target_height: int
  target_yolo_fps: float
  batch_size: int

  def to_dict(self) -> Dict[str, Any]:
    return {
      "mode": self.mode.value,
      "cpuCores": int(self.cpu_cores),
      "ramGb": round(float(self.ram_gb), 3),
      "gpuAvailable": bool(self.gpu_available),
      "gpuName": str(self.gpu_name),
      "gpuTotalGb": round(float(self.gpu_total_gb), 3),
      "frameStride": int(self.frame_stride),
      "faceDetectInterval": int(self.face_detect_interval),
      "targetHeight": int(self.target_height),
      "targetYoloFps": float(self.target_yolo_fps),
      "batchSize": int(self.batch_size),
    }


class HardwareDetector:
  @classmethod
  def detect(cls) -> HardwareProfile:
    cpu_cores = int(psutil.cpu_count(logical=True) or 0)
    ram_gb = float(psutil.virtual_memory().total) / (1024 ** 3)

    gpu_available = bool(torch.cuda.is_available())
    gpu_name = "N/A"
    gpu_total_gb = 0.0
    if gpu_available:
      try:
        idx = int(torch.cuda.current_device())
      except Exception:
        idx = 0
      try:
        gpu_name = str(torch.cuda.get_device_name(idx))
      except Exception:
        gpu_name = "Unknown GPU"
      try:
        props = torch.cuda.get_device_properties(idx)
        gpu_total_gb = float(getattr(props, "total_memory", 0.0)) / (1024 ** 3)
      except Exception:
        gpu_total_gb = 0.0

    mode = cls._choose_mode(
      gpu_available=gpu_available,
      gpu_total_gb=gpu_total_gb,
      cpu_cores=cpu_cores,
      ram_gb=ram_gb,
    )
    return cls._profile_for_mode(
      mode=mode,
      cpu_cores=cpu_cores,
      ram_gb=ram_gb,
      gpu_available=gpu_available,
      gpu_name=gpu_name,
      gpu_total_gb=gpu_total_gb,
    )

  @classmethod
  def fallback_cpu_profile(cls, previous: Optional[HardwareProfile] = None) -> HardwareProfile:
    cpu_cores = int(previous.cpu_cores) if previous is not None else int(psutil.cpu_count(logical=True) or 0)
    ram_gb = float(previous.ram_gb) if previous is not None else float(psutil.virtual_memory().total) / (1024 ** 3)
    return cls._profile_for_mode(
      mode=HardwareMode.CPU_MODE,
      cpu_cores=cpu_cores,
      ram_gb=ram_gb,
      gpu_available=False,
      gpu_name="CPU fallback",
      gpu_total_gb=0.0,
    )

  @staticmethod
  def _choose_mode(
    gpu_available: bool,
    gpu_total_gb: float,
    cpu_cores: int,
    ram_gb: float,
  ) -> HardwareMode:
    if not gpu_available:
      return HardwareMode.CPU_MODE

    # Conservative split that matches requested tiers.
    if gpu_total_gb >= 8.0 and cpu_cores >= 10 and ram_gb >= 16.0:
      return HardwareMode.HIGH_GPU_MODE
    return HardwareMode.MID_GPU_MODE

  @classmethod
  def _profile_for_mode(
    cls,
    mode: HardwareMode,
    cpu_cores: int,
    ram_gb: float,
    gpu_available: bool,
    gpu_name: str,
    gpu_total_gb: float,
  ) -> HardwareProfile:
    if mode == HardwareMode.CPU_MODE:
      return HardwareProfile(
        mode=mode,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpu_available=False,
        gpu_name="CPU",
        gpu_total_gb=0.0,
        frame_stride=4,
        face_detect_interval=10,
        target_height=480,
        target_yolo_fps=3.0,
        batch_size=1,
      )

    if mode == HardwareMode.MID_GPU_MODE:
      return HardwareProfile(
        mode=mode,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_total_gb=gpu_total_gb,
        frame_stride=2,
        face_detect_interval=5,
        target_height=720,
        target_yolo_fps=8.0,
        batch_size=2,
      )

    return HardwareProfile(
      mode=mode,
      cpu_cores=cpu_cores,
      ram_gb=ram_gb,
      gpu_available=gpu_available,
      gpu_name=gpu_name,
      gpu_total_gb=gpu_total_gb,
      frame_stride=1,
      face_detect_interval=5,
      target_height=1080,
      target_yolo_fps=12.0,
      batch_size=4,
    )

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np

from config import settings


@dataclass
class QualityWarning:
  code: str
  message: str
  severity: str


class FrameQualityAnalyzer:
  def __init__(self) -> None:
    self._streaks: Dict[str, int] = defaultdict(int)
    self._activation_frames = settings.quality_warning_activation_frames

  def analyze(self, frame_bgr: np.ndarray) -> List[QualityWarning]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    dark_ratio = float(np.mean(gray < 30))
    bright_ratio = float(np.mean(gray > 225))

    edges = cv2.Canny(gray, 60, 140)
    edge_density = float(np.mean(edges > 0))

    h, w = gray.shape[:2]
    y1 = int(h * 0.25)
    y2 = int(h * 0.75)
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)
    center = gray[y1:y2, x1:x2]
    center_edges = edges[y1:y2, x1:x2]
    center_dark_ratio = float(np.mean(center < 32))
    center_bright_ratio = float(np.mean(center > 223))
    center_edge_density = float(np.mean(center_edges > 0))
    center_entropy = self._entropy(center)

    low_light = (
      mean_brightness < settings.low_light_mean_threshold
      or dark_ratio > settings.low_light_dark_ratio_threshold
    )
    dirty_camera = (
      laplacian_var < settings.dirty_camera_laplacian_threshold
      and contrast < settings.dirty_camera_contrast_threshold
      and 35.0 < mean_brightness < 220.0
    )
    blocked_view = (
      dark_ratio > settings.blocked_view_dark_ratio_threshold
      or bright_ratio > settings.blocked_view_bright_ratio_threshold
      or center_dark_ratio > settings.blocked_view_center_dark_ratio_threshold
      or center_bright_ratio > settings.blocked_view_center_bright_ratio_threshold
      or (
        edge_density < settings.blocked_view_edge_density_threshold
        and contrast < settings.blocked_view_contrast_threshold
      )
      or (
        center_edge_density < settings.blocked_view_center_edge_density_threshold
        and center_entropy < settings.blocked_view_center_entropy_threshold
      )
    )

    active_flags = {
      "LOW_LIGHT": low_light,
      "DIRTY_CAMERA": dirty_camera,
      "BLOCKED_VIEW": blocked_view,
    }
    for key, active in active_flags.items():
      if active:
        self._streaks[key] += 1
      else:
        self._streaks[key] = max(self._streaks[key] - 1, 0)

    warnings: List[QualityWarning] = []
    if self._streaks["LOW_LIGHT"] >= self._activation_frames:
      warnings.append(
        QualityWarning(
          code="LOW_LIGHT",
          message="Low light detected. Increase room lighting.",
          severity="MEDIUM",
        )
      )
    if self._streaks["DIRTY_CAMERA"] >= self._activation_frames:
      warnings.append(
        QualityWarning(
          code="DIRTY_CAMERA",
          message="Camera appears blurry/dirty. Clean the lens.",
          severity="MEDIUM",
        )
      )
    blocked_activation = max(1, settings.blocked_view_warning_activation_frames)
    if self._streaks["BLOCKED_VIEW"] >= blocked_activation:
      warnings.append(
        QualityWarning(
          code="BLOCKED_VIEW",
          message="Camera view appears blocked.",
          severity="HIGH",
        )
      )
    return warnings

  def _entropy(self, image: np.ndarray) -> float:
    hist = cv2.calcHist([image], [0], None, [32], [0, 256]).ravel().astype(np.float64)
    total = float(hist.sum())
    if total <= 0.0:
      return 0.0
    p = hist / total
    p = p[p > 1e-9]
    return float(-np.sum(p * np.log2(p)))

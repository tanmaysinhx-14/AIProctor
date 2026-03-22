from typing import Optional, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]


class FaceEmbeddingExtractor:
  """
  Lightweight appearance descriptor for track re-identification.

  The repo does not currently bundle ArcFace/FaceNet weights, so this
  uses a deterministic fallback descriptor that is cheap enough for the
  real-time pipeline and stable enough for short-term identity recovery.
  """

  def __init__(self) -> None:
    self._patch_size = 24

  def extract(self, frame_bgr: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
    if frame_bgr is None or frame_bgr.size == 0:
      return None

    crop = self._crop(frame_bgr, bbox)
    if crop is None or crop.size == 0:
      return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    patch = cv2.resize(gray, (self._patch_size, self._patch_size), interpolation=cv2.INTER_AREA)
    patch_f = patch.astype(np.float32) / 255.0
    patch_f = patch_f - float(np.mean(patch_f))
    patch_std = float(np.std(patch_f))
    if patch_std > 1e-6:
      patch_f = patch_f / patch_std

    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    hist, _ = np.histogram(
      angle,
      bins=8,
      range=(0.0, float(2.0 * np.pi)),
      weights=mag,
    )

    descriptor = np.concatenate(
      [
        patch_f.flatten(),
        hist.astype(np.float32),
      ],
      axis=0,
    ).astype(np.float32)
    norm = float(np.linalg.norm(descriptor))
    if norm <= 1e-6:
      return None
    return descriptor / norm

  @staticmethod
  def similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
      return 0.0
    if a.shape != b.shape:
      return 0.0
    sim = float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-6))
    return float(np.clip(sim, -1.0, 1.0))

  @staticmethod
  def blend(
    previous: Optional[np.ndarray],
    current: Optional[np.ndarray],
    alpha: float = 0.35,
  ) -> Optional[np.ndarray]:
    if current is None:
      return previous
    if previous is None:
      return current
    mixed = ((1.0 - alpha) * previous) + (alpha * current)
    norm = float(np.linalg.norm(mixed))
    if norm <= 1e-6:
      return current
    return mixed / norm

  @staticmethod
  def _crop(frame_bgr: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
    frame_h, frame_w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    pad_x = int(round(width * 0.10))
    pad_y = int(round(height * 0.12))
    x1 = max(0, min(x1 - pad_x, frame_w - 1))
    y1 = max(0, min(y1 - pad_y, frame_h - 1))
    x2 = max(x1 + 2, min(x2 + pad_x, frame_w))
    y2 = max(y1 + 2, min(y2 + pad_y, frame_h))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.shape[0] < 20 or crop.shape[1] < 20:
      return None
    return crop

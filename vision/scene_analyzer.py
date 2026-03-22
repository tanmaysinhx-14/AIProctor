from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from config import settings

BBox = Tuple[int, int, int, int]


class SceneMotionAnalyzer:
  def __init__(self) -> None:
    self._bg_subtractor = self._create_subtractor()
    self._static_zones: List[Tuple[float, float, float, float]] = []
    self._last_fg_mask: np.ndarray | None = None
    self._last_frame_shape: Tuple[int, int] | None = None
    self._last_active_regions: List[BBox] = []

  def reset(self) -> None:
    self._bg_subtractor = self._create_subtractor()
    self._last_fg_mask = None
    self._last_frame_shape = None
    self._last_active_regions = []

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
    self._static_zones = parsed[:16]

  def analyze(self, frame_bgr: np.ndarray, person_boxes: List[BBox]) -> Dict[str, Any]:
    frame_h, frame_w = frame_bgr.shape[:2]
    fg_mask = self._bg_subtractor.apply(
      frame_bgr,
      learningRate=float(settings.scene_background_learning_rate),
    )
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), dtype=np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
    self._last_fg_mask = fg_mask.copy()
    self._last_frame_shape = (frame_h, frame_w)

    motion_pixels = float(np.count_nonzero(fg_mask))
    total_pixels = float(max(frame_w * frame_h, 1))

    person_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    for bbox in person_boxes:
      x1, y1, x2, y2 = self._expand_bbox(
        bbox,
        frame_w=frame_w,
        frame_h=frame_h,
        ratio=float(settings.scene_person_mask_expand_ratio),
      )
      person_mask[y1:y2, x1:x2] = 255

    static_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    for zone in self._static_zones:
      x1, y1, x2, y2 = self._zone_to_bbox(zone, frame_w, frame_h)
      static_mask[y1:y2, x1:x2] = 255

    person_motion = cv2.bitwise_and(fg_mask, person_mask)
    non_person_mask = cv2.bitwise_not(person_mask)
    background_motion = cv2.bitwise_and(fg_mask, non_person_mask)
    ignored_background = cv2.bitwise_and(background_motion, static_mask)
    unmasked_background = cv2.bitwise_and(background_motion, cv2.bitwise_not(static_mask))

    active_regions = self._extract_regions(
      mask=unmasked_background,
      frame_w=frame_w,
      frame_h=frame_h,
      limit=int(settings.scene_background_region_limit),
    )
    self._last_active_regions = [
      self._zone_to_bbox((item["x"], item["y"], item["w"], item["h"]), frame_w, frame_h)
      for item in active_regions
    ]

    person_area = float(np.count_nonzero(person_mask))
    background_area = float(total_pixels - person_area)
    return {
      "foregroundMotionRatio": round(motion_pixels / total_pixels, 5),
      "personMotionRatio": round(float(np.count_nonzero(person_motion)) / max(person_area, 1.0), 5),
      "backgroundMotionRatio": round(float(np.count_nonzero(unmasked_background)) / max(background_area, 1.0), 5),
      "ignoredBackgroundMotionRatio": round(float(np.count_nonzero(ignored_background)) / total_pixels, 5),
      "activeBackgroundRegions": active_regions,
      "staticBackgroundZoneCount": len(self._static_zones),
    }

  def region_motion_ratio(self, bbox: BBox, expand_ratio: float = 0.0) -> float:
    if self._last_fg_mask is None or self._last_frame_shape is None:
      return 0.0

    frame_h, frame_w = self._last_frame_shape
    x1, y1, x2, y2 = self._expand_bbox(
      bbox,
      frame_w=frame_w,
      frame_h=frame_h,
      ratio=max(0.0, float(expand_ratio)),
    )
    if x2 <= x1 or y2 <= y1:
      return 0.0

    region = self._last_fg_mask[y1:y2, x1:x2]
    motion = float(np.count_nonzero(region))
    area = float(max((x2 - x1) * (y2 - y1), 1))
    return motion / area

  def static_zone_overlap_ratio(self, bbox: BBox) -> float:
    if self._last_frame_shape is None or not self._static_zones:
      return 0.0

    frame_h, frame_w = self._last_frame_shape
    bbox_area = float(max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1))
    overlap_area = 0.0
    for zone in self._static_zones:
      overlap_area += self._intersection_area(bbox, self._zone_to_bbox(zone, frame_w, frame_h))
    return overlap_area / bbox_area

  def active_background_overlap_ratio(self, bbox: BBox) -> float:
    if not self._last_active_regions:
      return 0.0
    bbox_area = float(max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1))
    overlap_area = 0.0
    for region in self._last_active_regions:
      overlap_area += self._intersection_area(bbox, region)
    return overlap_area / bbox_area

  def _create_subtractor(self):
    return cv2.createBackgroundSubtractorMOG2(
      history=max(30, int(settings.scene_background_history)),
      varThreshold=max(4.0, float(settings.scene_background_var_threshold)),
      detectShadows=False,
    )

  def _extract_regions(
    self,
    mask: np.ndarray,
    frame_w: int,
    frame_h: int,
    limit: int,
  ) -> List[Dict[str, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(
      8.0,
      float(frame_w * frame_h) * float(settings.scene_background_contour_min_area_ratio),
    )
    boxes: List[Tuple[int, int, int, int, float]] = []
    for contour in contours:
      area = float(cv2.contourArea(contour))
      if area < min_area:
        continue
      x, y, w, h = cv2.boundingRect(contour)
      boxes.append((x, y, x + w, y + h, area))
    boxes.sort(key=lambda item: item[4], reverse=True)

    regions: List[Dict[str, float]] = []
    for x1, y1, x2, y2, _ in boxes[:max(0, limit)]:
      regions.append(
        {
          "x": round(x1 / max(frame_w, 1), 5),
          "y": round(y1 / max(frame_h, 1), 5),
          "w": round((x2 - x1) / max(frame_w, 1), 5),
          "h": round((y2 - y1) / max(frame_h, 1), 5),
        }
      )
    return regions

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
  def _zone_to_bbox(zone: Tuple[float, float, float, float], frame_w: int, frame_h: int) -> BBox:
    x, y, w, h = zone
    x1 = int(round(x * max(frame_w, 1)))
    y1 = int(round(y * max(frame_h, 1)))
    x2 = int(round((x + w) * max(frame_w, 1)))
    y2 = int(round((y + h) * max(frame_h, 1)))
    x1 = max(0, min(x1, frame_w))
    y1 = max(0, min(y1, frame_h))
    x2 = max(x1 + 1, min(x2, frame_w))
    y2 = max(y1 + 1, min(y2, frame_h))
    return (x1, y1, x2, y2)

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

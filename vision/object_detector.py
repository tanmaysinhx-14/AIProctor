from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import ClassVar, Dict, List, Optional, Tuple

import torch
from ultralytics import YOLO

from config import settings

BBox = Tuple[int, int, int, int]


@dataclass
class DetectedObject:
  label: str
  confidence: float
  bbox: Optional[BBox] = None


class YoloObjectDetector:
  _shared_model: ClassVar[Optional[YOLO]] = None
  _shared_device: ClassVar[Optional[str]] = None
  _shared_names: ClassVar[Optional[Dict[int, str]]] = None
  _shared_gadget_model: ClassVar[Optional[YOLO]] = None
  _shared_gadget_names: ClassVar[Optional[Dict[int, str]]] = None
  _shared_gadget_path: ClassVar[str] = ""
  _init_lock: ClassVar[Lock] = Lock()
  _predict_lock: ClassVar[Lock] = Lock()

  def __init__(self) -> None:
    self.confidence = settings.yolo_confidence
    self.iou = settings.yolo_iou
    self.min_interval = 1.0 / max(settings.yolo_inference_fps, 0.1)
    self.person_min_interval = 1.0 / max(settings.person_detector_inference_fps, 0.1)
    self.label_map = settings.object_label_map
    self.allowed_labels = set(self.label_map.values())

    self._last_inference_ts = 0.0
    self._cached_results: List[DetectedObject] = []
    self._last_person_inference_ts = 0.0
    self._cached_person_results: List[DetectedObject] = []
    self._phone_streak = 0
    self._phone_anchor_bbox: Optional[BBox] = None
    self._phone_anchor_ttl = 0
    self._person_min_confidence = float(settings.person_min_confidence)

    self._phone_min_confidence = float(settings.phone_min_confidence)
    self._phone_min_area_ratio = float(settings.phone_min_area_ratio)
    self._phone_min_aspect_ratio = float(settings.phone_min_aspect_ratio)
    self._phone_max_aspect_ratio = float(settings.phone_max_aspect_ratio)
    self._phone_confirm_frames = max(1, int(settings.phone_confirm_frames))

    self._primary_phone_classes = {"cell phone"}
    self._surrogate_phone_classes = {"remote", "laptop", "tv", "mouse", "book"}

    self._last_probe_ts = 0.0
    self._probe_min_gap_sec = 0.12
    self._last_phone_debug: Dict[str, float] = {}
    self._background_motion_zones: List[Tuple[float, float, float, float]] = []
    self._background_zone_iou_reject = 0.22
    self._gadget_min_confidence = float(settings.gadget_min_confidence)
    self._gadget_confirm_frames = max(1, int(settings.gadget_confirm_frames))
    self._gadget_roi_expand_ratio = float(settings.gadget_roi_expand_ratio)
    self._gadget_anchor_iou = float(settings.gadget_anchor_iou)
    self._gadget_ttl_frames = max(1, int(settings.gadget_ttl_frames))
    self._gadget_anchor_bbox: Optional[BBox] = None
    self._gadget_streak = 0
    self._gadget_ttl = 0
    self._last_gadget_debug: Dict[str, float] = {}
    self._gadget_enabled = self._ensure_gadget_model_loaded(settings.gadget_model_path)

    self._ensure_model_loaded(settings.yolo_model_path)
    self._primary_phone_class_ids = self._class_ids_for_labels(self._primary_phone_classes)
    self._probe_phone_class_ids = self._class_ids_for_labels(
      self._primary_phone_classes | self._surrogate_phone_classes
    )
    self._person_class_ids = self._class_ids_for_labels({"person"})

    self._refresh_phone_gates()
    self._reset_phone_debug()
    self._reset_gadget_debug()

  def set_inference_fps(self, fps: float) -> None:
    self.min_interval = 1.0 / max(float(fps), 0.1)

  def set_person_inference_fps(self, fps: float) -> None:
    self.person_min_interval = 1.0 / max(float(fps), 0.1)

  def set_runtime_device(self, device: str) -> str:
    with self._init_lock:
      if self._shared_model is None:
        return "uninitialized"

      target = str(device or "cpu").lower()
      if target.startswith("cuda"):
        if not torch.cuda.is_available():
          target = "cpu"
        else:
          target = "cuda:0"
      else:
        target = "cpu"

      if self._shared_device != target:
        self._shared_model.to(target)
        if self._shared_gadget_model is not None:
          self._shared_gadget_model.to(target)
        self._shared_device = target
      return str(self._shared_device)

  @classmethod
  def _ensure_model_loaded(cls, model_path: str) -> None:
    with cls._init_lock:
      if cls._shared_model is not None:
        return

      if torch.cuda.is_available():
        cls._shared_device = "cuda:0"
      else:
        cls._shared_device = "cpu"
      model = YOLO(model_path)
      model.to(cls._shared_device)
      cls._shared_model = model

      raw_names = model.model.names
      if isinstance(raw_names, dict):
        cls._shared_names = {int(k): str(v) for k, v in raw_names.items()}
      else:
        cls._shared_names = {idx: str(name) for idx, name in enumerate(raw_names)}

  @classmethod
  def _ensure_gadget_model_loaded(cls, model_path: str) -> bool:
    path_text = str(model_path or "").strip()
    if not path_text:
      return False
    path = Path(path_text)
    if not path.is_file():
      return False

    with cls._init_lock:
      if cls._shared_gadget_model is not None and cls._shared_gadget_path == str(path):
        return True

      device = cls._shared_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
      model = YOLO(str(path))
      model.to(device)
      cls._shared_gadget_model = model
      cls._shared_gadget_path = str(path)

      raw_names = model.model.names
      if isinstance(raw_names, dict):
        cls._shared_gadget_names = {int(k): str(v) for k, v in raw_names.items()}
      else:
        cls._shared_gadget_names = {idx: str(name) for idx, name in enumerate(raw_names)}

    return True

  def reset_phone_state(self) -> None:
    self._phone_streak = 0
    self._phone_anchor_bbox = None
    self._phone_anchor_ttl = 0
    self._last_probe_ts = 0.0
    self._gadget_anchor_bbox = None
    self._gadget_streak = 0
    self._gadget_ttl = 0

  def set_phone_overrides(self, overrides: Dict[str, float]) -> None:
    if not overrides:
      return

    if "phone_min_confidence" in overrides:
      # Keep calibration from becoming too strict for real-time phone recall.
      self._phone_min_confidence = float(max(0.0, min(0.62, overrides["phone_min_confidence"])))

    if "phone_min_area_ratio" in overrides:
      self._phone_min_area_ratio = float(max(0.0, min(0.06, overrides["phone_min_area_ratio"])))

    if "phone_min_aspect_ratio" in overrides:
      self._phone_min_aspect_ratio = float(max(0.05, min(10.0, overrides["phone_min_aspect_ratio"])))

    if "phone_max_aspect_ratio" in overrides:
      self._phone_max_aspect_ratio = float(
        max(self._phone_min_aspect_ratio, min(10.0, overrides["phone_max_aspect_ratio"]))
      )

    if "phone_confirm_frames" in overrides:
      self._phone_confirm_frames = max(1, min(2, int(round(overrides["phone_confirm_frames"]))))

    self._refresh_phone_gates()

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
      x = float(max(0.0, min(x, 1.0)))
      y = float(max(0.0, min(y, 1.0)))
      w = float(max(0.0, min(w, 1.0 - x)))
      h = float(max(0.0, min(h, 1.0 - y)))
      if w <= 0.0 or h <= 0.0:
        continue
      parsed.append((x, y, w, h))
    self._background_motion_zones = parsed[:12]

  def background_motion_zone_count(self) -> int:
    return len(self._background_motion_zones)

  def current_phone_thresholds(self) -> Dict[str, float]:
    return {
      "phone_min_confidence": round(float(self._phone_min_confidence), 3),
      "phone_min_area_ratio": round(float(self._phone_min_area_ratio), 5),
      "phone_min_aspect_ratio": round(float(self._phone_min_aspect_ratio), 3),
      "phone_max_aspect_ratio": round(float(self._phone_max_aspect_ratio), 3),
      "phone_confirm_frames": float(self._phone_confirm_frames),
      "phone_relaxed_confidence": round(float(self._phone_relaxed_confidence), 3),
      "phone_relaxed_area_ratio": round(float(self._phone_relaxed_area_ratio), 5),
      "phone_immediate_confidence": round(float(self._phone_immediate_confidence), 3),
      "phone_probe_confidence": round(float(self._phone_probe_confidence), 3),
    }

  def phone_debug(self) -> Dict[str, float]:
    return dict(self._last_phone_debug)

  def gadget_debug(self) -> Dict[str, float]:
    return dict(self._last_gadget_debug)

  def detect_persons(self, frame_bgr) -> List[DetectedObject]:
    now = monotonic()
    if now - self._last_person_inference_ts < self.person_min_interval:
      return list(self._cached_person_results)

    with self._predict_lock:
      now = monotonic()
      if now - self._last_person_inference_ts < self.person_min_interval:
        return list(self._cached_person_results)

      if self._shared_model is None or self._shared_device is None:
        return []

      try:
        results = self._shared_model.predict(
          source=frame_bgr,
          conf=min(self.confidence, self._person_min_confidence),
          iou=self.iou,
          device=self._shared_device,
          classes=self._person_class_ids or None,
          verbose=False,
        )
      except Exception as exc:
        if str(self._shared_device).startswith("cuda"):
          self._fallback_to_cpu()
          results = self._shared_model.predict(
            source=frame_bgr,
            conf=min(self.confidence, self._person_min_confidence),
            iou=self.iou,
            device=self._shared_device,
            classes=self._person_class_ids or None,
            verbose=False,
          )
        else:
          raise exc

      parsed = self._parse_person_predictions(results[0], frame_shape=frame_bgr.shape)
      self._cached_person_results = parsed
      self._last_person_inference_ts = now
      return list(parsed)

  def detect(
    self,
    frame_bgr,
    face_boxes: Optional[List[BBox]] = None,
    person_boxes_hint: Optional[List[BBox]] = None,
  ) -> List[DetectedObject]:
    now = monotonic()
    if now - self._last_inference_ts < self.min_interval:
      return list(self._cached_results)

    with self._predict_lock:
      now = monotonic()
      if now - self._last_inference_ts < self.min_interval:
        return list(self._cached_results)

      if self._shared_model is None or self._shared_device is None:
        return []

      try:
        primary_results = self._shared_model.predict(
          source=frame_bgr,
          conf=self.confidence,
          iou=self.iou,
          device=self._shared_device,
          verbose=False,
        )
      except Exception as exc:
        if str(self._shared_device).startswith("cuda"):
          self._fallback_to_cpu()
          primary_results = self._shared_model.predict(
            source=frame_bgr,
            conf=self.confidence,
            iou=self.iou,
            device=self._shared_device,
            verbose=False,
          )
        else:
          raise exc

      parsed, primary_debug = self._parse_predictions(
        result=primary_results[0],
        frame_shape=frame_bgr.shape,
        face_boxes=face_boxes or [],
        person_boxes_hint=person_boxes_hint or [],
        probe_mode=False,
      )

      probe_debug = self._default_debug_snapshot()
      has_phone = any(item.label == "phone" for item in parsed)
      if not has_phone and self._should_run_probe(now):
        probe_phones, probe_debug = self._probe_phone_candidates(frame_bgr, face_boxes or [])
        parsed.extend(probe_phones)

      gadget_objects = self._detect_gadgets(frame_bgr, face_boxes or [])
      parsed.extend(gadget_objects)

      stabilized = self._stabilize_phone(parsed)
      self._cached_results = stabilized
      self._last_inference_ts = now
      self._merge_phone_debug(primary_debug, probe_debug, stabilized)
      return list(stabilized)

  def runtime_device(self) -> str:
    if self._shared_model is None:
      return "uninitialized"
    try:
      return str(next(self._shared_model.model.parameters()).device)
    except Exception:
      return str(self._shared_device or "unknown")

  def _fallback_to_cpu(self) -> None:
    if self._shared_model is None:
      return
    try:
      self._shared_model.to("cpu")
      if self._shared_gadget_model is not None:
        self._shared_gadget_model.to("cpu")
      self._shared_device = "cpu"
    except Exception:
      self._shared_device = "cpu"

  def _refresh_phone_gates(self) -> None:
    self._phone_relaxed_confidence = max(0.18, self._phone_min_confidence - 0.20)
    self._phone_relaxed_area_ratio = max(0.0004, self._phone_min_area_ratio * 0.45)
    self._phone_immediate_confidence = min(0.95, max(0.62, self._phone_min_confidence + 0.12))
    self._phone_probe_confidence = max(0.12, self._phone_relaxed_confidence - 0.08)

  def _class_ids_for_labels(self, labels: set[str]) -> List[int]:
    if self._shared_names is None:
      return []
    ids: List[int] = []
    for idx, name in self._shared_names.items():
      if name in labels:
        ids.append(int(idx))
    return ids

  def _should_run_probe(self, now: float) -> bool:
    if now - self._last_probe_ts < self._probe_min_gap_sec:
      return False
    self._last_probe_ts = now
    return True

  def _probe_phone_candidates(
    self,
    frame_bgr,
    face_boxes: List[BBox],
  ) -> Tuple[List[DetectedObject], Dict[str, float]]:
    if self._shared_model is None or self._shared_device is None:
      return [], self._default_debug_snapshot()

    probe_iou = max(0.30, self.iou - 0.12)
    classes = self._probe_phone_class_ids or self._primary_phone_class_ids or None

    probe_results = self._shared_model.predict(
      source=frame_bgr,
      conf=self._phone_probe_confidence,
      iou=probe_iou,
      device=self._shared_device,
      classes=classes,
      imgsz=736,
      augment=True,
      verbose=False,
    )

    parsed, debug = self._parse_predictions(
        result=probe_results[0],
        frame_shape=frame_bgr.shape,
        face_boxes=face_boxes,
        person_boxes_hint=[],
        probe_mode=True,
      )
    phones = [obj for obj in parsed if obj.label == "phone"]
    debug["probeUsed"] = 1.0
    debug["probeAccepted"] = float(len(phones))
    return phones, debug

  def _parse_predictions(
    self,
    result,
    frame_shape: Tuple[int, ...],
    face_boxes: List[BBox],
    person_boxes_hint: List[BBox],
    probe_mode: bool,
  ) -> Tuple[List[DetectedObject], Dict[str, float]]:
    parsed: List[DetectedObject] = []
    debug = self._default_debug_snapshot()
    debug["probeMode"] = 1.0 if probe_mode else 0.0

    boxes = getattr(result, "boxes", None)
    if boxes is None:
      return parsed, debug

    frame_h, frame_w = frame_shape[:2]
    raw_detections: List[Dict[str, object]] = []

    for box in boxes:
      class_id = int(box.cls[0].item())
      confidence = float(box.conf[0].item())
      raw_label = self._label_for_class(class_id)
      normalized = self.label_map.get(raw_label, raw_label)

      if normalized not in self.allowed_labels:
        continue

      xyxy = box.xyxy[0].tolist()
      x1, y1, x2, y2 = (int(v) for v in xyxy)
      x1 = max(0, min(x1, frame_w - 1))
      x2 = max(0, min(x2, frame_w - 1))
      y1 = max(0, min(y1, frame_h - 1))
      y2 = max(0, min(y2, frame_h - 1))
      bbox = (x1, y1, x2, y2)

      raw_detections.append(
        {
          "label": normalized,
          "confidence": confidence,
          "bbox": bbox,
          "raw_label": raw_label,
        }
      )

    person_boxes = [item["bbox"] for item in raw_detections if item["label"] == "person"]
    person_boxes.extend([bbox for bbox in person_boxes_hint if bbox is not None])
    person_boxes = self._dedupe_boxes(person_boxes)
    focus_regions = self._build_focus_regions(face_boxes, person_boxes, frame_w, frame_h)
    debug["personContext"] = float(len(person_boxes))
    debug["focusRegions"] = float(len(focus_regions))

    for item in raw_detections:
      label = str(item["label"])
      confidence = float(item["confidence"])
      bbox = item["bbox"]
      raw_label = str(item["raw_label"])

      if label != "phone":
        parsed.append(
          DetectedObject(
            label=label,
            confidence=confidence,
            bbox=bbox,
          )
        )
        continue

      debug["candidates"] += 1.0
      in_background_motion_zone = self._is_in_background_motion_zone(bbox, frame_w, frame_h)
      accepted, accepted_by_near_camera = self._passes_phone_filters(
        confidence=confidence,
        bbox=bbox,
        frame_w=frame_w,
        frame_h=frame_h,
        raw_label=raw_label,
        focus_regions=focus_regions,
        person_boxes=person_boxes,
        probe_mode=probe_mode,
        background_motion_zone_hit=in_background_motion_zone,
      )
      if not accepted:
        continue

      debug["accepted"] += 1.0
      if accepted_by_near_camera:
        debug["nearCameraAccepted"] += 1.0
      if raw_label in self._primary_phone_classes:
        debug["primaryAccepted"] += 1.0
      else:
        debug["surrogateAccepted"] += 1.0

      parsed.append(
        DetectedObject(
          label=label,
          confidence=confidence,
          bbox=bbox,
        )
      )

    return parsed, debug

  def _parse_person_predictions(self, result, frame_shape: Tuple[int, ...]) -> List[DetectedObject]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
      return []

    frame_h, frame_w = frame_shape[:2]
    persons: List[DetectedObject] = []
    for box in boxes:
      class_id = int(box.cls[0].item())
      confidence = float(box.conf[0].item())
      raw_label = self._label_for_class(class_id)
      normalized = self.label_map.get(raw_label, raw_label)
      if normalized != "person" or confidence < self._person_min_confidence:
        continue

      xyxy = box.xyxy[0].tolist()
      x1, y1, x2, y2 = (int(v) for v in xyxy)
      x1 = max(0, min(x1, frame_w - 1))
      x2 = max(0, min(x2, frame_w - 1))
      y1 = max(0, min(y1, frame_h - 1))
      y2 = max(0, min(y2, frame_h - 1))
      if x2 <= x1 or y2 <= y1:
        continue
      width = float(max(x2 - x1, 1))
      height = float(max(y2 - y1, 1))
      height_width_ratio = height / width
      if height_width_ratio < max(0.95, float(settings.person_min_height_width_ratio) * 0.80):
        continue
      if height_width_ratio > float(settings.person_max_height_width_ratio) * 1.15:
        continue
      persons.append(
        DetectedObject(
          label="person",
          confidence=confidence,
          bbox=(x1, y1, x2, y2),
        )
      )
    return self._dedupe_detected_objects(persons)

  def _detect_gadgets(self, frame_bgr, face_boxes: List[BBox]) -> List[DetectedObject]:
    self._reset_gadget_debug()
    if not self._gadget_enabled or self._shared_gadget_model is None:
      self._decay_gadget_state()
      return []
    if not face_boxes:
      self._decay_gadget_state()
      return []

    frame_h, frame_w = frame_bgr.shape[:2]
    roi_boxes = [
      self._expand_bbox(
        bbox=bbox,
        frame_w=frame_w,
        frame_h=frame_h,
        x_ratio=self._gadget_roi_expand_ratio,
        y_ratio=self._gadget_roi_expand_ratio * 0.80,
      )
      for bbox in self._dedupe_boxes(face_boxes)
    ]

    candidates: List[DetectedObject] = []
    for roi_bbox in roi_boxes:
      x1, y1, x2, y2 = roi_bbox
      if x2 <= x1 or y2 <= y1:
        continue
      roi = frame_bgr[y1:y2, x1:x2]
      if roi.size == 0:
        continue
      try:
        results = self._shared_gadget_model.predict(
          source=roi,
          conf=self._gadget_min_confidence,
          iou=max(0.22, self.iou - 0.10),
          device=self._shared_device,
          verbose=False,
        )
      except Exception as exc:
        if str(self._shared_device).startswith("cuda"):
          self._fallback_to_cpu()
          results = self._shared_gadget_model.predict(
            source=roi,
            conf=self._gadget_min_confidence,
            iou=max(0.22, self.iou - 0.10),
            device=self._shared_device,
            verbose=False,
          )
        else:
          raise exc

      boxes = getattr(results[0], "boxes", None) if results else None
      if boxes is None:
        continue
      for box in boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        raw_label = self._gadget_label_for_class(class_id)
        normalized = self._normalize_gadget_label(raw_label)
        if normalized is None or confidence < self._gadget_min_confidence:
          continue
        xyxy = box.xyxy[0].tolist()
        rx1, ry1, rx2, ry2 = (int(v) for v in xyxy)
        bx1 = max(0, min(x1 + rx1, frame_w - 1))
        by1 = max(0, min(y1 + ry1, frame_h - 1))
        bx2 = max(0, min(x1 + rx2, frame_w - 1))
        by2 = max(0, min(y1 + ry2, frame_h - 1))
        if bx2 <= bx1 or by2 <= by1:
          continue
        candidates.append(
          DetectedObject(
            label="audio_device",
            confidence=confidence,
            bbox=(bx1, by1, bx2, by2),
          )
        )

    deduped = self._dedupe_detected_objects(candidates)
    confirmed = self._stabilize_gadgets(deduped)
    self._last_gadget_debug = self._default_gadget_debug_snapshot()
    self._last_gadget_debug["candidates"] = float(len(deduped))
    self._last_gadget_debug["confirmed"] = float(len(confirmed))
    self._last_gadget_debug["streak"] = float(self._gadget_streak)
    self._last_gadget_debug["gadget_detected"] = 1.0 if confirmed else 0.0
    return confirmed

  def _stabilize_gadgets(self, detections: List[DetectedObject]) -> List[DetectedObject]:
    if not detections:
      self._decay_gadget_state()
      return []

    best = max(
      detections,
      key=lambda item: float(item.confidence) * float(self._bbox_area(item.bbox)),
    )
    if self._gadget_anchor_bbox is not None and best.bbox is not None:
      if self._bbox_iou(self._gadget_anchor_bbox, best.bbox) >= self._gadget_anchor_iou:
        self._gadget_streak += 1
      else:
        self._gadget_streak = 1
    else:
      self._gadget_streak = 1
    self._gadget_anchor_bbox = best.bbox
    self._gadget_ttl = self._gadget_ttl_frames

    if self._gadget_streak < self._gadget_confirm_frames:
      return []
    return [best]

  def _decay_gadget_state(self) -> None:
    self._gadget_streak = max(0, self._gadget_streak - 1)
    self._gadget_ttl = max(0, self._gadget_ttl - 1)
    if self._gadget_ttl <= 0:
      self._gadget_anchor_bbox = None

  def _gadget_label_for_class(self, class_id: int) -> str:
    if self._shared_gadget_names is None:
      return ""
    return str(self._shared_gadget_names.get(int(class_id), ""))

  @staticmethod
  def _normalize_gadget_label(raw_label: str) -> Optional[str]:
    label = str(raw_label or "").strip().lower().replace("-", " ").replace("_", " ")
    if not label:
      return None
    gadget_terms = (
      "headphone",
      "headset",
      "earphone",
      "earbud",
      "ear piece",
      "earpiece",
      "airpod",
      "bluetooth",
    )
    return "audio_device" if any(term in label for term in gadget_terms) else None

  def _passes_phone_filters(
    self,
    confidence: float,
    bbox: BBox,
    frame_w: int,
    frame_h: int,
    raw_label: str,
    focus_regions: List[BBox],
    person_boxes: List[BBox],
    probe_mode: bool,
    background_motion_zone_hit: bool,
  ) -> Tuple[bool, bool]:
    x1, y1, x2, y2 = bbox
    width = float(max(x2 - x1, 1))
    height = float(max(y2 - y1, 1))
    frame_area = float(max(frame_w * frame_h, 1))
    area_ratio = (width * height) / frame_area
    aspect_ratio = width / height

    relaxed_min_aspect = max(0.08, self._phone_min_aspect_ratio * 0.55)
    relaxed_max_aspect = min(9.0, self._phone_max_aspect_ratio * 1.5)
    strict_geometry = (
      area_ratio >= self._phone_min_area_ratio
      and self._phone_min_aspect_ratio <= aspect_ratio <= self._phone_max_aspect_ratio
    )
    relaxed_geometry = (
      area_ratio >= self._phone_relaxed_area_ratio
      and relaxed_min_aspect <= aspect_ratio <= relaxed_max_aspect
    )

    near_focus = self._near_regions(bbox, focus_regions, frame_w, frame_h, radius_scale=1.8)
    near_person = self._near_regions(bbox, person_boxes, frame_w, frame_h, radius_scale=2.1)
    temporal_match = self._phone_anchor_bbox is not None and self._bbox_iou(self._phone_anchor_bbox, bbox) >= 0.12
    near_camera = self._is_near_camera_candidate(
      bbox=bbox,
      frame_w=frame_w,
      frame_h=frame_h,
      area_ratio=area_ratio,
      aspect_ratio=aspect_ratio,
    )

    if raw_label in self._primary_phone_classes:
      min_conf = self._phone_probe_confidence if probe_mode else self._phone_relaxed_confidence
      if confidence < min_conf:
        return False, False

      if not relaxed_geometry and not near_camera:
        return False, False

      near_camera_conf = max(
        self._phone_probe_confidence,
        self._phone_relaxed_confidence - (0.16 if probe_mode else 0.08),
      )
      if near_camera and confidence >= near_camera_conf:
        return True, True

      score = 0
      if confidence >= self._phone_min_confidence:
        score += 2
      else:
        score += 1

      if strict_geometry:
        score += 2
      elif relaxed_geometry:
        score += 1

      if near_focus:
        score += 1
      if near_person:
        score += 1
      if temporal_match:
        score += 1
      if near_camera:
        score += 2

      if confidence >= self._phone_immediate_confidence and (near_focus or near_person or strict_geometry or near_camera):
        return True, near_camera

      if probe_mode:
        if near_focus or near_person or near_camera or temporal_match:
          return score >= 2, near_camera
        return False, False

      if near_focus or near_person:
        if background_motion_zone_hit and confidence < (self._phone_min_confidence + 0.08):
          return False, False
        return score >= 3, near_camera

      if near_camera:
        return score >= 3, True

      if background_motion_zone_hit:
        return False, False

      return score >= 5 and confidence >= (self._phone_min_confidence + 0.02), False

    if raw_label not in self._surrogate_phone_classes:
      return False, False

    if raw_label == "mouse":
      surrogate_conf = max(0.28, self._phone_probe_confidence if probe_mode else self._phone_relaxed_confidence - 0.06)
    elif raw_label == "book":
      surrogate_conf = max(0.40, self._phone_relaxed_confidence + (0.04 if not probe_mode else -0.02))
    else:
      surrogate_conf = max(0.40, self._phone_relaxed_confidence + (0.06 if not probe_mode else 0.0))

    if confidence < surrogate_conf:
      return False, False

    min_area_ratio = max(
      self._phone_min_area_ratio * (1.2 if raw_label in {"remote", "mouse"} else 1.4),
      0.0018,
    )
    if area_ratio < min_area_ratio or area_ratio > 0.35:
      return False, False

    if aspect_ratio < 0.16 or aspect_ratio > 5.4:
      return False, False

    if raw_label == "laptop" and area_ratio > 0.30:
      return False, False

    if raw_label == "tv":
      if confidence < max(0.62, surrogate_conf + 0.1):
        return False, False
      if area_ratio > 0.22:
        return False, False

    if raw_label == "book" and not (near_camera or near_focus or temporal_match):
      return False, False

    if raw_label == "mouse" and area_ratio > 0.08 and not near_camera:
      return False, False

    if not (near_focus or near_person or temporal_match or near_camera):
      return False, False

    if background_motion_zone_hit and not (near_focus or near_person or near_camera):
      return False, False

    if not (temporal_match or near_camera or near_focus) and confidence < max(0.56, surrogate_conf + 0.04):
      return False, False

    return True, near_camera

  def _is_near_camera_candidate(
    self,
    bbox: BBox,
    frame_w: int,
    frame_h: int,
    area_ratio: float,
    aspect_ratio: float,
  ) -> bool:
    if area_ratio >= 0.032 and 0.08 <= aspect_ratio <= 7.0:
      return True

    min_area = max(self._phone_relaxed_area_ratio * 1.25, 0.0022)
    if area_ratio < min_area:
      return False

    if aspect_ratio < 0.08 or aspect_ratio > 7.0:
      return False

    cx, cy = self._center(bbox)
    dx = abs((cx / max(float(frame_w), 1.0)) - 0.5)
    dy = abs((cy / max(float(frame_h), 1.0)) - 0.5)
    center_distance = (dx * dx + dy * dy) ** 0.5
    return center_distance <= 0.62

  def _stabilize_phone(self, detections: List[DetectedObject]) -> List[DetectedObject]:
    phones = [det for det in detections if det.label == "phone"]
    others = [det for det in detections if det.label != "phone"]

    if phones:
      self._phone_streak = min(self._phone_streak + 1, self._phone_confirm_frames + 5)
      best_phone = max(phones, key=lambda det: float(det.confidence))
      if best_phone.bbox is not None:
        self._phone_anchor_bbox = best_phone.bbox
        self._phone_anchor_ttl = max(self._phone_confirm_frames + 3, 5)
    else:
      self._phone_streak = max(self._phone_streak - 2, 0)
      if self._phone_anchor_ttl > 0:
        self._phone_anchor_ttl -= 1
      if self._phone_anchor_ttl <= 0:
        self._phone_anchor_bbox = None

    filtered_phones: List[DetectedObject] = []
    for det in phones:
      immediate = float(det.confidence) >= self._phone_immediate_confidence
      anchor_match = (
        self._phone_anchor_bbox is not None
        and det.bbox is not None
        and self._bbox_iou(self._phone_anchor_bbox, det.bbox) >= 0.10
      )
      if immediate or anchor_match or self._phone_streak >= self._phone_confirm_frames:
        filtered_phones.append(det)

    return others + filtered_phones

  def _label_for_class(self, class_id: int) -> str:
    if self._shared_names is None:
      return str(class_id)
    return self._shared_names.get(class_id, str(class_id))

  def _build_focus_regions(
    self,
    face_boxes: List[BBox],
    person_boxes: List[BBox],
    frame_w: int,
    frame_h: int,
  ) -> List[BBox]:
    regions: List[BBox] = []
    for bbox in face_boxes:
      regions.append(self._expand_bbox(bbox, frame_w, frame_h, x_ratio=1.35, y_ratio=2.0))

    for bbox in person_boxes:
      x1, y1, x2, y2 = bbox
      person_h = max(y2 - y1, 1)
      upper_body = (
        x1,
        y1,
        x2,
        min(frame_h - 1, y1 + int(person_h * 0.82)),
      )
      regions.append(self._expand_bbox(upper_body, frame_w, frame_h, x_ratio=0.25, y_ratio=0.2))

    return regions

  def _near_regions(
    self,
    bbox: BBox,
    regions: List[BBox],
    frame_w: int,
    frame_h: int,
    radius_scale: float,
  ) -> bool:
    if not regions:
      return False

    cx, cy = self._center(bbox)
    bw = float(max(bbox[2] - bbox[0], 1))
    bh = float(max(bbox[3] - bbox[1], 1))
    base_radius = max(bw, bh) * 0.5

    for region in regions:
      if self._bbox_iou(bbox, region) > 0.01:
        return True

      rcx, rcy = self._center(region)
      rw = float(max(region[2] - region[0], 1))
      rh = float(max(region[3] - region[1], 1))
      region_radius = max(rw, rh) * 0.5
      allowed = (base_radius + region_radius) * max(radius_scale, 0.1)
      allowed_norm = allowed / max(float(max(frame_w, frame_h)), 1.0)

      dx = (cx - rcx) / max(float(frame_w), 1.0)
      dy = (cy - rcy) / max(float(frame_h), 1.0)
      distance = (dx * dx + dy * dy) ** 0.5
      if distance <= max(allowed_norm, 0.1):
        return True

    return False

  def _expand_bbox(
    self,
    bbox: BBox,
    frame_w: int,
    frame_h: int,
    x_ratio: float,
    y_ratio: float,
  ) -> BBox:
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    pad_x = int(round(bw * x_ratio))
    pad_y = int(round(bh * y_ratio))
    return (
      max(0, x1 - pad_x),
      max(0, y1 - pad_y),
      min(frame_w - 1, x2 + pad_x),
      min(frame_h - 1, y2 + pad_y),
    )

  def _center(self, bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

  def _dedupe_detected_objects(self, detections: List[DetectedObject]) -> List[DetectedObject]:
    if len(detections) <= 1:
      return list(detections)
    ordered = sorted(
      detections,
      key=lambda item: float(item.confidence) * float(self._bbox_area(item.bbox)),
      reverse=True,
    )
    kept: List[DetectedObject] = []
    kept_boxes: List[BBox] = []
    for detection in ordered:
      bbox = detection.bbox
      if bbox is None:
        continue
      if any(self._bbox_iou(bbox, other) >= 0.62 for other in kept_boxes):
        continue
      kept.append(detection)
      kept_boxes.append(bbox)
    return kept

  def _dedupe_boxes(self, boxes: List[BBox]) -> List[BBox]:
    if len(boxes) <= 1:
      return list(boxes)
    ordered = sorted(boxes, key=self._bbox_area, reverse=True)
    kept: List[BBox] = []
    for bbox in ordered:
      if any(self._bbox_iou(bbox, other) >= 0.62 for other in kept):
        continue
      kept.append(bbox)
    return kept

  @staticmethod
  def _bbox_area(bbox: Optional[BBox]) -> int:
    if bbox is None:
      return 0
    x1, y1, x2, y2 = bbox
    return max(1, x2 - x1) * max(1, y2 - y1)

  def _bbox_iou(self, a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = float(inter_w * inter_h)
    if inter_area <= 0.0:
      return 0.0

    area_a = float(max((ax2 - ax1) * (ay2 - ay1), 1))
    area_b = float(max((bx2 - bx1) * (by2 - by1), 1))
    union = max(area_a + area_b - inter_area, 1.0)
    return inter_area / union

  def _is_in_background_motion_zone(self, bbox: BBox, frame_w: int, frame_h: int) -> bool:
    if not self._background_motion_zones:
      return False
    for zone in self._background_motion_zones:
      zone_bbox = self._zone_to_bbox(zone, frame_w, frame_h)
      if self._bbox_iou(bbox, zone_bbox) >= self._background_zone_iou_reject:
        return True
    return False

  @staticmethod
  def _zone_to_bbox(zone: Tuple[float, float, float, float], frame_w: int, frame_h: int) -> BBox:
    x, y, w, h = zone
    x1 = int(round(x * max(frame_w - 1, 1)))
    y1 = int(round(y * max(frame_h - 1, 1)))
    x2 = int(round((x + w) * max(frame_w - 1, 1)))
    y2 = int(round((y + h) * max(frame_h - 1, 1)))
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(0, min(max(x2, x1 + 1), frame_w - 1))
    y2 = max(0, min(max(y2, y1 + 1), frame_h - 1))
    return (x1, y1, x2, y2)

  def _default_debug_snapshot(self) -> Dict[str, float]:
    return {
      "candidates": 0.0,
      "accepted": 0.0,
      "primaryAccepted": 0.0,
      "surrogateAccepted": 0.0,
      "nearCameraAccepted": 0.0,
      "personContext": 0.0,
      "focusRegions": 0.0,
      "probeMode": 0.0,
      "probeUsed": 0.0,
      "probeAccepted": 0.0,
    }

  def _default_gadget_debug_snapshot(self) -> Dict[str, float]:
    return {
      "enabled": 1.0 if self._gadget_enabled and self._shared_gadget_model is not None else 0.0,
      "candidates": 0.0,
      "confirmed": 0.0,
      "streak": float(self._gadget_streak),
      "gadget_detected": 0.0,
    }

  def _reset_gadget_debug(self) -> None:
    self._last_gadget_debug = self._default_gadget_debug_snapshot()

  def _reset_phone_debug(self) -> None:
    self._last_phone_debug = self._default_debug_snapshot()
    self._last_phone_debug.update(
      {
        "streak": float(self._phone_streak),
        "confirmFrames": float(self._phone_confirm_frames),
        "visibleAfterStabilize": 0.0,
      }
    )

  def _merge_phone_debug(
    self,
    primary_debug: Dict[str, float],
    probe_debug: Dict[str, float],
    stabilized: List[DetectedObject],
  ) -> None:
    visible_after = float(sum(1 for obj in stabilized if obj.label == "phone"))
    self._last_phone_debug = {
      "candidates": float(primary_debug.get("candidates", 0.0) + probe_debug.get("candidates", 0.0)),
      "accepted": float(primary_debug.get("accepted", 0.0) + probe_debug.get("accepted", 0.0)),
      "primaryAccepted": float(primary_debug.get("primaryAccepted", 0.0) + probe_debug.get("primaryAccepted", 0.0)),
      "surrogateAccepted": float(primary_debug.get("surrogateAccepted", 0.0) + probe_debug.get("surrogateAccepted", 0.0)),
      "nearCameraAccepted": float(primary_debug.get("nearCameraAccepted", 0.0) + probe_debug.get("nearCameraAccepted", 0.0)),
      "personContext": float(primary_debug.get("personContext", 0.0)),
      "focusRegions": float(primary_debug.get("focusRegions", 0.0)),
      "probeUsed": float(probe_debug.get("probeUsed", 0.0)),
      "probeAccepted": float(probe_debug.get("probeAccepted", 0.0)),
      "visibleAfterStabilize": visible_after,
      "streak": float(self._phone_streak),
      "confirmFrames": float(self._phone_confirm_frames),
    }

import asyncio
import base64
import binascii
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from config import settings
from hardware.hardware_detector import HardwareDetector, HardwareMode, HardwareProfile
from models.eye_tracker import create_eye_tracker
from models.face_detector import create_face_detector
from models.pose_model import create_pose_model
from models.schemas import FramePayload
from pipeline.scheduler import AdaptiveFrameScheduler
from tracking.face_tracker import AdaptiveFaceTracker
from tracking.person_tracker import CentroidPersonTracker
from utils.performance import RollingFps
from utils.performance_report import SessionPerformanceReporter
from utils.resource_monitor import ResourceMonitor
from vision.calibration import CalibrationManager
from vision.frame_quality import FrameQualityAnalyzer
from vision.hand_face_detector import HandFaceInteractionDetector
from vision.movement_analyzer import MovementAnalyzer
from vision.object_detector import DetectedObject, YoloObjectDetector
from vision.person_filter import PersonValidator
from vision.risk_engine import RiskEngine
from vision.scene_analyzer import SceneMotionAnalyzer

app = FastAPI(title="Advanced AI Proctoring Backend")
_frontend_path = Path(__file__).resolve().parent / "frontend" / "index.html"


def _warmup_models_sync() -> None:
  detector = YoloObjectDetector()
  warm_frame = np.zeros((480, 640, 3), dtype=np.uint8)
  try:
    detector.detect(warm_frame, face_boxes=[])
  except Exception:
    pass


class ProctoringSession:
  def __init__(self) -> None:
    self.hardware_profile: HardwareProfile = HardwareDetector.detect()
    self.scheduler = AdaptiveFrameScheduler(self.hardware_profile)
    self._gpu_fallback_active = False
    self._frame_index = 0

    self.face_detector = create_face_detector(self.hardware_profile.mode, max_faces=settings.max_faces)
    self.face_tracker = AdaptiveFaceTracker(
      max_disappeared=settings.tracker_max_disappeared,
      distance_threshold=settings.tracker_distance_threshold,
    )
    self.head_pose = create_pose_model()
    self.eye_tracker = create_eye_tracker()
    self.hand_face_detector = HandFaceInteractionDetector()
    self.frame_quality = FrameQualityAnalyzer()
    self.object_detector = YoloObjectDetector()
    self.object_detector.set_inference_fps(self.scheduler.target_yolo_fps())
    self.object_detector.set_person_inference_fps(
      max(1.5, min(settings.person_detector_inference_fps, self.scheduler.target_yolo_fps()))
    )
    if self.hardware_profile.mode == HardwareMode.CPU_MODE:
      self.object_detector.set_runtime_device("cpu")
    self.person_tracker = CentroidPersonTracker(
      max_disappeared=settings.person_tracker_max_disappeared,
      distance_threshold=settings.person_tracker_distance_threshold,
    )
    self.person_filter = PersonValidator()
    self.scene_analyzer = SceneMotionAnalyzer()
    self.movement_analyzer = MovementAnalyzer()
    self.risk_engine = RiskEngine()
    self.calibration = CalibrationManager()

    self._process_lock = asyncio.Lock()
    self._stats_lock = Lock()
    self._dropped_frames = 0
    self._processed_frames = 0
    self._fps = RollingFps(window_size=settings.fps_window)
    self.resource_monitor = ResourceMonitor()
    self._client_context: Dict[str, Any] = {}
    self._last_face_analytics: Dict[int, Dict[str, Any]] = {}
    self._last_object_payload: List[Dict[str, Any]] = []
    self._last_warning_payload: List[Dict[str, str]] = []
    self._secondary_face_streak = 0
    self.session_report: Optional[SessionPerformanceReporter] = None
    if settings.performance_report_enabled:
      self.session_report = SessionPerformanceReporter(
        device_profile=self.resource_monitor.device_profile(),
        sample_period_sec=settings.performance_report_sample_period_sec,
        timeline_limit=settings.performance_report_timeline_limit,
        event_limit=settings.performance_report_event_limit,
        export_dir=settings.performance_report_export_dir,
      )
    self._risk_snapshots: List[Dict[str, Any]] = []
    self._next_risk_snapshot_id = 1
    self._last_risk_snapshot_ts = 0.0

  async def process_frame(self, frame_base64: str) -> Dict[str, Any]:
    async with self._process_lock:
      return await asyncio.to_thread(self._process_frame_sync, frame_base64)

  def _process_frame_sync(self, frame_base64: str) -> Dict[str, Any]:
    total_start = perf_counter()
    self._frame_index += 1
    frame_index = int(self._frame_index)

    face_time_ms = 0.0
    pose_time_ms = 0.0
    yolo_time_ms = 0.0
    quality_time_ms = 0.0
    risk_time_ms = 0.0

    run_inference = self.scheduler.should_process_frame(frame_index)
    inference_skipped = not run_inference

    frame = self._decode_frame(frame_base64)
    tracked_payload: List[Dict[str, Any]] = []
    person_payload: List[Dict[str, Any]] = []
    person_debug: Dict[str, float] = {}
    objects: List[DetectedObject] = []
    warnings_payload: List[Dict[str, str]] = []
    scene_payload: Dict[str, Any] = {}
    face_count = 0
    person_count = 0
    should_update_risk = bool(run_inference)
    new_risk_snapshot: Optional[Dict[str, Any]] = None
    should_stop_monitoring = False
    face_boxes: List[Tuple[int, int, int, int]] = []

    if frame is not None:
      target_width = min(settings.max_frame_width, self.scheduler.max_width_for_frame(frame.shape))
      frame = self._resize_frame(frame, target_width)

      person_detections = self.object_detector.detect_persons(frame)
      tracked_persons = self.person_tracker.update(person_detections)
      person_focus_boxes = [person.bbox for person in tracked_persons if int(person.disappeared) <= 2]
      scene_payload = self.scene_analyzer.analyze(frame, person_focus_boxes)

      run_detection = self.scheduler.should_run_face_detection(frame_index, self.face_tracker.active_count())
      detections = None
      if run_detection:
        face_start = perf_counter()
        detections = self.face_detector.detect(frame, roi_boxes=person_focus_boxes)
        face_time_ms = (perf_counter() - face_start) * 1000.0

      tracked_faces = self.face_tracker.update(frame, detections=detections, run_detection=run_detection)
      if not tracked_faces and not run_detection:
        face_start = perf_counter()
        recovery_detections = self.face_detector.detect(frame)
        face_time_ms += (perf_counter() - face_start) * 1000.0
        tracked_faces = self.face_tracker.update(
          frame,
          detections=recovery_detections,
          run_detection=True,
        )
      movement_by_id = self.movement_analyzer.compute(tracked_faces, frame.shape)
      active_face_ids = {face.id for face in tracked_faces}
      self.head_pose.prune(active_face_ids)
      self.eye_tracker.prune(active_face_ids)

      if run_inference:
        pose_start = perf_counter()
        hand_on_face_by_id = self.hand_face_detector.detect(frame, tracked_faces)
        for face in tracked_faces:
          head_pose = self.head_pose.estimate(face.landmarks, frame.shape, face_id=face.id)
          gaze_result = self.eye_tracker.estimate(
            face.landmarks,
            face_id=face.id,
            head_pose=(head_pose.yaw, head_pose.pitch),
          )
          face_confidence = float(min(1.0, 0.35 + (0.13 * max(int(getattr(face, "persistence_frames", 1)), 1))))
          eye_symmetry = self._eye_symmetry(face.landmarks)
          pitch_ratio = self._pitch_ratio(face.landmarks)
          frontalness = self._frontalness(head_pose.yaw, eye_symmetry)
          tracked_payload.append(
            {
              "id": face.id,
              "bbox": self._bbox_payload(face.bbox),
              "movement": round(float(movement_by_id.get(face.id, 0.0)), 4),
              "headPose": {
                "yaw": round(float(head_pose.yaw), 2),
                "pitch": round(float(head_pose.pitch), 2),
                "roll": round(float(head_pose.roll), 2),
              },
              "gaze": gaze_result.direction,
              "eyeVisible": gaze_result.eye_visible,
              "gazeOffsetX": round(float(gaze_result.horizontal_offset), 4),
              "gazeOffsetY": round(float(gaze_result.vertical_offset), 4),
              "gazeX": round(float(gaze_result.normalized_x), 4),
              "gazeY": round(float(gaze_result.normalized_y), 4),
              "gazeConfidence": round(float(gaze_result.confidence), 4),
              "eyeVisibility": round(float(gaze_result.eye_visibility), 4),
              "landmarkStability": round(float(head_pose.stability), 4),
              "pitchRatio": round(float(pitch_ratio), 4),
              "frontalness": round(float(frontalness), 4),
              "faceFrontalness": round(float(frontalness), 4),
              "faceConfidence": round(face_confidence, 4),
              "eyeSymmetry": round(float(eye_symmetry), 4),
              "trackPersistenceFrames": int(getattr(face, "persistence_frames", 1)),
              "handOnFace": bool(hand_on_face_by_id.get(face.id, False)),
            }
          )
          self._last_face_analytics[face.id] = {
            "headPose": dict(tracked_payload[-1]["headPose"]),
            "gaze": str(gaze_result.direction),
            "eyeVisible": bool(gaze_result.eye_visible),
            "gazeOffsetX": round(float(gaze_result.horizontal_offset), 4),
            "gazeOffsetY": round(float(gaze_result.vertical_offset), 4),
            "gazeX": round(float(gaze_result.normalized_x), 4),
            "gazeY": round(float(gaze_result.normalized_y), 4),
            "gazeConfidence": round(float(gaze_result.confidence), 4),
            "eyeVisibility": round(float(gaze_result.eye_visibility), 4),
            "landmarkStability": round(float(head_pose.stability), 4),
            "pitchRatio": round(float(pitch_ratio), 4),
            "frontalness": round(float(frontalness), 4),
            "faceFrontalness": round(float(frontalness), 4),
            "faceConfidence": round(face_confidence, 4),
            "eyeSymmetry": round(float(eye_symmetry), 4),
            "trackPersistenceFrames": int(getattr(face, "persistence_frames", 1)),
            "handOnFace": bool(hand_on_face_by_id.get(face.id, False)),
          }
        tracked_payload = self._stabilize_secondary_faces(tracked_payload)
        face_boxes = [self._payload_to_bbox(face.get("bbox")) for face in tracked_payload if face.get("bbox")]
        face_count = len(tracked_payload)
        pose_time_ms = (perf_counter() - pose_start) * 1000.0

        quality_start = perf_counter()
        quality_warnings = self.frame_quality.analyze(frame)
        quality_time_ms = (perf_counter() - quality_start) * 1000.0
        warnings_payload = [
          {
            "code": warning.code,
            "message": warning.message,
            "severity": warning.severity,
          }
          for warning in quality_warnings
        ]
        self._last_warning_payload = list(warnings_payload)

        yolo_start = perf_counter()
        try:
          objects = self.object_detector.detect(
            frame,
            face_boxes=face_boxes,
            person_boxes_hint=person_focus_boxes,
          )
        except Exception as exc:
          if self.hardware_profile.mode != HardwareMode.CPU_MODE:
            self._activate_cpu_fallback(f"GPU inference unstable: {exc}")
            objects = self.object_detector.detect(
              frame,
              face_boxes=face_boxes,
              person_boxes_hint=person_focus_boxes,
            )
          else:
            objects = []
        yolo_time_ms = (perf_counter() - yolo_start) * 1000.0
      else:
        tracked_payload = self._tracking_only_payload(tracked_faces=tracked_faces, movement_by_id=movement_by_id)
        tracked_payload = self._stabilize_secondary_faces(tracked_payload)
        face_boxes = [self._payload_to_bbox(face.get("bbox")) for face in tracked_payload if face.get("bbox")]
        face_count = len(tracked_payload)
        warnings_payload = list(self._last_warning_payload)
        should_update_risk = False

      person_payload, person_debug = self.person_filter.filter(
        tracked_persons=tracked_persons,
        tracked_faces=tracked_faces,
        frame=frame,
        scene_analyzer=self.scene_analyzer,
      )
      person_count = len(person_payload)
    else:
      self.record_drop()
      should_update_risk = False

    calibration_payload = self.calibration.status_payload()
    if frame is not None and settings.calibration_enabled and run_inference:
      calibration_payload, threshold_overrides = self.calibration.update(
        tracked_faces=tracked_payload,
        objects=objects,
        frame_shape=frame.shape,
        frame=frame,
        client_context=self._client_context,
      )
      profile = calibration_payload.get("profile") or {}
      if threshold_overrides:
        self._apply_calibration_profile(profile)
      if threshold_overrides and bool(profile.get("ready", False)):
        self._apply_calibration_thresholds(threshold_overrides)

    risk_start = perf_counter()
    if self.calibration.active:
      risk_score = 0.0
      risk_level = "LOW"
      _, _, risk_breakdown = self.risk_engine.current()
      risk_breakdown = self._zero_risk_breakdown(risk_breakdown)
      risk_breakdown["calibrationActive"] = True
    elif should_update_risk:
      risk_score, risk_level, risk_breakdown = self.risk_engine.compute(
        face_count,
        tracked_payload,
        objects,
        face_signal_available=self.face_detector.is_available,
        context=self._risk_context(frame_shape=frame.shape if frame is not None else None),
        person_count=person_count,
      )
    else:
      risk_score, risk_level, risk_breakdown = self.risk_engine.current()
    risk_time_ms = (perf_counter() - risk_start) * 1000.0

    if frame is not None and run_inference and not self.calibration.active:
      new_risk_snapshot, should_stop_monitoring = self._update_risk_snapshots(
        frame=frame,
        risk_score=float(risk_score),
        risk_level=risk_level,
      )

    for face in tracked_payload:
      face_id = int(face.get("id", -1))
      if face_id < 0:
        continue
      analytics = self._last_face_analytics.setdefault(face_id, {})
      for key in (
        "attentionState",
        "signalConfidence",
        "attentionConfidence",
        "attentionScore",
        "baselineYaw",
        "baselinePitch",
        "deviationScore",
        "fastDeviationTriggered",
        "yawNoise",
        "pitchNoise",
        "gazeNoise",
        "landmarkStability",
        "pitchRatio",
        "gazeX",
        "gazeY",
        "eyeVisibility",
        "frontalness",
        "faceConfidence",
        "eyeSymmetry",
        "trackPersistenceFrames",
        "horizontalAttentionState",
        "gazeAttentionState",
        "orientationAttentionState",
      ):
        if key in face:
          analytics[key] = face.get(key)

    processed_frames, dropped_frames, fps = self._mark_processed()
    total_time_ms = (perf_counter() - total_start) * 1000.0
    frame_metrics = {
      "faceTimeMs": round(face_time_ms, 2),
      "yoloTimeMs": round(yolo_time_ms, 2),
      "poseTimeMs": round(pose_time_ms, 2),
      "qualityTimeMs": round(quality_time_ms, 2),
      "riskTimeMs": round(risk_time_ms, 2),
      "totalTimeMs": round(total_time_ms, 2),
      "fps": round(fps, 2),
      "droppedFrames": dropped_frames,
      "processedFrames": processed_frames,
      "inferenceSkipped": inference_skipped,
      "frameIndex": frame_index,
      "frameStride": self.scheduler.as_dict().get("effectiveStride", 1.0),
      "faceDetectInterval": self.scheduler.as_dict().get("faceDetectInterval", 1.0),
    }
    resources: Dict[str, Any] = {}
    if settings.include_metrics or self.session_report is not None:
      resources = self.resource_monitor.sample()
      self.scheduler.update_feedback(
        cpu_percent=float(resources.get("cpuSystemPercent") or 0.0),
        frame_latency_ms=float(frame_metrics["totalTimeMs"]),
      )
    if self.session_report is not None:
      self.session_report.record(resources=resources, metrics=frame_metrics)

    object_payload = [
      {
        "label": obj.label,
        "confidence": round(float(obj.confidence), 3),
        "bbox": (
          {
            "x": int(obj.bbox[0]),
            "y": int(obj.bbox[1]),
            "w": int(max(obj.bbox[2] - obj.bbox[0], 0)),
            "h": int(max(obj.bbox[3] - obj.bbox[1], 0)),
          }
          if obj.bbox is not None
          else None
        ),
      }
      for obj in objects
    ]
    if run_inference:
      self._last_object_payload = list(object_payload)
    else:
      object_payload = list(self._last_object_payload)

    response: Dict[str, Any] = {
      "personCount": person_count,
      "trackedPersons": person_payload,
      "faceCount": face_count,
      "trackedFaces": tracked_payload,
      "objects": object_payload,
      "warnings": warnings_payload,
      "riskScore": round(float(risk_score), 3),
      "riskLevel": risk_level,
      "calibration": calibration_payload,
      "scene": scene_payload,
      "riskSnapshotsInfo": {
        "count": len(self._risk_snapshots),
        "max": 0 if not settings.risk_capture_enabled else settings.risk_capture_max_images,
      },
      "riskSnapshots": self._risk_snapshot_summaries(),
    }

    if new_risk_snapshot is not None:
      response["newRiskSnapshot"] = new_risk_snapshot

    if should_stop_monitoring:
      response["monitoringControl"] = {
        "shouldStop": True,
        "reason": "MAX_RISK_CAPTURES",
        "message": (
          f"Risk snapshot limit reached ({settings.risk_capture_max_images}). "
          "Monitoring stopped automatically."
        ),
        "snapshotCount": len(self._risk_snapshots),
        "maxSnapshots": settings.risk_capture_max_images,
      }

    if settings.include_risk_breakdown:
      response["riskBreakdown"] = risk_breakdown

    if settings.include_metrics:
      response["metrics"] = {
        **frame_metrics,
        "faceDetectorAvailable": self.face_detector.is_available,
        "handDetectorAvailable": self.hand_face_detector.is_available,
        "yoloDevice": self.object_detector.runtime_device(),
        "phoneThresholds": self.object_detector.current_phone_thresholds(),
        "phoneDebug": self.object_detector.phone_debug(),
        "gadgetDebug": self.object_detector.gadget_debug(),
        "backgroundMotionZones": {
          "faceDetector": int(self.face_detector.background_motion_zone_count()),
          "objectDetector": int(self.object_detector.background_motion_zone_count()),
          "sceneAnalyzer": int(scene_payload.get("staticBackgroundZoneCount", 0)),
        },
        "scene": scene_payload,
        "personDebug": person_debug,
        "attentionModel": risk_breakdown.get("attentionModel", {}) if isinstance(risk_breakdown, dict) else {},
        "clientContext": dict(self._client_context),
        "resources": resources,
        "performanceReport": self.performance_report_status(),
        "hardwareProfile": self.hardware_profile.to_dict(),
        "scheduler": self.scheduler.as_dict(),
        "gpuFallbackActive": self._gpu_fallback_active,
      }

    return response

  def _activate_cpu_fallback(self, reason: str) -> None:
    if self._gpu_fallback_active:
      return
    self._gpu_fallback_active = True
    self.hardware_profile = HardwareDetector.fallback_cpu_profile(self.hardware_profile)
    self.scheduler.set_profile(self.hardware_profile)
    self.object_detector.set_runtime_device("cpu")
    self.object_detector.set_inference_fps(self.scheduler.target_yolo_fps())
    self.object_detector.set_person_inference_fps(
      max(1.5, min(settings.person_detector_inference_fps, self.scheduler.target_yolo_fps()))
    )
    self._last_warning_payload.append(
      {
        "code": "GPU_FALLBACK",
        "message": f"Switched to CPU mode: {reason}",
        "severity": "HIGH",
      }
    )
    self._last_warning_payload = self._last_warning_payload[-6:]

  def _tracking_only_payload(
    self,
    tracked_faces: List[Any],
    movement_by_id: Dict[int, float],
  ) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for face in tracked_faces:
      fallback = self._last_face_analytics.get(
        int(face.id),
        {
          "headPose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
          "gaze": "CENTER",
          "eyeVisible": False,
          "gazeOffsetX": 0.0,
          "gazeOffsetY": 0.0,
          "gazeX": 0.0,
          "gazeY": 0.0,
          "gazeConfidence": 0.0,
          "eyeVisibility": 0.0,
          "landmarkStability": 1.0,
          "pitchRatio": 0.5,
          "frontalness": 1.0,
          "faceFrontalness": 1.0,
          "faceConfidence": 0.48,
          "eyeSymmetry": 0.0,
          "trackPersistenceFrames": 1,
          "handOnFace": False,
          "attentionState": "FOCUSED",
          "signalConfidence": 0.0,
          "attentionConfidence": 0.0,
          "attentionScore": 0.0,
          "baselineYaw": None,
          "baselinePitch": None,
          "deviationScore": 0.0,
          "fastDeviationTriggered": False,
          "yawNoise": 0.0,
          "pitchNoise": 0.0,
          "gazeNoise": 0.0,
          "horizontalAttentionState": "FOCUSED",
          "gazeAttentionState": "FOCUSED",
          "orientationAttentionState": "FOCUSED",
        },
      )
      payload_item = {
        "id": int(face.id),
        "bbox": self._bbox_payload(face.bbox),
        "movement": round(float(movement_by_id.get(face.id, 0.0)), 4),
        "headPose": {
          "yaw": round(float(fallback["headPose"]["yaw"]), 2),
          "pitch": round(float(fallback["headPose"]["pitch"]), 2),
          "roll": round(float(fallback["headPose"]["roll"]), 2),
        },
        "gaze": str(fallback["gaze"]),
        "eyeVisible": bool(fallback["eyeVisible"]),
        "gazeOffsetX": round(float(fallback.get("gazeOffsetX", 0.0)), 4),
        "gazeOffsetY": round(float(fallback.get("gazeOffsetY", 0.0)), 4),
        "gazeX": round(float(fallback.get("gazeX", 0.0)), 4),
        "gazeY": round(float(fallback.get("gazeY", 0.0)), 4),
        "gazeConfidence": round(float(fallback.get("gazeConfidence", 0.0)), 4),
        "eyeVisibility": round(float(fallback.get("eyeVisibility", 0.0)), 4),
        "landmarkStability": round(float(fallback.get("landmarkStability", 1.0)), 4),
        "pitchRatio": round(float(fallback.get("pitchRatio", 0.5)), 4),
        "frontalness": round(float(fallback.get("frontalness", fallback.get("faceFrontalness", 1.0))), 4),
        "faceFrontalness": round(float(fallback.get("faceFrontalness", fallback.get("frontalness", 1.0))), 4),
        "faceConfidence": round(float(fallback.get("faceConfidence", 0.48)), 4),
        "eyeSymmetry": round(float(fallback.get("eyeSymmetry", 0.0)), 4),
        "trackPersistenceFrames": int(max(float(fallback.get("trackPersistenceFrames", 1)), 1)),
        "handOnFace": bool(fallback["handOnFace"]),
      }
      for key in (
        "attentionState",
        "signalConfidence",
        "attentionConfidence",
        "attentionScore",
        "baselineYaw",
        "baselinePitch",
        "deviationScore",
        "fastDeviationTriggered",
        "yawNoise",
        "pitchNoise",
        "gazeNoise",
        "pitchRatio",
        "gazeX",
        "gazeY",
        "eyeVisibility",
        "frontalness",
        "faceConfidence",
        "eyeSymmetry",
        "trackPersistenceFrames",
        "horizontalAttentionState",
        "gazeAttentionState",
        "orientationAttentionState",
      ):
        if key in fallback:
          payload_item[key] = fallback.get(key)
      payload.append(payload_item)
    return payload

  def _stabilize_secondary_faces(self, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(faces) <= 1:
      self._secondary_face_streak = 0
      return faces

    ordered = sorted(faces, key=self._payload_bbox_area, reverse=True)
    primary = ordered[0]
    secondary = ordered[1:]

    primary_area = self._payload_bbox_area(primary)
    filtered_secondary = []
    for face in secondary:
      area = self._payload_bbox_area(face)
      if area >= int(primary_area * 0.32):
        filtered_secondary.append(face)

    if not filtered_secondary:
      self._secondary_face_streak = 0
      return [primary]

    if self._secondary_face_streak < 1:
      self._secondary_face_streak += 1
      return [primary]

    self._secondary_face_streak = min(self._secondary_face_streak + 1, 5)
    return [primary, *filtered_secondary]

  @staticmethod
  def _payload_bbox_area(face_payload: Dict[str, Any]) -> int:
    bbox = face_payload.get("bbox") or {}
    width = int(max(float(bbox.get("w", 0)), 0))
    height = int(max(float(bbox.get("h", 0)), 0))
    return max(1, width * height)

  @staticmethod
  def _bbox_payload(bbox: Tuple[int, int, int, int]) -> Dict[str, int]:
    return {
      "x": int(bbox[0]),
      "y": int(bbox[1]),
      "w": int(max(bbox[2] - bbox[0], 0)),
      "h": int(max(bbox[3] - bbox[1], 0)),
    }

  @staticmethod
  def _payload_to_bbox(bbox_payload: Dict[str, Any]) -> Tuple[int, int, int, int]:
    x = int(float(bbox_payload.get("x", 0)))
    y = int(float(bbox_payload.get("y", 0)))
    w = int(max(float(bbox_payload.get("w", 0)), 0))
    h = int(max(float(bbox_payload.get("h", 0)), 0))
    return (x, y, x + w, y + h)

  @staticmethod
  def _eye_symmetry(landmarks: Any) -> float:
    if not isinstance(landmarks, np.ndarray) or landmarks.ndim != 2 or landmarks.shape[0] <= 291:
      return 0.0

    try:
      left_eye = landmarks[33]
      right_eye = landmarks[263]
      nose = landmarks[1]
    except Exception:
      return 0.0

    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    if eye_dist < 6.0:
      return 0.0

    left_dist = float(np.linalg.norm(left_eye - nose))
    right_dist = float(np.linalg.norm(right_eye - nose))
    return float(abs(left_dist - right_dist) / max(eye_dist, 1e-6))

  @staticmethod
  def _pitch_ratio(landmarks: Any) -> float:
    if not isinstance(landmarks, np.ndarray) or landmarks.ndim != 2 or landmarks.shape[0] <= 152:
      return 0.5

    try:
      nose = landmarks[1]
      chin = landmarks[152]
      forehead = landmarks[10]
    except Exception:
      return 0.5

    face_height = float(abs(chin[1] - forehead[1]))
    if face_height < 8.0:
      return 0.5
    return float(np.clip((float(chin[1]) - float(nose[1])) / face_height, 0.0, 1.5))

  @staticmethod
  def _frontalness(yaw: float, eye_symmetry: float) -> float:
    yaw_term = abs(float(yaw)) / max(float(settings.face_off_camera_yaw_gate), 1e-6)
    symmetry_term = float(eye_symmetry) / max(float(settings.face_orientation_eye_symmetry_threshold), 1e-6)
    return float(np.clip(1.0 - max(yaw_term, symmetry_term), 0.0, 1.0))

  def _zero_risk_breakdown(self, breakdown: Dict[str, Any]) -> Dict[str, Any]:
    zeroed = dict(breakdown)
    rule_states = []
    for state in breakdown.get("ruleStates", []):
      copied = dict(state)
      copied["active"] = False
      copied["streak"] = 0
      copied["repetitionFactor"] = 0.0
      copied["multiplier"] = 1.0
      copied["points"] = 0.0
      rule_states.append(copied)
    zeroed["ruleStates"] = rule_states
    zeroed["activeRules"] = []
    zeroed["dominantRule"] = None
    zeroed["rawScore"] = 0.0
    zeroed["emaScore"] = 0.0
    zeroed["smoothedScore"] = 0.0
    zeroed["riskLevel"] = "LOW"
    return zeroed

  def _apply_calibration_thresholds(self, overrides: Dict[str, float]) -> None:
    if not overrides:
      return
    self.risk_engine.set_adaptive_thresholds(overrides)
    self.object_detector.set_phone_overrides(overrides)
    self.eye_tracker.set_overrides(overrides)

  def _apply_calibration_profile(self, profile: Dict[str, Any]) -> None:
    if not profile:
      return
    self.risk_engine.set_calibration_profile(profile)
    zones = profile.get("backgroundMotionZones") or []
    if isinstance(zones, list):
      self.face_detector.set_background_motion_zones(zones)
      self.object_detector.set_background_motion_zones(zones)
      self.person_filter.set_background_motion_zones(zones)
      self.scene_analyzer.set_background_motion_zones(zones)

  def start_calibration(self) -> Dict[str, Any]:
    if not settings.calibration_enabled:
      return {
        "active": False,
        "completed": False,
        "mode": "DISABLED",
        "instruction": "Calibration is disabled in settings.",
      }

    self.calibration.start()
    self.risk_engine.reset()
    self.object_detector.reset_phone_state()
    self.person_tracker.reset()
    self.person_filter.reset()
    self.scene_analyzer.reset()
    self.eye_tracker.reset_overrides()
    self.face_detector.set_background_motion_zones([])
    self.object_detector.set_background_motion_zones([])
    self.person_filter.set_background_motion_zones([])
    self.scene_analyzer.set_background_motion_zones([])
    return self.calibration.status_payload()

  def cancel_calibration(self) -> Dict[str, Any]:
    self.calibration.cancel()
    return self.calibration.status_payload()

  def calibration_status(self) -> Dict[str, Any]:
    return self.calibration.status_payload()

  def set_client_context(self, client_context: Optional[Dict[str, Any]]) -> None:
    if not isinstance(client_context, dict):
      return
    sanitized: Dict[str, Any] = {}
    for key in (
      "viewportWidth",
      "viewportHeight",
      "screenWidth",
      "screenHeight",
      "devicePixelRatio",
      "captureWidth",
      "captureHeight",
      "videoWidth",
      "videoHeight",
    ):
      value = client_context.get(key)
      try:
        sanitized[key] = float(value)
      except (TypeError, ValueError):
        continue
    self._client_context = sanitized

  def _risk_context(self, frame_shape: Optional[Tuple[int, int, int]]) -> Dict[str, float]:
    context: Dict[str, float] = {}
    if frame_shape is not None:
      context["frameWidth"] = float(max(int(frame_shape[1]), 1))
      context["frameHeight"] = float(max(int(frame_shape[0]), 1))
    for key in ("viewportWidth", "viewportHeight", "screenWidth", "screenHeight", "devicePixelRatio"):
      value = self._client_context.get(key)
      try:
        context[key] = float(value)
      except (TypeError, ValueError):
        continue
    if "screenWidth" in context and "devicePixelRatio" in context:
      context["displayWidth"] = context["screenWidth"] * max(context["devicePixelRatio"], 1.0)
    if "screenHeight" in context and "devicePixelRatio" in context:
      context["displayHeight"] = context["screenHeight"] * max(context["devicePixelRatio"], 1.0)
    return context

  def record_drop(self, count: int = 1) -> None:
    with self._stats_lock:
      self._dropped_frames += max(0, int(count))

  def _mark_processed(self) -> Tuple[int, int, float]:
    self._fps.tick()
    fps = self._fps.value
    with self._stats_lock:
      self._processed_frames += 1
      return self._processed_frames, self._dropped_frames, fps

  def reset_stats(self) -> None:
    with self._stats_lock:
      self._dropped_frames = 0
      self._processed_frames = 0
    self._fps.reset()

  def performance_report_status(self) -> Dict[str, Any]:
    if self.session_report is None:
      return {"enabled": False}
    status = self.session_report.status()
    status["enabled"] = True
    return status

  def performance_report(self) -> Dict[str, Any]:
    if self.session_report is None:
      return {"enabled": False, "message": "Performance reporting is disabled."}
    report = self.session_report.build_report()
    report["enabled"] = True
    return report

  def export_performance_report(self) -> Dict[str, Any]:
    if self.session_report is None:
      return {"enabled": False, "message": "Performance reporting is disabled."}
    exported = self.session_report.export_to_file(directory=settings.performance_report_export_dir)
    exported["enabled"] = True
    return exported

  def reset_performance_report(self) -> Dict[str, Any]:
    if self.session_report is None:
      return {"enabled": False, "message": "Performance reporting is disabled."}
    status = self.session_report.reset()
    status["enabled"] = True
    return status

  def set_runtime_thresholds(self, overrides: Dict[str, float]) -> Dict[str, Any]:
    if not overrides:
        return {"applied": 0, "keys": []}
    valid: Dict[str, float] = {}
    for key, value in overrides.items():
        try:
            valid[key] = float(value)
        except (TypeError, ValueError):
            continue
    if valid:
        self.risk_engine.set_adaptive_thresholds(valid)
    eye_keys = {"eye_visibility_threshold","eye_horizontal_deadzone",
                "eye_vertical_up_threshold","eye_vertical_down_threshold","eye_baseline_alpha"}
    eye_ov = {k: v for k, v in valid.items() if k in eye_keys}
    if eye_ov:
        self.eye_tracker.set_overrides(eye_ov)
    phone_keys = {"phone_min_confidence","phone_min_area_ratio","phone_confirm_frames"}
    phone_ov = {k: v for k, v in valid.items() if k in phone_keys}
    if phone_ov:
        self.object_detector.set_phone_overrides(phone_ov)
    return {"applied": len(valid), "keys": sorted(valid.keys())}

  def get_current_thresholds(self) -> Dict[str, Any]:
    return {
        "adaptive": self.risk_engine.get_adaptive_thresholds(),
        "phone": self.object_detector.current_phone_thresholds(),
        "hasCalibration": bool(self.calibration.completed),
    }

  @staticmethod
  def _decode_frame(frame_base64: str) -> Optional[np.ndarray]:
    try:
      if "," in frame_base64:
        frame_base64 = frame_base64.split(",", 1)[1]
      frame_bytes = base64.b64decode(frame_base64, validate=True)
      frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
      frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
      return frame
    except (binascii.Error, ValueError):
      return None

  @staticmethod
  def _resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
      return frame

    scale = max_width / float(w)
    new_size = (max_width, int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

  def _update_risk_snapshots(
    self,
    frame: np.ndarray,
    risk_score: float,
    risk_level: str,
  ) -> Tuple[Optional[Dict[str, Any]], bool]:
    if not settings.risk_capture_enabled:
      return None, False
    threshold = float(settings.risk_capture_threshold)
    max_images = max(0, int(settings.risk_capture_max_images))

    if max_images <= 0 or risk_score < threshold:
      return None, False

    if len(self._risk_snapshots) >= max_images:
      return None, True

    now = perf_counter()
    min_gap = max(0.1, float(settings.risk_capture_cooldown_sec))
    if (now - self._last_risk_snapshot_ts) < min_gap:
      return None, False

    image_base64 = self._encode_risk_snapshot(frame)
    if not image_base64:
      return None, False

    snapshot = {
      "id": int(self._next_risk_snapshot_id),
      "capturedAt": datetime.now(timezone.utc).isoformat(),
      "riskScore": round(float(risk_score), 3),
      "riskLevel": str(risk_level),
      "mimeType": "image/jpeg",
      "imageBase64": image_base64,
    }
    self._risk_snapshots.append(snapshot)
    self._next_risk_snapshot_id += 1
    self._last_risk_snapshot_ts = now
    return snapshot, False

  def _encode_risk_snapshot(self, frame: np.ndarray) -> str:
    quality = int(max(30, min(95, int(settings.risk_capture_jpeg_quality))))
    ok, encoded = cv2.imencode(
      ".jpg",
      frame,
      [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
      return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")

  def _risk_snapshot_summaries(self) -> List[Dict[str, Any]]:
    return [
      {
        "id": int(snapshot.get("id", 0)),
        "capturedAt": str(snapshot.get("capturedAt", "")),
        "riskScore": float(snapshot.get("riskScore", 0.0)),
        "riskLevel": str(snapshot.get("riskLevel", "LOW")),
      }
      for snapshot in self._risk_snapshots
    ]

  def close(self) -> None:
    if self.session_report is not None and settings.performance_report_auto_export_on_close:
      try:
        self.session_report.export_to_file(directory=settings.performance_report_export_dir)
      except Exception:
        pass
    self.person_filter.close()
    self.face_detector.close()
    self.hand_face_detector.close()


@app.on_event("startup")
async def preload_models() -> None:
  await asyncio.to_thread(_warmup_models_sync)


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
  if _frontend_path.exists():
    return HTMLResponse(_frontend_path.read_text(encoding="utf-8"))
  return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)


@app.get("/health")
async def health() -> Dict[str, str]:
  return {"status": "ok"}


@app.websocket(settings.websocket_path)
async def websocket_proctoring(websocket: WebSocket) -> None:
  await websocket.accept()
  session = ProctoringSession()
  frame_queue: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=1)
  render_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue(maxsize=1)
  stop_event = asyncio.Event()

  async def inference_worker() -> None:
    while True:
      frame_base64 = await frame_queue.get()
      if frame_base64 is None:
        break
      response = await session.process_frame(frame_base64)

      if render_queue.full():
        try:
          _ = render_queue.get_nowait()
        except asyncio.QueueEmpty:
          pass
      try:
        render_queue.put_nowait(response)
      except asyncio.QueueFull:
        pass

      control = response.get("monitoringControl") if isinstance(response, dict) else None
      if isinstance(control, dict) and bool(control.get("shouldStop", False)):
        stop_event.set()
        break

  async def render_worker() -> None:
    while True:
      payload = await render_queue.get()
      if payload is None:
        break

      try:
        await websocket.send_json(payload)
      except Exception:
        stop_event.set()
        break

      control = payload.get("monitoringControl") if isinstance(payload, dict) else None
      if isinstance(control, dict) and bool(control.get("shouldStop", False)):
        try:
          await websocket.close(code=1000, reason="risk_snapshot_limit")
        except Exception:
          pass
        stop_event.set()
        break

  inference_task = asyncio.create_task(inference_worker())
  render_task = asyncio.create_task(render_worker())
  try:
    while True:
      if stop_event.is_set():
        break
      raw_payload = await websocket.receive_json()
      try:
        payload = FramePayload(**raw_payload)
        frame_base64 = payload.frame
        session.set_client_context(payload.client)
      except Exception:
        command = str(raw_payload.get("command", "")).strip().lower() if isinstance(raw_payload, dict) else ""
        if command == "reset_stats":
          session.reset_stats()
          await websocket.send_json({"type": "ack", "command": "reset_stats"})
          continue
        if command == "start_calibration":
          calibration = session.start_calibration()
          await websocket.send_json({"type": "ack", "command": "start_calibration", "calibration": calibration})
          continue
        if command == "cancel_calibration":
          calibration = session.cancel_calibration()
          await websocket.send_json({"type": "ack", "command": "cancel_calibration", "calibration": calibration})
          continue
        if command == "calibration_status":
          await websocket.send_json({"type": "calibration_status", "calibration": session.calibration_status()})
          continue
        if command == "performance_report_status":
          await websocket.send_json({"type": "performance_report_status", "status": session.performance_report_status()})
          continue
        if command == "performance_report":
          await websocket.send_json({"type": "performance_report", "report": session.performance_report()})
          continue
        if command == "performance_report_export":
          await websocket.send_json({"type": "performance_report_export", "export": session.export_performance_report()})
          continue
        if command == "performance_report_reset":
          status = session.reset_performance_report()
          await websocket.send_json({"type": "ack", "command": "performance_report_reset", "reportStatus": status})
          continue
        if command == "set_thresholds":
          threshold_data = raw_payload.get("thresholds", {}) if isinstance(raw_payload, dict) else {}
          result = session.set_runtime_thresholds(threshold_data)
          await websocket.send_json({"type": "ack", "command": "set_thresholds", "result": result})
          continue

        if command == "get_thresholds":
          current = session.get_current_thresholds()
          await websocket.send_json({"type": "thresholds_state", "thresholds": current})
          continue
        frame_base64 = ""

      if frame_queue.full():
        try:
          _ = frame_queue.get_nowait()
          session.record_drop()
        except asyncio.QueueEmpty:
          pass

      await frame_queue.put(frame_base64)
  except WebSocketDisconnect:
    stop_event.set()
  finally:
    if frame_queue.full():
      try:
        _ = frame_queue.get_nowait()
        session.record_drop()
      except asyncio.QueueEmpty:
        pass

    if not inference_task.done():
      try:
        frame_queue.put_nowait(None)
      except asyncio.QueueFull:
        pass

    if not render_task.done():
      try:
        render_queue.put_nowait(None)
      except asyncio.QueueFull:
        pass

    await asyncio.gather(inference_task, render_task, return_exceptions=True)
    await asyncio.to_thread(session.close)

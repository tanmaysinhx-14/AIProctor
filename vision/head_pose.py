from dataclasses import dataclass
from math import atan2, degrees, radians, sqrt, tan
from typing import Dict, Optional, Set, Tuple

import cv2
import numpy as np

from config import settings


@dataclass
class HeadPose:
  yaw: float
  pitch: float
  roll: float
  stability: float = 1.0
  used_fallback: bool = False


class _AngleEma:
  def __init__(self, alpha: float = 0.3) -> None:
    self._alpha = float(np.clip(alpha, 0.05, 0.95))
    self._initialized = False
    self._value = 0.0

  def update(self, value: float) -> float:
    if not self._initialized:
      self._value = float(value)
      self._initialized = True
      return float(value)
    self._value = ((1.0 - self._alpha) * self._value) + (self._alpha * float(value))
    return float(self._value)


class _PoseSmoother:
  def __init__(self) -> None:
    alpha = float(settings.pose_smoothing_alpha)
    self._yaw = _AngleEma(alpha=alpha)
    self._pitch = _AngleEma(alpha=alpha)
    self._roll = _AngleEma(alpha=alpha)

  def update(self, pose: HeadPose) -> HeadPose:
    return HeadPose(
      yaw=self._yaw.update(pose.yaw),
      pitch=self._pitch.update(pose.pitch),
      roll=self._roll.update(pose.roll),
    )


@dataclass
class _PoseState:
  smoother: _PoseSmoother
  rvec: Optional[np.ndarray] = None
  tvec: Optional[np.ndarray] = None
  last_pose: Optional[HeadPose] = None
  last_stability: float = 0.0


class HeadPoseEstimator:
  # Six canonical landmarks: nose tip, chin, eye corners, and mouth corners.
  _landmark_ids = np.array([1, 152, 33, 263, 61, 291], dtype=np.int32)

  # Canonical 3D facial model for solvePnP (arbitrary but consistent units).
  _model_points = np.array(
    [
      (0.0, 0.0, 0.0),  # Nose tip
      (0.0, -63.6, -12.5),  # Chin
      (-43.3, 32.7, -26.0),  # Left eye corner
      (43.3, 32.7, -26.0),  # Right eye corner
      (-28.9, -28.9, -24.1),  # Left mouth corner
      (28.9, -28.9, -24.1),  # Right mouth corner
    ],
    dtype=np.float64,
  )

  _left_eye_idx = 2
  _right_eye_idx = 3
  _horizontal_fov_deg = 60.0

  def __init__(self) -> None:
    self._states: Dict[int, _PoseState] = {}
    self._reprojection_error_threshold = 18.0

  def estimate(
    self,
    landmarks: np.ndarray,
    frame_shape: Tuple[int, int, int],
    face_id: Optional[int] = None,
  ) -> HeadPose:
    if face_id is None:
      pose, _, _, _, _ = self._estimate_raw_pose(
        landmarks=landmarks,
        frame_shape=frame_shape,
        prior_rvec=None,
        prior_tvec=None,
      )
      return pose

    state = self._states.get(face_id)
    if state is None:
      state = _PoseState(smoother=_PoseSmoother())
      self._states[face_id] = state

    pose, rvec, tvec, reproj_error, stability = self._estimate_raw_pose(
      landmarks=landmarks,
      frame_shape=frame_shape,
      prior_rvec=state.rvec,
      prior_tvec=state.tvec,
    )

    stable_measurement = (
      rvec is not None
      and tvec is not None
      and reproj_error <= self._reprojection_error_threshold
      and stability >= float(settings.pose_landmark_stability_min)
    )

    if stable_measurement:
      state.rvec = rvec
      state.tvec = tvec
      smoothed = state.smoother.update(pose)
      state.last_pose = HeadPose(
        yaw=smoothed.yaw,
        pitch=smoothed.pitch,
        roll=smoothed.roll,
        stability=stability,
        used_fallback=False,
      )
      state.last_stability = stability
      return state.last_pose

    if state.last_pose is not None:
      return HeadPose(
        yaw=state.last_pose.yaw,
        pitch=state.last_pose.pitch,
        roll=state.last_pose.roll,
        stability=stability,
        used_fallback=True,
      )

    smoothed = state.smoother.update(pose)
    state.last_pose = HeadPose(
      yaw=smoothed.yaw,
      pitch=smoothed.pitch,
      roll=smoothed.roll,
      stability=stability,
      used_fallback=stability < float(settings.pose_landmark_stability_min),
    )
    state.last_stability = stability
    return state.last_pose

  def prune(self, active_face_ids: Set[int]) -> None:
    stale_ids = [face_id for face_id in self._states if face_id not in active_face_ids]
    for face_id in stale_ids:
      del self._states[face_id]

  def _estimate_raw_pose(
    self,
    landmarks: np.ndarray,
    frame_shape: Tuple[int, int, int],
    prior_rvec: Optional[np.ndarray],
    prior_tvec: Optional[np.ndarray],
  ) -> Tuple[HeadPose, Optional[np.ndarray], Optional[np.ndarray], float, float]:
    if landmarks.shape[0] <= self._landmark_ids.max():
      return HeadPose(yaw=0.0, pitch=0.0, roll=0.0, stability=0.0), None, None, float("inf"), 0.0

    image_points = landmarks[self._landmark_ids].astype(np.float64)
    if not np.isfinite(image_points).all():
      return HeadPose(yaw=0.0, pitch=0.0, roll=0.0, stability=0.0), None, None, float("inf"), 0.0

    camera_matrix = self._estimate_camera_matrix(frame_shape=frame_shape)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vec, translation_vec = self._solve_pnp(
      image_points=image_points,
      camera_matrix=camera_matrix,
      dist_coeffs=dist_coeffs,
      prior_rvec=prior_rvec,
      prior_tvec=prior_tvec,
    )
    if not success or rotation_vec is None or translation_vec is None:
      return HeadPose(yaw=0.0, pitch=0.0, roll=0.0, stability=0.0), None, None, float("inf"), 0.0

    reproj_error = self._reprojection_error(
      image_points=image_points,
      rotation_vec=rotation_vec,
      translation_vec=translation_vec,
      camera_matrix=camera_matrix,
      dist_coeffs=dist_coeffs,
    )
    stability = self._landmark_stability(image_points=image_points, frame_shape=frame_shape, reproj_error=reproj_error)
    if reproj_error > self._reprojection_error_threshold and prior_rvec is not None and prior_tvec is not None:
      return HeadPose(yaw=0.0, pitch=0.0, roll=0.0, stability=stability), None, None, reproj_error, stability

    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)

    # Convert the rotation matrix into Euler angles:
    # pitch (x-axis), yaw (y-axis), roll (z-axis).
    pitch_deg, yaw_deg, raw_roll_deg = self._rotation_to_euler_deg(rotation_matrix)
    eye_roll_deg = self._eye_roll_deg(image_points=image_points)

    # Blend roll from PnP and eye-line geometry to reduce jitter while keeping physical roll.
    corrected_roll_deg = 0.75 * raw_roll_deg + 0.25 * eye_roll_deg

    pose = HeadPose(
      yaw=self._sanitize_angle(yaw_deg, min_value=-75.0, max_value=75.0),
      pitch=self._sanitize_angle(pitch_deg, min_value=-65.0, max_value=65.0),
      roll=self._sanitize_angle(corrected_roll_deg, min_value=-60.0, max_value=60.0),
      stability=stability,
      used_fallback=False,
    )
    return pose, rotation_vec, translation_vec, reproj_error, stability

  def _solve_pnp(
    self,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    prior_rvec: Optional[np.ndarray],
    prior_tvec: Optional[np.ndarray],
  ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    if prior_rvec is not None and prior_tvec is not None:
      success, rvec, tvec = cv2.solvePnP(
        self._model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        rvec=prior_rvec.copy(),
        tvec=prior_tvec.copy(),
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
      )
      if success:
        return True, rvec, tvec

    success, rvec, tvec = cv2.solvePnP(
      self._model_points,
      image_points,
      camera_matrix,
      dist_coeffs,
      flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
      return False, None, None

      # Refine from EPNP initialization for a more stable iterative solution.
    success, rvec, tvec = cv2.solvePnP(
      self._model_points,
      image_points,
      camera_matrix,
      dist_coeffs,
      rvec=rvec,
      tvec=tvec,
      useExtrinsicGuess=True,
      flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
      return False, None, None
    return True, rvec, tvec

  def _estimate_camera_matrix(self, frame_shape: Tuple[int, int, int]) -> np.ndarray:
    h, w = frame_shape[:2]
    half_fov = radians(self._horizontal_fov_deg / 2.0)
    fx = (w / 2.0) / max(tan(half_fov), 1e-6)
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    return np.array(
      [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
      ],
      dtype=np.float64,
    )

  def _eye_roll_deg(self, image_points: np.ndarray) -> float:
    left_eye = image_points[self._left_eye_idx]
    right_eye = image_points[self._right_eye_idx]
    # Eye-line angle provides a geometric roll estimate in image space.
    return float(degrees(atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))

  def _rotation_to_euler_deg(self, rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
    sy = sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
      pitch = atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
      yaw = atan2(-rotation_matrix[2, 0], sy)
      roll = atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
      pitch = atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
      yaw = atan2(-rotation_matrix[2, 0], sy)
      roll = 0.0

    return float(degrees(pitch)), float(degrees(yaw)), float(degrees(roll))

  def _reprojection_error(
    self,
    image_points: np.ndarray,
    rotation_vec: np.ndarray,
    translation_vec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
  ) -> float:
    projected, _ = cv2.projectPoints(
      self._model_points,
      rotation_vec,
      translation_vec,
      camera_matrix,
      dist_coeffs,
    )
    projected = projected.reshape(-1, 2)
    error = np.linalg.norm(projected - image_points, axis=1).mean()
    return float(error)

  def _landmark_stability(
    self,
    image_points: np.ndarray,
    frame_shape: Tuple[int, int, int],
    reproj_error: float,
  ) -> float:
    if image_points.shape[0] < 6 or not np.isfinite(image_points).all():
      return 0.0

    h, w = frame_shape[:2]
    nose = image_points[0]
    chin = image_points[1]
    left_eye = image_points[self._left_eye_idx]
    right_eye = image_points[self._right_eye_idx]
    mouth_l = image_points[4]
    mouth_r = image_points[5]

    eye_span = float(np.linalg.norm(right_eye - left_eye))
    face_height = float(np.linalg.norm(chin - nose))
    mouth_span = float(np.linalg.norm(mouth_r - mouth_l))
    ratio = mouth_span / max(eye_span, 1e-6)

    reproj_score = 1.0 - np.clip(reproj_error / max(self._reprojection_error_threshold * 1.25, 1e-6), 0.0, 1.0)
    eye_score = np.clip(eye_span / max(float(w) * 0.05, 1e-6), 0.0, 1.0)
    height_score = np.clip(face_height / max(float(h) * 0.08, 1e-6), 0.0, 1.0)
    ratio_score = 1.0 - np.clip(abs(ratio - 0.72) / 0.42, 0.0, 1.0)
    stability = (0.55 * reproj_score) + (0.20 * eye_score) + (0.15 * height_score) + (0.10 * ratio_score)
    return float(np.clip(stability, 0.0, 1.0))

  def _sanitize_angle(self, angle: float, min_value: float, max_value: float) -> float:
    normalized = (angle + 180.0) % 360.0 - 180.0
    return float(np.clip(normalized, min_value, max_value))

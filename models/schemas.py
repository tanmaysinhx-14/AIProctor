from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class FramePayload(BaseModel):
  frame: str = Field(..., description="Base64 encoded frame bytes")
  client: Optional[Dict[str, float]] = None


class BBoxSchema(BaseModel):
  x: int
  y: int
  w: int
  h: int


class HeadPoseSchema(BaseModel):
  yaw: float
  pitch: float
  roll: float


class TrackedFaceSchema(BaseModel):
  id: int
  bbox: Optional[BBoxSchema] = None
  movement: float
  headPose: HeadPoseSchema
  gaze: Literal["LEFT", "RIGHT", "CENTER", "DOWN", "UP"]
  eyeVisible: bool
  eyeVisibility: Optional[float] = None
  handOnFace: bool = False
  landmarkStability: Optional[float] = None
  faceFrontalness: Optional[float] = None
  frontalness: Optional[float] = None
  pitchRatio: Optional[float] = None
  gazeX: Optional[float] = None
  gazeY: Optional[float] = None
  eyeSymmetry: Optional[float] = None
  trackPersistenceFrames: Optional[int] = None
  faceConfidence: Optional[float] = None
  attentionState: Optional[str] = None
  signalConfidence: Optional[float] = None
  attentionConfidence: Optional[float] = None
  attentionScore: Optional[float] = None
  baselineYaw: Optional[float] = None
  baselinePitch: Optional[float] = None
  deviationScore: Optional[float] = None
  fastDeviationTriggered: Optional[bool] = None
  yawNoise: Optional[float] = None
  pitchNoise: Optional[float] = None
  gazeNoise: Optional[float] = None
  horizontalAttentionState: Optional[str] = None
  gazeAttentionState: Optional[str] = None
  orientationAttentionState: Optional[str] = None


class TrackedPersonSchema(BaseModel):
  id: int
  bbox: Optional[BBoxSchema] = None
  confidence: float
  disappeared: int = 0
  visibleFrames: int = 0
  poseKeypointsDetected: Optional[bool] = None
  skeletonConfidence: Optional[float] = None
  skeletonValid: Optional[bool] = None
  skeletonStability: Optional[float] = None
  shoulderWidthRatio: Optional[float] = None
  faceFrames: Optional[int] = None
  motionRatio: Optional[float] = None
  motionFrequency: Optional[float] = None
  flowVariance: Optional[float] = None
  frameHeightRatio: Optional[float] = None
  anchorValidated: Optional[bool] = None
  anchorRequired: Optional[bool] = None
  centroidRangePx: Optional[float] = None
  directionFlips: Optional[int] = None
  faceValidated: Optional[bool] = None
  activeBackgroundOverlap: Optional[float] = None
  edgeDensity: Optional[float] = None
  brightness: Optional[float] = None
  confidenceFloor: Optional[float] = None
  centroidDriftSatisfied: Optional[bool] = None
  faceTopRatio: Optional[float] = None
  calibrationZoneOverlap: Optional[float] = None
  suppressionZoneOverlap: Optional[float] = None
  confirmed: Optional[bool] = None


class ObjectSchema(BaseModel):
  label: Literal["phone", "person", "audio_device"]
  confidence: float
  bbox: Optional[BBoxSchema] = None


class WarningSchema(BaseModel):
  code: Literal["LOW_LIGHT", "DIRTY_CAMERA", "BLOCKED_VIEW", "GPU_FALLBACK"]
  message: str
  severity: Literal["LOW", "MEDIUM", "HIGH"]


class MetricsSchema(BaseModel):
  faceTimeMs: float
  yoloTimeMs: float
  poseTimeMs: float
  qualityTimeMs: float
  riskTimeMs: float
  totalTimeMs: float
  fps: float
  droppedFrames: int
  processedFrames: int
  faceDetectorAvailable: bool
  handDetectorAvailable: bool
  yoloDevice: str
  phoneThresholds: Optional[Dict[str, float]] = None
  phoneDebug: Optional[Dict[str, float]] = None
  gadgetDebug: Optional[Dict[str, float]] = None
  resources: Optional[Dict[str, object]] = None
  performanceReport: Optional[Dict[str, object]] = None
  hardwareProfile: Optional[Dict[str, object]] = None
  scheduler: Optional[Dict[str, object]] = None
  gpuFallbackActive: Optional[bool] = None
  inferenceSkipped: Optional[bool] = None
  frameIndex: Optional[int] = None
  frameStride: Optional[float] = None
  faceDetectInterval: Optional[float] = None
  scene: Optional[Dict[str, object]] = None
  personDebug: Optional[Dict[str, object]] = None
  attentionModel: Optional[Dict[str, object]] = None


class RiskRuleStateSchema(BaseModel):
  code: str
  label: str
  active: bool
  baseSeverity: float
  streak: int
  repetitionFactor: float
  multiplier: float
  points: float


class RiskBreakdownSchema(BaseModel):
  rawScore: float
  emaScore: float
  smoothedScore: float
  riskLevel: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
  activeRules: List[RiskRuleStateSchema]
  ruleStates: List[RiskRuleStateSchema]
  dominantRule: Optional[str] = None
  adaptiveThresholds: Optional[Dict[str, float]] = None
  calibrationActive: Optional[bool] = None
  attentionModel: Optional[Dict[str, object]] = None


class CalibrationPayloadSchema(BaseModel):
  mode: str
  active: bool
  completed: bool
  stageCode: str
  stageLabel: str
  instruction: str
  stageProgress: float
  overallProgress: float
  remainingSec: float
  stageElapsedSec: Optional[float] = None
  stageMinSec: Optional[float] = None
  stageMaxSec: Optional[float] = None
  stageReady: Optional[bool] = None
  stageFeedback: Optional[str] = None
  completedAgoSec: Optional[float] = None
  completedHoldSec: Optional[float] = None
  stageOutcomes: Optional[Dict[str, object]] = None
  profile: Optional[Dict[str, object]] = None
  profileThresholds: Optional[Dict[str, float]] = None


class ScenePayloadSchema(BaseModel):
  foregroundMotionRatio: Optional[float] = None
  personMotionRatio: Optional[float] = None
  backgroundMotionRatio: Optional[float] = None
  ignoredBackgroundMotionRatio: Optional[float] = None
  activeBackgroundRegions: Optional[List[Dict[str, float]]] = None
  staticBackgroundZoneCount: Optional[int] = None



class RiskSnapshotSchema(BaseModel):
  id: int
  capturedAt: str
  riskScore: float
  riskLevel: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
  mimeType: Optional[str] = None
  imageBase64: Optional[str] = None


class RiskSnapshotsInfoSchema(BaseModel):
  count: int
  max: int


class MonitoringControlSchema(BaseModel):
  shouldStop: bool
  reason: str
  message: str
  snapshotCount: int
  maxSnapshots: int

class RiskResponseSchema(BaseModel):
  personCount: Optional[int] = None
  trackedPersons: Optional[List[TrackedPersonSchema]] = None
  faceCount: int
  trackedFaces: List[TrackedFaceSchema]
  objects: List[ObjectSchema]
  warnings: List[WarningSchema] = []
  riskScore: float
  riskLevel: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
  calibration: Optional[CalibrationPayloadSchema] = None
  scene: Optional[ScenePayloadSchema] = None
  riskBreakdown: Optional[RiskBreakdownSchema] = None
  metrics: Optional[MetricsSchema] = None
  riskSnapshotsInfo: Optional[RiskSnapshotsInfoSchema] = None
  riskSnapshots: Optional[List[RiskSnapshotSchema]] = None
  newRiskSnapshot: Optional[RiskSnapshotSchema] = None
  monitoringControl: Optional[MonitoringControlSchema] = None



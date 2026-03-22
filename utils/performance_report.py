import json
import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
  try:
    number = float(value)
  except (TypeError, ValueError):
    return None
  if math.isnan(number) or math.isinf(number):
    return None
  return number


def _percentile(values: List[float], pct: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(values)
  if len(ordered) == 1:
    return ordered[0]

  rank = (len(ordered) - 1) * (pct / 100.0)
  low = int(math.floor(rank))
  high = int(math.ceil(rank))
  if low == high:
    return ordered[low]
  weight = rank - low
  return ordered[low] * (1.0 - weight) + ordered[high] * weight


@dataclass
class MetricSeries:
  max_samples: int = 120000
  count: int = 0
  total: float = 0.0
  minimum: float = math.inf
  maximum: float = -math.inf
  first: Optional[float] = None
  last: Optional[float] = None
  samples: List[float] = field(default_factory=list)

  def add(self, value: float) -> None:
    self.count += 1
    self.total += value
    self.minimum = min(self.minimum, value)
    self.maximum = max(self.maximum, value)
    if self.first is None:
      self.first = value
    self.last = value
    if len(self.samples) >= self.max_samples:
      # Keep memory bounded while preserving long-session shape.
      self.samples = self.samples[::2]
    self.samples.append(value)

  def summary(self) -> Optional[Dict[str, float]]:
    if self.count <= 0:
      return None

    p50 = _percentile(self.samples, 50.0)
    p95 = _percentile(self.samples, 95.0)
    avg = self.total / float(self.count)
    first = self.first if self.first is not None else avg
    last = self.last if self.last is not None else avg
    delta = last - first
    delta_pct = (delta / first * 100.0) if abs(first) > 1e-6 else 0.0

    return {
      "count": float(self.count),
      "avg": round(avg, 4),
      "min": round(self.minimum, 4),
      "p50": round(p50, 4),
      "p95": round(p95, 4),
      "max": round(self.maximum, 4),
      "first": round(first, 4),
      "last": round(last, 4),
      "delta": round(delta, 4),
      "deltaPercent": round(delta_pct, 2),
    }


class SessionPerformanceReporter:
  """Aggregates live per-frame metrics into a report-ready session summary."""

  def __init__(
    self,
    device_profile: Dict[str, Any],
    sample_period_sec: float = 1.0,
    timeline_limit: int = 3600,
    event_limit: int = 250,
    export_dir: str = "reports",
  ) -> None:
    self._device_profile = dict(device_profile or {})
    self._sample_period_sec = max(0.2, float(sample_period_sec))
    self._timeline_limit = max(100, int(timeline_limit))
    self._event_limit = max(50, int(event_limit))
    self._export_dir = str(export_dir or "reports")

    self.session_id = str(uuid.uuid4())
    self.started_at = _utc_now_iso()
    self._start_mono = monotonic()
    self._last_timeline_sec = -1.0
    self._sample_count = 0
    self._metric_series: Dict[str, MetricSeries] = {}
    self._timeline: List[Dict[str, Any]] = []
    self._events: List[Dict[str, Any]] = []
    self._threshold_active: Dict[str, bool] = {}
    self._last_dropped_frames: Optional[float] = None

  def reset(self) -> Dict[str, Any]:
    self.session_id = str(uuid.uuid4())
    self.started_at = _utc_now_iso()
    self._start_mono = monotonic()
    self._last_timeline_sec = -1.0
    self._sample_count = 0
    self._metric_series.clear()
    self._timeline.clear()
    self._events.clear()
    self._threshold_active.clear()
    self._last_dropped_frames = None
    return self.status()

  def status(self) -> Dict[str, Any]:
    return {
      "sessionId": self.session_id,
      "startedAt": self.started_at,
      "durationSec": round(self._elapsed_sec(), 2),
      "samples": int(self._sample_count),
      "timelinePoints": int(len(self._timeline)),
      "events": int(len(self._events)),
    }

  def record(self, resources: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    elapsed = self._elapsed_sec()
    sample = self._compose_sample(resources=resources, metrics=metrics)
    if not sample:
      return

    self._sample_count += 1
    for key, value in sample.items():
      numeric = _safe_float(value)
      if numeric is None:
        continue
      series = self._metric_series.setdefault(key, MetricSeries())
      series.add(numeric)

    self._check_threshold_events(sample, elapsed)
    self._append_timeline(sample, elapsed)

  def build_report(self) -> Dict[str, Any]:
    ended_at = _utc_now_iso()
    duration_sec = self._elapsed_sec()

    metric_summary: Dict[str, Dict[str, float]] = {}
    for key, series in self._metric_series.items():
      summary = series.summary()
      if summary is not None:
        metric_summary[key] = summary

    report = {
      "schemaVersion": "1.0",
      "session": {
        "id": self.session_id,
        "startedAt": self.started_at,
        "endedAt": ended_at,
        "durationSec": round(duration_sec, 3),
        "samples": int(self._sample_count),
      },
      "device": dict(self._device_profile),
      "comparisonVector": self._comparison_vector(metric_summary),
      "summary": self._build_summary(metric_summary),
      "observations": self._build_observations(metric_summary),
      "eventLog": list(self._events),
      "timeline": list(self._timeline),
      "llmReady": self._llm_ready(metric_summary),
    }
    return report

  def export_to_file(self, directory: Optional[str] = None) -> Dict[str, Any]:
    report = self.build_report()
    export_root = Path(directory or self._export_dir)
    export_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    device_tag = self._slug(str(self._device_profile.get("deviceTag", "device")))
    filename = f"perf-report-{device_tag}-{ts}-{self.session_id[:8]}.json"
    path = export_root / filename
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
      "path": str(path.resolve()),
      "sizeBytes": int(path.stat().st_size),
      "sessionId": self.session_id,
      "createdAt": _utc_now_iso(),
    }

  def _elapsed_sec(self) -> float:
    return max(0.0, monotonic() - self._start_mono)

  def _compose_sample(self, resources: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, float]:
    base: Dict[str, float] = {}

    resource_keys = [
      "cpuSystemPercent",
      "cpuProcessPercent",
      "cpuLogicalCores",
      "memorySystemPercent",
      "memoryUsedGb",
      "memoryAvailableGb",
      "processRssMb",
      "processVmsMb",
      "processThreads",
      "gpuUtilPercent",
      "gpuMemoryUtilPercent",
      "gpuTempC",
      "gpuPowerW",
      "gpuMemoryUsedMb",
      "gpuMemoryTotalMb",
      "gpuMemoryAllocatedMb",
      "gpuMemoryReservedMb",
      "gpuMemoryAllocatedPercent",
      "gpuMemoryReservedPercent",
      "gpuSmiMemoryUsedMb",
      "gpuSmiMemoryTotalMb",
    ]

    metric_keys = [
      "fps",
      "totalTimeMs",
      "faceTimeMs",
      "yoloTimeMs",
      "poseTimeMs",
      "qualityTimeMs",
      "riskTimeMs",
      "droppedFrames",
      "processedFrames",
    ]

    for key in resource_keys:
      numeric = _safe_float(resources.get(key))
      if numeric is not None:
        base[key] = numeric

    for key in metric_keys:
      numeric = _safe_float(metrics.get(key))
      if numeric is not None:
        base[key] = numeric

    fps = base.get("fps")
    cpu_proc = base.get("cpuProcessPercent")
    cpu_sys = base.get("cpuSystemPercent")
    frame_ms = base.get("totalTimeMs")

    if fps is not None and cpu_proc is not None and cpu_proc > 0.01:
      base["fpsPerCpuProcess"] = fps / cpu_proc
    if fps is not None and cpu_sys is not None and cpu_sys > 0.01:
      base["fpsPerCpuSystem"] = fps / cpu_sys
    if frame_ms is not None and cpu_proc is not None and cpu_proc > 0.01:
      base["frameMsPerCpuProcess"] = frame_ms / cpu_proc
    if frame_ms is not None and cpu_sys is not None and cpu_sys > 0.01:
      base["frameMsPerCpuSystem"] = frame_ms / cpu_sys

    return base

  def _append_timeline(self, sample: Dict[str, float], elapsed: float) -> None:
    if self._last_timeline_sec >= 0 and (elapsed - self._last_timeline_sec) < self._sample_period_sec:
      return

    self._last_timeline_sec = elapsed
    row = {
      "tSec": round(elapsed, 2),
      "fps": round(sample.get("fps", 0.0), 3),
      "frameMs": round(sample.get("totalTimeMs", 0.0), 3),
      "cpuProcessPercent": round(sample.get("cpuProcessPercent", 0.0), 3),
      "memorySystemPercent": round(sample.get("memorySystemPercent", 0.0), 3),
      "processRssMb": round(sample.get("processRssMb", 0.0), 3),
      "gpuUtilPercent": round(sample.get("gpuUtilPercent", 0.0), 3),
      "gpuTempC": round(sample.get("gpuTempC", 0.0), 3),
      "droppedFrames": round(sample.get("droppedFrames", 0.0), 3),
      "fpsPerCpuProcess": round(sample.get("fpsPerCpuProcess", 0.0), 4),
    }
    self._timeline.append(row)
    if len(self._timeline) > self._timeline_limit:
      self._timeline = self._timeline[-self._timeline_limit:]

  def _check_threshold_events(self, sample: Dict[str, float], elapsed: float) -> None:
    self._check_crossing(
      key="cpu_process_hot",
      value=sample.get("cpuProcessPercent"),
      threshold=90.0,
      elapsed=elapsed,
      category="cpu",
      severity="high",
      title_on="Process CPU spike",
      title_off="Process CPU recovered",
      metric_name="cpuProcessPercent",
    )
    self._check_crossing(
      key="ram_pressure",
      value=sample.get("memorySystemPercent"),
      threshold=90.0,
      elapsed=elapsed,
      category="memory",
      severity="high",
      title_on="System RAM pressure",
      title_off="System RAM normalized",
      metric_name="memorySystemPercent",
    )
    self._check_crossing(
      key="gpu_temp_hot",
      value=sample.get("gpuTempC"),
      threshold=85.0,
      elapsed=elapsed,
      category="gpu",
      severity="high",
      title_on="GPU temperature high",
      title_off="GPU temperature recovered",
      metric_name="gpuTempC",
    )
    self._check_crossing(
      key="gpu_mem_pressure",
      value=sample.get("gpuMemoryUtilPercent"),
      threshold=92.0,
      elapsed=elapsed,
      category="gpu",
      severity="medium",
      title_on="GPU memory pressure",
      title_off="GPU memory pressure reduced",
      metric_name="gpuMemoryUtilPercent",
    )
    self._check_crossing(
      key="low_fps",
      value=sample.get("fps"),
      threshold=5.0,
      elapsed=elapsed,
      category="throughput",
      severity="high",
      title_on="Low FPS",
      title_off="FPS recovered",
      metric_name="fps",
      lower_is_bad=True,
    )
    self._check_crossing(
      key="slow_frame",
      value=sample.get("totalTimeMs"),
      threshold=220.0,
      elapsed=elapsed,
      category="throughput",
      severity="medium",
      title_on="Frame latency spike",
      title_off="Frame latency normalized",
      metric_name="totalTimeMs",
    )

    dropped = sample.get("droppedFrames")
    if dropped is not None:
      if self._last_dropped_frames is not None and dropped > self._last_dropped_frames:
        delta = dropped - self._last_dropped_frames
        self._add_event(
          elapsed=elapsed,
          severity="medium",
          category="queue",
          title="Dropped frame increment",
          details=f"Dropped frame counter increased by {int(delta)}.",
          evidence={"droppedFrames": round(dropped, 2)},
        )
      self._last_dropped_frames = dropped

  def _check_crossing(
    self,
    key: str,
    value: Optional[float],
    threshold: float,
    elapsed: float,
    category: str,
    severity: str,
    title_on: str,
    title_off: str,
    metric_name: str,
    lower_is_bad: bool = False,
  ) -> None:
    if value is None:
      return

    active = value <= threshold if lower_is_bad else value >= threshold
    prev_active = self._threshold_active.get(key, False)
    if active and not prev_active:
      op = "<=" if lower_is_bad else ">="
      self._add_event(
        elapsed=elapsed,
        severity=severity,
        category=category,
        title=title_on,
        details=f"{metric_name} {op} {threshold}",
        evidence={metric_name: round(value, 3), "threshold": threshold},
      )
    elif (not active) and prev_active:
      self._add_event(
        elapsed=elapsed,
        severity="low",
        category=category,
        title=title_off,
        details=f"{metric_name} returned inside expected range.",
        evidence={metric_name: round(value, 3), "threshold": threshold},
      )
    self._threshold_active[key] = active

  def _add_event(
    self,
    elapsed: float,
    severity: str,
    category: str,
    title: str,
    details: str,
    evidence: Optional[Dict[str, Any]] = None,
  ) -> None:
    event = {
      "tSec": round(elapsed, 2),
      "timestamp": _utc_now_iso(),
      "severity": severity.upper(),
      "category": category,
      "title": title,
      "details": details,
      "evidence": dict(evidence or {}),
    }
    self._events.append(event)
    if len(self._events) > self._event_limit:
      self._events = self._events[-self._event_limit:]

  def _build_summary(self, metric_summary: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    return {
      "throughput": {
        "fps": metric_summary.get("fps"),
        "frameTimeMs": metric_summary.get("totalTimeMs"),
        "droppedFrames": metric_summary.get("droppedFrames"),
      },
      "cpu": {
        "systemPercent": metric_summary.get("cpuSystemPercent"),
        "processPercent": metric_summary.get("cpuProcessPercent"),
      },
      "memory": {
        "systemPercent": metric_summary.get("memorySystemPercent"),
        "processRssMb": metric_summary.get("processRssMb"),
      },
      "gpu": {
        "utilPercent": metric_summary.get("gpuUtilPercent"),
        "memoryUtilPercent": metric_summary.get("gpuMemoryUtilPercent"),
        "temperatureC": metric_summary.get("gpuTempC"),
        "powerW": metric_summary.get("gpuPowerW"),
        "memoryUsedMb": metric_summary.get("gpuMemoryUsedMb"),
      },
      "efficiency": {
        "fpsPerCpuProcess": metric_summary.get("fpsPerCpuProcess"),
        "fpsPerCpuSystem": metric_summary.get("fpsPerCpuSystem"),
        "frameMsPerCpuProcess": metric_summary.get("frameMsPerCpuProcess"),
      },
    }

  def _comparison_vector(self, metric_summary: Dict[str, Dict[str, float]]) -> Dict[str, Optional[float]]:
    def take(metric: str, stat: str = "avg") -> Optional[float]:
      item = metric_summary.get(metric)
      if not item:
        return None
      value = item.get(stat)
      return float(value) if value is not None else None

    return {
      "fpsAvg": take("fps", "avg"),
      "fpsP95": take("fps", "p95"),
      "frameTimeAvgMs": take("totalTimeMs", "avg"),
      "frameTimeP95Ms": take("totalTimeMs", "p95"),
      "cpuProcessAvgPercent": take("cpuProcessPercent", "avg"),
      "cpuProcessPeakPercent": take("cpuProcessPercent", "max"),
      "memorySystemPeakPercent": take("memorySystemPercent", "max"),
      "processRssPeakMb": take("processRssMb", "max"),
      "gpuUtilAvgPercent": take("gpuUtilPercent", "avg"),
      "gpuTempPeakC": take("gpuTempC", "max"),
      "gpuMemoryUtilPeakPercent": take("gpuMemoryUtilPercent", "max"),
      "fpsPerCpuProcessAvg": take("fpsPerCpuProcess", "avg"),
    }

  def _build_observations(self, metric_summary: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []

    def stat(metric: str, field: str) -> Optional[float]:
      bucket = metric_summary.get(metric)
      if not bucket:
        return None
      value = bucket.get(field)
      if value is None:
        return None
      return float(value)

    fps_avg = stat("fps", "avg")
    fps_p95 = stat("fps", "p95")
    frame_p95 = stat("totalTimeMs", "p95")
    cpu_proc_avg = stat("cpuProcessPercent", "avg")
    cpu_proc_peak = stat("cpuProcessPercent", "max")
    ram_sys_peak = stat("memorySystemPercent", "max")
    rss_delta = stat("processRssMb", "delta")
    drop_max = stat("droppedFrames", "max")
    gpu_temp_peak = stat("gpuTempC", "max")
    gpu_mem_peak = stat("gpuMemoryUtilPercent", "max")

    if fps_avg is not None:
      if fps_avg < 5.0:
        observations.append(self._obs("HIGH", "throughput", "Low average FPS", f"Average FPS is {fps_avg:.2f}."))
      elif fps_avg < 10.0:
        observations.append(self._obs("MEDIUM", "throughput", "Moderate FPS", f"Average FPS is {fps_avg:.2f}."))
      else:
        observations.append(self._obs("LOW", "throughput", "Stable FPS", f"Average FPS is {fps_avg:.2f}."))

    if frame_p95 is not None and frame_p95 > 220.0:
      observations.append(
        self._obs("HIGH", "latency", "High frame latency p95", f"p95 frame time is {frame_p95:.2f} ms.")
      )

    if cpu_proc_peak is not None and cpu_proc_peak >= 90.0:
      observations.append(
        self._obs("HIGH", "cpu", "CPU saturation risk", f"Process CPU peaked at {cpu_proc_peak:.2f}%.")
      )
    elif cpu_proc_avg is not None and cpu_proc_avg >= 70.0:
      observations.append(
        self._obs("MEDIUM", "cpu", "Elevated process CPU", f"Average process CPU is {cpu_proc_avg:.2f}%.")
      )

    if ram_sys_peak is not None and ram_sys_peak >= 90.0:
      observations.append(
        self._obs("HIGH", "memory", "System memory pressure", f"System RAM peaked at {ram_sys_peak:.2f}%.")
      )

    if rss_delta is not None and rss_delta > 300.0:
      observations.append(
        self._obs("MEDIUM", "memory", "Process memory growth", f"Process RSS grew by {rss_delta:.2f} MB.")
      )

    if drop_max is not None and drop_max > 0:
      observations.append(
        self._obs("MEDIUM", "queue", "Frames dropped", f"Dropped frame counter reached {drop_max:.0f}.")
      )

    if gpu_temp_peak is not None and gpu_temp_peak >= 85.0:
      observations.append(
        self._obs("HIGH", "gpu", "GPU thermal pressure", f"GPU temperature peaked at {gpu_temp_peak:.1f}C.")
      )

    if gpu_mem_peak is not None and gpu_mem_peak >= 92.0:
      observations.append(
        self._obs("MEDIUM", "gpu", "GPU memory pressure", f"GPU memory utilization peaked at {gpu_mem_peak:.1f}%.")
      )

    if not observations:
      observations.append(self._obs("LOW", "general", "No significant anomalies", "No major threshold breach observed."))

    # Prioritize highest severity first for quick downstream summarization.
    weight = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    observations.sort(key=lambda item: weight.get(str(item.get("severity")), 0), reverse=True)

    if fps_p95 is not None:
      observations.append(
        self._obs("LOW", "throughput", "Upper FPS envelope", f"p95 FPS is {fps_p95:.2f}.")
      )

    return observations[:16]

  def _llm_ready(self, metric_summary: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    device_name = str(self._device_profile.get("deviceTag", "unknown-device"))
    headline = [
      f"Session {self.session_id[:8]} on {device_name}",
      f"Duration {self._elapsed_sec():.1f}s, samples {self._sample_count}.",
    ]
    return {
      "headline": headline,
      "keyMetrics": self._comparison_vector(metric_summary),
      "eventCount": len(self._events),
      "observationCount": len(self._build_observations(metric_summary)),
      "reportUseHint": (
        "Use device + comparisonVector across laptops to compare throughput, CPU/GPU pressure, "
        "and memory behavior for the same model/scenario."
      ),
    }

  @staticmethod
  def _obs(severity: str, category: str, title: str, details: str) -> Dict[str, str]:
    return {
      "severity": severity,
      "category": category,
      "title": title,
      "details": details,
    }

  @staticmethod
  def _slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return cleaned or "device"

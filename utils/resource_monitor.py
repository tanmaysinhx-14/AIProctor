import hashlib
import os
import platform
import socket
import subprocess
import sys
from threading import Lock
from time import monotonic
from typing import Any, Dict, List, Optional

import psutil
import torch


class ResourceMonitor:
  """Collect lightweight process/system/GPU resource snapshots for per-frame metrics."""

  def __init__(self, smi_refresh_sec: float = 1.0) -> None:
    self._process = psutil.Process(os.getpid())
    self._smi_refresh_sec = max(0.5, float(smi_refresh_sec))
    self._smi_lock = Lock()
    self._last_smi_ts = 0.0
    self._last_smi: Dict[str, Any] = {}
    self._device_profile = self._collect_device_profile()

    # Warm-up for non-blocking cpu_percent deltas.
    psutil.cpu_percent(interval=None)
    self._process.cpu_percent(interval=None)

  def device_profile(self) -> Dict[str, Any]:
    profile = dict(self._device_profile)
    gpus = self._device_profile.get("gpuDevices")
    if isinstance(gpus, list):
      profile["gpuDevices"] = [dict(item) if isinstance(item, dict) else item for item in gpus]
    return profile

  def sample(self) -> Dict[str, Any]:
    resources: Dict[str, Any] = {}

    cpu_system_pct = float(psutil.cpu_percent(interval=None))
    cpu_process_pct = float(self._process.cpu_percent(interval=None))
    vm = psutil.virtual_memory()
    proc_mem = self._process.memory_info()

    resources.update(
      {
        "cpuSystemPercent": round(cpu_system_pct, 2),
        "cpuProcessPercent": round(cpu_process_pct, 2),
        "cpuLogicalCores": float(psutil.cpu_count(logical=True) or 0),
        "memorySystemPercent": round(float(vm.percent), 2),
        "memoryUsedGb": round(float(vm.used) / (1024 ** 3), 3),
        "memoryAvailableGb": round(float(vm.available) / (1024 ** 3), 3),
        "processRssMb": round(float(proc_mem.rss) / (1024 ** 2), 2),
        "processVmsMb": round(float(proc_mem.vms) / (1024 ** 2), 2),
        "processThreads": float(self._process.num_threads()),
      }
    )

    if torch.cuda.is_available():
      gpu_idx = 0
      try:
        gpu_idx = int(torch.cuda.current_device())
      except Exception:
        gpu_idx = 0

      resources.update(self._gpu_snapshot(gpu_idx))
    else:
      resources.update(
        {
          "gpuAvailable": False,
          "gpuName": "N/A",
        }
      )

    return resources

  def _gpu_snapshot(self, gpu_idx: int) -> Dict[str, Any]:
    gpu: Dict[str, Any] = {
      "gpuAvailable": True,
      "gpuIndex": float(gpu_idx),
    }

    try:
      gpu_name = torch.cuda.get_device_name(gpu_idx)
    except Exception:
      gpu_name = "Unknown GPU"
    gpu["gpuName"] = gpu_name

    try:
      free_b, total_b = torch.cuda.mem_get_info(gpu_idx)
      used_b = max(float(total_b - free_b), 0.0)
      total_mb = float(total_b) / (1024 ** 2)
      free_mb = float(free_b) / (1024 ** 2)
      used_mb = used_b / (1024 ** 2)
      gpu.update(
        {
          "gpuMemoryTotalMb": round(total_mb, 2),
          "gpuMemoryUsedMb": round(used_mb, 2),
          "gpuMemoryFreeMb": round(free_mb, 2),
        }
      )
    except Exception:
      pass

    try:
      alloc_mb = float(torch.cuda.memory_allocated(gpu_idx)) / (1024 ** 2)
      reserved_mb = float(torch.cuda.memory_reserved(gpu_idx)) / (1024 ** 2)
      max_alloc_mb = float(torch.cuda.max_memory_allocated(gpu_idx)) / (1024 ** 2)
      gpu.update(
        {
          "gpuMemoryAllocatedMb": round(alloc_mb, 2),
          "gpuMemoryReservedMb": round(reserved_mb, 2),
          "gpuMaxMemoryAllocatedMb": round(max_alloc_mb, 2),
        }
      )

      total_mb = float(gpu.get("gpuMemoryTotalMb", 0.0))
      if total_mb > 0:
        gpu["gpuMemoryAllocatedPercent"] = round((alloc_mb / total_mb) * 100.0, 2)
        gpu["gpuMemoryReservedPercent"] = round((reserved_mb / total_mb) * 100.0, 2)
    except Exception:
      pass

    gpu.update(self._nvidia_smi_snapshot(gpu_idx))
    return gpu

  def _collect_device_profile(self) -> Dict[str, Any]:
    logical_cores = int(psutil.cpu_count(logical=True) or 0)
    physical_cores = int(psutil.cpu_count(logical=False) or 0)
    vm = psutil.virtual_memory()

    gpu_devices: List[Dict[str, Any]] = []
    gpu_available = bool(torch.cuda.is_available())
    if gpu_available:
      count = int(torch.cuda.device_count())
      for idx in range(count):
        gpu_devices.append(self._cuda_device_profile(idx))

    profile: Dict[str, Any] = {
      "deviceTag": str(socket.gethostname() or "unknown-host"),
      "hostName": str(socket.gethostname() or "unknown-host"),
      "os": platform.system(),
      "osVersion": platform.version(),
      "platform": platform.platform(),
      "machine": platform.machine(),
      "pythonVersion": platform.python_version(),
      "pythonExecutable": sys.executable,
      "torchVersion": getattr(torch, "__version__", "unknown"),
      "cudaVersion": getattr(getattr(torch, "version", None), "cuda", None),
      "cpuModel": platform.processor() or "Unknown CPU",
      "cpuPhysicalCores": physical_cores,
      "cpuLogicalCores": logical_cores,
      "memoryTotalGb": round(float(vm.total) / (1024 ** 3), 3),
      "gpuAvailable": gpu_available,
      "gpuCount": len(gpu_devices),
      "gpuDevices": gpu_devices,
    }

    gpu_names = ",".join(str(item.get("name", "")) for item in gpu_devices)
    raw_fingerprint = (
      f"{profile['hostName']}|{profile['platform']}|{profile['cpuModel']}|"
      f"{profile['cpuLogicalCores']}|{profile['memoryTotalGb']}|{gpu_names}"
    )
    profile["deviceFingerprint"] = hashlib.sha1(raw_fingerprint.encode("utf-8")).hexdigest()[:16]
    return profile

  @staticmethod
  def _cuda_device_profile(index: int) -> Dict[str, Any]:
    item: Dict[str, Any] = {
      "index": int(index),
      "name": "Unknown GPU",
    }
    try:
      item["name"] = str(torch.cuda.get_device_name(index))
    except Exception:
      item["name"] = "Unknown GPU"

    try:
      props = torch.cuda.get_device_properties(index)
      total_gb = float(getattr(props, "total_memory", 0.0)) / (1024 ** 3)
      item["memoryTotalGb"] = round(total_gb, 3)
      mp_count = int(getattr(props, "multi_processor_count", 0))
      if mp_count > 0:
        item["multiProcessorCount"] = mp_count
    except Exception:
      pass
    return item

  def _nvidia_smi_snapshot(self, gpu_idx: int) -> Dict[str, Any]:
    with self._smi_lock:
      now = monotonic()
      if (now - self._last_smi_ts) < self._smi_refresh_sec and self._last_smi:
        return dict(self._last_smi)

      self._last_smi_ts = now
      smi: Dict[str, Any] = {}
      try:
        result = subprocess.run(
          [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu_idx),
          ],
          capture_output=True,
          text=True,
          timeout=0.9,
          check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
          parts = [part.strip() for part in result.stdout.strip().split(",")]
          if len(parts) >= 6:
            smi["gpuUtilPercent"] = self._to_float(parts[0])
            smi["gpuMemoryUtilPercent"] = self._to_float(parts[1])
            smi["gpuTempC"] = self._to_float(parts[2])
            smi["gpuPowerW"] = self._to_float(parts[3])
            smi["gpuSmiMemoryUsedMb"] = self._to_float(parts[4])
            smi["gpuSmiMemoryTotalMb"] = self._to_float(parts[5])
      except Exception:
        smi = {}

      self._last_smi = smi
      return dict(self._last_smi)

  @staticmethod
  def _to_float(raw: str) -> Optional[float]:
    try:
      value = float(raw)
    except (TypeError, ValueError):
      return None
    return round(value, 2)

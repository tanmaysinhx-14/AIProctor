from collections import deque
from threading import Lock
from time import perf_counter
from typing import Deque, Optional


class RollingFps:
  def __init__(self, window_size: int = 30) -> None:
    if window_size < 2:
      raise ValueError("window_size must be >= 2")
    self._timestamps: Deque[float] = deque(maxlen=window_size)
    self._lock = Lock()

  def tick(self, timestamp: Optional[float] = None) -> None:
    ts = perf_counter() if timestamp is None else timestamp
    with self._lock:
      self._timestamps.append(ts)

  @property
  def value(self) -> float:
    with self._lock:
      if len(self._timestamps) < 2:
        return 0.0
      duration = self._timestamps[-1] - self._timestamps[0]
      if duration <= 0.0:
        return 0.0
      return (len(self._timestamps) - 1) / duration

  def reset(self) -> None:
    with self._lock:
      self._timestamps.clear()

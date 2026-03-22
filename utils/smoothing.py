from collections import deque
from typing import Deque


class MovingAverage:
  def __init__(self, window_size: int) -> None:
    if window_size <= 0:
      raise ValueError("window_size must be positive")
    self.window_size = window_size
    self.values: Deque[float] = deque(maxlen=window_size)

  def add(self, value: float) -> float:
    self.values.append(float(value))
    return self.current

  @property
  def current(self) -> float:
    if not self.values:
      return 0.0
    return sum(self.values) / len(self.values)

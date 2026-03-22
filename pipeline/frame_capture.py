from queue import Empty, Full, Queue
from threading import Event, Thread
from time import sleep
from typing import Optional, Tuple

import cv2
import numpy as np


class FrameCaptureThread:
  """Dedicated capture thread that always keeps the newest camera frame."""

  def __init__(self, camera_index: int = 0, queue_size: int = 2) -> None:
    self.camera_index = int(camera_index)
    self._queue: Queue[Tuple[int, np.ndarray]] = Queue(maxsize=max(1, int(queue_size)))
    self._stop_event = Event()
    self._thread: Optional[Thread] = None
    self._capture: Optional[cv2.VideoCapture] = None
    self._seq = 0

  def start(self) -> None:
    if self._thread and self._thread.is_alive():
      return

    self._stop_event.clear()
    self._capture = cv2.VideoCapture(self.camera_index)
    self._thread = Thread(target=self._loop, name="frame-capture", daemon=True)
    self._thread.start()

  def stop(self) -> None:
    self._stop_event.set()
    if self._thread and self._thread.is_alive():
      self._thread.join(timeout=1.2)
    self._thread = None
    if self._capture is not None:
      self._capture.release()
      self._capture = None

    while not self._queue.empty():
      try:
        self._queue.get_nowait()
      except Empty:
        break

  def latest(self, timeout_sec: float = 0.05) -> Optional[Tuple[int, np.ndarray]]:
    try:
      item = self._queue.get(timeout=max(0.0, float(timeout_sec)))
      return item
    except Empty:
      return None

  def _loop(self) -> None:
    while not self._stop_event.is_set():
      if self._capture is None:
        sleep(0.05)
        continue

      ok, frame = self._capture.read()
      if not ok or frame is None:
        sleep(0.01)
        continue

      self._seq += 1
      packet = (self._seq, frame)
      if self._queue.full():
        try:
          self._queue.get_nowait()
        except Empty:
          pass
      try:
        self._queue.put_nowait(packet)
      except Full:
        pass

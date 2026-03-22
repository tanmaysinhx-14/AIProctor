from queue import Empty, Full, Queue
from typing import Any, Dict, Optional


class LatestResultRenderer:
  """
  Non-blocking renderer buffer:
  always keeps only the latest result so UI publishing never slows inference.
  """

  def __init__(self, maxsize: int = 1) -> None:
    self._queue: Queue[Dict[str, Any]] = Queue(maxsize=max(1, int(maxsize)))

  def publish(self, payload: Dict[str, Any]) -> None:
    if self._queue.full():
      try:
        self._queue.get_nowait()
      except Empty:
        pass
    try:
      self._queue.put_nowait(payload)
    except Full:
      pass

  def poll_latest(self) -> Optional[Dict[str, Any]]:
    latest: Optional[Dict[str, Any]] = None
    while True:
      try:
        latest = self._queue.get_nowait()
      except Empty:
        break
    return latest

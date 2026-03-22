from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic
from typing import Any, Callable, Dict, List, Optional


class ThreadedInferenceEngine:
  """
  Threaded inference executor with bounded queues.
  Keeps camera/input ingestion non-blocking by dropping oldest work when saturated.
  """

  def __init__(
    self,
    infer_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    max_in_queue: int = 2,
    max_out_queue: int = 2,
    batch_size: int = 1,
    batch_wait_ms: float = 6.0,
  ) -> None:
    self._infer_fn = infer_fn
    self._in_q: Queue[Dict[str, Any]] = Queue(maxsize=max(1, int(max_in_queue)))
    self._out_q: Queue[Dict[str, Any]] = Queue(maxsize=max(1, int(max_out_queue)))
    self._batch_size = max(1, int(batch_size))
    self._batch_wait_ms = max(0.0, float(batch_wait_ms))
    self._stop_event = Event()
    self._worker: Optional[Thread] = None

  def start(self) -> None:
    if self._worker and self._worker.is_alive():
      return
    self._stop_event.clear()
    self._worker = Thread(target=self._loop, name="inference-engine", daemon=True)
    self._worker.start()

  def stop(self) -> None:
    self._stop_event.set()
    if self._worker and self._worker.is_alive():
      self._worker.join(timeout=1.2)
    self._worker = None

  def submit(self, item: Dict[str, Any]) -> None:
    if self._in_q.full():
      try:
        self._in_q.get_nowait()
      except Empty:
        pass
    try:
      self._in_q.put_nowait(item)
    except Full:
      pass

  def get_latest(self, timeout_sec: float = 0.0) -> Optional[Dict[str, Any]]:
    if timeout_sec > 0:
      try:
        return self._out_q.get(timeout=timeout_sec)
      except Empty:
        return None

    latest: Optional[Dict[str, Any]] = None
    while True:
      try:
        latest = self._out_q.get_nowait()
      except Empty:
        break
    return latest

  def _loop(self) -> None:
    while not self._stop_event.is_set():
      first = self._poll_next(timeout_sec=0.05)
      if first is None:
        continue

      batch: List[Dict[str, Any]] = [first]
      if self._batch_size > 1:
        deadline = monotonic() + (self._batch_wait_ms / 1000.0)
        while len(batch) < self._batch_size and monotonic() < deadline:
          nxt = self._poll_next(timeout_sec=0.0)
          if nxt is None:
            continue
          batch.append(nxt)

      for entry in batch:
        if self._stop_event.is_set():
          break
        try:
          result = self._infer_fn(entry)
        except Exception as exc:
          result = {
            "ok": False,
            "error": str(exc),
          }

        if self._out_q.full():
          try:
            self._out_q.get_nowait()
          except Empty:
            pass
        try:
          self._out_q.put_nowait(result)
        except Full:
          pass

  def _poll_next(self, timeout_sec: float) -> Optional[Dict[str, Any]]:
    try:
      return self._in_q.get(timeout=max(0.0, timeout_sec))
    except Empty:
      return None

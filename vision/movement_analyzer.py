from math import sqrt
from typing import Dict, List, Tuple

import numpy as np

from vision.face_tracker import TrackedFace


class MovementAnalyzer:
  def __init__(self) -> None:
    self._previous_centroids: Dict[int, Tuple[float, float]] = {}

  def compute(self, tracked_faces: List[TrackedFace], frame_shape: Tuple[int, int, int]) -> Dict[int, float]:
    h, w = frame_shape[:2]
    frame_diagonal = max(sqrt(float(w * w + h * h)), 1.0)
    active_ids = set()
    movement_by_id: Dict[int, float] = {}

    for face in tracked_faces:
      active_ids.add(face.id)
      previous = self._previous_centroids.get(face.id)
      current = face.centroid

      if previous is None:
        movement = 0.0
      else:
        delta = np.linalg.norm(np.array(current) - np.array(previous))
        movement = float(delta / frame_diagonal)

      self._previous_centroids[face.id] = current
      movement_by_id[face.id] = movement

    stale_ids = [track_id for track_id in self._previous_centroids if track_id not in active_ids]
    for track_id in stale_ids:
      del self._previous_centroids[track_id]

    return movement_by_id

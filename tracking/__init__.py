__all__ = ["AdaptiveFaceTracker"]


def __getattr__(name: str):
  if name == "AdaptiveFaceTracker":
    from tracking.face_tracker import AdaptiveFaceTracker

    return AdaptiveFaceTracker
  raise AttributeError(f"module 'tracking' has no attribute {name!r}")

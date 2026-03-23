from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Deque, Dict, FrozenSet, List, Optional, Set


@dataclass
class TierEvent:
  tier: int
  rule: str
  label: str
  timestamp_iso: str
  session_elapsed_sec: float
  evidence: Dict[str, Any]
  escalated_from: Optional[int] = None

  def to_dict(self) -> Dict[str, Any]:
    return {
      "tier": self.tier,
      "rule": self.rule,
      "label": self.label,
      "timestampIso": self.timestamp_iso,
      "sessionElapsedSec": round(self.session_elapsed_sec, 2),
      "evidence": self.evidence,
      "escalatedFrom": self.escalated_from,
    }


class TierEngine:
  """
  Classifies risk rule activations into three response tiers:

  Tier 1 — Single brief trigger.
            Logged silently. No score penalty. Available for
            post-exam review only.

  Tier 2 — Same rule fires TIER2_MIN_TRIGGERS times within a
            rolling TIER2_WINDOW_SEC window.
            Flagged for human review.

  Tier 3 — High-confidence violation.
            Either a dangerous combination of two rules active
            simultaneously, or one high-severity rule sustained
            for too long.  Immediate alert.
  """

  # ── Tier 2 ─────────────────────────────────────────────────
  TIER2_WINDOW_SEC: float = 120.0
  TIER2_MIN_TRIGGERS: int = 3

  # ── Tier 3 combinations ────────────────────────────────────
  TIER3_COMBOS: List[Dict[str, Any]] = [
    {
      "id": "phone_gaze",
      "rules": frozenset({"phone_detected", "extreme_gaze"}),
      "label": "Phone + Looking Away",
      "description": "Phone visible while student eyes are off-screen.",
    },
    {
      "id": "phone_orientation",
      "rules": frozenset({"phone_detected", "face_not_facing_camera"}),
      "label": "Phone + Facing Away",
      "description": "Phone visible while student turned away from camera.",
    },
    {
      "id": "phone_missing",
      "rules": frozenset({"phone_detected", "face_missing"}),
      "label": "Phone + Face Hidden",
      "description": "Phone visible while student face is hidden.",
    },
    {
      "id": "persons_gaze",
      "rules": frozenset({"multiple_persons", "extreme_gaze"}),
      "label": "Extra Person + Gaze Away",
      "description": "Another person present while student looks away.",
    },
    {
      "id": "persons_orientation",
      "rules": frozenset({"multiple_persons", "face_not_facing_camera"}),
      "label": "Extra Person + Facing Away",
      "description": "Another person present while student faces away.",
    },
    {
      "id": "persons_missing",
      "rules": frozenset({"multiple_persons", "face_missing"}),
      "label": "Extra Person + Face Hidden",
      "description": "Another person present while student face disappears.",
    },
  ]

  # ── Tier 3 sustained rules ─────────────────────────────────
  # Rule must be continuously active for at least this many seconds.
  TIER3_SUSTAINED_SEC: Dict[str, float] = {
    "face_missing":           10.0,
    "face_not_facing_camera": 12.0,
    "multiple_persons":        8.0,
    "phone_detected":          6.0,
  }

  EVENT_LOG_LIMIT: int = 300

  def __init__(self, session_start_mono: Optional[float] = None) -> None:
    self._session_start: float = (
      session_start_mono if session_start_mono is not None else monotonic()
    )
    self._activation_times: Dict[str, Deque[float]] = defaultdict(deque)
    self._rule_active_since: Dict[str, Optional[float]] = {}
    self._combo_active_since: Dict[str, Optional[float]] = {}
    self._tier3_emitted: Set[str] = set()
    self._tier2_rules: Set[str] = set()
    self._event_log: List[TierEvent] = []
    self._activation_counts: Dict[str, int] = defaultdict(int)
    self._prev_active: Set[str] = set()

  # ── Public ─────────────────────────────────────────────────

  def update(self, rule_states: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Call once per processed frame with ruleStates from RiskEngine.
    Returns a tierStatus dict ready to embed in the frame response.
    """
    now = monotonic()
    elapsed = now - self._session_start
    now_iso = datetime.now(timezone.utc).isoformat()

    currently_active: Set[str] = {
      str(r["code"]) for r in rule_states if bool(r.get("active", False))
    }
    rule_labels: Dict[str, str] = {
      str(r["code"]): str(r.get("label", r["code"])) for r in rule_states
    }

    rising_edges = currently_active - self._prev_active
    falling_edges = self._prev_active - currently_active
    new_events: List[TierEvent] = []

    # Track continuous activity per rule
    for rule in currently_active:
      if self._rule_active_since.get(rule) is None:
        self._rule_active_since[rule] = now
    for rule in falling_edges:
      self._rule_active_since[rule] = None
      self._tier3_emitted.discard(f"sustained_{rule}")

    # Rising edges → Tier 1 or Tier 2
    for rule in rising_edges:
      self._activation_counts[rule] += 1
      self._activation_times[rule].append(now)
      self._prune_window(rule, now)
      recent = len(self._activation_times[rule])
      label = rule_labels.get(rule, rule)

      if recent >= self.TIER2_MIN_TRIGGERS:
        self._tier2_rules.add(rule)
        new_events.append(TierEvent(
          tier=2, rule=rule, label=label,
          timestamp_iso=now_iso, session_elapsed_sec=elapsed,
          evidence={
            "triggersInWindow": recent,
            "windowSec": self.TIER2_WINDOW_SEC,
            "totalActivations": self._activation_counts[rule],
          },
          escalated_from=1,
        ))
      else:
        new_events.append(TierEvent(
          tier=1, rule=rule, label=label,
          timestamp_iso=now_iso, session_elapsed_sec=elapsed,
          evidence={
            "triggersInWindow": recent,
            "totalActivations": self._activation_counts[rule],
          },
        ))

    # Re-evaluate Tier 2 for already-active rules whose window filled up
    for rule in currently_active - rising_edges:
      self._prune_window(rule, now)
      if len(self._activation_times[rule]) >= self.TIER2_MIN_TRIGGERS:
        self._tier2_rules.add(rule)

    # Clean Tier 2 set for cooled-down rules
    for rule in list(self._tier2_rules):
      self._prune_window(rule, now)
      if len(self._activation_times[rule]) < self.TIER2_MIN_TRIGGERS:
        self._tier2_rules.discard(rule)

    # Tier 3: sustained single rules
    tier3_sustained: List[Dict[str, Any]] = []
    for rule, req_sec in self.TIER3_SUSTAINED_SEC.items():
      active_since = self._rule_active_since.get(rule)
      if active_since is not None:
        sustained = now - active_since
        if sustained >= req_sec:
          tier3_sustained.append({
            "rule": rule,
            "label": rule_labels.get(rule, rule),
            "sustainedSec": round(sustained, 1),
            "requiredSec": req_sec,
          })
          key = f"sustained_{rule}"
          if key not in self._tier3_emitted:
            self._tier3_emitted.add(key)
            new_events.append(TierEvent(
              tier=3, rule=rule,
              label=f"Sustained: {rule_labels.get(rule, rule)}",
              timestamp_iso=now_iso, session_elapsed_sec=elapsed,
              evidence={
                "sustainedSec": round(sustained, 1),
                "requiredSec": req_sec,
                "type": "sustained",
              },
              escalated_from=2 if rule in self._tier2_rules else 1,
            ))

    # Tier 3: dangerous combinations
    tier3_combos: List[Dict[str, str]] = []
    for combo in self.TIER3_COMBOS:
      cid: str = combo["id"]
      required: FrozenSet[str] = combo["rules"]
      if required.issubset(currently_active):
        tier3_combos.append({"id": cid, "label": str(combo["label"])})
        if self._combo_active_since.get(cid) is None:
          self._combo_active_since[cid] = now
        if cid not in self._tier3_emitted:
          self._tier3_emitted.add(cid)
          new_events.append(TierEvent(
            tier=3, rule=cid, label=str(combo["label"]),
            timestamp_iso=now_iso, session_elapsed_sec=elapsed,
            evidence={
              "rules": sorted(required),
              "description": str(combo["description"]),
              "type": "combination",
            },
          ))
      else:
        if self._combo_active_since.get(cid) is not None:
          self._combo_active_since[cid] = None
          self._tier3_emitted.discard(cid)

    # Persist events
    self._event_log.extend(new_events)
    if len(self._event_log) > self.EVENT_LOG_LIMIT:
      self._event_log = self._event_log[-self.EVENT_LOG_LIMIT:]

    self._prev_active = currently_active

    tier3_ids: Set[str] = (
      {item["rule"] for item in tier3_sustained}
      | {c["id"] for c in tier3_combos}
    )
    tier2_active = list(self._tier2_rules & currently_active)
    tier1_active = list(currently_active - self._tier2_rules - tier3_ids)

    active_tier = 0
    if tier3_ids:
      active_tier = 3
    elif tier2_active:
      active_tier = 2
    elif tier1_active:
      active_tier = 1

    return {
      "activeTier": active_tier,
      "tier1Rules": tier1_active,
      "tier2Rules": tier2_active,
      "tier3Rules": list(tier3_ids),
      "tier3Combos": tier3_combos,
      "tier3Sustained": tier3_sustained,
      "newEvents": [e.to_dict() for e in new_events],
      "recentEventLog": [e.to_dict() for e in self._event_log[-60:]],
      "activationCounts": dict(self._activation_counts),
      "sessionElapsedSec": round(elapsed, 2),
    }

  def reset(self) -> None:
    self._session_start = monotonic()
    self._activation_times.clear()
    self._rule_active_since.clear()
    self._combo_active_since.clear()
    self._tier3_emitted.clear()
    self._tier2_rules.clear()
    self._event_log.clear()
    self._activation_counts.clear()
    self._prev_active.clear()

  def full_event_log(self) -> List[Dict[str, Any]]:
    return [e.to_dict() for e in self._event_log]

  # ── Private ────────────────────────────────────────────────

  def _prune_window(self, rule: str, now: float) -> None:
    cutoff = now - self.TIER2_WINDOW_SEC
    q = self._activation_times[rule]
    while q and q[0] < cutoff:
      q.popleft()

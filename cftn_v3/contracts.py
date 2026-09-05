from __future__ import annotations

from dataclasses import dataclass, field
from .config import TOWERS


@dataclass(frozen=True)
class Call:
    tower: str
    round: int
    request: str
    depends_on: tuple[str, ...] = ()


@dataclass
class ExecutionPlan:
    calls: list[Call] = field(default_factory=list)
    confidence: float = 1.0

    def validate(self, active, *, threshold=0.95):
        if not threshold <= self.confidence <= 1:
            raise ValueError("routing confidence below threshold")
        seen = {}
        counts = {}
        for call in sorted(self.calls, key=lambda c: c.round):
            if call.tower not in active or call.tower not in TOWERS:
                raise ValueError("inactive or unknown tower")
            if call.tower in seen or not 0 <= call.round < 4 or not call.request:
                raise ValueError("duplicate tower, invalid round, or empty request")
            counts[call.round] = counts.get(call.round, 0) + 1
            if counts[call.round] > 2:
                raise ValueError("round capacity exceeded")
            for dep in call.depends_on:
                if dep not in seen or seen[dep] >= call.round:
                    raise ValueError("invalid or cyclic dependency")
            seen[call.tower] = call.round
        return self


@dataclass(frozen=True)
class Evidence:
    record_id: str
    tower: str
    verified: bool
    verifier: str


@dataclass
class UpdatePlan:
    mode: str
    targets: tuple[str, ...]
    evidence: tuple[Evidence, ...]

    def validate(self):
        if self.mode not in {"specialist", "continual", "routing", "communication", "integration"}:
            raise ValueError("unknown training mode")
        if not self.targets or len(set(self.targets)) != len(self.targets) or set(self.targets)-set(TOWERS):
            raise ValueError("invalid update targets")
        if self.mode in {"specialist", "continual"} and len(self.targets) != 1:
            raise ValueError("ordinary update changes one specialist only")
        for target in self.targets:
            if not any(e.tower == target and e.verified and e.verifier and e.record_id for e in self.evidence):
                raise ValueError("update requires independently verified evidence")
        return self

    def allows(self, name):
        if self.mode in {"specialist", "continual", "integration"}:
            if any(name.startswith(f"towers.{tower}.") for tower in self.targets):
                return True
        if self.mode == "routing":
            return name.startswith("dispatcher.")
        if self.mode in {"communication", "integration"}:
            if name.startswith("coordinator.receivers."):
                return True
            if any(name.startswith(f"bridges.{tower}.") for tower in self.targets):
                return True
        if self.mode == "integration" and name.startswith("coordinator.adapter."):
            return True
        return False

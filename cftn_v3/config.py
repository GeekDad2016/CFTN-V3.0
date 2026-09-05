from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

TOWERS = ("math", "string", "code", "formal_logic", "science", "retrieval",
          "long_context", "multilingual", "tool_use", "structured_data",
          "information_extraction", "commonsense")
LANGUAGES = ("en", "ro")
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
PROFILES = {"tiny": (2, 32, 4), "small": (8, 512, 8),
            "medium": (12, 768, 12), "large": (16, 1024, 16)}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclasses.dataclass
class Config:
    profile: str = "small"
    coordinator: str = MODEL_ID
    revision: str = REVISION
    languages: tuple = LANGUAGES
    towers: tuple = TOWERS
    seed: int = 719
    context: int = 4096
    long_context: int = 16384
    message_tokens: int = 8
    rounds: int = 4
    towers_per_round: int = 2
    threshold: float = 0.95
    effective_batch: int = 32
    epoch_examples: int = 4096
    replay_fraction: float = 0.25
    replay_capacity: int = 4096
    specialist_lr: float = 3e-4
    bridge_lr: float = 1e-4
    adapter_lr: float = 1e-5
    continual_lr: float = 3e-5
    max_acquisition_steps: int = 10000
    max_continual_steps: int = 200
    native_threshold: float = 0.90
    retention_drop: float = 0.01
    routing_accuracy: float = 0.99
    validity_threshold: float = 0.995
    minimum_panel: int = 1000
    memory_fraction: float = 0.80
    hf_config: dict | None = None
    active: tuple = ()

    def __post_init__(self):
        self.towers = tuple(self.towers)
        self.languages = tuple(self.languages)
        self.active = tuple(self.active)
        if self.profile not in PROFILES or self.towers != TOWERS or self.languages != LANGUAGES:
            raise ValueError("unsupported profile, tower registry, or languages")
        if set(self.active) - set(TOWERS):
            raise ValueError("unknown active capability")

    def as_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def tiny(cls):
        return cls(profile="tiny", coordinator="tiny", context=4096,
                   long_context=4096, message_tokens=2, effective_batch=2)

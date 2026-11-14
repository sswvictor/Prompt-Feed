from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.config.env import apply_env_overrides, load_project_env


@dataclass
class SentimentConfig:
    enabled: bool = True
    negative_threshold: float = -0.3
    consecutive_negative_turns: int = 2
    top_category_penalty: float = 0.5
    other_category_bonus: float = 1.1


@dataclass
class DialogueConfig:
    enable_gpt: bool = True
    model: str = "gpt-4o-mini"
    max_history_turns: int = 6
    gpt_router_fallback: bool = True
    inject_intention_block: bool = True


@dataclass
class M2RuntimeConfig:
    """Belief update + dialogue-policy knobs (S0/S1 redesign)."""

    converge_threshold: float = 0.55
    slot_evidence_power: float = 0.30
    belief_power_decay: float = 0.80
    belief_recency_alpha: float = 0.70
    prior_reinjection: float = 0.15
    top1_suggest_threshold: float = 0.40
    high_entropy_ratio: float = 0.99
    clarify_budget: int = 1
    # Hard lexical label overrides: OFF by default (eval upper-bound only).
    enable_domain_guards: bool = False
    enable_m1_voting: bool = True
    guard_confidence_cap: float = 0.85
    vote_recency_decay: float = 0.75
    # Soft evidence quality: dampen confidence on cross-turn category flips.
    enable_conflict_dampen: bool = False
    conflict_dampen_factor: float = 0.65


@dataclass
class PipelineConfig:
    extractor_mode: str = "pipeline"
    enable_m3: bool = False
    user_id: str | None = None
    memory_backend: str = "inmemory"
    sqlite_path: str = "data/memory/profiles.db"
    config_path: str = "config.yaml"
    hypothesis_max_size: int = 120
    converge_threshold: float = 0.55
    m3_weight_when_ambiguous: float = 0.0
    m2: M2RuntimeConfig = field(default_factory=M2RuntimeConfig)
    dialogue: DialogueConfig = field(default_factory=DialogueConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)

    def resolve_paths(self, root: Path | None = None) -> PipelineConfig:
        base = root or Path(__file__).resolve().parents[2]
        if not Path(self.config_path).is_absolute():
            object.__setattr__(self, "config_path", str(base / self.config_path))
        if not Path(self.sqlite_path).is_absolute():
            object.__setattr__(self, "sqlite_path", str(base / self.sqlite_path))
        return self


def load_pipeline_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    path = Path(config_path) if config_path else root / "config.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def pipeline_config_from_yaml(config_path: Optional[str] = None) -> PipelineConfig:
    load_project_env()
    root = Path(__file__).resolve().parents[2]
    path = Path(config_path) if config_path else root / "config.yaml"
    raw = load_pipeline_config(str(path))
    pipe = raw.get("pipeline", {})
    m3 = raw.get("m3", {})
    m2 = raw.get("m2", {})
    dlg = raw.get("dialogue", {})
    sent = raw.get("sentiment", {})
    dialogue = DialogueConfig(
        enable_gpt=bool(dlg.get("enable_gpt", True)),
        model=str(dlg.get("model", "gpt-4o-mini")),
        max_history_turns=int(dlg.get("max_history_turns", 6)),
        gpt_router_fallback=bool(dlg.get("gpt_router_fallback", True)),
        inject_intention_block=bool(dlg.get("inject_intention_block", True)),
    )
    sentiment = SentimentConfig(
        enabled=bool(sent.get("enabled", True)),
        negative_threshold=float(sent.get("negative_threshold", -0.3)),
        consecutive_negative_turns=int(sent.get("consecutive_negative_turns", 2)),
        top_category_penalty=float(sent.get("top_category_penalty", 0.5)),
        other_category_bonus=float(sent.get("other_category_bonus", 1.1)),
    )
    m2_runtime = M2RuntimeConfig(
        converge_threshold=float(m2.get("converge_threshold", 0.55)),
        slot_evidence_power=float(m2.get("slot_evidence_power", 0.30)),
        belief_power_decay=float(m2.get("belief_power_decay", 0.80)),
        belief_recency_alpha=float(m2.get("belief_recency_alpha", 0.70)),
        prior_reinjection=float(m2.get("prior_reinjection", 0.15)),
        top1_suggest_threshold=float(m2.get("top1_suggest_threshold", 0.40)),
        high_entropy_ratio=float(m2.get("high_entropy_ratio", 0.99)),
        clarify_budget=int(m2.get("clarify_budget", 1)),
        enable_domain_guards=bool(m2.get("enable_domain_guards", False)),
        enable_m1_voting=bool(m2.get("enable_m1_voting", True)),
        guard_confidence_cap=float(m2.get("guard_confidence_cap", 0.85)),
        vote_recency_decay=float(m2.get("vote_recency_decay", 0.75)),
        enable_conflict_dampen=bool(m2.get("enable_conflict_dampen", False)),
        conflict_dampen_factor=float(m2.get("conflict_dampen_factor", 0.65)),
    )
    cfg = PipelineConfig(
        extractor_mode=str(pipe.get("extractor_mode", "pipeline")),
        memory_backend=str(pipe.get("memory_backend", "inmemory")),
        sqlite_path=str(pipe.get("sqlite_path", "data/memory/profiles.db")),
        config_path=str(path),
        hypothesis_max_size=int(pipe.get("hypothesis_max_size", 120)),
        converge_threshold=m2_runtime.converge_threshold,
        m3_weight_when_ambiguous=float(m3.get("weight_when_ambiguous", 0.0)),
        m2=m2_runtime,
        dialogue=dialogue,
        sentiment=sentiment,
    )
    return apply_env_overrides(cfg.resolve_paths(root))

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.m2_reasoning.belief import normalize, uniform_prior
from src.m2_reasoning.hypothesis import GoalHypothesis
from src.m3_memory.profile import AggregatedUserProfile


def load_m3_config(config_path: Optional[str] = None) -> Dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    path = Path(config_path) if config_path else root / "config.yaml"
    if not path.exists():
        return {
            "decay_half_life_days": 90,
            "smoothing_alpha": 0.1,
            "weight_when_ambiguous": 0.0,
            "cold_start_uniform_weight": 1.0,
        }
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    m3 = raw.get("m3", {})
    return {
        "decay_half_life_days": float(m3.get("decay_half_life_days", 90)),
        "smoothing_alpha": float(m3.get("smoothing_alpha", 0.1)),
        "weight_when_ambiguous": float(m3.get("weight_when_ambiguous", 0.0)),
        "cold_start_uniform_weight": float(m3.get("cold_start_uniform_weight", 1.0)),
    }


def normalize_with_smoothing(
    counts: Dict[str, int],
    decay: float,
    alpha: float = 0.1,
) -> Dict[str, float]:
    if not counts:
        return {}
    keys = list(counts.keys())
    total = sum(counts.values()) * decay + alpha * len(keys)
    if total <= 0:
        uniform = 1.0 / len(keys)
        return {k: uniform for k in keys}
    return {k: (counts[k] * decay + alpha) / total for k in keys}


def _decay_factor(last_updated: float, half_life_days: float) -> float:
    days_since = max(0.0, (time.time() - last_updated) / 86400.0)
    return 0.5 ** (days_since / half_life_days) if half_life_days > 0 else 1.0


def compute_prior(
    profile: AggregatedUserProfile,
    hypotheses: List[GoalHypothesis],
    config_path: Optional[str] = None,
) -> Dict[str, float]:
    """Map aggregate statistics to a normalized prior over hypothesis IDs."""
    cfg = load_m3_config(config_path)
    alpha = float(cfg["smoothing_alpha"])
    half_life = float(cfg["decay_half_life_days"])

    if not hypotheses:
        return {}

    uniform = uniform_prior(hypotheses)
    if profile.total_sessions <= 0 and not any(
        profile.category_counts.values()
        or profile.temporal_counts.values()
        or profile.price_tier_counts.values()
    ):
        return uniform

    decay = _decay_factor(profile.last_updated, half_life)
    cat_probs = normalize_with_smoothing(profile.category_counts, decay, alpha)
    temp_probs = normalize_with_smoothing(profile.temporal_counts, decay, alpha)
    price_probs = normalize_with_smoothing(profile.price_tier_counts, decay, alpha)

    n_cats = len({h.service_category for h in hypotheses}) or 1
    n_temps = len({h.temporal_slot for h in hypotheses}) or 1
    n_prices = len({h.price_tier for h in hypotheses}) or 1

    prior: Dict[str, float] = {}
    for h in hypotheses:
        cat_p = cat_probs.get(h.service_category, 1.0 / n_cats) if cat_probs else 1.0 / n_cats
        temp_p = temp_probs.get(h.temporal_slot, 1.0 / n_temps) if temp_probs else 1.0 / n_temps
        price_p = price_probs.get(h.price_tier, 1.0 / n_prices) if price_probs else 1.0 / n_prices
        prior[h.hypothesis_id] = cat_p * temp_p * price_p

    return normalize(prior)


def blend_priors(
    uniform: Dict[str, float],
    history: Dict[str, float],
    history_weight: float,
) -> Dict[str, float]:
    """Convex blend: (1-w)*uniform + w*history."""
    w = min(1.0, max(0.0, history_weight))
    blended = {
        h_id: (1.0 - w) * uniform.get(h_id, 0.0) + w * history.get(h_id, 0.0)
        for h_id in uniform
    }
    return normalize(blended)

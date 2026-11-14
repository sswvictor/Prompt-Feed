from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class M1Sample:
    text: str
    service_category: str
    temporal_preference: str | None
    urgency_level: str
    location_context: str | None
    sentiment_tags: List[str]


BIO_LABELS = [
    "O",
    "B-TEMPORAL",
    "I-TEMPORAL",
    "B-URGENCY",
    "I-URGENCY",
    "B-LOCATION",
    "I-LOCATION",
]

TEMPORAL_LEXICON = {
    # OLD:
    # "asap": ["asap", "immediately", "now"],
    # NEW: include surface variants used in data
    "asap": ["asap", "immediately", "immediate", "now", "rightaway", "right", "away"],
    "weekend": ["weekend", "saturday", "sunday"],
    "morning": ["morning"],
    "afternoon": ["afternoon"],
    "evening": ["evening", "tonight"],
    "tomorrow": ["tomorrow"],
}

LOCATION_LEXICON = {
    "nearby": ["nearby", "near", "nearme", "close"],
    "city_center": ["city", "center", "downtown", "office"],
}

URGENCY_LEXICON = {
    # OLD:
    # "high": ["urgent", "asap", "immediately", "emergency", "now"],
    # NEW: cover common morphology/adverbs in user text
    "high": ["urgent", "urgently", "asap", "immediately", "immediate", "emergency", "now"],
    "medium": [],
    "low": ["not", "later", "whenever"],
}


def simple_tokenize(text: str) -> List[str]:
    cleaned = text.lower().replace(",", " ").replace(".", " ").replace("?", " ")
    cleaned = cleaned.replace("!", " ").replace(";", " ").replace(":", " ").replace("-", " ")
    return [t for t in cleaned.split() if t]


def _apply_span(labels: List[str], start_idx: int, span_len: int, b_label: str, i_label: str) -> None:
    if start_idx < 0 or start_idx >= len(labels):
        return
    labels[start_idx] = b_label
    for i in range(1, span_len):
        if start_idx + i < len(labels):
            labels[start_idx + i] = i_label


def build_bio_labels(sample: M1Sample) -> Tuple[List[str], List[str]]:
    tokens = simple_tokenize(sample.text)
    labels = ["O"] * len(tokens)

    # temporal
    if sample.temporal_preference:
        # OLD:
        # keys = TEMPORAL_LEXICON.get(sample.temporal_preference, [sample.temporal_preference])
        # NEW: normalize canonical value first (e.g., ASAP -> asap)
        canonical_temporal = str(sample.temporal_preference).lower()
        keys = TEMPORAL_LEXICON.get(canonical_temporal, [canonical_temporal])
        for idx, tok in enumerate(tokens):
            if tok in keys:
                _apply_span(labels, idx, 1, "B-TEMPORAL", "I-TEMPORAL")
                break

    # urgency
    canonical_urgency = str(sample.urgency_level).lower()
    for idx, tok in enumerate(tokens):
        # OLD:
        # if tok in URGENCY_LEXICON.get(sample.urgency_level, []):
        if tok in URGENCY_LEXICON.get(canonical_urgency, []):
            _apply_span(labels, idx, 1, "B-URGENCY", "I-URGENCY")
            break

    # location
    if sample.location_context:
        canonical_location = str(sample.location_context).lower()
        keys = LOCATION_LEXICON.get(canonical_location, [canonical_location])
        for idx, tok in enumerate(tokens):
            if tok in keys:
                _apply_span(labels, idx, 1, "B-LOCATION", "I-LOCATION")
                break

    return tokens, labels


def load_samples(path: str) -> List[M1Sample]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out: List[M1Sample] = []
    for r in rows:
        out.append(
            M1Sample(
                text=r["text"],
                service_category=r["service_category"],
                temporal_preference=r.get("temporal_preference"),
                urgency_level=r.get("urgency_level", "medium"),
                location_context=r.get("location_context"),
                sentiment_tags=r.get("sentiment_tags", []),
            )
        )
    return out


def split_samples(
    samples: List[M1Sample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[M1Sample], List[M1Sample], List[M1Sample]]:
    if not samples:
        return [], [], []
    xs = list(samples)
    rnd = random.Random(seed)
    rnd.shuffle(xs)

    n = len(xs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = xs[:n_train]
    val = xs[n_train : n_train + n_val]
    test = xs[n_train + n_val :]
    return train, val, test


def category_distribution(samples: List[M1Sample]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for s in samples:
        dist[s.service_category] = dist.get(s.service_category, 0) + 1
    return dist


def write_split_manifest(
    train: List[M1Sample],
    val: List[M1Sample],
    test: List[M1Sample],
    path: str,
    seed: int = 42,
) -> None:
    manifest = {
        "seed": seed,
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "train_distribution": category_distribution(train),
        "val_distribution": category_distribution(val),
        "test_distribution": category_distribution(test),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

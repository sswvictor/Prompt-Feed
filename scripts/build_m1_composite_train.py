from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.m1_nlu.dataset import (  # noqa: E402
    LOCATION_LEXICON,
    M1Sample,
    TEMPORAL_LEXICON,
    URGENCY_LEXICON,
    category_distribution,
    simple_tokenize,
    split_samples,
    write_split_manifest,
)

IN_DOMAIN_PATH = ROOT / "data" / "annotations" / "m1_train.json"
OUT_PATH = ROOT / "data" / "annotations" / "m1_train_composite.json"
MANIFEST_PATH = ROOT / "data" / "annotations" / "m1_split_manifest.json"
ATIS_BENCHMARK_PATH = ROOT / "data" / "benchmarks" / "atis_intent_mapped.json"

SNIPS_HF_DATASET = "snips_built_in_intents"
ATIS_HF_DATASET = "tuetschek/atis"
HASD_HF_DATASET = "andreaceto/hasd"
SNIPS_GITHUB_URL = (
    "https://raw.githubusercontent.com/snipsco/nlu-benchmark/master/"
    "2017-06-custom-intent-engines/BookRestaurant/train_BookRestaurant_full.json"
)
SGD_GITHUB_API = (
    "https://api.github.com/repos/google-research-datasets/"
    "dstc8-schema-guided-dialogue/contents"
)
SGD_GITHUB_RAW = (
    "https://raw.githubusercontent.com/google-research-datasets/"
    "dstc8-schema-guided-dialogue/master"
)
SGD_CACHE_DIR = ROOT / "data" / "external" / "sgd"
LUCID_DEFAULT_PATH = (
    ROOT / "data" / "external" / "lucid" / "ml-lucid-datagen-main" / "lucid_v1.0" / "LUCID_data.json"
)

TRAINING_CATEGORIES = frozenset({"fitness", "wellness", "beauty", "medical", "dining", "travel"})
ATIS_TRAVEL_INTENTS = frozenset(
    {
        "flight",
        "airfare",
        "airline",
        "airport",
        "city",
        "ground_service",
        "ground_fare",
        "flight_no",
        "flight_time",
        "aircraft",
        "arrival_time",
        "departure_time",
        "distance",
        "meal",
        "capacity",
        "restriction",
        "abbreviation",
        "quantity",
        "cheapest",
    }
)
HASD_MEDICAL_INTENTS = frozenset({"schedule", "cancel", "reschedule", "query_avail"})
SNIPS_INTENT_MAP = {"BookRestaurant": "dining"}
LUCID_INTENT_MAP = {
    "book_massage": "wellness",
    "book_spa_appointment": "wellness",
    "find_massages": "wellness",
    "book_swimming_lesson": "fitness",
    "book_triathlon": "fitness",
    "create_workout_regime": "fitness",
    "log_exercise": "fitness",
    "find_exercises": "fitness",
    "find_workouts": "fitness",
    "find_triathlons": "fitness",
}
SOURCE_PRIORITY = {"in_domain": 0, "augment": 0, "lucid": 1, "snips": 2, "atis": 3, "hasd": 4, "sgd": 5}
_LUCID_EXPRESSION_RE = re.compile(r"^([a-z_]+)\(")

SGD_SKIP_SERVICE_PREFIXES = (
    "Events_",
    "Music_",
    "Media_",
    "Movies_",
    "Weather_",
    "Banks_",
    "Calendar_",
    "Messaging_",
    "Homes_",
    "Alarm_",
    "Payment_",
)
SGD_DINING_PREFIXES = ("Restaurants_",)
SGD_TRAVEL_PREFIXES = (
    "Hotels_",
    "Flights_",
    "Trains_",
    "Buses_",
    "RentalCars_",
    "Travel_",
    "RideSharing_",
)
SGD_SERVICE_UTTERANCE_HINTS: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("salon", "hair", "haircut", "barber", "manicure", "nail", "beauty", "facial"), "beauty"),
    (("spa", "massage", "wellness", "relaxation", "sauna"), "wellness"),
    (("gym", "fitness", "workout", "trainer", "yoga", "pilates"), "fitness"),
    (
        (
            "doctor",
            "dentist",
            "physician",
            "clinic",
            "hospital",
            "medical",
            "practitioner",
            "therapist",
            "counsel",
            "health",
        ),
        "medical",
    ),
)

FILLER_EXACT = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "no",
        "nope",
        "ok",
        "okay",
        "sure",
        "correct",
        "6pm",
        "6 pm",
        "7pm",
        "7 pm",
        "8pm",
        "8 pm",
    }
)
_TIME_ONLY_RE = re.compile(r"^(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)?(?:\s+please)?$", re.I)


@dataclass
class CompositeRow:
    text: str
    service_category: str
    source: str
    source_intent: str
    temporal_preference: str | None = None
    urgency_level: str = "medium"
    location_context: str | None = None
    sentiment_tags: List[str] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "service_category": self.service_category,
            "source": self.source,
            "source_intent": self.source_intent,
            "temporal_preference": self.temporal_preference,
            "urgency_level": self.urgency_level,
            "location_context": self.location_context,
            "sentiment_tags": self.sentiment_tags or [],
        }


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def is_filler(text: str) -> bool:
    norm = normalize_text(text)
    if norm in FILLER_EXACT:
        return True
    if _TIME_ONLY_RE.match(norm):
        return True
    tokens = simple_tokenize(text)
    if len(tokens) < 3:
        return True
    time_words = {"at", "for", "around", "about", "on"}
    if all(tok.isdigit() or tok in {"am", "pm"} or tok in time_words for tok in tokens):
        return True
    return False


def infer_slots_from_text(text: str) -> Tuple[str | None, str, str | None]:
    tokens = simple_tokenize(text)
    temporal: str | None = None
    urgency = "medium"
    location: str | None = None

    for canon, keys in TEMPORAL_LEXICON.items():
        if any(tok in keys for tok in tokens):
            temporal = "ASAP" if canon == "asap" else canon
            break

    for level, keys in URGENCY_LEXICON.items():
        if keys and any(tok in keys for tok in tokens):
            urgency = level
            break

    for loc, keys in LOCATION_LEXICON.items():
        if any(tok in keys for tok in tokens):
            location = loc
            break

    return temporal, urgency, location


def _with_inferred_slots(row: CompositeRow) -> CompositeRow:
    if row.source == "in_domain":
        return row
    temporal, urgency, location = infer_slots_from_text(row.text)
    return CompositeRow(
        text=row.text,
        service_category=row.service_category,
        source=row.source,
        source_intent=row.source_intent,
        temporal_preference=temporal,
        urgency_level=urgency,
        location_context=location,
        sentiment_tags=[],
    )


def load_augmented_rows(
    seed_path: Path,
    *,
    fitness_target: int = 200,
    wellness_target: int = 200,
    dining_colloquial_target: int = 80,
    seed: int = 42,
) -> List[CompositeRow]:
    from scripts.augment_dining_colloquial import generate_dining_colloquial_rows
    from scripts.augment_fitness_wellness import generate_augmented_rows

    fitness_wellness = generate_augmented_rows(
        seed_path,
        fitness_target=fitness_target,
        wellness_target=wellness_target,
        seed=seed,
    )
    dining = generate_dining_colloquial_rows(target=dining_colloquial_target, seed=seed)
    return fitness_wellness + dining


def load_in_domain(path: Path) -> List[CompositeRow]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: List[CompositeRow] = []
    for row in rows:
        label = row["service_category"]
        if label not in TRAINING_CATEGORIES:
            continue
        out.append(
            CompositeRow(
                text=row["text"].strip(),
                service_category=label,
                source="in_domain",
                source_intent=label,
                temporal_preference=row.get("temporal_preference"),
                urgency_level=row.get("urgency_level", "medium"),
                location_context=row.get("location_context"),
                sentiment_tags=row.get("sentiment_tags", []),
            )
        )
    return out


def _snips_text_from_parts(sample: Dict[str, Any]) -> str:
    if "text" in sample and isinstance(sample["text"], str):
        return sample["text"].strip()
    if "data" in sample:
        return "".join(part.get("text", "") for part in sample["data"]).strip()
    return ""


def _iter_snips_hf() -> Iterable[Tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset(SNIPS_HF_DATASET, split="train")
    cols = set(ds.column_names)
    text_col = "text" if "text" in cols else "utterance" if "utterance" in cols else None
    intent_col = "intent" if "intent" in cols else "label" if "label" in cols else None
    if not text_col or not intent_col:
        raise ValueError(f"unexpected SNIPS columns: {ds.column_names}")

    for row in ds:
        intent = str(row[intent_col])
        if intent not in SNIPS_INTENT_MAP:
            continue
        text = str(row[text_col]).strip()
        if text:
            yield text, intent


def _iter_snips_github() -> Iterable[Tuple[str, str]]:
    with urllib.request.urlopen(SNIPS_GITHUB_URL, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    samples = payload.get("BookRestaurant", payload)
    if not isinstance(samples, list):
        raise ValueError("unexpected SNIPS GitHub payload")
    for sample in samples:
        text = _snips_text_from_parts(sample)
        if text:
            yield text, "BookRestaurant"


def load_snips_rows() -> Tuple[List[CompositeRow], str]:
    source_note = ""
    pairs: List[Tuple[str, str]] = []
    try:
        pairs = list(_iter_snips_hf())
        source_note = f"hf:{SNIPS_HF_DATASET}"
    except Exception as exc:
        source_note = f"hf_failed:{type(exc).__name__};fallback=github"
        pairs = list(_iter_snips_github())

    rows: List[CompositeRow] = []
    for text, intent in pairs:
        label = SNIPS_INTENT_MAP[intent]
        rows.append(
            CompositeRow(
                text=text,
                service_category=label,
                source="snips",
                source_intent=intent,
            )
        )
    return rows, source_note


def _normalize_atis_intent(intent: str) -> str:
    raw = intent.strip().lower()
    if raw.startswith("atis_"):
        raw = raw[5:]
    return raw.replace(" ", "_")


def _is_atis_travel_intent(intent: str) -> bool:
    src = _normalize_atis_intent(intent)
    if not src or src == "in_domain":
        return False
    if src in ATIS_TRAVEL_INTENTS:
        return True
    parts = src.split("+")
    return all(part in ATIS_TRAVEL_INTENTS for part in parts if part)


def _atis_row(text: str, intent: str) -> Optional[CompositeRow]:
    src_intent = _normalize_atis_intent(intent)
    if not _is_atis_travel_intent(intent):
        return None
    return CompositeRow(
        text=text.strip(),
        service_category="travel",
        source="atis",
        source_intent=src_intent,
    )


def _iter_atis_hf() -> Iterable[Tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset(ATIS_HF_DATASET, split="train")
    cols = set(ds.column_names)
    text_col = next((c for c in ("text", "sentence", "utterance") if c in cols), None)
    intent_col = next((c for c in ("intent", "label", "category") if c in cols), None)
    if not text_col or not intent_col:
        raise ValueError(f"unexpected ATIS columns: {ds.column_names}")
    for row in ds:
        yield str(row[text_col]).strip(), str(row[intent_col])


def _iter_atis_vendored(path: Path) -> Iterable[Tuple[str, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        yield row["text"].strip(), str(row.get("source_intent", "unknown"))


def _build_atis_benchmark(vendored: List[Dict[str, Any]], hf_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    benchmark_rows: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for row in vendored:
        text = row["text"].strip()
        norm = normalize_text(text)
        if norm in seen:
            continue
        seen.add(norm)
        src_intent = _normalize_atis_intent(str(row.get("source_intent", "")))
        if src_intent == "in_domain":
            benchmark_rows.append(
                {
                    "text": text,
                    "source_intent": "in_domain",
                    "service_category": row["service_category"],
                }
            )
        elif _is_atis_travel_intent(src_intent):
            benchmark_rows.append(
                {"text": text, "source_intent": src_intent, "service_category": "travel"}
            )
        else:
            benchmark_rows.append(
                {"text": text, "source_intent": src_intent, "service_category": "other"}
            )

    for text, intent in hf_pairs:
        src_intent = _normalize_atis_intent(intent)
        if src_intent == "in_domain" or not _is_atis_travel_intent(intent):
            continue
        norm = normalize_text(text)
        if norm in seen:
            continue
        seen.add(norm)
        benchmark_rows.append(
            {"text": text.strip(), "source_intent": src_intent, "service_category": "travel"}
        )

    return benchmark_rows


def load_atis_rows(benchmark_path: Path) -> Tuple[List[CompositeRow], str, List[Dict[str, Any]]]:
    vendored = (
        json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.exists() else []
    )
    pairs: List[Tuple[str, str]] = []
    source_note = ""
    try:
        pairs = list(_iter_atis_hf())
        source_note = f"hf:{ATIS_HF_DATASET}"
    except Exception as exc:
        source_note = f"hf_failed:{type(exc).__name__};fallback=vendored:{benchmark_path.name}"
        pairs = list(_iter_atis_vendored(benchmark_path))

    rows: List[CompositeRow] = []
    for text, intent in pairs:
        row = _atis_row(text, intent)
        if row is not None:
            rows.append(row)

    benchmark_rows = _build_atis_benchmark(vendored, pairs if source_note.startswith("hf:") else [])
    return rows, source_note, benchmark_rows



def load_hasd_rows() -> Tuple[List[CompositeRow], str]:
    from datasets import load_dataset

    ds = load_dataset(HASD_HF_DATASET, split="train")
    intent_names = ds.features["intent"].names
    rows: List[CompositeRow] = []
    for row in ds:
        intent_idx = row["intent"]
        intent = intent_names[intent_idx] if isinstance(intent_idx, int) else str(intent_idx)
        if intent not in HASD_MEDICAL_INTENTS:
            continue
        text = str(row["text"]).strip()
        if not text:
            continue
        rows.append(
            CompositeRow(
                text=text,
                service_category="medical",
                source="hasd",
                source_intent=intent,
            )
        )
    return rows, f"hf:{HASD_HF_DATASET}"


def _sgd_category_from_service(service: str, active_intent: str, utterance: str) -> Optional[str]:
    if any(service.startswith(prefix) for prefix in SGD_SKIP_SERVICE_PREFIXES):
        return None
    if any(service.startswith(prefix) for prefix in SGD_DINING_PREFIXES):
        if active_intent in {"FindRestaurants", "ReserveRestaurant"} or active_intent.endswith("Restaurants"):
            return "dining"
        return None
    if any(service.startswith(prefix) for prefix in SGD_TRAVEL_PREFIXES):
        if active_intent and active_intent != "NONE":
            return "travel"
        return None
    if service.startswith("Services_"):
        text = utterance.lower()
        for keywords, category in SGD_SERVICE_UTTERANCE_HINTS:
            if any(kw in text for kw in keywords):
                return category
        if active_intent in {"FindProvider", "BookAppointment", "ReserveAppointment"}:
            return "medical"
    return None


def _github_json(url: str, timeout: int = 60, retries: int = 4) -> Any:
    import time

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ethical-intention-booking"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise last_err  # type: ignore[misc]


def _list_sgd_dialogue_files(split: str) -> List[str]:
    cache_meta = SGD_CACHE_DIR / f"{split}_files.json"
    if cache_meta.exists():
        return json.loads(cache_meta.read_text(encoding="utf-8"))
    payload = _github_json(f"{SGD_GITHUB_API}/{split}")
    files = sorted(x["name"] for x in payload if x["name"].startswith("dialogues_"))
    SGD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_meta.write_text(json.dumps(files), encoding="utf-8")
    return files


def _load_sgd_dialogue_file(split: str, filename: str) -> List[Dict[str, Any]]:
    local_path = SGD_CACHE_DIR / split / filename
    if local_path.exists():
        return json.loads(local_path.read_text(encoding="utf-8"))
    url = f"{SGD_GITHUB_RAW}/{split}/{filename}"
    dialogues = _github_json(url)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(json.dumps(dialogues), encoding="utf-8")
    return dialogues


def load_sgd_rows(splits: Iterable[str] = ("train", "dev", "test")) -> Tuple[List[CompositeRow], str]:
    rows: List[CompositeRow] = []
    files_loaded = 0
    for split in splits:
        for filename in _list_sgd_dialogue_files(split):
            dialogues = _load_sgd_dialogue_file(split, filename)
            files_loaded += 1
            for dialogue in dialogues:
                for turn in dialogue.get("turns", []):
                    if turn.get("speaker") != "USER":
                        continue
                    utterance = str(turn.get("utterance", "")).strip()
                    if not utterance:
                        continue
                    for frame in turn.get("frames", []):
                        service = str(frame.get("service", ""))
                        active_intent = str(frame.get("state", {}).get("active_intent", "") or "")
                        if not service or active_intent == "NONE":
                            continue
                        category = _sgd_category_from_service(service, active_intent, utterance)
                        if category is None:
                            continue
                        rows.append(
                            CompositeRow(
                                text=utterance,
                                service_category=category,
                                source="sgd",
                                source_intent=f"{service}:{active_intent}",
                            )
                        )
                        break
    return rows, f"github:dstc8-schema-guided-dialogue files={files_loaded}"


def _parse_lucid_intent(expression: str) -> Optional[str]:
    match = _LUCID_EXPRESSION_RE.match(expression.strip())
    return match.group(1) if match else None


def load_lucid_rows(path: Path) -> Tuple[List[CompositeRow], str]:
    if not path.exists():
        return [], f"missing:{path}"

    dialogues = json.loads(path.read_text(encoding="utf-8"))
    rows: List[CompositeRow] = []
    for dialogue in dialogues:
        turns = dialogue.get("turns", [])
        for idx, turn in enumerate(turns):
            if turn.get("author") != "User":
                continue
            text = str(turn.get("query", "")).strip()
            if not text:
                continue
            source_intent: Optional[str] = None
            for follow in turns[idx + 1 :]:
                if follow.get("author") != "System":
                    continue
                expression = str(follow.get("expression", "")).strip()
                if not expression:
                    continue
                source_intent = _parse_lucid_intent(expression)
                break
            if not source_intent or source_intent not in LUCID_INTENT_MAP:
                continue
            rows.append(
                CompositeRow(
                    text=text,
                    service_category=LUCID_INTENT_MAP[source_intent],
                    source="lucid",
                    source_intent=source_intent,
                )
            )
    return rows, f"local:{path.relative_to(ROOT)}"


def count_by_source_category(rows: List[CompositeRow]) -> Dict[str, Dict[str, int]]:
    table: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        table[row.service_category][row.source] += 1
    return {cat: dict(sources) for cat, sources in sorted(table.items())}


def filter_rows(rows: List[CompositeRow]) -> List[CompositeRow]:
    kept: List[CompositeRow] = []
    for row in rows:
        if is_filler(row.text):
            continue
        kept.append(_with_inferred_slots(row))
    return kept


def dedupe_rows(rows: List[CompositeRow]) -> List[CompositeRow]:
    best: Dict[str, CompositeRow] = {}
    for row in rows:
        key = normalize_text(row.text)
        if not key:
            continue
        existing = best.get(key)
        if existing is None or SOURCE_PRIORITY[row.source] < SOURCE_PRIORITY[existing.source]:
            best[key] = row
    return list(best.values())


def balance_rows(
    rows: List[CompositeRow],
    per_class_cap: int,
    seed: int,
    strategy: str = "cap",
) -> List[CompositeRow]:
    by_class: Dict[str, List[CompositeRow]] = defaultdict(list)
    for row in rows:
        by_class[row.service_category].append(row)

    effective_cap = per_class_cap
    if strategy == "equal" and by_class:
        effective_cap = min(min(len(items) for items in by_class.values()), per_class_cap)

    rnd = random.Random(seed)
    balanced: List[CompositeRow] = []
    for category in sorted(by_class):
        items = by_class[category]
        in_domain = [r for r in items if r.source == "in_domain"]
        external = [r for r in items if r.source != "in_domain"]
        external.sort(key=lambda r: SOURCE_PRIORITY.get(r.source, 99))
        groups: Dict[int, List[CompositeRow]] = defaultdict(list)
        for row in external:
            groups[SOURCE_PRIORITY.get(row.source, 99)].append(row)
        ordered_external: List[CompositeRow] = []
        for prio in sorted(groups):
            bucket = groups[prio]
            rnd.shuffle(bucket)
            ordered_external.extend(bucket)

        remaining = max(0, effective_cap - len(in_domain))
        balanced.extend(in_domain + ordered_external[:remaining])
    return balanced


def rows_to_samples(rows: List[CompositeRow]) -> List[M1Sample]:
    return [
        M1Sample(
            text=row.text,
            service_category=row.service_category,
            temporal_preference=row.temporal_preference,
            urgency_level=row.urgency_level,
            location_context=row.location_context,
            sentiment_tags=row.sentiment_tags or [],
        )
        for row in rows
    ]


def build_composite(
    in_domain_path: Path,
    out_path: Path,
    manifest_path: Path,
    atis_benchmark_path: Path,
    per_class_cap: int,
    seed: int,
    balance_strategy: str = "cap",
    include_sgd: bool = True,
    include_hasd: bool = True,
    include_lucid: bool = True,
    include_augment: bool = True,
    augment_fitness_target: int = 200,
    augment_wellness_target: int = 200,
    augment_dining_colloquial_target: int = 80,
    lucid_path: Path = LUCID_DEFAULT_PATH,
) -> Dict[str, Any]:
    in_domain = load_in_domain(in_domain_path)
    augmented: List[CompositeRow] = []
    augment_note = "skipped"
    if include_augment:
        augmented = load_augmented_rows(
            in_domain_path,
            fitness_target=augment_fitness_target,
            wellness_target=augment_wellness_target,
            dining_colloquial_target=augment_dining_colloquial_target,
            seed=seed,
        )
        augment_note = (
            f"fitness={augment_fitness_target},wellness={augment_wellness_target},"
            f"dining_colloquial={augment_dining_colloquial_target}"
        )
    snips, snips_note = load_snips_rows()
    atis, atis_note, benchmark_rows = load_atis_rows(atis_benchmark_path)
    hasd: List[CompositeRow] = []
    hasd_note = "skipped"
    sgd: List[CompositeRow] = []
    sgd_note = "skipped"
    lucid: List[CompositeRow] = []
    lucid_note = "skipped"
    if include_hasd:
        hasd, hasd_note = load_hasd_rows()
    if include_sgd:
        sgd, sgd_note = load_sgd_rows()
    if include_lucid:
        lucid, lucid_note = load_lucid_rows(lucid_path)

    raw_merged = in_domain + augmented + lucid + snips + atis + hasd + sgd
    merged = filter_rows(raw_merged)
    deduped = dedupe_rows(merged)
    balanced = balance_rows(
        deduped,
        per_class_cap=per_class_cap,
        seed=seed,
        strategy=balance_strategy,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([row.to_dict() for row in balanced], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    atis_benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    atis_benchmark_path.write_text(
        json.dumps(benchmark_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    samples = rows_to_samples(balanced)
    train, val, test = split_samples(samples, seed=seed)
    write_split_manifest(train, val, test, str(manifest_path), seed=seed)

    raw_by_source = count_by_source_category(filter_rows(raw_merged))
    balanced_by_source = count_by_source_category(balanced)

    def _rel(path: Path) -> str:
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(ROOT.resolve()))
        except ValueError:
            return str(resolved)

    return {
        "output": _rel(out_path),
        "total": len(balanced),
        "per_class": category_distribution(samples),
        "per_class_by_source_raw": raw_by_source,
        "per_class_by_source_balanced": balanced_by_source,
        "balance_strategy": balance_strategy,
        "per_class_cap": per_class_cap,
        "sources": {
            "in_domain": len(in_domain),
            "augment_raw": len(augmented),
            "augment_note": augment_note,
            "lucid_raw": len(lucid),
            "snips_raw": len(snips),
            "atis_raw": len(atis),
            "hasd_raw": len(hasd),
            "sgd_raw": len(sgd),
            "lucid_note": lucid_note,
            "snips_note": snips_note,
            "atis_note": atis_note,
            "hasd_note": hasd_note,
            "sgd_note": sgd_note,
            "after_filter_dedupe": len(deduped),
        },
        "split": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "atis_benchmark_updated": _rel(atis_benchmark_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build M1 composite training corpus")
    parser.add_argument("--in-domain", default=str(IN_DOMAIN_PATH))
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--atis-benchmark", default=str(ATIS_BENCHMARK_PATH))
    parser.add_argument("--per-class-cap", type=int, default=400)
    parser.add_argument(
        "--balance-strategy",
        choices=["cap", "equal"],
        default="cap",
        help="cap: per-class ceiling; equal: min class count across categories",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-sgd", action="store_true")
    parser.add_argument("--skip-hasd", action="store_true")
    parser.add_argument("--skip-lucid", action="store_true")
    parser.add_argument("--skip-augment", action="store_true")
    parser.add_argument("--augment-fitness-target", type=int, default=120)
    parser.add_argument("--augment-wellness-target", type=int, default=60)
    parser.add_argument("--augment-dining-colloquial-target", type=int, default=80)
    parser.add_argument("--lucid-path", default=str(LUCID_DEFAULT_PATH))
    args = parser.parse_args()

    report = build_composite(
        in_domain_path=Path(args.in_domain),
        out_path=Path(args.out),
        manifest_path=Path(args.manifest),
        atis_benchmark_path=Path(args.atis_benchmark),
        per_class_cap=args.per_class_cap,
        seed=args.seed,
        balance_strategy=args.balance_strategy,
        include_sgd=not args.skip_sgd,
        include_hasd=not args.skip_hasd,
        include_lucid=not args.skip_lucid,
        include_augment=not args.skip_augment,
        augment_fitness_target=args.augment_fitness_target,
        augment_wellness_target=args.augment_wellness_target,
        augment_dining_colloquial_target=args.augment_dining_colloquial_target,
        lucid_path=Path(args.lucid_path),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

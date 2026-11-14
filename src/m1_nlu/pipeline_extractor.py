from __future__ import annotations

import json
from pathlib import Path

from src.m1_nlu.model import JointNLUConfig, JointNLUModel
from src.m1_nlu.types import StructuredTaskVariable, normalize_stv


class PipelineExtractor:
    # Learns from train artifact when available; otherwise uses heuristics fallback.
    def __init__(self, model_path: str = "artifacts/m1/model.json") -> None:
        self.model: JointNLUModel | None = None
        p = Path(model_path)
        if p.exists():
            payload = json.loads(p.read_text(encoding="utf-8"))
            cfg = payload.get("config", {})
            model = JointNLUModel(
                JointNLUConfig(
                    num_intents=int(cfg.get("num_intents", 1)),
                    num_slot_labels=int(cfg.get("num_slot_labels", 16)),
                    backbone=str(cfg.get("backbone", "bert-base-uncased")),
                )
            )
            model.is_trained = True
            model.label_priors = payload.get("label_priors", {})
            model.intent_log_priors = payload.get("intent_log_priors", {})
            model.intent_log_likelihoods = payload.get("intent_log_likelihoods", {})
            model.default_log_prob = payload.get("default_log_prob", {})
            model.slot_token_tag = payload.get("slot_token_tag", {})
            self.model = model

    def _fallback_extract(self, user_input: str) -> StructuredTaskVariable:
        text = user_input.lower()
        category = "other"
        if "gym" in text or "workout" in text or "fitness" in text:
            category = "fitness"
        elif "spa" in text or "relax" in text or "massage" in text:
            category = "wellness"
        elif "beauty" in text or "skin" in text:
            category = "beauty"
        elif "clinic" in text or "doctor" in text or "medical" in text:
            category = "medical"
        elif (
            "restaurant" in text
            or "dining" in text
            or "dinner" in text
            or "lunch" in text
            or "table for" in text
        ):
            category = "dining"
        elif (
            "flight" in text
            or "hotel" in text
            or "travel" in text
            or "airport" in text
            or "train" in text
        ):
            category = "travel"

        category_conf = 0.8 if category != "other" else 0.35
        urgency = "high" if ("asap" in text or "urgent" in text) else "medium"
        temporal = "weekend" if "weekend" in text else None
        location = "nearby" if ("near me" in text or "nearby" in text) else None
        tags = []
        if "affordable" in text or "cheap" in text:
            tags.append("affordable")
        if "quick" in text:
            tags.append("quick")
        if "relax" in text:
            tags.append("relaxing")

        stv = StructuredTaskVariable(
            service_category=category,
            category_confidence=category_conf,
            temporal_preference=temporal,
            urgency_level=urgency,
            time_window_days=1 if urgency == "high" else 7,
            location_context=location,
            sentiment_tags=tags,
            extraction_confidence=0.75 if category != "other" else 0.3,
            raw_input=user_input,
            clarification_needed=False,
        )
        return normalize_stv(stv)

    def extract(self, user_input: str) -> StructuredTaskVariable:
        if not self.model or not self.model.is_trained:
            return self._fallback_extract(user_input)

        label, conf = self.model.predict_intent(user_input)
        text = user_input.lower()
        tokens, tags = self.model.predict_slot_bio(user_input)
        temporal = None
        urgency = "medium"
        location = None
        for tok, tag in zip(tokens, tags):
            if tag.startswith("B-TEMPORAL"):
                if tok in {"asap", "immediately", "now", "urgent", "urgently"}:
                    temporal = "ASAP"
                elif tok in {"weekend", "saturday", "sunday"}:
                    temporal = "weekend"
                elif tok in {"morning", "afternoon", "evening", "tomorrow", "tonight"}:
                    temporal = "evening" if tok == "tonight" else tok
            elif tag.startswith("B-URGENCY"):
                urgency = "high" if tok in {"asap", "urgent", "urgently", "immediately", "emergency", "now"} else urgency
            elif tag.startswith("B-LOCATION"):
                if tok in {"downtown", "city", "center", "office"}:
                    location = "city_center"
                elif tok in {"nearby", "near", "close"}:
                    location = "nearby"

        if temporal is None:
            for key in ["weekend", "morning", "afternoon", "evening", "tomorrow", "asap"]:
                if key in text:
                    temporal = "ASAP" if key == "asap" else key
                    break
        if urgency == "medium" and any(k in text for k in ["asap", "urgent", "urgently", "immediately", "emergency", "now"]):
            urgency = "high"
        if location is None:
            location = "nearby" if any(k in text for k in ["nearby", "near me"]) else ("city_center" if any(k in text for k in ["downtown", "city center", "office"]) else None)
        tags = [x for x in ["relaxing", "affordable", "quick", "intense", "premium", "gentle"] if x in text]
        stv = StructuredTaskVariable(
            service_category=label,
            category_confidence=conf,
            temporal_preference=temporal,
            urgency_level=urgency,
            time_window_days=1 if urgency == "high" else 7,
            location_context=location,
            sentiment_tags=tags,
            extraction_confidence=conf,
            raw_input=user_input,
            clarification_needed=False,
        )
        return normalize_stv(stv)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.m1_nlu.dataset import BIO_LABELS, M1Sample, build_bio_labels, simple_tokenize


@dataclass
class JointNLUConfig:
    num_intents: int
    num_slot_labels: int
    backbone: str = "bert-base-uncased"


class JointNLUModel:
    """
    Lightweight placeholder for Joint intent + slot architecture.
    Keeps interface stable for later HF/PyTorch implementation.
    """

    def __init__(self, config: JointNLUConfig) -> None:
        self.config = config
        self.is_trained = False
        self.label_priors: Dict[str, float] = {}
        self.intent_vocab: Dict[str, int] = {}
        self.intent_log_priors: Dict[str, float] = {}
        self.intent_log_likelihoods: Dict[str, Dict[str, float]] = {}
        self.default_log_prob: Dict[str, float] = {}
        self.slot_token_tag: Dict[str, str] = {}

    def fit(self, samples: List[M1Sample]) -> None:
        intent_counts: Dict[str, int] = {}
        token_counts_by_intent: Dict[str, Dict[str, int]] = {}
        total_tokens_by_intent: Dict[str, int] = {}

        token_tag_counts: Dict[str, Dict[str, int]] = {}
        for s in samples:
            intent = s.service_category
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            if intent not in token_counts_by_intent:
                token_counts_by_intent[intent] = {}
                total_tokens_by_intent[intent] = 0
            tokens = simple_tokenize(s.text)
            for t in tokens:
                self.intent_vocab[t] = self.intent_vocab.get(t, 0) + 1
                token_counts_by_intent[intent][t] = token_counts_by_intent[intent].get(t, 0) + 1
                total_tokens_by_intent[intent] += 1

            bio_tokens, bio_labels = build_bio_labels(s)
            for tok, tag in zip(bio_tokens, bio_labels):
                if tok not in token_tag_counts:
                    token_tag_counts[tok] = {}
                token_tag_counts[tok][tag] = token_tag_counts[tok].get(tag, 0) + 1

        total_samples = sum(intent_counts.values()) or 1
        self.label_priors = {k: v / total_samples for k, v in intent_counts.items()}
        self.intent_log_priors = {k: math.log(v) for k, v in self.label_priors.items()}

        vocab_size = max(1, len(self.intent_vocab))
        for intent, counts in token_counts_by_intent.items():
            total = total_tokens_by_intent[intent]
            denom = total + vocab_size
            self.default_log_prob[intent] = math.log(1.0 / denom)
            self.intent_log_likelihoods[intent] = {}
            for token, c in counts.items():
                self.intent_log_likelihoods[intent][token] = math.log((c + 1.0) / denom)

        for token, tag_count in token_tag_counts.items():
            self.slot_token_tag[token] = sorted(tag_count.items(), key=lambda kv: kv[1], reverse=True)[0][0]
        self.is_trained = True

    def predict_intent(self, text: str) -> tuple[str, float]:
        if not self.is_trained or not self.label_priors:
            return "other", 0.3
        tokens = simple_tokenize(text)
        scores: Dict[str, float] = {}
        for intent, log_prior in self.intent_log_priors.items():
            s = log_prior
            default = self.default_log_prob[intent]
            ll = self.intent_log_likelihoods.get(intent, {})
            for t in tokens:
                s += ll.get(t, default)
            scores[intent] = s

        top_label, top_score = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[0]
        exp_scores = {k: math.exp(v - top_score) for k, v in scores.items()}
        z = sum(exp_scores.values()) or 1.0
        conf = exp_scores[top_label] / z
        return top_label, conf

    def predict_slot_bio(self, text: str) -> Tuple[List[str], List[str]]:
        tokens = simple_tokenize(text)
        tags = [self.slot_token_tag.get(t, "O") for t in tokens]
        # normalize illegal I- starts
        fixed: List[str] = []
        for i, tag in enumerate(tags):
            if tag.startswith("I-") and (i == 0 or fixed[-1] == "O"):
                tag = "B-" + tag[2:]
            if tag not in BIO_LABELS:
                tag = "O"
            fixed.append(tag)
        return tokens, fixed

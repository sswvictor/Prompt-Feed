from __future__ import annotations

from typing import Protocol

from src.m1_nlu.types import StructuredTaskVariable


class NLUExtractor(Protocol):
    def extract(self, user_input: str) -> StructuredTaskVariable:
        ...

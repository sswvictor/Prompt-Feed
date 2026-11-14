from __future__ import annotations

from src.m1_nlu.extractor import NLUExtractor
from src.m1_nlu.llm_extractor import LLMExtractor
from src.m1_nlu.pipeline_extractor import PipelineExtractor


def get_extractor(mode: str) -> NLUExtractor:
    if mode == "pipeline":
        return PipelineExtractor()
    if mode == "llm":
        return LLMExtractor()
    raise ValueError(f"Unsupported extractor mode: {mode}")

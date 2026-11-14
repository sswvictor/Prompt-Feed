from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from src.pipeline.config import PipelineConfig

_ENV_LOADED = False
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_project_env(root: Path | None = None) -> bool:
    """Load .env once. Variables already in the environment are not overwritten."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return True

    base = root or project_root()
    env_path = base / ".env"
    if env_path.is_file():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path, override=False)
        except ImportError:
            pass

    _ENV_LOADED = True
    return env_path.is_file()


def env_str(key: str) -> Optional[str]:
    value = os.getenv(key)
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def env_bool(key: str) -> Optional[bool]:
    raw = env_str(key)
    if raw is None:
        return None
    lower = raw.lower()
    if lower in _TRUTHY:
        return True
    if lower in _FALSY:
        return False
    return None


def env_int(key: str) -> Optional[int]:
    raw = env_str(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def apply_env_overrides(cfg: PipelineConfig) -> PipelineConfig:
    load_project_env()

    dialogue = cfg.dialogue
    dlg_updates: Dict[str, Any] = {}
    if (v := env_bool("DIALOGUE_ENABLE_GPT")) is not None:
        dlg_updates["enable_gpt"] = v
    if (v := env_str("DIALOGUE_MODEL")) is not None:
        dlg_updates["model"] = v
    if (v := env_bool("DIALOGUE_INJECT_INTENTION_BLOCK")) is not None:
        dlg_updates["inject_intention_block"] = v
    if (v := env_int("DIALOGUE_MAX_HISTORY_TURNS")) is not None:
        dlg_updates["max_history_turns"] = v
    if (v := env_bool("DIALOGUE_GPT_ROUTER_FALLBACK")) is not None:
        dlg_updates["gpt_router_fallback"] = v
    if dlg_updates:
        dialogue = replace(dialogue, **dlg_updates)

    pipe_updates: Dict[str, Any] = {"dialogue": dialogue}
    if (v := env_str("PIPELINE_EXTRACTOR_MODE")) is not None:
        pipe_updates["extractor_mode"] = v
    if (v := env_int("PIPELINE_HYPOTHESIS_MAX_SIZE")) is not None:
        pipe_updates["hypothesis_max_size"] = v
    if (v := env_str("M3_MEMORY_BACKEND")) is not None:
        pipe_updates["memory_backend"] = v
    if (v := env_str("M3_SQLITE_PATH")) is not None:
        pipe_updates["sqlite_path"] = v

    return replace(cfg, **pipe_updates)


def gpt_connection_status(model: str = "gpt-4o-mini", enable_gpt: bool = True) -> Dict[str, Any]:
    load_project_env()
    api_key = env_str("OPENAI_API_KEY") or ""
    if not enable_gpt:
        return {
            "connected": False,
            "mode": "mock",
            "model": model,
            "reason": "DIALOGUE_ENABLE_GPT is false",
        }
    if not api_key:
        return {
            "connected": False,
            "mode": "mock",
            "model": model,
            "reason": "OPENAI_API_KEY not set",
        }
    return {
        "connected": True,
        "mode": "gpt",
        "model": model,
        "reason": None,
    }

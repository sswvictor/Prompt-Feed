from __future__ import annotations

from pathlib import Path

from src.config.env import project_root


def rel_to_root(path: str | Path, root: Path | None = None) -> str:
    """Path relative to the repo root, with forward slashes."""
    base = (root or project_root()).resolve()
    raw = Path(path)
    candidate = raw if raw.is_absolute() else (base / raw)
    try:
        return candidate.resolve().relative_to(base).as_posix()
    except ValueError:
        text = str(path).replace("\\", "/")
        lower = text.lower()
        marker = "/code/"
        if marker in lower:
            return text[lower.rfind(marker) + len(marker) :]
        return raw.as_posix().lstrip("/")

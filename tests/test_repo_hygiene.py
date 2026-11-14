"""Fail if tracked text files contain live API keys or home-directory paths."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from src.config.env import project_root

_TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".yml",
    ".yaml",
    ".toml",
    ".txt",
    ".json",
    ".example",
    ".cff",
    ".gitignore",
}
_LIVE_KEY = re.compile(r"sk-(?:proj|live)-[A-Za-z0-9_-]{16,}")
_ABS_WIN = re.compile(r"(?i)[A-Z]:\\Users\\")
_ABS_UNIX = re.compile(r"(?<![A-Za-z])/Users/[A-Za-z0-9._-]+/")


def _tracked_text_files(root: Path) -> list[Path]:
    raw = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=root,
        stderr=subprocess.DEVNULL,
    )
    names = [n for n in raw.decode("utf-8").split("\0") if n]
    out: list[Path] = []
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        if path.suffix.lower() in _TEXT_SUFFIXES or path.name in {
            "LICENSE",
            "NOTICE.md",
            "README.md",
            ".gitignore",
            ".env.example",
        }:
            out.append(path)
    return out


def test_gitignore_excludes_dotenv() -> None:
    text = (project_root() / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in text
    assert "!.env.example" in text


def test_tracked_text_has_no_live_keys_or_home_paths() -> None:
    root = project_root()
    files = _tracked_text_files(root)
    assert files, "git ls-files returned no text files"
    hits: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if _LIVE_KEY.search(text) or _ABS_WIN.search(text) or _ABS_UNIX.search(text):
            hits.append(str(path.relative_to(root).as_posix()))
    assert hits == []

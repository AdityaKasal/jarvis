"""Config loading: config.yaml for behaviour, .env for secrets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_root"] = ROOT
    return cfg


HINTS = {
    "ANTHROPIC_API_KEY": "Create a key at console.anthropic.com/settings/keys "
                         "and put it in .env.",
    "ELEVENLABS_API_KEY": "Create a key at elevenlabs.io/app/settings/api-keys "
                          "and put it in .env.",
}


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        hint = HINTS.get(name, "Copy .env.example to .env and fill it in.")
        raise RuntimeError(f"{name} is not set. {hint}")
    return val

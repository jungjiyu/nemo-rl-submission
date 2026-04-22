"""Shared helpers for reward modules: locate .env, load creds, build headers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ENV_LOADED = False


def _find_dotenv() -> Optional[Path]:
    """Walk upward from this file to find a `.env` at the repo root.

    The repo root is identified as the first ancestor that contains either
    `pyproject.toml` or a `.env` file. Returns the `.env` Path if present,
    else None.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
        if (parent / "pyproject.toml").exists():
            # Repo root reached; no .env present.
            return None
    return None


def load_dotenv_once() -> None:
    """Load key=value pairs from the repo-root `.env` into os.environ exactly once."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    env_file = _find_dotenv()
    if env_file is None:
        return
    try:
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    except Exception as e:
        print(f"[hackaton._env] failed to parse {env_file}: {e!r}")


def openrouter_headers(app_suffix: str) -> tuple[str, dict[str, str]]:
    """Return (base_url, headers) for an OpenRouter chat/completions call.

    `app_suffix` is appended to the default X-Title ("vuvlm-render-reward") and
    should identify the caller (e.g. "judge", "solve") for ledgering.
    """
    load_dotenv_once()
    base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    app_name_env = os.environ.get("OPENROUTER_APP_NAME", "vuvlm-render-reward")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", ""),
        "X-Title": f"{app_name_env}/{app_suffix}",
    }
    return base, headers

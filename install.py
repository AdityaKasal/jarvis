#!/usr/bin/env python3
"""One-shot setup for Jarvis. Run once, then use the launcher it makes.

    python3 install.py

Creates the virtualenv, installs the dependencies, downloads the speech model,
writes your .env, and generates a double-clickable launcher for whichever
platform you are on. Safe to run again - it skips whatever is already done.

Deliberately dependency-free and stdlib-only: this is the script that runs
*before* anything is installed, so it cannot import anything that isn't already
in a stock Python.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"

# ctranslate2, which faster-whisper is built on, publishes wheels only up to
# 3.13. On 3.14 pip falls back to building from source and fails with a wall of
# compiler errors that says nothing about the real problem.
MIN_PYTHON = (3, 9)
MAX_PYTHON = (3, 13)


def say(message: str = "") -> None:
    print(message, flush=True)


def step(n: int, total: int, message: str) -> None:
    say(f"\n[{n}/{total}] {message}")


def venv_python(venv: Path) -> Path:
    if platform.system() == "Windows":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def suitable(version: tuple[int, int]) -> bool:
    return MIN_PYTHON <= version <= MAX_PYTHON


def find_interpreter() -> tuple[Path, tuple[int, int]] | tuple[None, None]:
    """An interpreter that dependencies actually have wheels for.

    Prefers the one running this script, then looks for named versions on
    PATH, newest first - a machine on 3.14 very often still has 3.12 or 3.13
    installed alongside it.
    """
    here = sys.version_info[:2]
    if suitable(here):
        return Path(sys.executable), here

    say(f"  This is Python {here[0]}.{here[1]}; looking for a supported one...")
    for minor in range(MAX_PYTHON[1], MIN_PYTHON[1] - 1, -1):
        name = f"python3.{minor}"
        found = shutil.which(name)
        if not found:
            continue
        try:
            out = subprocess.run(
                [found, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
                capture_output=True, text=True, timeout=15)
            major, got_minor = (int(x) for x in out.stdout.split())
        except Exception:
            continue
        if suitable((major, got_minor)):
            return Path(found), (major, got_minor)
    return None, None


def run(cmd: list[str], what: str) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        say(f"\n  {what} failed:\n")
        tail = (result.stderr or result.stdout).strip().splitlines()[-15:]
        for line in tail:
            say(f"    {line}")
        raise SystemExit(1)


def make_env(root: Path) -> bool:
    """Write .env, asking for the keys. Returns True if both are set.

    The keys are typed straight into a local file with owner-only permissions.
    Nothing is sent anywhere, and the file is in .gitignore.
    """
    env = root / ".env"
    existing: dict[str, str] = {}
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, _, value = line.partition("=")
                existing[key.strip()] = value.strip()

    wanted = {
        "ANTHROPIC_API_KEY": ("Claude, for the conversation",
                              "console.anthropic.com/settings/keys"),
        "ELEVENLABS_API_KEY": ("ElevenLabs, for the voice",
                               "elevenlabs.io/app/settings/api-keys"),
    }

    values: dict[str, str] = {}
    for name, (what, where) in wanted.items():
        current = existing.get(name, "")
        if current and not current.endswith("..."):
            say(f"  {name} is already set - keeping it.")
            values[name] = current
            continue
        say(f"\n  {name} - {what}")
        say(f"  Get one at {where}")
        try:
            typed = input("  Paste it here (or press Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            typed = ""
        values[name] = typed or f"{name.split('_')[0].lower()}-..."

    body = ["# Written by install.py. Never commit this file.", ""]
    for name, value in values.items():
        body.append(f"{name}={value}")
    env.write_text("\n".join(body) + "\n")
    try:
        env.chmod(0o600)   # no-op on Windows, which has no POSIX mode bits
    except OSError:
        pass

    return all(not v.endswith("...") for v in values.values())


def main() -> int:
    total = 6
    say("Setting up Jarvis.")
    say(f"  {ROOT}")

    step(1, total, "Finding a suitable Python")
    interpreter, version = find_interpreter()
    if interpreter is None:
        say(f"  No Python between {MIN_PYTHON[0]}.{MIN_PYTHON[1]} and "
            f"{MAX_PYTHON[0]}.{MAX_PYTHON[1]} was found.")
        say("  faster-whisper's engine has no wheels outside that range, and")
        say("  building it from source is not something to do by accident.")
        say("\n  Install one, then run this again:")
        say("    macOS    brew install python@3.13")
        say("    Windows  python.org/downloads (pick 3.13)")
        say("    Linux    sudo apt install python3.13-venv")
        return 1
    say(f"  Using Python {version[0]}.{version[1]} at {interpreter}")

    step(2, total, "Creating the virtualenv")
    if venv_python(VENV).exists():
        say("  Already there - skipping.")
    else:
        run([str(interpreter), "-m", "venv", str(VENV)], "Creating the virtualenv")
        say(f"  {VENV}")

    python = venv_python(VENV)

    step(3, total, "Installing dependencies (a few minutes the first time)")
    run([str(python), "-m", "pip", "install", "--upgrade", "--quiet", "pip"],
        "Upgrading pip")
    run([str(python), "-m", "pip", "install", "--quiet", "-r",
         str(ROOT / "requirements.txt")], "Installing dependencies")
    say("  Done.")

    step(4, total, "Setting up your API keys")
    complete = make_env(ROOT)

    step(5, total, "Downloading the speech model (about 500 MB, once)")
    result = subprocess.run(
        [str(python), "-c",
         "import sys; sys.path.insert(0, r'%s');"
         "from jarvis.config import load_config;"
         "from jarvis.stt import Transcriber;"
         "Transcriber(load_config()).warm_up()" % ROOT],
        capture_output=True, text=True)
    if result.returncode == 0:
        say("  Ready.")
    else:
        # Not fatal: it downloads on first use anyway, just less pleasantly.
        say("  Could not fetch it now; it will download the first time you talk.")

    step(6, total, "Making the launcher")
    run([str(python), str(ROOT / "scripts" / "make_launchers.py")],
        "Generating launchers")

    system = platform.system()
    launcher = {"Darwin": "launchers/Jarvis.app",
                "Windows": "launchers\\Jarvis.bat"}.get(
                    system, "launchers/jarvis.desktop")

    say("\n" + "-" * 58)
    say("Jarvis is installed.")
    say(f"\n  Open: {ROOT / launcher}")
    if system == "Darwin":
        say("  Double-click it, or drag it to your Dock.")
        say("  The first launch: right-click it and choose Open, so macOS lets it run.")
    elif system == "Windows":
        say("  Double-click it, or pin it to Start.")
    else:
        say("  Copy it to ~/.local/share/applications/ for a menu entry.")

    if not complete:
        say("\n  One thing first: your API keys are still placeholders.")
        say(f"  Edit {ROOT / '.env'} and put the real ones in,")
        say("  then run this installer again to check them.")
    say("-" * 58)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        say("\nStopped.")
        raise SystemExit(130)

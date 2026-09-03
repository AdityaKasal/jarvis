#!/bin/bash
# Run this to set up Jarvis:  ./install.sh
cd "$(dirname "$0")" || exit 1

PY=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
done

if [ -z "$PY" ]; then
    echo "No Python found. Try: sudo apt install python3.13-venv"
    exit 1
fi

exec "$PY" install.py

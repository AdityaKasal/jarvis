#!/bin/bash
# Double-click this to set up Jarvis.
cd "$(dirname "$0")" || exit 1

PY=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
done

if [ -z "$PY" ]; then
    echo "No Python found. Install it from python.org (pick 3.13), then run this again."
    echo
    read -r -p "Press Return to close."
    exit 1
fi

"$PY" install.py
status=$?
echo
read -r -p "Press Return to close."
exit $status

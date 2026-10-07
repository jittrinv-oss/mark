#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "$(readlink -f -- "$0")")"
[[ -x .venv/bin/python ]] || { echo 'Run: bash setup.sh first.' >&2; exit 1; }
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo 'No graphical desktop. Run from the Pi desktop Terminal (not plain SSH).' >&2; exit 1
fi
exec .venv/bin/python mark_ui.py

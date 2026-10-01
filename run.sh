#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "$(readlink -f -- "$0")")"
if [[ ! -x .venv/bin/python ]]; then echo 'Run bash setup.sh first.' >&2; exit 1; fi
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo 'No graphical desktop detected. Run from the Pi desktop terminal, not plain SSH.' >&2; exit 1
fi
exec .venv/bin/python mark_ui.py

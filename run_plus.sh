#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "$(readlink -f -- "$0")")"
[[ -x .venv/bin/python ]] || { echo 'Run: bash setup.sh first.' >&2; exit 1; }
exec .venv/bin/python mark_ui_plus.py

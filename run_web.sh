#!/usr/bin/env bash
cd -- "$(dirname -- "$(readlink -f -- "$0")")"
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
exec "$PY" web_app.py "$@"

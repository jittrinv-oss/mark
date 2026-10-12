#!/usr/bin/env bash
# เปิดโปรแกรมตรวจมาร์คแบบเต็มจอ (ใช้กับ autostart)
cd -- "$(dirname -- "$(readlink -f -- "$0")")"
[[ -x .venv/bin/python ]] || { echo 'Run: bash setup.sh first.' >&2; exit 1; }
exec .venv/bin/python mark_ui_kiosk.py "$@"

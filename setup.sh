#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "$(readlink -f -- "$0")")"
trap 'printf "ERROR: setup stopped at line %s. No PLC output was enabled.\n" "$LINENO" >&2' ERR

if [[ $(uname -s) != Linux ]]; then echo 'Run this on Raspberry Pi OS, not Windows.' >&2; exit 2; fi
if [[ $(uname -m) != aarch64 ]]; then echo 'Expected 64-bit Raspberry Pi OS (aarch64).' >&2; exit 2; fi
if [[ $(id -u) -eq 0 ]]; then echo 'Run as normal desktop user: bash setup.sh (not sudo bash setup.sh).' >&2; exit 2; fi
if ! command -v apt-get >/dev/null; then echo 'Raspberry Pi OS / Debian APT required.' >&2; exit 2; fi
if [[ ! -f mark_ui.py || ! -f glyph_check.py || ! -f mark_match.py ]]; then
  echo 'Missing mark_ui.py, glyph_check.py, or mark_match.py. Copy your existing app here first.' >&2; exit 2
fi
if [[ ! -f requirements.txt ]]; then echo 'Missing requirements.txt.' >&2; exit 2; fi

# Do not modify OS or network settings; obtain IT approval before APT uses company network.
packages=(python3-venv python3-pip python3-numpy python3-opencv python3-tk)
if [[ -f plc_out.py || -f find_do_pin.py ]]; then packages+=(python3-lgpio); fi
missing=()
for p in "${packages[@]}"; do dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -qx 'install ok installed' || missing+=("$p"); done
if ((${#missing[@]})); then
  printf 'Missing OS packages: %s\n' "${missing[*]}"
  echo 'APT may require network access. Stop here if IT has not approved access.'
  if [[ "${1:-}" == --check-only ]]; then exit 3; fi
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends "${missing[@]}"
fi
if [[ "${1:-}" == --check-only ]]; then echo 'OS packages installed.'; exit 0; fi

# APT OpenCV and Tkinter remain visible in this virtual environment.
if [[ -e .venv && ! -x .venv/bin/python ]]; then echo '.venv exists but is invalid; move it aside manually.' >&2; exit 4; fi
if [[ ! -d .venv ]]; then python3 -m venv --system-site-packages .venv; fi
if ! grep -Eq "^include-system-site-packages = true$" .venv/pyvenv.cfg; then
  echo "Existing .venv hides OS packages; rename .venv and rerun setup.sh." >&2; exit 4
fi
if ! .venv/bin/python -c 'import cv2, numpy, tkinter' >/dev/null 2>&1; then
  echo 'venv cannot import APT packages. Remove/rename .venv, then rerun setup.sh.' >&2; exit 4
fi
# --no-index prevents unexpected PyPI downloads; installed APT numpy satisfies requirements.
.venv/bin/python -m pip install --no-index --disable-pip-version-check -r requirements.txt
.venv/bin/python - <<'PY'
import cv2, numpy, tkinter
print('PASS: cv2',cv2.__version__,'numpy',numpy.__version__,'Tk',tkinter.TkVersion)
try:
    import lgpio
    print('PASS: lgpio available')
except ImportError:
    print('NOTE: lgpio absent; needed only for optional PLC scripts')
PY
.venv/bin/python -m py_compile mark_ui.py glyph_check.py mark_match.py
printf '\nSetup complete. Launch on the Pi desktop: ./run.sh\n'

#!/usr/bin/env bash
# bash setup.sh --check-only : report missing OS packages only (installs nothing)
# bash setup.sh              : install missing packages (needs IT-approved APT access)
set -Eeuo pipefail
cd -- "$(dirname -- "$(readlink -f -- "$0")")"
trap 'printf "ERROR: setup stopped at line %s. Nothing was sent to the PLC.\n" "$LINENO" >&2' ERR
[[ $(uname -s) == Linux ]]   || { echo 'Run this on Raspberry Pi OS, not Windows.' >&2; exit 2; }
[[ $(uname -m) == aarch64 ]] || { echo 'Expected 64-bit Raspberry Pi OS (aarch64).' >&2; exit 2; }
[[ $(id -u) -ne 0 ]]         || { echo 'Run as normal user: bash setup.sh (not sudo).' >&2; exit 2; }
command -v apt-get >/dev/null || { echo 'APT required.' >&2; exit 2; }
for f in mark_ui.py glyph_check.py mark_match.py pdf_master.py camera.py requirements.txt; do
  [[ -f $f ]] || { echo "Missing $f in $(pwd)" >&2; exit 2; }
done
installed() { dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -qx 'install ok installed'; }
available() { apt-cache show "$1" >/dev/null 2>&1; }
packages=(python3-venv python3-pip python3-numpy python3-opencv python3-tk poppler-utils)
if   installed python3-pymupdf || available python3-pymupdf; then packages+=(python3-pymupdf)
elif installed python3-fitz    || available python3-fitz;    then packages+=(python3-fitz); fi
[[ -f plc_out.py || -f find_do_pin.py ]] && packages+=(python3-lgpio)
missing=()
for p in "${packages[@]}"; do installed "$p" || missing+=("$p"); done
if ((${#missing[@]})); then
  printf 'Missing OS packages: %s\n' "${missing[*]}"
  if [[ "${1:-}" == --check-only ]]; then
    echo '(--check-only: nothing installed. Run "bash setup.sh" when IT-approved APT access is available.)'; exit 3
  fi
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends "${missing[@]}"
else
  echo 'All OS packages installed.'
fi
[[ "${1:-}" == --check-only ]] && exit 0
if [[ -e .venv && ! -x .venv/bin/python ]]; then echo '.venv is invalid; rename it and rerun.' >&2; exit 4; fi
[[ -d .venv ]] || python3 -m venv --system-site-packages .venv
grep -Eq '^include-system-site-packages = true$' .venv/pyvenv.cfg || { echo 'Existing .venv hides OS packages; rename .venv and rerun.' >&2; exit 4; }
.venv/bin/python -m pip install --no-index --disable-pip-version-check -r requirements.txt
.venv/bin/python -c "import cv2, numpy, tkinter; print('PASS: cv2', cv2.__version__, '| numpy', numpy.__version__)"
.venv/bin/python -c "import fitz; print('PASS: PyMuPDF')" 2>/dev/null || echo 'NOTE: no PyMuPDF -> PDF uses pdftoppm raster mode'
.venv/bin/python -m py_compile mark_ui.py glyph_check.py mark_match.py pdf_master.py camera.py
printf '\nSetup complete. Start from the Pi desktop:  bash run.sh\n'

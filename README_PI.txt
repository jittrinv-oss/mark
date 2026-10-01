MARK INSPECTION - RASPBERRY PI 5 MIGRATION

This archive contains installation helpers, NOT the full application.
Keep your tested mark_ui.py, glyph_check.py, mark_match.py and your existing
masters/ and config JSON files. Do not replace your tested algorithm with
an older example. Back up the full Windows folder first.

1. Install Raspberry Pi OS 64-bit Desktop on boot media; boot to Desktop.
2. Obtain IT approval before connecting the Pi to company network or using APT.
3. Copy the existing app folder to /home/<your-user>/mark_inspection on the Pi.
   Copy these four helper files into that same folder.
   Example: cp -a "/media/$USER/USB_LABEL/mark_inspection" ~/mark_inspection
   Use the actual USB mount point; do not copy Windows .venv or __pycache__.
4. In the Pi desktop Terminal:
     cd ~/mark_inspection
     bash setup.sh --check-only   # optional preflight (exit 3 = missing packages)
     bash setup.sh
     bash run.sh
   USB/Windows transfer can remove executable bits; bash run.sh works regardless.
5. Test with stored images before connecting camera or PLC. Existing config
   coordinates remain valid ONLY for the same master image/crop dimensions.
6. Camera: check /dev/video* and test the Camera tab on the Pi. The code must
   use cv2.VideoCapture(index) on Linux, not cv2.CAP_DSHOW (Windows only).
7. PLC: do NOT connect or enable outputs until pin mapping, HAT wiring,
   input polarity, fail-safe logic and operator changeover reset are validated
   by authorized maintenance. setup.sh does not start plc_out.py.

Requirements: numpy is installed by APT and checked by pip -r requirements.txt
in a venv that exposes APT site packages. cv2, tkinter and optional lgpio
are APT packages because pip OpenCV ARM wheels may not be available and
Tkinter is not installed via pip. No --break-system-packages used.

No network/IT access? Do not run setup.sh on the corporate network without
approval. Ask IT to provision the listed APT packages from an approved source.
A folder of .whl files is not enough to install APT-managed cv2/tkinter.

Note: setup.sh installs dependencies and validates syntax; it does not flash
an SD card, modify display overlays, set up a service, or integrate PLC I/O.

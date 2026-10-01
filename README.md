# Mark Inspection: Raspberry Pi 5 setup

This guide accompanies the **Pi 5 migration helpers** (`requirements.txt`, `setup.sh`, and `run.sh`). Put these files alongside your **existing, tested** `mark_ui.py`, `glyph_check.py`, and `mark_match.py`. The helpers install dependencies; they do **not** replace your inspection algorithm or connect its OK/NG result to the PLC.

## Before you begin

- Boot **64-bit Raspberry Pi OS Desktop** on the Pi 5. A graphical desktop is needed to see the Tkinter UI.
- Back up your current `mark_inspection` folder, including Master images and configuration JSON files. Do not copy a Windows `.venv` to the Pi.
- Obtain IT approval before connecting the Pi to a company network or using its package source. `setup.sh` can call APT to install missing packages.
- Leave the PLC disconnected during software setup and image-only testing.

Expected layout on the Pi:

```text
~/mark_inspection/
├── mark_ui.py
├── glyph_check.py
├── mark_match.py
├── requirements.txt
├── setup.sh
├── run.sh
├── masters/             # if your existing app uses it
└── *.json               # your existing model-specific configs, if any
```

If you have `plc_out.py` or `find_do_pin.py`, keep those with your existing project, but **do not run or wire the outputs during installation**.

## Install and launch

Open **Terminal on the Pi desktop**, then run:

```bash
cd ~/mark_inspection
bash setup.sh --check-only
```

`--check-only` **does not install anything**. If it prints `Missing OS packages: python3-opencv` (or other names), it is reporting what is not installed. A nonzero exit status in that case is intentional. The package name is **`python3-opencv`**, not `pyhon3-opencv`.

Once an approved APT source is available, run:

```bash
bash setup.sh
bash run.sh
```

`setup.sh` checks for the app files, installs missing OS packages via APT, creates `.venv` with access to OS-installed packages, processes `requirements.txt`, checks imports and Python syntax, and stops if a check fails. `bash run.sh` launches the UI from the Pi desktop. Use `bash run.sh` even if the file lost executable permissions during transfer from Windows or USB.

### Why `requirements.txt` does not list OpenCV

The migration helper installs `python3-opencv` (provides `cv2`) and `python3-tk` (provides Tkinter) using APT. Its `requirements.txt` checks NumPy in a virtual environment that can see those OS packages. Do not mix another `opencv-python` wheel into that same environment without first revising and testing the dependency strategy. A `wheels` folder by itself does **not** satisfy the script's APT package check.

## Test in stages

1. **Dependencies:** `bash setup.sh` finishes without an error.
2. **UI:** `bash run.sh` opens the inspection window on the Pi display.
3. **Saved images:** load a known Master and known OK/NG Test images. Confirm the displayed result and the correct model-specific config. Config coordinates are valid only for the same Master image and crop dimensions.
4. **Camera, when available:** connect it and test the live-camera tab. If the app opens a camera using a Windows-only backend, adapt that code for Linux before testing.
5. **PLC, separately:** have authorized maintenance validate the actual HAT pin mapping, electrical input polarity, changeover reset, fault behavior, and interlock before connecting outputs. **Installation success is not PLC validation.**

## Troubleshooting

### `Missing OS packages: python3-opencv`

If this came from `bash setup.sh --check-only`, it is an **inventory result, not a failed installation**. With IT-approved package access, run `bash setup.sh` without `--check-only`. If that fails, retain the complete output from `apt-get update` or `apt-get install` for diagnosis. Do not assume a downloaded wheel fixes a missing APT package.

### `No module named cv2`

Check that setup completed, then use the app's interpreter:

```bash
cd ~/mark_inspection
.venv/bin/python -c "import cv2; print(cv2.__version__)"
```

If that fails, capture the full error. Do not use `sudo pip install` or `--break-system-packages` as a workaround.

### Existing `.venv` is invalid or hides OS packages

The helper will stop instead of silently using an incompatible environment. Back up or rename `.venv`, then rerun `bash setup.sh`; do not delete an environment containing work you need to keep.

### `bash run.sh` says no graphical desktop

Run it from Terminal **on the Pi desktop**, not from a plain SSH session without graphical forwarding. `setup.sh --check-only` and dependency checks can be run in a terminal session without opening the UI.

## Scope and safety

This guide does not flash an SD card, change display overlays, configure networking, install a background service, or modify PLC logic. Do not treat a simulated OK/NG result or a successful dependency check as permission to run the production line. Keep the existing first-piece and QA approval process in place until the full system has been validated.

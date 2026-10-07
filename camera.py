#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera.py — ค้นหาและเปิดกล้อง USB (Windows และ Raspberry Pi 5)
  - list_cameras() ไล่ทุก index / ทุก backend และ "ทดลองอ่านภาพจริง" ก่อนนับว่าใช้ได้
  - บน Linux ตัดอุปกรณ์ภายในของ Pi 5 ออก และแสดงชื่อรุ่นกล้อง
  - open_camera() ลอง MJPG + ความละเอียดสูงก่อน ถ้าอ่านไม่ได้ใช้ค่าเริ่มต้นของกล้อง
"""
import glob
import sys

import cv2

IS_WIN = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")
_PI_INTERNAL = ("pispbe", "rp1-cfe", "rpivid", "hevc", "bcm2835-codec", "bcm2835-isp")


def _linux_name(idx):
    try:
        with open(f"/sys/class/video4linux/video{idx}/name", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _try_read(cap, tries=8):
    for _ in range(tries):
        ok, fr = cap.read()
        if ok and fr is not None and fr.size and fr.mean() > 1:
            return fr
    return None


def _backends():
    if IS_WIN:
        return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
    if IS_LINUX:
        return [("V4L2", cv2.CAP_V4L2)]
    return [("ANY", cv2.CAP_ANY)]


def list_cameras(max_index=8):
    found = []
    if IS_LINUX:
        idxs = sorted(int(p[10:]) for p in glob.glob("/dev/video*") if p[10:].isdigit())
    else:
        idxs = list(range(max_index))
    for idx in idxs:
        name = _linux_name(idx) if IS_LINUX else ""
        if IS_LINUX and any(k in name.lower() for k in _PI_INTERNAL):
            continue
        for bname, bid in _backends():
            cap = cv2.VideoCapture(idx, bid)
            if not cap.isOpened():
                cap.release(); continue
            fr = _try_read(cap)
            cap.release()
            if fr is None:
                continue
            h, w = fr.shape[:2]
            label = f"กล้อง {idx}"
            if name:
                label += f" — {name}"
            elif IS_WIN:
                label += " — กล้องในตัวเครื่อง?" if idx == 0 else " — USB?"
            label += f" ({w}x{h}, {bname})"
            found.append(dict(index=idx, backend=bname, backend_id=bid, name=name, size=(w, h), label=label))
            break
    return found


def open_camera(index, backend_id=None, width=1920, height=1080):
    backends = [backend_id] if backend_id is not None else [b for _, b in _backends()]
    for bid in backends:
        cap = cv2.VideoCapture(index, bid)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            try:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            except Exception:
                pass
            fr = _try_read(cap)
            if fr is not None:
                return cap, (fr.shape[1], fr.shape[0])
            cap.release()
        cap = cv2.VideoCapture(index, bid)
        if cap.isOpened():
            fr = _try_read(cap)
            if fr is not None:
                return cap, (fr.shape[1], fr.shape[0])
            cap.release()
    return None, None


if __name__ == "__main__":
    cams = list_cameras()
    print("\n".join(c["label"] for c in cams) if cams else "ไม่พบกล้องที่อ่านภาพได้")

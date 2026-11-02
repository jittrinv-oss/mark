#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera.py v2 — ค้นหาและเปิดกล้อง USB (Windows และ Raspberry Pi 5) แบบได้ 2K จริง
  - ชื่อฟังก์ชันและค่าที่คืนกลับเหมือนเดิม: list_cameras(), open_camera() -> (cap, (w, h))
  - แก้: ขอ MJPG + 2560x1440 ทั้งตอน list และตอนเปิดจริง
  - แก้: ตั้ง FOURCC ทั้งก่อนและหลังตั้งความละเอียด (ไดรเวอร์ DSHOW บางตัวต้องการแบบนี้)
  - แก้: รอให้กล้องอุ่นเครื่องนานขึ้น (ที่ 2K เฟรมแรก ๆ มักเป็นสีดำ ทำให้ระบบเดิมยอมถอยไปใช้ 640x480)
  - แก้: ถ้า 2K ไม่ได้ ให้ลด 1080p -> 720p ทีละขั้น และ "ไม่" กระโดดลงไป 640x480 เงียบ ๆ อีก
  - ทดสอบ:  python camera.py          (แสดงรายการกล้อง)
            python camera.py --test 1 (ทดสอบกล้อง index 1 ทุกความละเอียด + บันทึก snapshot)
"""
import glob
import os
import sys

IS_WIN = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")
if IS_WIN:
    # ให้ MSMF เปิดเร็วขึ้นและไม่แปลงภาพเอง
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2  # noqa: E402

# ความละเอียดที่จะลอง จากสูงไปต่ำ (ค่าแรก = เป้าหมาย 2K)
RES_CANDIDATES = [(2560, 1440), (1920, 1080), (1280, 720)]

_PI_INTERNAL = ("pispbe", "rp1-cfe", "rpivid", "hevc", "bcm2835-codec", "bcm2835-isp")


def _linux_name(idx):
    try:
        with open(f"/sys/class/video4linux/video{idx}/name", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _fourcc_str(cap):
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    s = "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))
    return s if s.strip("\x00").isprintable() and s.strip("\x00") else "?"


def _try_read(cap, tries=30, need_w=None):
    """อ่านเฟรมจนได้ภาพจริง (ไม่ดำ). ถ้า need_w ระบุ จะรอจนได้ความกว้างนั้นด้วย"""
    last = None
    for _ in range(tries):
        ok, fr = cap.read()
        if not ok or fr is None or not fr.size:
            continue
        last = fr
        if fr.mean() <= 1:          # เฟรมดำช่วงปรับแสง -> อ่านต่อ
            continue
        if need_w and fr.shape[1] != need_w:
            continue
        return fr
    # ไม่ได้ตามที่ขอ แต่ถ้ามีภาพที่สว่างพอให้คืนไป (ผู้เรียกจะเช็คขนาดเอง)
    if last is not None and last.mean() > 1:
        return last
    return None


def _apply_mode(cap, w, h, fourcc="MJPG"):
    cc = cv2.VideoWriter_fourcc(*fourcc)
    cap.set(cv2.CAP_PROP_FOURCC, cc)          # ก่อน (V4L2 / ส่วนใหญ่)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FOURCC, cc)          # หลัง (DSHOW บางรุ่น)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass


def _backends():
    if IS_WIN:
        # DSHOW รับ MJPG ความละเอียดสูงได้ดีกว่า MSMF
        return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
    if IS_LINUX:
        return [("V4L2", cv2.CAP_V4L2)]
    return [("ANY", cv2.CAP_ANY)]


def _best_mode(cap, candidates, verbose=False):
    """ลองทีละความละเอียดบน cap ที่เปิดอยู่ คืน (frame) ตัวที่ใหญ่ที่สุดที่ได้"""
    best = None
    for w, h in candidates:
        _apply_mode(cap, w, h)
        fr = _try_read(cap, tries=30, need_w=w)
        got = (fr.shape[1], fr.shape[0]) if fr is not None else None
        if verbose:
            print(f"   ขอ {w}x{h} -> ได้ {got} fourcc={_fourcc_str(cap)}")
        if fr is None:
            continue
        if best is None or fr.shape[1] > best.shape[1]:
            best = fr
        if got == (w, h):
            break                     # ได้ตามที่ขอแล้ว ไม่ต้องลดลง
    return best


def list_cameras(max_index=8, probe_hd=True):
    """probe_hd=True: ป้ายจะแสดงความละเอียดสูงสุดที่เปิดได้จริง (ไม่ใช่ 640x480 ค่าเริ่มต้น)"""
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
                cap.release()
                continue
            fr = _best_mode(cap, RES_CANDIDATES) if probe_hd else None
            if fr is None:
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
            found.append(dict(index=idx, backend=bname, backend_id=bid, name=name,
                              size=(w, h), label=label))
            break
    return found


def open_camera(index, backend_id=None, width=2560, height=1440, verbose=False):
    """เปิดกล้องที่ความละเอียดสูงสุดที่ได้จริง เริ่มจาก width x height แล้วลดทีละขั้น
    คืน (cap, (w, h)) หรือ (None, None)"""
    # ไม่ว่าโปรแกรมหลักจะส่ง 640x480 / 1920x1080 มา ก็ลองค่าที่สูงกว่าก่อนเสมอ
    cands = [(width, height)] + [r for r in RES_CANDIDATES if r[0] < width]
    if width < RES_CANDIDATES[0][0]:
        cands = RES_CANDIDATES[:]
    backends = [backend_id] if backend_id is not None else [b for _, b in _backends()]

    best_cap, best_size = None, None
    for bid in backends:
        cap = cv2.VideoCapture(index, bid)
        if not cap.isOpened():
            cap.release()
            continue
        if verbose:
            print(f"[open_camera] index={index} backend={bid}")
        fr = _best_mode(cap, cands, verbose=verbose)
        if fr is not None:
            size = (fr.shape[1], fr.shape[0])
            # ตั้งโหมดให้ตรงกับที่ได้จริงอีกครั้ง (กันไดรเวอร์ค้างโหมดสุดท้ายที่ลอง)
            _apply_mode(cap, *size)
            _try_read(cap, tries=10)
            try:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            except Exception:
                pass
            if best_size is None or size[0] > best_size[0]:
                if best_cap is not None:
                    best_cap.release()
                best_cap, best_size = cap, size
            else:
                cap.release()
            if best_size == (width, height):
                break
        else:
            cap.release()

    if best_cap is not None:
        if best_size != (width, height):
            print(f"[camera] เตือน: ขอ {width}x{height} แต่กล้องให้ได้ {best_size[0]}x{best_size[1]}")
        return best_cap, best_size

    # ทางสุดท้าย: ค่าเริ่มต้นของกล้อง (อาจเป็น 640x480) — แจ้งเตือนชัดเจน
    for bid in backends:
        cap = cv2.VideoCapture(index, bid)
        if cap.isOpened():
            fr = _try_read(cap)
            if fr is not None:
                print(f"[camera] เตือน: ใช้ค่าเริ่มต้นของกล้อง {fr.shape[1]}x{fr.shape[0]} "
                      f"(เปิด MJPG ความละเอียดสูงไม่ได้)")
                return cap, (fr.shape[1], fr.shape[0])
        cap.release()
    return None, None


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--test":
        idx = int(sys.argv[2])
        cap, size = open_camera(idx, verbose=True)
        if cap is None:
            print("เปิดกล้องไม่ได้")
            sys.exit(1)
        ok, fr = cap.read()
        print(f"ผลสุดท้าย: {size}  fourcc={_fourcc_str(cap)}  "
              f"fps={cap.get(cv2.CAP_PROP_FPS):.0f}")
        if ok:
            cv2.imwrite(f"snapshot_cam{idx}_{size[0]}x{size[1]}.jpg", fr)
            print("บันทึก snapshot แล้ว — เปิดดูว่าเส้นมาร์คบาง ๆ ชัดไหม")
        cap.release()
    else:
        cams = list_cameras()
        print("\n".join(c["label"] for c in cams) if cams else "ไม่พบกล้องที่อ่านภาพได้")

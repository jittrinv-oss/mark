#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_polarity.py v2 — ส่วนเสริมสำหรับมาร์คบน "สกรีนสี" (ตัวหนังสือขาว/เหลืองบนกระจกเขียว)
ไม่แก้ไขโค้ดเดิม: แปลงภาพเป็น "ตัวดำบนพื้นขาว" ก่อนส่งเข้าการตรวจเดิม
  - แปลงทั้ง Master และ Test / ตัดเฉพาะแผ่นสกรีนสีในภาพ Test / ขยาย Test ให้ใกล้ขนาด Master
  - ภาพตัวดำพื้นขาวแบบเดิม ไม่ถูกแตะ
"""
import numpy as np
import cv2

MODES = ("อัตโนมัติ", "ตัวเข้ม พื้นอ่อน (เดิม)", "ตัวอ่อน พื้นเข้ม/พื้นสี")


def crop_color_patch(bgr, sat_thr=45, min_frac=0.04, max_frac=0.85, pad_frac=0.04):
    if bgr is None:
        return bgr, None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    col = ((hsv[:, :, 1] > sat_thr) & (hsv[:, :, 2] > 40)).astype(np.uint8) * 255
    H, W = col.shape
    k = max(5, (min(H, W) // 25) | 1)
    col = cv2.morphologyEx(col, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    col = cv2.morphologyEx(col, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(col, 8)
    if n <= 1:
        return bgr, None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, a = (int(v) for v in st[i])
    frac = (w * h) / float(H * W)
    if frac < min_frac or frac > max_frac:
        return bgr, None
    px, py = int(w * pad_frac), int(h * pad_frac)
    x0, y0 = min(W - 1, x + px), min(H - 1, y + py)
    x1, y1 = max(x0 + 1, x + w - px), max(y0 + 1, y + h - py)
    return bgr[y0:y1, x0:x1].copy(), (x0, y0, x1 - x0, y1 - y0)


def _flatten(ch):
    ch = ch.astype(np.float32)
    k = max(15, (min(ch.shape[:2]) // 8) | 1)
    return ch - cv2.medianBlur(np.clip(ch, 0, 255).astype(np.uint8), k).astype(np.float32)


def _channels(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(bgr)
    yield "L", lab[:, :, 0]
    yield "B", b
    yield "G", g
    yield "R", r
    yield "R-G", np.clip(r.astype(np.int16) - g.astype(np.int16) + 128, 0, 255).astype(np.uint8)


def detect_light_marks(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    use = (g > 20) & (g < 250)
    if use.mean() < 0.2:
        use = np.ones(use.shape, bool)
    best_name, best_c, best_hi, best_lo = None, -1.0, 0.0, 0.0
    for name, ch in _channels(bgr):
        d = _flatten(cv2.GaussianBlur(ch, (3, 3), 0))[use]
        hi, lo = float(np.percentile(d, 98.5)), float(np.percentile(d, 1.5))
        c = max(hi, -lo)
        if c > best_c:
            best_name, best_c, best_hi, best_lo = name, c, hi, lo
    return bool(best_hi > -best_lo * 1.15), best_name, best_c


def is_color_screen(bgr, sat_thr=45, min_frac=0.35):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(((hsv[:, :, 1] > sat_thr) & (hsv[:, :, 2] > 40)).mean()) >= min_frac


def normalize_test(bgr, mode="อัตโนมัติ"):
    if bgr is None or mode == MODES[1]:
        return bgr, ""
    light, name, _ = detect_light_marks(bgr)
    if mode == MODES[0] and not light:
        return bgr, ""
    d = _flatten(cv2.GaussianBlur(dict(_channels(bgr))[name], (3, 3), 0))
    if not light:
        d = -d
    top = max(float(np.percentile(d, 99.5)), 8.0)
    out = (255 * (1 - np.clip(d / top, 0, 1))).astype(np.uint8)
    out = cv2.createCLAHE(2.0, (8, 8)).apply(out)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR), f"กลับสี (ช่อง {name})"


def clean_master(bgr, min_frac=0.00015):
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    _, ink = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= max(30, int(min_frac * g.size))
    return cv2.cvtColor(np.where(keep[lab], 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def prepare_pair(master, test, mode="อัตโนมัติ"):
    """คืน (master2, test2, ข้อความ) — master2 ขนาดเท่าเดิม (โซน IGNORE ยังใช้ได้)"""
    if mode == MODES[1]:
        return master, test, ""
    msgs = []
    t = test
    need_crop = test is not None and (
        mode == MODES[2]
        or (master is not None and is_color_screen(master, min_frac=0.25))
        or is_color_screen(test, min_frac=0.04))
    if need_crop:
        t2, box = crop_color_patch(test)
        if box is not None:
            t = t2
            msgs.append("ตัดเฉพาะแผ่นสกรีนสีของ Test")
    if t is not None and master is not None and t.shape[1] < 0.8 * master.shape[1]:
        s = min(4.0, master.shape[1] / float(t.shape[1]))
        t = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    m2, mm = normalize_test(master, mode)
    t3, tm = normalize_test(t, mode)
    if mm:
        m2 = clean_master(m2)
        msgs.append("Master " + mm)
    if tm:
        msgs.append("Test " + tm)
    if not mm and not tm:
        return master, test, ""
    return m2, t3, "  /  ".join(msgs)

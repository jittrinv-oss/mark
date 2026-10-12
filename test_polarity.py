#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_polarity.py v3 — ส่วนเสริมสำหรับมาร์คบน "สกรีนสี" (กระจกเขียว ฯลฯ)
ไม่แก้ไขโค้ดเดิม: แปลงภาพเป็น "ตัวดำบนพื้นขาว สีเทาล้วน" ก่อนส่งเข้าการตรวจเดิม

v3:
  1) ตัวหนังสือ "สีเข้ม" บนพื้นเขียว ก็ต้องแปลงด้วย
     (เดิมแปลงเฉพาะตัวสีอ่อน -> พื้นเขียวถูกโปรแกรมเดิมเข้าใจว่าเป็น "รอยปากกาสี" 77% ของภาพ
      แล้วลบทิ้ง + ขยายขอบ -> กินขอบตัวอักษร -> ขึ้นสีส้ม (จางขาดหาย) เต็มภาพ)
  2) ทำขอบคม: ตัดพื้นหลัง/noise ให้ขาวสนิท (ไม่มีแถบเทา/เส้นลายจอ) + unsharp mask
  3) ภาพตัวดำพื้นขาว ไม่มีสี (แบบเดิม) -> ไม่แตะเลย
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
    if bgr is None:
        return False
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(((hsv[:, :, 1] > sat_thr) & (hsv[:, :, 2] > 40)).mean()) >= min_frac


def _crisp(d, sharpen=True):
    """
    d = ค่าความเข้มของหมึก (บวก = หมึก) -> ภาพตัวดำพื้นขาว ขอบคม
      - ต่ำกว่าระดับ noise พื้นหลัง -> ขาวสนิท (ตัดแถบเทา/ลายจอ)
      - สูงกว่าระดับหมึกเต็ม -> ดำสนิท
      - ช่วงกลางยืดแบบชัน -> ขอบคม
    """
    d = cv2.bilateralFilter(d.astype(np.float32), 5, 12, 5)          # ลด noise แต่คงขอบ
    pos = d[d > 0]
    if pos.size < 50:
        pos = np.abs(d).ravel()
    noise = float(np.percentile(np.abs(d), 60))                     # ระดับพื้นหลัง
    full = max(float(np.percentile(pos, 97)), noise + 8.0)          # ระดับหมึกเต็ม
    lo = noise + 0.15 * (full - noise)
    hi = noise + 0.70 * (full - noise)
    ink = np.clip((d - lo) / max(hi - lo, 1.0), 0, 1)
    out = (255 * (1 - ink)).astype(np.uint8)
    if sharpen:
        blur = cv2.GaussianBlur(out, (0, 0), 1.2)
        out = cv2.addWeighted(out, 1.6, blur, -0.6, 0)
    return out


def normalize_test(bgr, mode="อัตโนมัติ", force=False):
    """
    คืน (ภาพตัวดำพื้นขาว สีเทาล้วน, ข้อความ)
    แปลงเมื่อ: ตัวหนังสือสีอ่อน  หรือ  ภาพเป็นสกรีนสี (force/ตรวจพบ)  หรือ  โหมดบังคับ
    ภาพตัวเข้ม พื้นขาว ไม่มีสี ในโหมดอัตโนมัติ -> คืนภาพเดิม
    """
    if bgr is None or mode == MODES[1]:
        return bgr, ""
    light, name, _ = detect_light_marks(bgr)
    colored = force or is_color_screen(bgr, min_frac=0.25)
    if mode == MODES[0] and not light and not colored:
        return bgr, ""
    d = _flatten(cv2.GaussianBlur(dict(_channels(bgr))[name], (3, 3), 0))
    if not light:
        d = -d
    out = _crisp(d)
    kind = "ตัวอ่อน" if light else "ตัวเข้ม"
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR), f"แปลง{kind}บนพื้นสี -> ขาวดำ (ช่อง {name})"


def clean_master(bgr, min_area=12):
    """Master ที่แปลงแล้ว -> ขาวดำ + ลบจุดเล็กมาก (ฝุ่น/ลายจอ) โดยไม่ตัดขีด/วงเล็บเล็ก ๆ"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = (g < 140).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return cv2.cvtColor(np.where(keep[lab], 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _normalize_master(master, mode, m_col):
    """
    Master ที่เป็นรูปถ่ายสกรีนสี: แปลงเฉพาะ "แผ่นสกรีน" (ไม่เอาพื้นเทา/ตัวหนังสือกลับด้านด้านหลังมาคิด)
    แล้ววางกลับในภาพขาวขนาดเดิม -> พิกัดเท่าเดิม โซน IGNORE ที่ลากไว้ยังใช้ได้
    """
    if master is None:
        return master, ""
    if m_col:
        crop, box = crop_color_patch(master, max_frac=0.95)
        if box is not None:
            # ภาพเล็ก -> ขยายก่อนแปลง (เส้นบางไม่ขาด) แล้วย่อกลับขนาดเดิม
            up = max(1.0, min(4.0, 900.0 / max(1, crop.shape[1])))
            big = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up > 1 else crop
            c2, msg = normalize_test(big, mode, force=True)
            if msg and up > 1:
                c2 = cv2.resize(c2, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)
            if msg:
                x, y, w, h = box
                out = np.full_like(master, 255)
                out[y:y + h, x:x + w] = c2
                return out, msg + " (เฉพาะแผ่นสกรีน)"
    return normalize_test(master, mode, force=m_col)


def prepare_pair(master, test, mode="อัตโนมัติ"):
    """คืน (master2, test2, ข้อความ) — master2 ขนาดเท่าเดิม (โซน IGNORE ยังใช้ได้)"""
    if mode == MODES[1]:
        return master, test, ""
    msgs = []
    t = test
    m_col = is_color_screen(master, min_frac=0.08)
    t_col = test is not None and is_color_screen(test, min_frac=0.04)
    if test is not None and (mode == MODES[2] or m_col or t_col):
        t2, box = crop_color_patch(test)
        if box is not None:
            t = t2
            msgs.append("ตัดเฉพาะแผ่นสกรีนสีของ Test")
    if t is not None and master is not None and t.shape[1] < 0.8 * master.shape[1]:
        s = min(4.0, master.shape[1] / float(t.shape[1]))
        t = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    m2, mm = _normalize_master(master, mode, m_col)
    t3, tm = normalize_test(t, mode, force=t_col or m_col)
    if mm:
        m2 = clean_master(m2)
        msgs.append("Master " + mm)
    if tm:
        msgs.append("Test " + tm)
    if not mm and not tm:
        return master, test, ""
    return m2, t3, "  /  ".join(msgs)

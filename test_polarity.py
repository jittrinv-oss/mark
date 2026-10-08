#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_polarity.py — ส่วนเสริม (add-on) สำหรับภาพ Test ที่ "ตัวหนังสือสีอ่อน บนพื้นเข้ม/พื้นสี"
เช่น ตัวหนังสือขาว-เหลืองบนกระจกสีเขียว

ไม่แก้ไขโค้ดเดิม: แปลงภาพ Test ให้เป็น "ตัวหนังสือดำบนพื้นขาว" แบบเดียวกับ Master
แล้วส่งเข้า inspect_glyph() ตัวเดิมตามปกติ

ขั้นตอน
  1) เลือกช่องสีที่ตัวหนังสือตัดกับพื้นมากที่สุด (B, G, R, L, หรือ ความต่างจากสีพื้น)
  2) ปรับพื้นหลังให้เรียบ (ลบแสงไม่สม่ำเสมอ)
  3) ตรวจขั้วสี: ถ้าตัวหนังสือสว่างกว่าพื้น -> กลับสี
  4) ยืดความเข้มให้พื้น = ขาว, ตัวหนังสือ = ดำ
ถ้าภาพเป็น "ตัวเข้มบนพื้นอ่อน" อยู่แล้ว (โหมดอัตโนมัติ) -> คืนภาพเดิมไม่แตะเลย
"""
import numpy as np
import cv2

MODES = ("อัตโนมัติ", "ตัวเข้ม พื้นอ่อน (เดิม)", "ตัวอ่อน พื้นเข้ม/พื้นสี")


def _flatten(ch):
    """ลบพื้นหลัง: คืน (ภาพ - พื้นหลัง) เป็น float  (+ = สว่างกว่าพื้น, - = เข้มกว่าพื้น)"""
    ch = ch.astype(np.float32)
    k = max(15, (min(ch.shape[:2]) // 8) | 1)
    bg = cv2.medianBlur(np.clip(ch, 0, 255).astype(np.uint8), k).astype(np.float32)
    return ch - bg


def _channels(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(bgr)
    yield "L", lab[:, :, 0]
    yield "B", b
    yield "G", g
    yield "R", r
    # ช่อง "ความเหลือง/ความขาว เทียบกับเขียว": ตัวหนังสือขาว-เหลืองบนเขียว R สูงขึ้นชัดกว่า G
    yield "R-G", np.clip(r.astype(np.int16) - g.astype(np.int16) + 128, 0, 255).astype(np.uint8)


def detect_light_marks(bgr):
    """
    คืน (is_light_on_dark, best_channel_name, contrast)
    วัดจากส่วนที่ "เบี่ยงจากพื้นหลังมาก ๆ" (หาง 1.5% บน/ล่าง) — ตัวหนังสืออยู่ฝั่งไหนมากกว่า
    """
    best = None
    for name, ch in _channels(bgr):
        d = _flatten(cv2.GaussianBlur(ch, (3, 3), 0))
        hi, lo = np.percentile(d, 98.5), np.percentile(d, 1.5)
        contrast = max(hi, -lo)
        if best is None or contrast > best[2]:
            best = (name, d, contrast, hi, lo)
    name, d, contrast, hi, lo = best
    light = hi > -lo * 1.15          # ฝั่งสว่างเด่นกว่าฝั่งเข้มชัดเจน = ตัวหนังสือสีอ่อน
    return bool(light), name, float(contrast)


def normalize_test(bgr, mode="อัตโนมัติ"):
    """
    คืน (ภาพ BGR ตัวดำพื้นขาว, ข้อความอธิบาย)
    mode = "อัตโนมัติ" | "ตัวเข้ม พื้นอ่อน (เดิม)" | "ตัวอ่อน พื้นเข้ม/พื้นสี"
    """
    if bgr is None or mode == MODES[1]:
        return bgr, "ใช้ภาพ Test ตามเดิม"
    light, name, contrast = detect_light_marks(bgr)
    if mode == MODES[0] and not light:
        return bgr, "ภาพ Test เป็นตัวเข้มบนพื้นอ่อน — ใช้ตามเดิม"

    # ใช้ช่องที่ตัดกันมากที่สุด แล้วทำให้ "ตัวหนังสือ = ค่าบวก"
    ch = dict(_channels(bgr))[name]
    d = _flatten(cv2.GaussianBlur(ch, (3, 3), 0))
    if not light:                     # โหมดบังคับ แต่ภาพจริงเป็นตัวเข้ม -> กลับขั้วให้ถูก
        d = -d
    # ยืด: พื้น (0) -> 255 ขาว, ตัวหนังสือเต็ม (percentile 99) -> 0 ดำ
    top = max(float(np.percentile(d, 99.5)), 8.0)
    ink = np.clip(d / top, 0, 1)
    out = (255 * (1 - ink)).astype(np.uint8)
    out = cv2.createCLAHE(2.0, (8, 8)).apply(out)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR), f"กลับสีภาพ Test (ตัวสีอ่อน, ช่อง {name})"

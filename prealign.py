#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prealign.py — ส่วนเสริม: จัดภาพ Test ให้ตรงกับ Master "ก่อน" ส่งเข้าการตรวจเดิม + ตรวจคุณภาพภาพ

แก้ 3 อาการ:
  1) align RESIZE-ONLY(!)  — ภาพกลับด้าน (mirror) / กลับหัว / หมุน / คนละสี (เขียว vs ขาว)
     การจัดตำแหน่งเดิม (ORB) หาจุดเด่นไม่เจอ -> ยืดภาพเฉย ๆ -> NG ทั้งภาพ
  2) เส้นพุ่งเป็นรัศมี — การจัดตำแหน่งเดิมได้ผลผิด (มุมมองบิดสุดขั้ว)
  3) ผลตรวจเบี้ยว / เบลอ

วิธี:
  - แปลงทั้ง 2 ภาพเป็น "แผนที่หมึก" ที่ไม่สนสี/ขั้วสี (ตัวขาวบนเขียว = ตัวดำบนขาว)
  - หาจุดเด่น SIFT แล้วจับคู่ ลองทั้ง "ปกติ" และ "กลับด้าน"
  - ใช้แบบ หมุน+ย่อขยาย+เลื่อน เท่านั้น (ไม่บิดมุมมอง -> ไม่มีเส้นพุ่ง)
  - บิดภาพ Test ให้ตรง Master (+ เผื่อขอบ) แล้วค่อยส่งให้โปรแกรมเดิมตรวจ
  - ด่านคุณภาพ: จับคู่ไม่ได้ / ภาพเบลอ / มาร์คเล็กเกิน -> แจ้ง "ถ่ายใหม่" แทนการฟ้อง NG มั่ว
"""
import math

import numpy as np
import cv2

MARGIN = 0.10            # เผื่อขอบรอบภาพ 10% (ตัวริมขอบไม่โดนตัด)
MIN_INLIERS = 15         # จุดอ้างอิงขั้นต่ำ ต่ำกว่านี้ = จับคู่ไม่ได้
MIN_MARK_PX = 450        # ความกว้างมาร์คในภาพ Test ขั้นต่ำ (พิกเซล) ต่ำกว่านี้ = ความละเอียดไม่พอ
BLUR_RATIO = 0.25        # ความคมของ Test เทียบ Master ต่ำกว่านี้ = เบลอ


# ---------------------------------------------------------------- แผนที่หมึก (ไม่สนสี)
def ink_map(bgr, long_side=900):
    """
    คืน (ภาพเทา: หมึก=ดำ พื้น=ขาว, scale ที่ย่อ/ขยาย)
    ใช้ "ความต่างจากพื้นหลัง" แบบค่าสัมบูรณ์ -> ตัวขาวบนเขียว และตัวดำบนขาว ได้ผลเหมือนกัน
    """
    h, w = bgr.shape[:2]
    s = long_side / float(max(h, w))
    img = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    k = max(15, (min(img.shape[:2]) // 8) | 1)
    best, best_c = None, -1.0
    for ch in (lab[:, :, 0], img[:, :, 2], img[:, :, 0]):            # ความสว่าง, แดง, น้ำเงิน
        c = cv2.GaussianBlur(ch, (3, 3), 0).astype(np.float32)
        d = np.abs(c - cv2.medianBlur(ch, k).astype(np.float32))
        contrast = float(np.percentile(d, 99)) - float(np.percentile(d, 50))
        if contrast > best_c:
            best, best_c = d, contrast
    lo = float(np.percentile(best, 50))
    hi = max(float(np.percentile(best, 99.3)), lo + 6.0)
    ink = np.clip((best - lo) / (hi - lo), 0, 1)
    return (255 * (1 - ink)).astype(np.uint8), s


def _sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------------------- จัดตำแหน่ง
def _detector():
    try:
        return cv2.SIFT_create(nfeatures=6000), cv2.NORM_L2
    except Exception:
        return cv2.ORB_create(6000), cv2.NORM_HAMMING


def _match(a, b, det, norm, ratio=0.78):
    ka, da = det.detectAndCompute(a, None)
    kb, db = det.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None, 0
    good = [p[0] for p in cv2.BFMatcher(norm).knnMatch(da, db, k=2)
            if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < 8:
        return None, 0
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0,
                                         maxIters=5000, confidence=0.999)
    if A is None or inl is None or math.hypot(A[0, 0], A[1, 0]) < 1e-3:
        return None, 0
    return A, int(inl.sum())


def prealign(master_bgr, test_bgr, margin=MARGIN):
    """
    คืน dict:
      ok (bool), test (ภาพ Test ที่บิดให้ตรง Master + เผื่อขอบ),
      mirrored, angle, inliers, mark_px (ความกว้างมาร์คใน Test เดิม), sharp_ratio,
      problems (list ข้อความปัญหาคุณภาพ)
    """
    mh, mw = master_bgr.shape[:2]
    mi, ms = ink_map(master_bgr)
    ti, ts = ink_map(test_bgr)
    det, norm = _detector()
    best = None
    for mirrored in (False, True):
        a = cv2.flip(mi, 1) if mirrored else mi
        A, n = _match(a, ti, det, norm)
        if A is not None and (best is None or n > best[2]):
            best = (mirrored, A, n)
    out = dict(ok=False, test=None, mirrored=False, angle=0.0, inliers=0, mark_px=0,
               sharp_ratio=1.0, problems=[])
    if best is None or best[2] < MIN_INLIERS:
        out["inliers"] = 0 if best is None else best[2]
        out["problems"].append("จับคู่ Test กับ Master ไม่ได้ — คนละโมเดล / ภาพเบลอมาก / มาร์คไม่อยู่ในภาพ")
        return out
    mirrored, A, n = best
    # A: master_inkmap(อาจกลับด้าน) -> test_inkmap   แปลงเป็น master(เต็ม) -> test(เต็ม)
    W_small = mi.shape[1]
    F = np.array([[-1, 0, W_small - 1], [0, 1, 0], [0, 0, 1]], np.float64) if mirrored else np.eye(3)
    S_m = np.diag([ms, ms, 1.0])            # master เต็ม -> master เล็ก
    S_t = np.diag([1 / ts, 1 / ts, 1.0])    # test เล็ก -> test เต็ม
    M = S_t @ np.vstack([A, [0, 0, 1]]) @ F @ S_m            # master(เต็ม) -> test(เต็ม)
    mx, my = int(round(mw * margin)), int(round(mh * margin))
    T = np.array([[1, 0, -mx], [0, 1, -my], [0, 0, 1]], np.float64)
    Minv = np.linalg.inv(M @ T)                              # test -> ภาพผลลัพธ์
    warped = cv2.warpAffine(test_bgr, Minv[:2], (mw + 2 * mx, mh + 2 * my),
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    scale = math.hypot(M[0, 0], M[1, 0])                     # px ของ Test ต่อ px ของ Master
    mark_px = int(mw * scale)

    # ความคม: เทียบแผนที่หมึกบริเวณมาร์คของ Test กับ Master ที่ขนาดเดียวกัน
    wi, _ = ink_map(warped[my:my + mh, mx:mx + mw], long_side=max(mi.shape))
    sharp_ratio = _sharpness(wi) / max(_sharpness(mi), 1e-6)

    problems = []
    if mark_px < MIN_MARK_PX:
        problems.append(f"มาร์คในภาพเล็กเกินไป ({mark_px}px < {MIN_MARK_PX}px) — ขยับกล้องเข้าใกล้ / ใช้กล้องความละเอียดสูงขึ้น")
    if sharp_ratio < BLUR_RATIO:
        problems.append(f"ภาพเบลอ (คม {sharp_ratio:.0%} ของ Master) — ปรับโฟกัส / ไฟ / กล้องนิ่ง")
    out.update(ok=True, test=warped, mirrored=mirrored,
               angle=round(math.degrees(math.atan2(A[1, 0], A[0, 0])), 1),
               inliers=n, mark_px=mark_px, sharp_ratio=round(sharp_ratio, 2), problems=problems)
    return out

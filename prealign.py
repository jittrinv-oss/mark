#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prealign.py v4 — ส่วนเสริม: จัดภาพ Test ให้ตรงกับ Master "ก่อน" ส่งเข้าการตรวจเดิม + ตรวจคุณภาพภาพ

v1: แผนที่หมึก (ไม่สนสี) + SIFT ลองทั้งปกติ/กลับด้าน + หมุน/ย่อขยาย/เลื่อน + ด่านคุณภาพ
v2: MASK_OUTSIDE — นอกพื้นที่มาร์คของ Master ใน Test เป็นสีพื้น (เฉพาะ Master สะอาด เช่น จาก PDF)
v3: CLEAN_COLOR_LINES — เติมทับเส้นสีบาง ๆ (เส้นประแดง / เส้นบอกระยะ) ที่พาดตัวอักษร
v4: สำหรับ "รูปถ่ายจากกล้อง / มือถือ"
    1) ปรับตั้งฉาก (perspective) — ถ่ายเอียง/มุมเฉียง ถูกดัดให้ตรงเหมือนภาพสแกน
       (ใช้ homography จากจุดอ้างอิง + ตรวจความสมเหตุสมผล ถ้าแปลกจะใช้แบบหมุน/ย่อขยายตามเดิม)
    2) เขตตรวจอัตโนมัติ (ROI) เมื่อ Master เป็นรูปถ่าย:
       - หากรอบเส้นประสีแดงรอบมาร์ค (ช่อง MARK PATTERN) -> ตรวจเฉพาะในกรอบ (+ขอบเล็กน้อย ให้ตัว V ที่ทับเส้นยังอยู่)
       - ไม่เจอกรอบแดง -> ใช้กลุ่มตัวอักษรหลักกลางภาพ
       -> หัวตาราง / "57.5 + 5.0" / ขอบกระดาษ / เงา ไม่ขึ้น alarm อีก โดยไม่ต้องลากโซน IGNORE
    3) บริเวณที่อยู่นอกรูปถ่าย Test (ถ่ายไม่ครบ) ไม่นับ
    คืนค่า master ที่ถูกจำกัดเขตแล้วด้วย (r["master"]) ขนาด/ตำแหน่งเท่าเดิม -> โซน IGNORE เดิมยังใช้ได้
"""
import math

import numpy as np
import cv2

MARGIN = 0.10
MIN_INLIERS = 15
MIN_MARK_PX = 450
BLUR_RATIO = 0.25
MASK_OUTSIDE = True        # v2/v4
CLEAN_COLOR_LINES = True   # v3
USE_PERSPECTIVE = True     # v4
AUTO_ROI = True            # v4


# ---------------------------------------------------------------- v3 เส้นสี
def clean_color_lines(bgr, max_frac=0.08):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    chroma = bgr.max(axis=2).astype(np.int16) - bgr.min(axis=2).astype(np.int16)
    col = ((hsv[:, :, 1] > 70) & (chroma >= 50) & (hsv[:, :, 2] > 60)).astype(np.uint8)
    frac = float(col.mean())
    if frac == 0 or frac > max_frac:
        return bgr, 0.0
    thick = cv2.dilate(cv2.erode(col, np.ones((7, 7), np.uint8)), np.ones((9, 9), np.uint8))
    lines = col & (1 - thick)
    if lines.sum() == 0:
        return bgr, 0.0
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    out = cv2.inpaint(bgr, lines * 255, 3, cv2.INPAINT_TELEA)
    return out, float(lines.mean())


# ---------------------------------------------------------------- แผนที่หมึก
def ink_map(bgr, long_side=900):
    h, w = bgr.shape[:2]
    s = long_side / float(max(h, w))
    img = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    k = max(15, (min(img.shape[:2]) // 8) | 1)
    best, best_c = None, -1.0
    for ch in (lab[:, :, 0], img[:, :, 2], img[:, :, 0]):
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


def _detector():
    try:
        return cv2.SIFT_create(nfeatures=6000), cv2.NORM_L2
    except Exception:
        return cv2.ORB_create(6000), cv2.NORM_HAMMING


def _match(a, b, det, norm, ratio=0.78):
    """คืน (A similarity, จำนวน inlier, src pts, dst pts)"""
    ka, da = det.detectAndCompute(a, None)
    kb, db = det.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None, 0, None, None
    good = [p[0] for p in cv2.BFMatcher(norm).knnMatch(da, db, k=2)
            if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < 8:
        return None, 0, None, None
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0,
                                         maxIters=5000, confidence=0.999)
    if A is None or inl is None or math.hypot(A[0, 0], A[1, 0]) < 1e-3:
        return None, 0, None, None
    return A, int(inl.sum()), src, dst


def _homography_ok(Hm, Sm, w, h):
    """v4: homography ต้องไม่บิดผิดรูป — เทียบกับแบบ similarity: มุมกรอบไม่ไขว้ ขนาดใกล้เคียง"""
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    try:
        ph = cv2.perspectiveTransform(c, Hm).reshape(-1, 2)
        ps = cv2.perspectiveTransform(c, Sm).reshape(-1, 2)
    except cv2.error:
        return False
    if not cv2.isContourConvex(ph.astype(np.float32)):
        return False
    ah, as_ = abs(cv2.contourArea(ph)), abs(cv2.contourArea(ps))
    if as_ <= 0 or not (0.6 < ah / as_ < 1.6):
        return False
    diag = math.hypot(*(ps[2] - ps[0]))
    return float(np.max(np.linalg.norm(ph - ps, axis=1))) < 0.25 * diag


# ---------------------------------------------------------------- v4 เขตตรวจ
def _is_clean(master_bgr):
    g = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY)
    return (g > 200).mean() >= 0.6


def mark_region(master_bgr, grow=0.035):
    """v2: Master สะอาด (ขาว-ดำ) -> convex hull ของหมึกทั้งหมด"""
    g = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY)
    ink = (g < 128).astype(np.uint8)
    if not _is_clean(master_bgr):
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 6
    pts = np.column_stack(np.nonzero(keep[lab]))[:, ::-1]
    if len(pts) < 20:
        return None
    hull = cv2.convexHull(pts.astype(np.int32))
    m = np.zeros(g.shape, np.uint8)
    cv2.fillConvexPoly(m, hull, 255)
    r = max(3, int(grow * max(g.shape)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    if m.mean() / 255 > 0.97:
        return None
    return m


def red_box_region(master_bgr, grow=0.025):
    """v4: กรอบเส้นประสีแดงรอบมาร์ค ในรูปถ่าย Master -> mask (255 = ในกรอบ) หรือ None"""
    H, W = master_bgr.shape[:2]
    hsv = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2HSV)
    red = (((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 70)).astype(np.uint8)
    if red.sum() < 50:
        return None
    k = max(9, int(max(H, W) * 0.03)) | 1
    joined = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 0.05 * W * H or area > 0.92 * W * H or w < 0.15 * W or h < 0.15 * H:
            continue
        # ต้องเป็น "กรอบ" (มีเส้นแดงอย่างน้อย 3 ด้าน) ไม่ใช่ก้อนแดงทึบ / โลโก้
        b = max(3, int(0.06 * min(w, h)))
        sides = [red[y:y + h, x:x + b].any(axis=1).mean(), red[y:y + h, x + w - b:x + w].any(axis=1).mean(),
                 red[y:y + b, x:x + w].any(axis=0).mean(), red[y + h - b:y + h, x:x + w].any(axis=0).mean()]
        if sum(s > 0.35 for s in sides) < 3:
            continue
        inner = red[y + h // 4:y + 3 * h // 4, x + w // 4:x + 3 * w // 4].mean()
        if inner > 0.15:
            continue
        if best is None or area > best[2] * best[3]:
            best = (x, y, w, h)
    if best is None:
        return None
    x, y, w, h = best
    gx, gy = int(grow * W), int(grow * H)
    m = np.zeros((H, W), np.uint8)
    m[max(0, y - gy):min(H, y + h + gy), max(0, x - gx):min(W, x + w + gx)] = 255
    return m


def auto_roi(master_bgr):
    """v4: เขตตรวจ — กรอบเส้นประแดงก่อน (ช่อง MARK PATTERN) / ไม่เจอ: Master สะอาด ใช้ hull หมึก"""
    m = red_box_region(master_bgr)
    if m is not None:
        return m, "กรอบเส้นประแดง"
    m = mark_region(master_bgr)
    return (m, "พื้นที่มาร์ค") if m is not None else (None, "")


def _bg_color(img, mask):
    px = img[mask > 0]
    if len(px) < 10:
        px = img.reshape(-1, 3)
    s = px.sum(axis=1)
    return np.median(px[s >= np.percentile(s, 60)], axis=0).astype(np.uint8)


# ---------------------------------------------------------------- หลัก
def prealign(master_bgr, test_bgr, margin=MARGIN):
    cleaned = 0.0
    if CLEAN_COLOR_LINES:
        test_bgr, cleaned = clean_color_lines(test_bgr)
        master_for_match, _ = clean_color_lines(master_bgr)
    else:
        master_for_match = master_bgr
    mh, mw = master_bgr.shape[:2]
    mi, ms = ink_map(master_for_match)
    ti, ts = ink_map(test_bgr)
    det, norm = _detector()
    best = None
    for mirrored in (False, True):
        a = cv2.flip(mi, 1) if mirrored else mi
        A, n, src, dst = _match(a, ti, det, norm)
        if A is not None and (best is None or n > best[2]):
            best = (mirrored, A, n, src, dst)
    out = dict(ok=False, test=None, master=master_bgr, mirrored=False, angle=0.0, inliers=0, mark_px=0,
               sharp_ratio=1.0, problems=[], masked=False, color_cleaned=cleaned, warp="", roi="")
    if best is None or best[2] < MIN_INLIERS:
        out["inliers"] = 0 if best is None else best[2]
        out["problems"].append("จับคู่ Test กับ Master ไม่ได้ — คนละโมเดล / ภาพเบลอมาก / มาร์คไม่อยู่ในภาพ")
        return out
    mirrored, A, n, src, dst = best
    W_small = mi.shape[1]
    F = np.array([[-1, 0, W_small - 1], [0, 1, 0], [0, 0, 1]], np.float64) if mirrored else np.eye(3)
    S_m = np.diag([ms, ms, 1.0])
    S_t = np.diag([1 / ts, 1 / ts, 1.0])
    Sim = np.vstack([A, [0, 0, 1]])
    warp = "หมุน/ย่อขยาย"
    if USE_PERSPECTIVE:
        Hs, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if Hs is not None and inl is not None and int(inl.sum()) >= max(MIN_INLIERS, 0.6 * n) \
                and _homography_ok(Hs, Sim, mi.shape[1], mi.shape[0]):
            Sim = Hs
            n = int(inl.sum())
            warp = "ปรับตั้งฉาก (perspective)"
    M = S_t @ Sim @ F @ S_m                                # master(เต็ม) -> test(เต็ม)
    mx, my = int(round(mw * margin)), int(round(mh * margin))
    T = np.array([[1, 0, -mx], [0, 1, -my], [0, 0, 1]], np.float64)
    Minv = np.linalg.inv(M @ T)
    size = (mw + 2 * mx, mh + 2 * my)
    warped = cv2.warpPerspective(test_bgr, Minv, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpPerspective(np.full(test_bgr.shape[:2], 255, np.uint8), Minv, size,
                                flags=cv2.INTER_NEAREST, borderValue=0)
    valid = cv2.erode(valid, np.ones((7, 7), np.uint8))
    scale = math.sqrt(abs(np.linalg.det(M[:2, :2])))
    mark_px = int(mw * scale)
    wi, _ = ink_map(warped[my:my + mh, mx:mx + mw], long_side=max(mi.shape))
    sharp_ratio = _sharpness(wi) / max(_sharpness(mi), 1e-6)

    master_out = master_for_match          # v4: Master ที่ลบเส้นสีแล้ว (ตัว V ที่ทับเส้นแดงไม่ถูกตัด)
    masked, roi_name = False, ""
    if MASK_OUTSIDE:
        reg, roi_name = auto_roi(master_bgr) if AUTO_ROI else (mark_region(master_for_match), "พื้นที่มาร์ค")
        full = np.zeros(warped.shape[:2], np.uint8)
        if reg is not None:
            full[my:my + mh, mx:mx + mw] = reg
        else:
            full[my:my + mh, mx:mx + mw] = 255
        # v4: ส่วนที่รูป Test ถ่ายไม่ครบ ไม่นับ
        keep = cv2.bitwise_and(full, valid)
        if reg is not None or (keep[my:my + mh, mx:mx + mw] == 0).any():
            warped = warped.copy()
            warped[keep == 0] = _bg_color(warped, keep)
            mk = keep[my:my + mh, mx:mx + mw]
            if (mk == 0).any():
                master_out = master_out.copy()
                master_out[mk == 0] = _bg_color(master_out, mk)
            masked = True

    problems = []
    if mark_px < MIN_MARK_PX and sharp_ratio < 1.5:
        problems.append(f"มาร์คในภาพเล็กเกินไป ({mark_px}px < {MIN_MARK_PX}px) — ขยับกล้องเข้าใกล้ / ใช้กล้องความละเอียดสูงขึ้น")
    if sharp_ratio < BLUR_RATIO:
        problems.append(f"ภาพเบลอ (คม {sharp_ratio:.0%} ของ Master) — ปรับโฟกัส / ไฟ / กล้องนิ่ง")
    out.update(ok=True, test=warped, master=master_out, mirrored=mirrored,
               angle=round(math.degrees(math.atan2(A[1, 0], A[0, 0])), 1),
               inliers=n, mark_px=mark_px, sharp_ratio=round(sharp_ratio, 2), problems=problems,
               masked=masked, warp=warp, roi=roi_name)
    return out

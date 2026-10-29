#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sharp_check.py v1 — ตัวตรวจมาร์คแบบ "ทนภาพเบลอ / กล้องความละเอียดต่ำ"  (ไฟล์เสริม)
ไม่แก้ไฟล์เดิม: mark_ui.py, glyph_check.py (v9), mark_match.py, mark_ui_plus.py ทำงานเหมือนเดิม
ถูกเรียกจาก mark_ui_sharp.py (จอ Pi / PC) และ web_sharp.py (มือถือ) เมื่อติ๊ก "ใช้ตัวตรวจคมชัด"

ขั้นตอน (ทุกค่าอยู่ใน sharp_config.json ไฟล์เดียว):
  0) quality gate      : Laplacian variance + ความกว้างมาร์ค -> "ภาพเบลอ ถ่ายใหม่" แทนการตัด NG
  1) preprocess        : เลือกช่องสีคอนทราสต์ดีสุด -> gamma -> ลบพื้นหลัง (Gaussian / top-hat / black-hat)
                         -> CLAHE -> denoise เบา -> unsharp mask (รัศมี 1-2 px, amount <= 1.0 กัน ringing)
                         -> ทำให้เป็น "ตัวเข้มบนพื้นขาว" เสมอ (ตัวขาวบนเขียวก็ได้)
  2) threshold         : Sauvola (หรือ adaptive) แทน threshold ค่าเดียว
  3) align             : จุดเด่น SIFT/ORB + cornerSubPix (sub-pixel) -> homography -> ECC ทั้งภาพ
                         -> ECC/phase correlation ต่อ "บรรทัดข้อความ" (block refine)
  4) ROI               : ตรวจเฉพาะพื้นที่มาร์ค ตัดกรอบเส้นประ / วัตถุนอกกรอบ / เงา
  5) skip QR           : หา QR code ใน Master แล้วไม่ตรวจบริเวณนั้น
  6) compare           : มี tolerance ทั้ง 2 ทิศ (missing / extra) ด้วย dilate หรือ distance transform
                         + ตัดสินระดับชิ้นส่วน (connected component) ไม่ใช่ XOR ทีละพิกเซล
  7) debug             : บันทึกภาพทุกขั้น + ภาพก่อน/หลัง sharpen
  (เสริม) best_frame   : ถ่ายหลายเฟรม เลือกเฟรมคมสุด หรือเฉลี่ยหลัง align
  (เสริม) OCR          : ถ้ามี pytesseract อ่านข้อความเทียบ (ปิดไว้เป็นค่าเริ่มต้น)
"""
import json
import math
import os
import time

import numpy as np
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE, "sharp_config.json")

DEFAULT = {
    "_note": "ค่าทั้งหมดของตัวตรวจคมชัด แก้ที่นี่ที่เดียว (ลบไฟล์นี้ = กลับค่าเริ่มต้น)",
    "proc_long_side": 1400,          # ขยาย/ย่อ Master ให้ด้านยาวเท่านี้ก่อนประมวลผล (Test ถูก align มาขนาดเดียวกัน)
    # ---- 0) quality gate
    "blur_check": True,
    "blur_lap_min": 25.0,             # Laplacian variance ขั้นต่ำ (บนภาพ Test ที่ align แล้ว ในพื้นที่มาร์ค)
    "blur_ratio_min": 0.20,           # หรือ ความคม Test / Master ต่ำกว่านี้ = เบลอ
    "min_mark_px": 350,               # ความกว้างมาร์คในภาพ Test ต้นฉบับ (px) ต่ำกว่านี้ = เตือนความละเอียดต่ำ
    "continue_if_blur": False,        # True = เบลอก็ตรวจต่อ (ติ๊กในหน้าจอได้)
    # ---- 1) preprocess
    "channel": "auto",                # auto / L / R / G / B
    "gamma": 0.85,                    # < 1 ดึงเงาให้สว่างขึ้น
    "bg_method": "gaussian",          # gaussian / tophat
    "bg_sigma_frac": 0.04,            # ขนาด Gaussian/kernel พื้นหลัง เทียบด้านยาว
    "clahe_clip": 2.0,
    "clahe_grid": 8,
    "denoise": "bilateral",           # bilateral / gaussian / none
    "denoise_strength": 5,
    "unsharp_radius": 1.2,            # px (1-2)
    "unsharp_amount": 0.7,            # 0.5-1.0  (บังคับไม่เกิน 1.0)
    # ---- 2) threshold
    "threshold": "sauvola",           # sauvola / adaptive
    "sauvola_window_frac": 0.035,
    "sauvola_k": 0.20,
    "test_threshold": "match_master", # match_master = ให้ Test มีพื้นที่หมึกเท่า Master (เส้นหนาเท่ากัน ไม่บวมจากภาพเบลอ) / same = ใช้แบบเดียวกับ Master
    "min_blob_frac": 0.00001,         # ชิ้นหมึกเล็กกว่า (สัดส่วนพื้นที่ภาพ) ตัดทิ้ง
    # ---- 3) align
    "min_inliers": 25,               # จุดอ้างอิงน้อยกว่านี้ = จับคู่ไม่มั่นใจ -> ถ่ายใหม่
    "subpixel": True,
    "global_ecc": True,
    "block_refine": True,
    "block_max_shift_frac": 0.012,    # ขยับต่อบรรทัดได้ไม่เกิน (สัดส่วนด้านยาว)
    # ---- 4) ROI
    "roi_auto": True,
    "drop_dashed_frame": True,
    "roi_margin_frac": 0.02,
    # ---- 5) QR
    "skip_qr": True,
    "qr_margin_frac": 0.02,
    # ---- 6) compare
    "compare": "dilate",              # dilate / distance
    "tol_px": 2,                      # tolerance (px ที่ขนาด proc_long_side)
    "cover_min": 0.70,                # ชิ้นตัวอักษรต้องมีหมึกใน Test ครอบคลุม >= 70%
    "missing_blob_letter_frac": 0.30, # หรือ มีก้อนที่หายใหญ่ >= 30% ของพื้นที่ตัวอักษรเฉลี่ย (ตัวอักษรติดกันเป็นชิ้นเดียว)
    "shape_ncc_min": 0.50,            # รูปร่างตัวอักษร Master vs Test (correlation) ต่ำกว่านี้ = ผิดตัว เช่น 1 กับ 2
    "min_missing_area_frac": 0.00006, # พื้นที่ส่วนหายขั้นต่ำต่อชิ้น (สัดส่วนพื้นที่ภาพ)
    "extra_area_frac": 0.00012,       # หมึกเกินขั้นต่ำ
    "check_extra": True,
    # ---- 7) debug / อื่น ๆ
    "debug_save": True,
    "multi_frames": 5,                # จำนวนเฟรมตอนกด จับภาพ+ตรวจ
    "frame_mode": "best",             # best / average
    "ocr": False,
}


# ===================================================================== config
def load_config():
    cfg = dict(DEFAULT)
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:
        save_config(cfg)
    cfg["unsharp_amount"] = float(min(1.0, max(0.0, cfg["unsharp_amount"])))   # กัน ringing
    cfg["unsharp_radius"] = float(min(2.0, max(0.5, cfg["unsharp_radius"])))
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ===================================================================== utils
def lap_var(gray, mask=None):
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    return float(lap[mask > 0].var()) if mask is not None and mask.any() else float(lap.var())


def _resize_long(img, long_side):
    h, w = img.shape[:2]
    s = long_side / float(max(h, w))
    if abs(s - 1) < 1e-3:
        return img, 1.0
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC), s


def _channels(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(bgr)
    return {"L": lab[:, :, 0], "R": r, "G": g, "B": b}


# ===================================================================== 1) preprocess
def preprocess(bgr, cfg, sharpen=True):
    """คืน (ภาพเทา uint8 ตัวเข้มพื้นขาว, ข้อมูล)"""
    H, W = bgr.shape[:2]
    big = max(H, W)
    sig = max(5.0, cfg["bg_sigma_frac"] * big)
    best = None
    chans = _channels(bgr)
    order = list(chans) if cfg["channel"] == "auto" else [cfg["channel"]]
    for name in order:
        c = chans[name].astype(np.float32) / 255.0
        c = np.power(np.clip(c, 1e-4, 1), cfg["gamma"])
        if cfg["bg_method"] == "tophat":
            k = int(sig * 2) | 1
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            th = cv2.morphologyEx(c, cv2.MORPH_TOPHAT, se)     # ตัวสว่าง
            bh = cv2.morphologyEx(c, cv2.MORPH_BLACKHAT, se)   # ตัวเข้ม (black-hat)
            d = th - bh
        else:
            # พื้นหลัง = Gaussian ใหญ่ (คำนวณบนภาพย่อ 1/4 แล้วขยายกลับ -> เร็วขึ้นมาก)
            small = cv2.resize(c, (max(8, c.shape[1] // 4), max(8, c.shape[0] // 4)), interpolation=cv2.INTER_AREA)
            bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), sig / 4), (c.shape[1], c.shape[0]), interpolation=cv2.INTER_LINEAR)
            d = c - bg
        hi, lo = float(np.percentile(d, 99.5)), float(-np.percentile(d, 0.5))
        score = max(hi, lo) / (float(np.std(d)) + 1e-6)
        if best is None or score > best[0]:
            best = (score, name, d, hi, lo)
    _, name, d, hi, lo = best
    light = hi > lo * 1.10                       # ตัวสว่างบนพื้นเข้ม/พื้นสี
    ink = d if light else -d                     # ink > 0
    top = max(float(np.percentile(ink, 99.7)), 1e-3)
    g = (255 * (1 - np.clip(ink / top, 0, 1))).astype(np.uint8)
    g = cv2.createCLAHE(cfg["clahe_clip"], (cfg["clahe_grid"],) * 2).apply(g)
    s = cfg["denoise_strength"]
    if cfg["denoise"] == "bilateral" and s > 0:
        g = cv2.bilateralFilter(g, 5, s * 6, 3)
    elif cfg["denoise"] == "gaussian" and s > 0:
        g = cv2.GaussianBlur(g, (3, 3), 0.6)
    before = g.copy()
    if sharpen and cfg["unsharp_amount"] > 0:
        f = g.astype(np.float32)
        bl = cv2.GaussianBlur(f, (0, 0), cfg["unsharp_radius"])
        g = np.clip(f + cfg["unsharp_amount"] * (f - bl), 0, 255).astype(np.uint8)
    return g, dict(channel=name, light_text=bool(light), before_sharpen=before)


# ===================================================================== 2) threshold
def binarize(gray, cfg):
    H, W = gray.shape
    if cfg["threshold"] == "adaptive":
        blk = max(15, int(cfg["sauvola_window_frac"] * max(H, W)) | 1)
        m = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk, 12)
    else:
        win = max(15, int(cfg["sauvola_window_frac"] * max(H, W)) | 1)
        f = gray.astype(np.float32)
        mean = cv2.boxFilter(f, -1, (win, win))
        sq = cv2.boxFilter(f * f, -1, (win, win))
        std = np.sqrt(np.maximum(sq - mean * mean, 0))
        T = mean * (1 + cfg["sauvola_k"] * (std / 128.0 - 1))
        m = ((f < T) & (f < 200)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, 4] >= max(6, cfg["min_blob_frac"] * H * W)
    return np.where(keep[lab], 255, 0).astype(np.uint8)


# ===================================================================== 4) ROI / 5) QR
def find_qr(gray_ink_dark, binm, cfg):
    """คืน list กรอบ (x0,y0,x1,y1) ของ QR code ใน Master"""
    boxes = []
    try:
        ok, pts = cv2.QRCodeDetector().detect(gray_ink_dark)
        if ok and pts is not None:
            p = pts.reshape(-1, 2)
            boxes.append((p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()))
    except Exception:
        pass
    if not boxes:                                  # หา finder pattern (สี่เหลี่ยมซ้อน 3 ชั้น)
        cnts, hier = cv2.findContours(binm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        if hier is not None:
            hier = hier[0]
            for i, c in enumerate(cnts):
                x, y, w, h = cv2.boundingRect(c)
                if w < 8 or h < 8 or not (0.75 < w / h < 1.33):
                    continue
                k, depth = hier[i][2], 0
                while k != -1 and depth < 3:
                    depth += 1
                    k = hier[k][2]
                if depth >= 2 and cv2.contourArea(c) > 0.8 * w * h:
                    ap = cv2.approxPolyDP(c, 0.06 * cv2.arcLength(c, True), True)
                    if len(ap) == 4:                       # สี่เหลี่ยมจริง (ไม่ใช่วงกลม E6 / มอก.)
                        cands.append((x, y, w, h))
        if len(cands) >= 3:
            cands.sort(key=lambda b: -b[2] * b[3])
            ref = cands[0][2]
            same = [b for b in cands if 0.6 < b[2] / ref < 1.6]
            # finder 3 ตัวต้องอยู่ใกล้กัน (ภายใน ~8 เท่าขนาด finder)
            pick = None
            for a_ in same:
                near = [b for b in same if abs(b[0] - a_[0]) < 8 * ref and abs(b[1] - a_[1]) < 8 * ref]
                if len(near) >= 3:
                    pick = near[:3]
                    break
            same = pick or []
            if len(same) == 3:
                xs = [b[0] for b in same] + [b[0] + b[2] for b in same]
                ys = [b[1] for b in same] + [b[1] + b[3] for b in same]
                boxes.append((min(xs), min(ys), max(xs), max(ys)))
    H, W = binm.shape
    m = int(cfg["qr_margin_frac"] * max(H, W))
    return [(max(0, int(a) - m), max(0, int(b) - m), min(W, int(c) + m), min(H, int(d) + m)) for a, b, c, d in boxes]


def _dashed_lines(st, txt_h):
    """หาแนวเส้นประ: ขีดบางสั้น ๆ >= 4 ชิ้น เรียงอยู่แนวเดียวกัน -> list ('h', y, x0, x1) / ('v', x, y0, y1)"""
    thin_t = max(3, 0.35 * txt_h)
    hs = [s for s in st[1:] if s[3] <= thin_t and s[2] >= 2.5 * s[3]]
    vs = [s for s in st[1:] if s[2] <= thin_t and s[3] >= 2.5 * s[2]]
    lines = []
    for kind, arr, key in (("h", hs, lambda s: s[1] + s[3] / 2), ("v", vs, lambda s: s[0] + s[2] / 2)):
        arr = sorted(arr, key=key)
        grp = []
        for s in arr + [None]:
            if s is not None and (not grp or abs(key(s) - key(grp[-1])) <= 0.3 * txt_h):
                grp.append(s)
                continue
            if len(grp) >= 4:
                c = float(np.median([key(g) for g in grp]))
                if kind == "h":
                    lines.append(("h", c, min(g[0] for g in grp), max(g[0] + g[2] for g in grp)))
                else:
                    lines.append(("v", c, min(g[1] for g in grp), max(g[1] + g[3] for g in grp)))
            grp = [s] if s is not None else []
    return lines


def mark_roi(mbin, cfg):
    """พื้นที่มาร์ค + ตัดกรอบเส้นประ (รวมมุมกรอบ) คืน (roi mask, ink ที่ใช้ตรวจ)"""
    H, W = mbin.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(mbin, 8)
    if n <= 1:
        return np.full((H, W), 255, np.uint8), mbin
    hs = st[1:, 3]
    txt_h = float(np.median(hs[hs > 3])) if (hs > 3).any() else 10.0
    keep = mbin.copy()
    if cfg["drop_dashed_frame"]:
        tol = max(4, 0.35 * txt_h)
        for kind, c, a0, a1 in _dashed_lines(st, txt_h):
            for i in range(1, n):
                x, y, w, h, area = st[i]
                if max(w, h) > 2.5 * txt_h:
                    continue
                if kind == "h" and y - tol <= c <= y + h + tol and x + w >= a0 - txt_h and x <= a1 + txt_h:
                    keep[lab == i] = 0
                if kind == "v" and x - tol <= c <= x + w + tol and y + h >= a0 - txt_h and y <= a1 + txt_h:
                    keep[lab == i] = 0
    ys, xs = np.nonzero(keep)
    roi = np.zeros((H, W), np.uint8)
    if len(xs):
        mg = int(cfg["roi_margin_frac"] * max(H, W))
        hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
        cv2.fillConvexPoly(roi, hull, 255)
        roi = cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * mg + 1,) * 2))
    return roi, keep


# ===================================================================== 3) align
def _detector():
    try:
        return cv2.SIFT_create(nfeatures=6000), cv2.NORM_L2
    except Exception:
        return cv2.ORB_create(6000), cv2.NORM_HAMMING


def _match(a, b, det, norm, subpix):
    ka, da = det.detectAndCompute(a, None)
    kb, db = det.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None, 0
    good = [p[0] for p in cv2.BFMatcher(norm).knnMatch(da, db, k=2)
            if len(p) == 2 and p[0].distance < 0.78 * p[1].distance]
    if len(good) < 10:
        return None, len(good)
    pa = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pb = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    if subpix:                                     # 7.3 sub-pixel refinement ของจุดอ้างอิง
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        try:
            pa = cv2.cornerSubPix(a, pa.copy(), (4, 4), (-1, -1), crit)
            pb = cv2.cornerSubPix(b, pb.copy(), (4, 4), (-1, -1), crit)
        except cv2.error:
            pass
    H, inl = cv2.findHomography(pb, pa, cv2.RANSAC, 3.0)      # test -> master
    if H is None or inl is None:
        return None, 0
    return H, int(inl.sum())


def _ecc(ref, mov, warp, mode, iters=60):
    a = cv2.GaussianBlur(ref, (0, 0), 1.5).astype(np.float32) / 255
    b = cv2.GaussianBlur(mov, (0, 0), 1.5).astype(np.float32) / 255
    try:
        _, w = cv2.findTransformECC(a, b, warp.astype(np.float32), mode,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-5), None, 3)
        return w
    except cv2.error:
        return None


def _text_bands(mink):
    """แบ่งบรรทัดข้อความจาก projection แนวนอนของ Master"""
    H = mink.shape[0]
    prof = (mink > 0).sum(axis=1)
    on = prof > max(2, 0.01 * prof.max())
    bands, s = [], None
    for y in range(H):
        if on[y] and s is None:
            s = y
        elif not on[y] and s is not None:
            if y - s > 4:
                bands.append((s, y))
            s = None
    if s is not None:
        bands.append((s, H))
    return bands


def align(mpre, tpre, tbgr_raw_pre, mink, cfg):
    """
    คืน (test_pre ที่ซ้อน Master, valid mask, info)
    tpre = ภาพ Test ที่ preprocess แล้ว (ขนาดเดิม)
    """
    Hm, Wm = mpre.shape
    det, norm = _detector()
    best = None
    for mir in (False, True):
        t = cv2.flip(tpre, 1) if mir else tpre
        Hh, n = _match(mpre, t, det, norm, cfg["subpixel"])
        if Hh is not None and (best is None or n > best[2]):
            best = (mir, Hh, n)
    if best is None or best[2] < 12:
        return None, None, dict(ok=False, inliers=0 if best is None else best[2])
    mir, Hh, n = best
    if mir:
        F = np.array([[-1, 0, tpre.shape[1] - 1], [0, 1, 0], [0, 0, 1]], np.float64)
        Hh = Hh @ F
    warped = cv2.warpPerspective(tpre, Hh, (Wm, Hm), flags=cv2.INTER_CUBIC, borderValue=255)
    valid = cv2.warpPerspective(np.full(tpre.shape, 255, np.uint8), Hh, (Wm, Hm), flags=cv2.INTER_NEAREST)
    method = "SIFT+subpix" if cfg["subpixel"] else "SIFT"
    if cfg["global_ecc"]:
        S = np.diag([0.5, 0.5, 1.0]).astype(np.float32)
        w = _ecc(cv2.resize(mpre, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                 cv2.resize(warped, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                 np.eye(3, dtype=np.float32), cv2.MOTION_HOMOGRAPHY)
        if w is not None:
            w = np.linalg.inv(S) @ w @ S                       # กลับเป็นสเกลเต็ม
        if w is not None and np.abs(w - np.eye(3)).max() < 0.08:
            warped = cv2.warpPerspective(warped, w, (Wm, Hm), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderValue=255)
            valid = cv2.warpPerspective(valid, w, (Wm, Hm), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
            method += "+ECC"
    shifts = []
    if cfg["block_refine"]:
        lim = cfg["block_max_shift_frac"] * max(Hm, Wm)
        dy_map = np.zeros(Hm, np.float32)
        dx_map = np.zeros(Hm, np.float32)
        wt = np.zeros(Hm, np.float32)
        for (y0, y1) in _text_bands(mink):
            pad = max(4, (y1 - y0) // 2)
            a0, a1 = max(0, y0 - pad), min(Hm, y1 + pad)
            ref, mov = mpre[a0:a1], warped[a0:a1]
            w = _ecc(ref, mov, np.eye(2, 3, dtype=np.float32), cv2.MOTION_TRANSLATION, 40)
            if w is not None:
                dx, dy = float(w[0, 2]), float(w[1, 2])
            else:
                (dx, dy), _ = cv2.phaseCorrelate(np.float32(ref), np.float32(mov))
            if abs(dx) <= lim and abs(dy) <= lim:
                dx_map[a0:a1] += dx
                dy_map[a0:a1] += dy
                wt[a0:a1] += 1
                shifts.append((y0, y1, round(dx, 2), round(dy, 2)))
        if shifts:
            ok = wt > 0
            dx_map[ok] /= wt[ok]
            dy_map[ok] /= wt[ok]
            idx = np.arange(Hm)
            if (~ok).any():
                dx_map[~ok] = np.interp(idx[~ok], idx[ok], dx_map[ok])
                dy_map[~ok] = np.interp(idx[~ok], idx[ok], dy_map[ok])
            dx_map = cv2.GaussianBlur(dx_map.reshape(-1, 1), (1, 0), 4).ravel() if Hm > 8 else dx_map
            dy_map = cv2.GaussianBlur(dy_map.reshape(-1, 1), (1, 0), 4).ravel() if Hm > 8 else dy_map
            gx, gy = np.meshgrid(np.arange(Wm, dtype=np.float32), np.arange(Hm, dtype=np.float32))
            mx = gx + dx_map[:, None]
            my = gy + dy_map[:, None]
            warped = cv2.remap(warped, mx, my, cv2.INTER_CUBIC, borderValue=255)
            valid = cv2.remap(valid, mx, my, cv2.INTER_NEAREST, borderValue=0)
            method += f"+block({len(shifts)})"
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8))
    return warped, valid, dict(ok=True, inliers=n, mirrored=mir, method=method, block_shifts=shifts, H=Hh)


# ===================================================================== 6) compare
def binarize_matched(test_gray, mink, check, cfg):
    """
    ให้หมึกของ Test มีพื้นที่เท่ากับ Master ในบริเวณที่ตรวจ (เลือกระดับตัดตามสัดส่วน)
    -> เส้นตัวอักษรหนาเท่ากัน ภาพเบลอไม่ทำให้ตัวบวมจนกลบจุดที่หายไป
    """
    reg = check > 0
    if reg.sum() < 100:
        return binarize(test_gray, cfg)
    frac = float((mink[reg] > 0).mean())
    vals = test_gray[reg]
    thr = float(np.percentile(vals, min(60.0, max(0.5, 100 * frac))))
    m = ((test_gray <= thr) & reg).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    H, W = m.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, 4] >= max(6, cfg["min_blob_frac"] * H * W)
    return np.where(keep[lab], 255, 0).astype(np.uint8)


def compare(mink, tink, check_mask, cfg):
    H, W = mink.shape
    tol = int(cfg["tol_px"])
    if cfg["compare"] == "distance":
        dt_t = cv2.distanceTransform(255 - tink, cv2.DIST_L2, 3)
        dt_m = cv2.distanceTransform(255 - mink, cv2.DIST_L2, 3)
        miss = ((mink > 0) & (dt_t > tol)).astype(np.uint8) * 255
        extra = ((tink > 0) & (dt_m > tol)).astype(np.uint8) * 255
    else:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1,) * 2)
        miss = cv2.bitwise_and(mink, cv2.bitwise_not(cv2.dilate(tink, k)))
        extra = cv2.bitwise_and(tink, cv2.bitwise_not(cv2.dilate(mink, k)))
    miss = cv2.bitwise_and(miss, check_mask)
    extra = cv2.bitwise_and(extra, check_mask)
    op = np.ones((2, 2), np.uint8)
    miss = cv2.morphologyEx(miss, cv2.MORPH_OPEN, op)
    extra = cv2.morphologyEx(extra, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # ระดับชิ้นส่วน: ต่อ connected component ของ Master
    min_area = max(8, cfg["min_missing_area_frac"] * H * W)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.bitwise_and(mink, check_mask), 8)
    areas = st[1:, 4]
    letter = float(np.median(areas[areas >= 10])) if (areas >= 10).any() else 100.0
    big_blob = max(min_area, cfg["missing_blob_letter_frac"] * letter)
    hs_ = st[1:, 3][areas >= 10]
    med_h = float(np.median(hs_)) if len(hs_) else 10.0
    nm, lm, sm, _ = cv2.connectedComponentsWithStats(miss, 8)
    rows = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if a < 10:
            continue
        comp = lab[y:y + h, x:x + w] == i
        mcut = miss[y:y + h, x:x + w]
        lost = int((mcut[comp] > 0).sum())
        cover = 1 - lost / max(a, 1)
        ids = np.unique(lm[y:y + h, x:x + w][comp & (mcut > 0)])
        biggest = max([int(sm[j, 4]) for j in ids if j > 0], default=0)
        # รูปร่างตัวอักษร: correlation ระหว่างชิ้นของ Master กับหมึก Test ในเขตของชิ้นนั้น (ทนเบลอ)
        p = 2 * tol + 3
        X0, Y0, X1, Y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
        mz = (lab[Y0:Y1, X0:X1] == i).astype(np.uint8)
        zone = cv2.dilate(mz, np.ones((2 * p + 1,) * 2, np.uint8))
        tz = ((tink[Y0:Y1, X0:X1] > 0) & (zone > 0)).astype(np.float32)
        a_ = cv2.GaussianBlur(mz.astype(np.float32), (0, 0), 1.5)
        b_ = cv2.GaussianBlur(tz, (0, 0), 1.5)
        a_ -= a_.mean(); b_ -= b_.mean()
        ncc = float((a_ * b_).sum() / (np.sqrt((a_ * a_).sum() * (b_ * b_).sum()) + 1e-6))
        shape_bad = h >= 0.5 * med_h and ncc < cfg["shape_ncc_min"]
        if (lost >= min_area and cover < cfg["cover_min"]) or biggest >= big_blob or shape_bad:
            status = "MISSING" if cover < 0.25 else "PARTIAL"
            note = f"หาย {1 - cover:.0%} ({lost}px) รูปร่างตรงกัน {ncc:.2f}"
        else:
            status, note = "OK", f"ครบ {cover:.0%} รูปร่าง {ncc:.2f}"
        rows.append(dict(id=len(rows) + 1, kind="TEXT", status=status, note=note, x=x, y=y, w=w, h=h,
                         area=a, cov=round(cover, 3), vfrac=None, hfrac=None, dx=0, dy=0))
    extras = []
    if cfg["check_extra"]:
        ea = max(20, cfg["extra_area_frac"] * H * W)
        n2, _, st2, _ = cv2.connectedComponentsWithStats(extra, 8)
        extras = [dict(x=int(st2[i, 0]), y=int(st2[i, 1]), w=int(st2[i, 2]), h=int(st2[i, 3]), area=int(st2[i, 4]))
                  for i in range(1, n2) if st2[i, 4] >= ea]
    return rows, extras, miss, extra


# ===================================================================== (เสริม) หลายเฟรม
def best_frame(frames, mode="best"):
    """เลือกเฟรมคมสุด หรือเฉลี่ยหลัง align (ECC translation) คืน (ภาพ, คะแนนความคมแต่ละเฟรม)"""
    frames = [f for f in frames if f is not None]
    if not frames:
        return None, []
    scores = [lap_var(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]
    i = int(np.argmax(scores))
    if mode != "average" or len(frames) < 2:
        return frames[i], scores
    ref = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
    acc = frames[i].astype(np.float32)
    cnt = 1
    for j, f in enumerate(frames):
        if j == i or scores[j] < 0.6 * scores[i]:
            continue
        w = _ecc(ref, cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), np.eye(2, 3, dtype=np.float32), cv2.MOTION_EUCLIDEAN, 50)
        if w is None:
            continue
        acc += cv2.warpAffine(f, w, (f.shape[1], f.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        cnt += 1
    return (acc / cnt).astype(np.uint8), scores


def before_after(before, after, zoom=3):
    """ภาพเทียบก่อน/หลัง sharpen (ซูมบริเวณที่มีหมึกมากสุด) — ใช้เช็คว่าไม่มี ringing (ขอบขาว/ดำซ้อน)"""
    g = 255 - before
    k = max(40, min(before.shape) // 4)
    s = cv2.boxFilter(g.astype(np.float32), -1, (k, k))
    y, x = np.unravel_index(int(np.argmax(s)), s.shape)
    y0, x0 = max(0, y - k // 2), max(0, x - k // 2)
    a = before[y0:y0 + k, x0:x0 + k]
    b = after[y0:y0 + k, x0:x0 + k]
    a = cv2.resize(a, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    b = cv2.resize(b, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    pa = cv2.copyMakeBorder(a, 30, 0, 0, 8, cv2.BORDER_CONSTANT, value=255)
    pb = cv2.copyMakeBorder(b, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=255)
    cv2.putText(pa, "BEFORE sharpen", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)
    cv2.putText(pb, "AFTER sharpen", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)
    return np.hstack([pa, pb])


def ocr_text(gray):
    try:
        import pytesseract
        return " ".join(pytesseract.image_to_string(gray, config="--psm 6").split())
    except Exception:
        return None


# ===================================================================== หลัก
COLOR = {"OK": (0, 170, 0), "PARTIAL": (0, 165, 255), "MISSING": (0, 0, 255), "SKIP": (150, 150, 150)}


def draw(base_gray, rows, extras, skips, show_ok=False):
    R = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    for (x0, y0, x1, y1) in skips:
        cv2.rectangle(R, (x0, y0), (x1, y1), (160, 160, 160), 2)
        cv2.putText(R, "QR skip", (x0 + 4, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1, cv2.LINE_AA)
    for r in rows:
        bad = r["status"] != "OK"
        if bad or show_ok:
            cv2.rectangle(R, (r["x"] - 3, r["y"] - 3), (r["x"] + r["w"] + 3, r["y"] + r["h"] + 3), COLOR[r["status"]], 3 if bad else 1)
    for e in extras:
        cv2.rectangle(R, (e["x"] - 3, e["y"] - 3), (e["x"] + e["w"] + 3, e["y"] + e["h"] + 3), (255, 0, 255), 3)
    return R


def _imw(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)


def inspect(master_bgr, test_bgr, cfg=None, ignore_zones=None, debug_dir=None, force_continue=None):
    """
    คืน dict: verdict (OK / NG / RETAKE), report (โครงสร้างเดียวกับ glyph_check), overlay (BGR),
              aligned (BGR ภาพ Test หลังประมวลผล), notes, debug_dir
    """
    cfg = cfg or load_config()
    t0 = time.time()
    notes = []
    m, sc = _resize_long(master_bgr, cfg["proc_long_side"])
    mpre, minfo = preprocess(m, cfg)
    tpre, tinfo = preprocess(test_bgr, cfg)
    tpre_soft, _ = preprocess(test_bgr, cfg, sharpen=False)
    notes.append(f"Master ช่อง {minfo['channel']}{' (ตัวสว่าง)' if minfo['light_text'] else ''} | "
                 f"Test ช่อง {tinfo['channel']}{' (ตัวสว่าง)' if tinfo['light_text'] else ''}")
    mbin = binarize(mpre, cfg)
    roi = np.full(mbin.shape, 255, np.uint8)
    mink = mbin
    if cfg["roi_auto"]:
        roi, mink = mark_roi(mbin, cfg)
    qrs = find_qr(mpre, mbin, cfg) if cfg["skip_qr"] else []
    if qrs:
        notes.append(f"ข้าม QR code {len(qrs)} จุด")

    aligned, valid, ainfo = align(mpre, tpre, None, mink, cfg)
    if aligned is not None and ainfo["inliers"] < cfg["min_inliers"]:
        aligned = None
    if aligned is None:
        return dict(verdict="RETAKE", report=None, overlay=None, aligned=None, debug_dir=None,
                    notes=notes + [f"จับคู่ Test กับ Master ไม่ได้ (จุดอ้างอิง {ainfo['inliers']}) — คนละโมเดล / เบลอมาก / มาร์คไม่อยู่ในภาพ"])
    soft = cv2.warpPerspective(tpre_soft, ainfo["H"], (mpre.shape[1], mpre.shape[0]), borderValue=255)
    notes.append(f"จัดภาพ: {ainfo['method']} จุดอ้างอิง {ainfo['inliers']}{' กลับด้าน' if ainfo['mirrored'] else ''}")

    # ---- 0) quality gate (วัดบนพื้นที่มาร์คหลัง align -> ไม่ขึ้นกับขนาดภาพ)
    blurry = False
    area = cv2.bitwise_and(roi, valid)
    lv_t = lap_var(soft, area)
    lv_m = lap_var(cv2.GaussianBlur(mpre, (0, 0), 0.8), roi)
    ratio = lv_t / max(lv_m, 1e-6)
    mark_px = int(math.sqrt(abs(np.linalg.det(ainfo["H"][:2, :2]))) ** -1 * mpre.shape[1]) if ainfo["H"] is not None else 0
    notes.append(f"ความคม Test {lv_t:.0f} ({ratio:.0%} ของ Master) | มาร์คในภาพต้นฉบับกว้าง ~{mark_px}px")
    if cfg["blur_check"] and (lv_t < cfg["blur_lap_min"] or ratio < cfg["blur_ratio_min"]):
        blurry = True
    if mark_px and mark_px < cfg["min_mark_px"]:
        notes.append(f"⚠ ความละเอียดต่ำ: มาร์คกว้าง {mark_px}px < {cfg['min_mark_px']}px — ขยับกล้องเข้าใกล้ / ตั้งกล้องความละเอียดสูง")
    cont = cfg["continue_if_blur"] if force_continue is None else force_continue

    check = cv2.bitwise_and(roi, valid)
    for (x0, y0, x1, y1) in qrs:
        check[y0:y1, x0:x1] = 0
    for z in ignore_zones or []:
        zx, zy = int(z["x"] * sc), int(z["y"] * sc)
        check[zy:zy + int(z["h"] * sc), zx:zx + int(z["w"] * sc)] = 0
    if cfg.get("test_threshold") == "match_master":
        tbin = binarize_matched(aligned, mink, check, cfg)
    else:
        tbin = binarize(aligned, cfg)
    rows, extras, miss, extra = compare(mink, tbin, check, cfg)
    bad = [r for r in rows if r["status"] != "OK"]
    verdict = "NG" if (bad or extras) else "OK"
    if blurry:
        if cont:
            notes.append("⚠ ภาพเบลอ — ตรวจต่อตามที่ติ๊กไว้ (ผลอาจคลาดเคลื่อน)")
        else:
            verdict = "RETAKE"
            notes.append("⚠ ภาพเบลอ ถ่ายใหม่ (ไม่ตัดสิน NG)")
    if cfg["ocr"]:
        a, b = ocr_text(mpre), ocr_text(aligned)
        if a is None:
            notes.append("OCR: ไม่มี pytesseract — ข้าม")
        else:
            notes.append(f"OCR Master: {a[:60]} | Test: {b[:60]}")
    overlay = draw(aligned, rows, extras, qrs)
    report = dict(verdict="NG" if verdict == "NG" else ("OK" if verdict == "OK" else "RETAKE"),
                  align_method=ainfo["method"], align_inliers=ainfo["inliers"],
                  n_elements=len(rows), n_ok=len(rows) - len(bad),
                  n_missing=sum(r["status"] == "MISSING" for r in rows),
                  n_partial=sum(r["status"] == "PARTIAL" for r in rows),
                  n_skip=len(qrs), n_blur=int(blurry), n_dash=0, n_extra=len(extras),
                  elements=rows, extras=extras, engine="sharp_check v1", notes=notes,
                  secs=round(time.time() - t0, 2))

    dd = None
    if cfg["debug_save"] and debug_dir:
        dd = debug_dir
        os.makedirs(dd, exist_ok=True)
        _imw(os.path.join(dd, "d1_master_pre.png"), mpre)
        _imw(os.path.join(dd, "d2_test_pre_before_sharpen.png"), cv2.warpPerspective(tinfo["before_sharpen"], ainfo["H"], (mpre.shape[1], mpre.shape[0]), borderValue=255))
        _imw(os.path.join(dd, "d3_test_pre_after_sharpen_aligned.png"), aligned)
        _imw(os.path.join(dd, "d4_master_threshold.png"), 255 - mink)
        _imw(os.path.join(dd, "d5_test_threshold.png"), 255 - tbin)
        dm = np.full(mpre.shape + (3,), 255, np.uint8)
        dm[mink > 0] = (200, 200, 200)
        dm[check == 0] = (235, 235, 245)
        dm[miss > 0] = (0, 0, 255)
        dm[extra > 0] = (255, 0, 255)
        _imw(os.path.join(dd, "d6_diff_map.png"), dm)
        _imw(os.path.join(dd, "d7_result.png"), overlay)
        _imw(os.path.join(dd, "d8_before_after_sharpen.png"), before_after(tinfo["before_sharpen"], tpre))
        with open(os.path.join(dd, "d0_info.json"), "w", encoding="utf-8") as f:
            json.dump(dict(verdict=verdict, notes=notes, block_shifts=ainfo["block_shifts"], config=cfg),
                      f, indent=2, ensure_ascii=False)
    return dict(verdict=verdict, report=report, overlay=overlay, aligned=cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR),
                notes=notes, debug_dir=dd, before=tinfo["before_sharpen"], after=tpre)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="sharp_debug")
    a = ap.parse_args()
    rd = lambda p: cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    r = inspect(rd(a.master), rd(a.test), debug_dir=a.out)
    print("RESULT", r["verdict"])
    for n in r["notes"]:
        print("  •", n)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mark_match.py — preprocess / align / auto-crop (v7)"""
import numpy as np
import cv2


def remove_pen_ink(bgr, sat_thr=60, val_thr=40, chroma_thr=45):
    """
    ลบรอยปากกาสี (น้ำเงิน/แดง) ที่เขียนกำกับบนกระดาษเทส
    v7: ต้องเป็น "สีจัดจริง" (ผลต่างระหว่างช่องสี R,G,B ≥ chroma_thr) ด้วย
        เดิมดูแค่ Saturation -> หมึกดำในภาพถ่ายที่ติดโทนน้ำตาล/เหลืองเล็กน้อย
        มี Saturation สูง (เพราะมืด) จึงถูกลบทิ้งเหมือนปากกา -> ตัวอักษรขาดเป็นท่อน -> ฟ้องหายผิด
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    chroma = bgr.max(axis=2).astype(np.int16) - bgr.min(axis=2).astype(np.int16)
    colored = (hsv[:, :, 1] > sat_thr) & (hsv[:, :, 2] > val_thr) & (chroma >= chroma_thr)
    colored = cv2.morphologyEx(colored.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    colored = cv2.dilate(colored, np.ones((5, 5), np.uint8))
    out = bgr.copy()
    out[colored > 0] = (255, 255, 255)
    return out


def to_ink_mask(bgr, drop_pen=True, C=15, min_blob=8, valid=None):
    """
    แปลงเป็น mask หมึก (หมึก=255)
    v7: "ปรับพื้นหลังให้เรียบก่อน แล้วค่อยตัดด้วย Otsu"
        - ประมาณความสว่างพื้นหลังด้วย morphological closing ขนาดใหญ่ (ลบตัวอักษรออก)
        - หารภาพด้วยพื้นหลัง -> แสงไม่สม่ำเสมอ/เงา/กระดาษเทาหายไป
        - Otsu เลือกเกณฑ์ตัดเองทั้งภาพ -> หมึกสีเทาจาง ๆ ทั้งตัวอักษรถูกนับครบ
          (adaptive threshold เดิมทำให้ตัวอักษรเส้นบาง/หมึกเทาแตกเป็นท่อน -> ฟ้องหายผิด)
        valid = mask บริเวณที่มีภาพจริง (หลัง warp) — นอกบริเวณนี้ตัดทิ้ง
    """
    img = remove_pen_ink(bgr) if drop_pen else bgr
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = max(15, (min(g.shape[:2]) // 12) | 1)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (k, k), 0)
    norm = cv2.divide(g, np.maximum(bg, 1), scale=255)
    norm = cv2.GaussianBlur(norm, (3, 3), 0)
    sel = norm if valid is None else norm[valid > 0]
    thr, _ = cv2.threshold(sel.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = np.where(norm < thr, 255, 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if valid is not None:
        m[valid == 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_blob
    return np.where(keep[lab], 255, 0).astype(np.uint8)


def _orb_h(src_bgr, dst_bgr, nfeat=6000, ratio=0.75, min_match=12, thr=3.0):
    g1 = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeat, 1.2, 8)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < min_match or len(k2) < min_match:
        return None, 0, None, None
    raw = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [p[0] for p in raw if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < min_match:
        return None, len(good), None, None
    s = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    d = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(s, d, cv2.RANSAC, thr)
    if H is None:
        return None, len(good), None, None
    return H, int(inl.sum()), s, d


def align(test_bgr, master_bgr, return_valid=False):
    """
    จัดภาพ test ให้ซ้อน master
    v7: ทำ 2 รอบ (หยาบ -> ละเอียด) + ใช้ขอบแบบ replicate และคืน mask บริเวณที่มีภาพจริง
        เพื่อไม่ให้ขอบสีขาวจากการ warp กลายเป็น "หมึกเกิน" สีม่วงตามขอบภาพ
    """
    h, w = master_bgr.shape[:2]
    H, inl, _, _ = _orb_h(test_bgr, master_bgr)
    method = "ORB+RANSAC"
    if H is None or inl < 12:
        aligned = cv2.resize(test_bgr, (w, h))
        valid = np.full((h, w), 255, np.uint8)
        return (aligned, "RESIZE-ONLY(!)", 0, valid) if return_valid else (aligned, "RESIZE-ONLY(!)", 0)
    warped = cv2.warpPerspective(test_bgr, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # รอบ 2: จูนละเอียดด้วย ECC บนภาพที่ warp แล้ว (แก้คลาดเคลื่อน 1-3 px)
    try:
        g1 = cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32) / 255
        g2 = cv2.GaussianBlur(cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32) / 255
        W = np.eye(3, dtype=np.float32)
        _, W = cv2.findTransformECC(g2, g1, W, cv2.MOTION_HOMOGRAPHY,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5), None, 5)
        H2 = np.linalg.inv(W.astype(np.float64)) @ H
        # ยอมรับเฉพาะเมื่อปรับเพียงเล็กน้อย (กันกระโดดผิด)
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        shift = np.abs(cv2.perspectiveTransform(corners, np.linalg.inv(W.astype(np.float64)))
                       - corners).max()
        if shift < max(w, h) * 0.03:
            H = H2
            method = "ORB+ECC"
            warped = cv2.warpPerspective(test_bgr, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
    except cv2.error:
        pass
    valid = cv2.warpPerspective(np.full(test_bgr.shape[:2], 255, np.uint8), H, (w, h))
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
    if return_valid:
        return warped, method, inl, valid
    return warped, method, inl


def auto_detect_crop(master_bgr, test_bgr, pad_ratio=0.12, min_match=15):
    """หาบริเวณใน Master (เอกสารทั้งหน้า) ที่ตรงกับภาพ Test -> (x, y, w, h), n"""
    H, inl, s, d = _orb_h(test_bgr, master_bgr, nfeat=8000, min_match=min_match, thr=5.0)
    if H is None:
        return None, inl
    tw, th = test_bgr.shape[1], test_bgr.shape[0]
    c = cv2.perspectiveTransform(np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2), H)
    x0, y0 = c.reshape(-1, 2).min(axis=0)
    x1, y1 = c.reshape(-1, 2).max(axis=0)
    Hh, Ww = master_bgr.shape[:2]
    px, py = (x1 - x0) * pad_ratio, (y1 - y0) * pad_ratio
    x0, y0 = max(0, int(x0 - px)), max(0, int(y0 - py))
    x1, y1 = min(Ww, int(x1 + px)), min(Hh, int(y1 + py))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None, inl
    return (x0, y0, x1 - x0, y1 - y0), inl

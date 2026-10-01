#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mark_match.py - โมดูลพื้นฐาน: preprocess / align"""
import numpy as np
import cv2


def remove_pen_ink(bgr, sat_thr=60, val_thr=40):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    colored = (hsv[:, :, 1] > sat_thr) & (hsv[:, :, 2] > val_thr)
    colored = cv2.morphologyEx(colored.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    colored = cv2.dilate(colored, np.ones((5, 5), np.uint8))
    out = bgr.copy()
    out[colored > 0] = (255, 255, 255)
    return out


def to_ink_mask(bgr, drop_pen=True, block=51, C=15, min_blob=8):
    img = remove_pen_ink(bgr) if drop_pen else bgr
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    g = cv2.bilateralFilter(g, 7, 50, 50)
    m = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, block | 1, C)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_blob:
            out[lab == i] = 255
    return out


def align_orb(test_bgr, master_bgr, nfeat=5000, ratio=0.75, min_match=12):
    g1 = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeat, 1.2, 8)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < min_match or len(k2) < min_match:
        return None, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < ratio * n.distance]
    if len(good) < min_match:
        return None, len(good)
    s = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    d = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(s, d, cv2.RANSAC, 3.0)
    if H is None:
        return None, len(good)
    return H, int(inl.sum())


def align_ecc(test_bgr, master_bgr, iters=300, eps=1e-6):
    g1 = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    g2 = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        _, warp = cv2.findTransformECC(g2, g1, warp, cv2.MOTION_AFFINE, crit, None, 5)
    except cv2.error:
        return None
    return np.vstack([warp, [0, 0, 1]]).astype(np.float64)


def align(test_bgr, master_bgr):
    h, w = master_bgr.shape[:2]
    H, inl = align_orb(test_bgr, master_bgr)
    if H is not None and inl >= 12:
        return cv2.warpPerspective(test_bgr, H, (w, h), borderValue=(255, 255, 255)), "ORB+RANSAC", inl
    H = align_ecc(test_bgr, master_bgr)
    if H is not None:
        return cv2.warpPerspective(test_bgr, H, (w, h), borderValue=(255, 255, 255)), "ECC-affine", -1
    return cv2.resize(test_bgr, (w, h)), "RESIZE-ONLY(!)", 0

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_autofind.py v3 — หา "มาร์ค" ในไฟล์ PDF (Order screen / APP Film) อัตโนมัติ
แล้ว zoom + กลับด้าน (mirror) + หมุน ให้ตรงกับภาพ Master
v3:
  - แก้ "เจอมาร์คไม่เต็ม" (เช่น OK9 C857 Order screen): เดิมตัดภาพ Test เท่ากรอบ Master พอดี
      -> มาร์คใน PDF ที่ "ใหญ่/เรียงต่างจาก Master" (มี S27413, เลขพาร์ท, QR code) ถูกตัดหัว-ท้ายหาย
    ใหม่: หากลุ่มตัวอักษรทั้งก้อนของมาร์ค แล้วขยายภาพ Test ให้ครอบมาร์คเต็ม (ทิศ/ขนาดเหมือน Master)
  - ตรวจ "Layout ต่างจาก Master": บอก % หมึกที่อยู่นอกกรอบ Master และ % ของ Master ที่ไม่เจอใน PDF
      -> r["layout_note"], r["layout_diff"] (True = ควรเช็ค Master ว่ารุ่น/Rev. เดียวกับ PDF)
  - ยังคืน r["test_window"] = ภาพขนาดเท่ากรอบ Master (แบบ v2) ไว้ให้ใช้ได้
  - หน้าที่มีมาร์ค 2 ตัว (ซ้าย/ขวา) ไม่รวมมาร์คตัวที่ 2 เข้ามา
v2:
  - หามาร์คตัวเล็กมากบนหน้าใหญ่ได้ (หา "กลุ่มตัวหนังสือ" แล้ว render เฉพาะกลุ่ม)
  - หน้าที่มีมาร์คหลายตำแหน่ง -> เลือกตำแหน่งที่ตรงกับ Master มากที่สุด
"""
import argparse
import math
import numpy as np
import cv2
try:
    import pymupdf as fitz
except Exception:
    import fitz

VERSION = "3"


# ---------------------------------------------------------------- render
def render_pdf(path, page_no=0, long_side=3000, clip=None, gray=False):
    """คืน (ภาพ BGR หรือเทา (gray=True), scale px/pt, (x0,y0) ของ clip เป็น pt)"""
    doc = fitz.open(path)
    try:
        page = doc[page_no]
        r = page.rect if clip is None else (fitz.Rect(clip) & page.rect)
        if r.is_empty or r.width < 1 or r.height < 1:
            raise ValueError("บริเวณที่ตัดว่างเปล่า")
        s = long_side / max(r.width, r.height)
        if gray:
            pix = page.get_pixmap(matrix=fitz.Matrix(s, s), clip=r, alpha=False, colorspace=fitz.csGRAY)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
            return img, s, (r.x0, r.y0)
        pix = page.get_pixmap(matrix=fitz.Matrix(s, s), clip=r, alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n).copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_GRAY2BGR)
        return img, s, (r.x0, r.y0)
    finally:
        doc.close()


def _render_scale(path, page_no, scale, clip):
    """render ที่ scale (px/pt) กำหนด — คืน (ภาพ, (x0,y0) ของ clip เป็น pt)"""
    doc = fitz.open(path)
    try:
        page = doc[page_no]
        r = fitz.Rect(clip) & page.rect
        if r.is_empty:
            raise ValueError("บริเวณที่ตัดว่างเปล่า")
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=r, alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n).copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_GRAY2BGR)
        return img, (r.x0, r.y0)
    finally:
        doc.close()


def page_size(path, page_no=0):
    doc = fitz.open(path)
    try:
        r = doc[page_no].rect
        return r.width, r.height
    finally:
        doc.close()


# ---------------------------------------------------------------- matching
def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _clean_master(master_bgr):
    """ลบเส้นประ/กรอบสีแดงใน Master (ถ้ามี) ก่อนใช้หาจุดเด่น"""
    b, g, r = [c.astype(np.int16) for c in cv2.split(master_bgr)]
    red = (r - np.maximum(g, b)) > 40
    out = master_bgr.copy()
    out[cv2.dilate(red.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (255, 255, 255)
    return out


def _detector():
    try:
        return cv2.SIFT_create(nfeatures=8000), cv2.NORM_L2
    except Exception:
        return cv2.ORB_create(8000), cv2.NORM_HAMMING


def _match(master_g, page_g, det, norm, ratio=0.75):
    km, dm = det.detectAndCompute(master_g, None)
    kp, dp = det.detectAndCompute(page_g, None)
    if dm is None or dp is None or len(km) < 8 or len(kp) < 8:
        return None, 0
    good = [p[0] for p in cv2.BFMatcher(norm).knnMatch(dm, dp, k=2)
            if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < 8:
        return None, 0
    src = np.float32([km[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0,
                                         maxIters=5000, confidence=0.999)
    if A is None or inl is None:
        return None, 0
    if math.hypot(A[0, 0], A[1, 0]) < 1e-3:          # ขนาดเป็นศูนย์ = จับคู่ผิด
        return None, 0
    return A, int(inl.sum())


def find_mark(page_bgr, master_bgr, min_inliers=12):
    """
    คืน dict(test, mirrored, angle, scale, inliers, box, M) หรือ None
    test = ภาพจาก page ที่บิดให้ซ้อนทับ Master พอดี (ขนาดเท่า Master)
    M    = 3x3 พิกัด Master -> พิกัด page
    """
    det, norm = _detector()
    mh, mw = master_bgr.shape[:2]
    pg = _gray(page_bgr)
    mg = _gray(_clean_master(master_bgr))
    best = None
    for mirrored in (False, True):
        mg2 = cv2.flip(mg, 1) if mirrored else mg
        A, n = _match(mg2, pg, det, norm)
        if A is not None and (best is None or n > best[2]):
            best = (mirrored, A, n)
    if best is None or best[2] < min_inliers:
        return None
    mirrored, A, n = best
    F = np.array([[-1, 0, mw - 1], [0, 1, 0], [0, 0, 1]], np.float64) if mirrored else np.eye(3)
    M = np.vstack([A, [0, 0, 1]]) @ F
    Minv = np.linalg.inv(M)
    test = cv2.warpAffine(page_bgr, Minv[:2], (mw, mh), flags=cv2.INTER_AREA, borderValue=(255, 255, 255))
    corners = cv2.transform(np.float32([[0, 0], [mw, 0], [mw, mh], [0, mh]]).reshape(-1, 1, 2), M[:2])
    return dict(test=test, mirrored=mirrored, angle=round(math.degrees(math.atan2(A[1, 0], A[0, 0])), 1),
                scale=round(math.hypot(A[0, 0], A[1, 0]), 3), inliers=n, box=corners.reshape(-1, 2), M=M)


# ---------------------------------------------------------------- หา "กลุ่มตัวหนังสือ" บนหน้า
def _text_mask(g, big):
    """หมึกที่เป็นตัวอักษรเล็ก ๆ (ตัดของใหญ่ + จุด dot pattern ออก) -> (mask, ความสูงตัวอักษรกลาง)"""
    ink = (g < 150).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    del ink
    a = st[:, cv2.CC_STAT_AREA]
    w_, h_ = st[:, 2], st[:, 3]
    small = np.zeros(n, bool)
    small[1:] = (a[1:] >= 4) & (w_[1:] < big) & (h_[1:] < big)
    dotlike = (np.abs(w_ - h_) <= 1) & (a >= 0.6 * w_ * h_)
    small &= ~dotlike
    mask = small[lab].view(np.uint8) * np.uint8(255)
    hs = h_[1:][small[1:]]
    ch = float(np.median(hs)) if hs.size else 0.0
    return mask, lab, ch


def text_candidates(page_bgr, max_cand=12):
    """
    หาบริเวณที่มี "ตัวอักษรเล็ก ๆ หลายตัวอยู่รวมกัน" (= มาร์ค) บนหน้า PDF
    คืน list ของ (x, y, w, h) หน่วยพิกเซลของภาพ page_bgr  เรียงจากน่าจะใช่มากสุด
    v3: ใช้หน่วยความจำน้อยลง (นับตัวอักษรต่อกลุ่มด้วย bincount แทนการวนทั้งภาพ)
    """
    g = _gray(page_bgr)
    H, W = g.shape
    big = max(W, H) * 0.02                         # ตัวอักษรต้องเล็กกว่า 2% ของหน้า
    mask, lab, _ = _text_mask(g, big)
    del g
    k = max(5, int(max(W, H) * 0.006)) | 1
    blob = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m2, lab2, st2, _ = cv2.connectedComponentsWithStats(blob, 8)
    del blob
    sel = mask > 0
    a_ = lab2[sel].astype(np.int64)
    b_ = lab[sel].astype(np.int64)
    del lab, lab2, sel, mask
    pairs = np.unique(a_ * (int(b_.max()) + 1 if b_.size else 1) + b_) if a_.size else np.zeros(0, np.int64)
    counts = np.bincount(pairs // (int(b_.max()) + 1 if b_.size else 1), minlength=m2) if pairs.size else np.zeros(m2, int)
    del a_, b_, pairs
    cands = []
    for i in range(1, m2):
        x, y, w, h, area = st2[i]
        if w < 10 or h < 10 or max(w, h) > max(W, H) * 0.25:
            continue
        cnt = int(counts[i])                                   # จำนวนตัวอักษรในกลุ่ม
        if cnt < 8:
            continue
        cands.append((cnt, (int(x), int(y), int(w), int(h))))
    cands.sort(key=lambda c: -c[0])
    return [b for _, b in cands[:max_cand]]


# ---------------------------------------------------------------- v3: มาร์คเต็มก้อน
def _master_char_h(master_bgr):
    g = _gray(_clean_master(master_bgr))
    ink = (g < 150).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    h = st[1:, 3]
    h = h[(h > 3) & (h < g.shape[0] * 0.3)]
    return float(np.median(h)) if h.size else g.shape[0] / 12.0


def _full_mark(reg, r, master_bgr):
    """
    reg = ภาพ PDF รอบมาร์ค (กว้างกว่ากรอบ Master), r = ผล find_mark บน reg
    คืน (test_full, info)  test_full = ภาพทิศเดียวกับ Master ครอบมาร์คทั้งก้อน
    """
    mh, mw = master_bgr.shape[:2]
    M = r["M"]
    Minv = np.linalg.inv(M)
    sc = float(r["scale"])
    g = _gray(reg)
    H, W = g.shape
    ch = _master_char_h(master_bgr) * sc                       # ความสูงตัวอักษร (px ของ reg)
    mask, _, _ = _text_mask(g, max(ch * 6, 12))                # ของใหญ่กว่าตัวอักษร 6 เท่า = แผ่นสีดำ/เส้นยาว
    # เส้นสี (ลูกศร/ระยะ/กรอบแดง/เขียว ของแบบงาน) ไม่ใช่มาร์ค
    b16 = reg.astype(np.int16)
    colored = (b16.max(axis=2) - b16.min(axis=2)) > 60
    mask[colored] = 0
    # QR code = ก้อนหมึกใหญ่ — เติมกลับ ถ้าอยู่ติดกลุ่มตัวอักษร (ใช้กล่องสี่เหลี่ยมจัตุรัส)
    ink = (g < 150).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if ch * 2 < max(w, h) < ch * 12 and 0.75 < w / max(h, 1) < 1.33 and a > 0.25 * w * h:
            mask[lab == i] = 255
    # รวมตัวอักษรเป็นบรรทัด/ก้อน: ระยะห่างไม่เกิน ~1.3 เท่าความสูงตัวอักษร
    k = max(3, int(ch * 1.3)) | 1
    blob = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # เริ่มจากก้อนที่ทับกรอบ Master (ที่จับคู่ได้) แล้วรวมก้อนที่ติดกัน
    win = np.zeros_like(blob)
    cv2.fillConvexPoly(win, r["box"].astype(np.int32), 255)
    nb, lb, sb, _ = cv2.connectedComponentsWithStats(blob, 8)
    keep = np.zeros(nb, bool)
    for i in range(1, nb):
        ov = int(((lb == i) & (win > 0)).sum())
        if ov > 0.02 * sb[i, 4] or ov > ch * ch:
            keep[i] = True
    if not keep.any():
        return r["test"], dict(full=False)
    sel = keep[lb] & (mask > 0)
    ys, xs = np.nonzero(sel)
    pts = np.float32(np.c_[xs, ys]).reshape(-1, 1, 2)
    mp = cv2.transform(pts, Minv[:2]).reshape(-1, 2)            # พิกัด Master
    # จำกัดไม่ให้ไกลเกิน 1.2 เท่ากรอบ Master (กันมาร์คอีกตัว / ตัวหนังสือแบบงาน)
    lim = (mp[:, 0] > -1.2 * mw) & (mp[:, 0] < 2.2 * mw) & (mp[:, 1] > -1.2 * mh) & (mp[:, 1] < 2.2 * mh)
    mp = mp[lim]
    if mp.size == 0:
        return r["test"], dict(full=False)
    x0 = min(0.0, float(np.percentile(mp[:, 0], 0.2)))
    y0 = min(0.0, float(np.percentile(mp[:, 1], 0.2)))
    x1 = max(float(mw), float(np.percentile(mp[:, 0], 99.8)))
    y1 = max(float(mh), float(np.percentile(mp[:, 1], 99.8)))
    pad = 0.04 * max(mw, mh)
    x0 = x0 - pad if x0 < 0 else 0.0
    y0 = y0 - pad if y0 < 0 else 0.0
    x1 = x1 + pad if x1 > mw else float(mw)
    y1 = y1 + pad if y1 > mh else float(mh)
    ox, oy = int(round(-x0)), int(round(-y0))
    FW, FH = int(round(x1 - x0)), int(round(y1 - y0))
    T = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], np.float64)
    full = cv2.warpAffine(reg, (T @ Minv)[:2], (FW, FH), flags=cv2.INTER_AREA, borderValue=(255, 255, 255))
    f16 = full.astype(np.int16)
    full[(f16.max(axis=2) - f16.min(axis=2)) > 60] = (255, 255, 255)   # ลบเส้นสีของแบบงาน (ลูกศร/ระยะ)
    del f16
    # layout: หมึกนอกกรอบ Master / Master ที่ไม่เจอใน PDF
    fi = (_gray(full) < 150)
    inside = np.zeros_like(fi)
    inside[oy:oy + mh, ox:ox + mw] = True
    tot = max(1, int(fi.sum()))
    out_ratio = float((fi & ~inside).sum()) / tot
    mg = _gray(_clean_master(master_bgr)) < 150
    tw = fi[oy:oy + mh, ox:ox + mw]
    tol = cv2.dilate(tw.astype(np.uint8), np.ones((max(3, int(ch * 0.25)) | 1,) * 2, np.uint8)) > 0
    miss_ratio = float((mg & ~tol).sum()) / max(1, int(mg.sum()))
    return full, dict(full=True, offset=(ox, oy), out_ratio=round(out_ratio, 3),
                      miss_ratio=round(miss_ratio, 3), size=(FW, FH))


# ---------------------------------------------------------------- หลัก
def find_mark_in_pdf(pdf_path, master_bgr, page_no=0, full_mark=True):
    """
    1) หากลุ่มตัวหนังสือทั่วหน้า -> render แต่ละกลุ่มละเอียด -> จับคู่ -> เลือกกลุ่มที่ดีที่สุด
    2) สำรอง: ทั้งหน้า
    3) จูนละเอียดรอบตำแหน่งที่เจอ (render กว้างกว่ากรอบ Master เพื่อให้ได้มาร์คเต็มก้อน)
    """
    mh, mw = master_bgr.shape[:2]
    best = None          # (result, scale, offset)
    GOOD = 60            # จุดอ้างอิงพอแล้ว -> หยุดค้นต่อ (ประหยัดเวลาบน Pi)
    # --- 1) กลุ่มตัวหนังสือทั่วหน้า
    page, s, off = render_pdf(pdf_path, page_no, 7000, gray=True)
    cands = text_candidates(page, max_cand=8)
    del page
    for (x, y, w, h) in cands:
        pad = 0.6 * max(w, h)
        clip = (off[0] + (x - pad) / s, off[1] + (y - pad) / s,
                off[0] + (x + w + pad) / s, off[1] + (y + h + pad) / s)
        try:
            reg, s2, off2 = render_pdf(pdf_path, page_no, 2000, clip)
        except ValueError:
            continue
        rr = find_mark(reg, master_bgr)
        if rr is not None and (best is None or rr["inliers"] > best[0]["inliers"]):
            best = (rr, s2, off2)
            if rr["inliers"] >= GOOD * 2:
                break
    # --- 2) สำรอง: ทั้งหน้า
    if best is None or best[0]["inliers"] < GOOD:
        page, s, off = render_pdf(pdf_path, page_no, 2200)    # v3: 3000 -> 2200 (SIFT ใช้หน่วยความจำ ~ครึ่งเดียว)
        r = find_mark(page, master_bgr)
        del page
        if r is not None and (best is None or r["inliers"] > best[0]["inliers"]):
            best = (r, s, off)
    if best is None:
        raise RuntimeError(
            "หามาร์คใน PDF ไม่เจอ\n"
            "- เช็คว่า Master เป็น 'โมเดลเดียวกับ PDF' (เลข 43R-…, รหัสรุ่น เช่น J0/T1 ต้องตรงกัน)\n"
            "- PDF บางแบบมีมาร์คไม่ครบ (เช่น มีแค่โลโก้ + 1 2 4 8) จะเทียบกับ Master เต็มไม่ได้")
    # --- 3) จูนละเอียด: render รอบตำแหน่งที่เจอ ให้มาร์คใหญ่ ~2 เท่าของ Master (เหมือน v2)
    r, s, off = best
    box_pt = r["box"] / s + np.array(off)
    x0, y0 = box_pt.min(axis=0)
    x1, y1 = box_pt.max(axis=0)
    span = max(x1 - x0, y1 - y0, 1.0)
    pad = 0.15 * span
    clip = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    try:
        reg, s3, off3 = render_pdf(pdf_path, page_no, int(min(6000, max(800, 2.0 * max(mw, mh) * 1.3))), clip)
        fine = find_mark(reg, master_bgr)
        del reg
        if fine is not None and fine["inliers"] >= r["inliers"] * 0.5:
            r, s, off = fine, s3, off3
    except ValueError:
        pass
    r["coarse_inliers"] = best[0]["inliers"]
    r["quality"] = "ดี" if r["inliers"] >= 60 else ("พอใช้" if r["inliers"] >= 25 else "ต่ำ — ควรเช็คว่าโมเดลตรงกัน")
    r["test_window"] = r["test"]
    r["layout_note"] = ""
    r["layout_diff"] = False
    if full_mark:
        # v3: render กว้างขึ้น (ไม่ทำ SIFT ซ้ำ — ประหยัดหน่วยความจำบน Pi) ที่ความละเอียดเดียวกัน
        #     แล้วเลื่อนพิกัดให้ตรง -> หามาร์คทั้งก้อน
        try:
            box_pt = r["box"] / s + np.array(off)
            x0, y0 = box_pt.min(axis=0)
            x1, y1 = box_pt.max(axis=0)
            span = max(x1 - x0, y1 - y0, 1.0)
            wpad = 0.8 * span
            wide, off4 = _render_scale(pdf_path, page_no, s, (x0 - wpad, y0 - wpad, x1 + wpad, y1 + wpad))
            dx, dy = (off[0] - off4[0]) * s, (off[1] - off4[1]) * s
            rw = dict(r)
            rw["M"] = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64) @ r["M"]
            rw["box"] = r["box"] + np.array([dx, dy])
            full, info = _full_mark(wide, rw, master_bgr)
            del wide
        except RuntimeError:
            raise
        except Exception as ex:                       # ไม่ให้ทั้งระบบล้มเพราะส่วนเสริม
            full, info = r["test"], dict(full=False, err=str(ex))
        r["full_info"] = info
        if info.get("full"):
            r["test"] = full
            o, m = info["out_ratio"], info["miss_ratio"]
            grown = info["size"] != (mw, mh)
            if o > 0.08 or m > 0.10:
                r["layout_diff"] = True
                r["layout_note"] = (f"⚠ มาร์คใน PDF ต่างจาก Master: หมึกนอกกรอบ Master {o*100:.0f}% | "
                                    f"Master ไม่เจอใน PDF {m*100:.0f}% — เช็ค Master รุ่น/Rev. เดียวกับ PDF")
            elif grown:
                r["layout_note"] = "ขยายภาพให้เห็นมาร์คเต็ม"
            if r["inliers"] < 25 and m > 0.6:
                # จุดอ้างอิงน้อย + Master เกือบทั้งหมดไม่อยู่ในภาพ = จับคู่ผิดที่ (เช่น หน้า OUTER ไม่มีมาร์ค)
                raise RuntimeError(
                    f"หามาร์คใน PDF ไม่เจอ (จุดที่ใกล้สุดตรงกับ Master แค่ {100 - m * 100:.0f}%, จุดอ้างอิง {r['inliers']})\n"
                    "- PDF นี้อาจไม่มีมาร์ค (เช่น หน้า OUTER) หรือ Master คนละโมเดล/คนละ Rev.\n"
                    "- เช็คเลข 43R-…, รหัสรุ่น (J0/T1/P0) ให้ตรงกับ PDF")
    r.pop("M", None)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf"); ap.add_argument("master")
    ap.add_argument("--out", default="test_from_pdf.png")
    a = ap.parse_args()
    master = cv2.imdecode(np.fromfile(a.master, np.uint8), cv2.IMREAD_COLOR)
    r = find_mark_in_pdf(a.pdf, master)
    cv2.imencode(".png", r["test"])[1].tofile(a.out)
    print(f"OK  mirror={r['mirrored']}  angle={r['angle']}°  scale={r['scale']}  "
          f"inliers={r['inliers']}  quality={r['quality']}  {r['layout_note']}  -> {a.out}")


if __name__ == "__main__":
    main()

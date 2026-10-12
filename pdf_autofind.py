#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_autofind.py v3 — หา "มาร์ค" ในไฟล์ PDF (Order screen / APP Film) อัตโนมัติ
แล้ว zoom + กลับด้าน (mirror) + หมุน ให้ตรงกับภาพ Master

v2: หามาร์คตัวเล็กบนหน้าใหญ่ (หา "กลุ่มตัวหนังสือ" แล้วจับคู่ทีละกลุ่ม), แก้ division by zero
v3: เผื่อขอบรอบภาพ Test (margin) — ตัวอักษรที่อยู่ชิดขอบ Master (เช่น "V" ด้านซ้าย)
    ไม่ถูกตัดขาดอีก  (เดิมตัดภาพ Test พอดีขนาด Master -> ตัวริมขอบโดนตัดครึ่ง)
    การตรวจเดิมจะจัดตำแหน่ง (align) ภาพ Test เข้าหา Master เองอยู่แล้ว
"""
import argparse
import math

import numpy as np
import cv2

try:
    import pymupdf as fitz
except Exception:
    import fitz

MARGIN = 0.12          # เผื่อขอบรอบภาพ Test 12% ของขนาด Master ทุกด้าน


# ---------------------------------------------------------------- render
def render_pdf(path, page_no=0, long_side=3000, clip=None):
    """คืน (ภาพ BGR, scale px/pt, (x0,y0) ของ clip เป็น pt)"""
    doc = fitz.open(path)
    try:
        page = doc[page_no]
        r = page.rect if clip is None else (fitz.Rect(clip) & page.rect)
        if r.is_empty or r.width < 1 or r.height < 1:
            raise ValueError("บริเวณที่ตัดว่างเปล่า")
        s = long_side / max(r.width, r.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(s, s), clip=r, alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n).copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_GRAY2BGR)
        return img, s, (r.x0, r.y0)
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
    if math.hypot(A[0, 0], A[1, 0]) < 1e-3:
        return None, 0
    return A, int(inl.sum())


def find_mark(page_bgr, master_bgr, min_inliers=12, margin=MARGIN):
    """
    คืน dict(test, mirrored, angle, scale, inliers, box, margin_px) หรือ None
    test = ภาพจาก page ที่บิดให้ตรงทิศ/ขนาด Master + เผื่อขอบ margin ทุกด้าน
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
    M = np.vstack([A, [0, 0, 1]]) @ F                   # master -> page
    mx, my = int(round(mw * margin)), int(round(mh * margin))
    T = np.array([[1, 0, -mx], [0, 1, -my], [0, 0, 1]], np.float64)   # ภาพผลลัพธ์ -> master (เลื่อนเผื่อขอบ)
    Minv = np.linalg.inv(M @ T)                          # page -> ภาพผลลัพธ์
    test = cv2.warpAffine(page_bgr, Minv[:2], (mw + 2 * mx, mh + 2 * my),
                          flags=cv2.INTER_AREA, borderValue=(255, 255, 255))
    corners = cv2.transform(np.float32([[0, 0], [mw, 0], [mw, mh], [0, mh]]).reshape(-1, 1, 2), M[:2])
    return dict(test=test, mirrored=mirrored, angle=round(math.degrees(math.atan2(A[1, 0], A[0, 0])), 1),
                scale=round(math.hypot(A[0, 0], A[1, 0]), 3), inliers=n, box=corners.reshape(-1, 2),
                margin_px=(mx, my))


# ---------------------------------------------------------------- หา "กลุ่มตัวหนังสือ" บนหน้า
def text_candidates(page_bgr, max_cand=12):
    """บริเวณที่มีตัวอักษรเล็ก ๆ หลายตัวอยู่รวมกัน (= มาร์ค) — คืน list (x, y, w, h) เป็นพิกเซล"""
    g = _gray(page_bgr)
    H, W = g.shape
    ink = (g < 150).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    a = st[:, cv2.CC_STAT_AREA]
    w_, h_ = st[:, 2], st[:, 3]
    big = max(W, H) * 0.02
    small = np.zeros(n, bool)
    small[1:] = (a[1:] >= 4) & (w_[1:] < big) & (h_[1:] < big)
    dotlike = (np.abs(w_ - h_) <= 1) & (a >= 0.6 * w_ * h_)      # dot pattern ขอบกระจก
    small &= ~dotlike
    mask = np.where(small[lab], 255, 0).astype(np.uint8)
    k = max(5, int(max(W, H) * 0.006)) | 1
    blob = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m2, lab2, st2, _ = cv2.connectedComponentsWithStats(blob, 8)
    cands = []
    for i in range(1, m2):
        x, y, w, h, _a = st2[i]
        if w < 10 or h < 10 or max(w, h) > max(W, H) * 0.25:
            continue
        cnt = len(np.unique(lab[(lab2 == i) & (mask > 0)]))
        if cnt < 8:
            continue
        cands.append((cnt, (int(x), int(y), int(w), int(h))))
    cands.sort(key=lambda c: -c[0])
    return [b for _, b in cands[:max_cand]]


# ---------------------------------------------------------------- หลัก
def find_mark_in_pdf(pdf_path, master_bgr, page_no=0):
    mh, mw = master_bgr.shape[:2]
    best = None
    GOOD = 60

    # 1) กลุ่มตัวหนังสือทั่วหน้า
    page, s, off = render_pdf(pdf_path, page_no, 7000)
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

    # 2) สำรอง: ทั้งหน้า
    if best is None or best[0]["inliers"] < GOOD:
        page, s, off = render_pdf(pdf_path, page_no, 3000)
        r = find_mark(page, master_bgr)
        del page
        if r is not None and (best is None or r["inliers"] > best[0]["inliers"]):
            best = (r, s, off)

    if best is None:
        raise RuntimeError(
            "หามาร์คใน PDF ไม่เจอ\n"
            "- เช็คว่า Master เป็น 'โมเดลเดียวกับ PDF' (เลข 43R-…, รหัสรุ่น ต้องตรงกัน)\n"
            "- PDF บางแบบมีมาร์คไม่ครบ จะเทียบกับ Master เต็มไม่ได้")

    # 3) จูนละเอียด: render รอบตำแหน่งที่เจอ (เผื่อขอบกว้างพอสำหรับ margin)
    r, s, off = best
    box_pt = r["box"] / s + np.array(off)
    x0, y0 = box_pt.min(axis=0)
    x1, y1 = box_pt.max(axis=0)
    span = max(x1 - x0, y1 - y0, 1.0)
    pad = (0.15 + MARGIN) * span
    clip = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    try:
        reg, s3, off3 = render_pdf(pdf_path, page_no, int(min(6000, max(800, 2.0 * max(mw, mh) * 1.6))), clip)
        fine = find_mark(reg, master_bgr)
        if fine is not None and fine["inliers"] >= r["inliers"] * 0.5:
            r = fine
    except ValueError:
        pass
    r["coarse_inliers"] = best[0]["inliers"]
    r["quality"] = "ดี" if r["inliers"] >= 60 else ("พอใช้" if r["inliers"] >= 25 else "ต่ำ — ควรเช็คว่าโมเดลตรงกัน")
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
          f"inliers={r['inliers']}  quality={r['quality']}  -> {a.out}")


if __name__ == "__main__":
    main()

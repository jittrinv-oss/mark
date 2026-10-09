#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_autofind.py — หา "มาร์ค" ในไฟล์ PDF แบบงานจริง (Order screen) อัตโนมัติ
แล้ว zoom + กลับด้าน (mirror) + หมุน ให้ตรงกับภาพ Master

เหตุผล: PDF ของสกรีนเป็นภาพจากด้านพิมพ์ -> มาร์ค "กลับด้าน" และ "เอียง" ตามรูปกระจก
         ส่วน Master เป็นภาพมาร์คตรง อ่านได้ปกติ

วิธีทำ (อัตโนมัติทั้งหมด):
  1) render หน้า PDF ความละเอียดสูง
  2) หาจุดเด่น (SIFT) ทั้งใน Master และ PDF — ลองทั้ง "ปกติ" และ "กลับด้าน"
  3) จับคู่ + RANSAC -> ได้ตำแหน่ง มุมหมุน ขนาด ของมาร์คใน PDF
  4) เลือกแบบที่จับคู่ได้มากที่สุด (บอกได้ว่ากลับด้านหรือไม่ หมุนกี่องศา)
  5) บิดภาพ PDF ให้ซ้อนทับ Master พอดี -> ได้ภาพ Test ขนาด/ทิศเดียวกับ Master

ใช้เดี่ยว ๆ:
  python pdf_autofind.py "Order screen C857.pdf" "C857 master mark.png" --out test_from_pdf.png
"""
import argparse
import math

import numpy as np
import cv2

try:
    import pymupdf as fitz
except Exception:
    import fitz


def render_pdf(path, page_no=0, long_side=3000, clip=None):
    """render หน้า PDF (หรือเฉพาะบริเวณ clip ในหน่วย pt) — คืน (ภาพ, scale px/pt, จุดเริ่ม clip)"""
    doc = fitz.open(path)
    try:
        page = doc[page_no]
        r = page.rect if clip is None else (fitz.Rect(clip) & page.rect)
        s = long_side / max(r.width, r.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(s, s), clip=r, alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n).copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_GRAY2BGR)
        return img, s, (r.x0, r.y0)
    finally:
        doc.close()


def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _detector():
    try:
        return cv2.SIFT_create(nfeatures=8000), cv2.NORM_L2
    except Exception:                         # OpenCV รุ่นเก่ามากไม่มี SIFT
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
    # similarity (หมุน + ย่อขยาย + เลื่อน) — เหมาะกับงานพิมพ์ ไม่บิดเบี้ยว
    A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0,
                                         maxIters=5000, confidence=0.999)
    if A is None:
        return None, 0
    return A, int(inl.sum())


def find_mark(page_bgr, master_bgr, min_inliers=12):
    """
    คืน dict(test=ภาพที่ตรงกับ Master, mirrored, angle, scale, inliers, box=4 มุมในหน้า PDF)
    หรือ None ถ้าหาไม่เจอ
    """
    det, norm = _detector()
    mh, mw = master_bgr.shape[:2]
    # ย่อหน้า PDF ให้มาร์คมีขนาดใกล้ Master คร่าว ๆ ก่อน (เร็วขึ้น) แล้วคำนวณกลับ
    pg = _gray(page_bgr)
    mg = _gray(master_bgr)
    best = None
    for mirrored in (False, True):
        mg2 = cv2.flip(mg, 1) if mirrored else mg
        A, n = _match(mg2, pg, det, norm)
        if A is not None and (best is None or n > best[2]):
            best = (mirrored, A, n)
    if best is None or best[2] < min_inliers:
        return None
    mirrored, A, n = best
    # A: master(ที่อาจถูก flip) -> page   =>   รวม flip เข้าไป: master -> page
    F = np.array([[-1, 0, mw - 1], [0, 1, 0], [0, 0, 1]], np.float64) if mirrored else np.eye(3)
    M = np.vstack([A, [0, 0, 1]]) @ F
    Minv = np.linalg.inv(M)
    test = cv2.warpAffine(page_bgr, Minv[:2], (mw, mh), flags=cv2.INTER_AREA,
                          borderValue=(255, 255, 255))
    scale = math.hypot(A[0, 0], A[1, 0])
    angle = math.degrees(math.atan2(A[1, 0], A[0, 0]))
    corners = cv2.transform(np.float32([[0, 0], [mw, 0], [mw, mh], [0, mh]]).reshape(-1, 1, 2), M[:2])
    return dict(test=test, mirrored=mirrored, angle=round(angle, 1), scale=round(scale, 3),
                inliers=n, box=corners.reshape(-1, 2))


def find_mark_in_pdf(pdf_path, master_bgr, page_no=0):
    """
    2 รอบ (ประหยัดหน่วยความจำ — ใช้ได้บน Raspberry Pi):
      รอบหยาบ: render ทั้งหน้าความละเอียดปานกลาง -> หาตำแหน่งมาร์คคร่าว ๆ
      รอบละเอียด: render เฉพาะบริเวณมาร์คให้มาร์คใหญ่กว่า Master ~2 เท่า -> หาใหม่ให้แม่น
    """
    mh, mw = master_bgr.shape[:2]
    coarse = None
    for long_side in (2500, 4000, 6000):
        page, s, _ = render_pdf(pdf_path, page_no, long_side)
        coarse = find_mark(page, master_bgr)
        del page
        if coarse is not None:
            break
    if coarse is None:
        raise RuntimeError("หามาร์คใน PDF ไม่เจอ — เช็คว่า Master เป็นรุ่นเดียวกับ PDF")
    box_pt = coarse["box"] / s
    x0, y0 = box_pt.min(axis=0)
    x1, y1 = box_pt.max(axis=0)
    pad = 0.15 * max(x1 - x0, y1 - y0)
    clip = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    # ให้มาร์คในภาพละเอียดยาวราว 2 เท่าของ Master (คมชัด แต่ไม่ใหญ่เกิน)
    side = 2.0 * max(mw, mh) * (max(x1 - x0, y1 - y0) + 2 * pad) / max(x1 - x0, y1 - y0)
    region, s2, _ = render_pdf(pdf_path, page_no, int(min(6000, max(800, side))), clip)
    fine = find_mark(region, master_bgr)
    r = fine if fine is not None and fine["inliers"] >= coarse["inliers"] * 0.5 else coarse
    r["coarse_inliers"] = coarse["inliers"]
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
          f"inliers={r['inliers']} (coarse {r['coarse_inliers']})  -> {a.out}")


if __name__ == "__main__":
    main()

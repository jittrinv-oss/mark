#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_master.py — โหลด "CAD FOR ORDER SCREEN" (PDF) แล้ว Auto-zoom เฉพาะ MARK PATTERN

  1) หา "กรอบเส้นประสีแดง" ที่ล้อมมาร์ค
     - ทางหลัก: อ่านเส้น vector สีแดงจาก PDF โดยตรง (แม่นยำ คมชัดที่สุด)
     - ทางสำรอง: render เป็นภาพแล้วหากรอบสีแดง (PDF สแกน / ไม่มี PyMuPDF)
     เจอหลายกรอบ -> เลือกกรอบใหญ่สุด = ช่อง MARK PATTERN
  2) Render เฉพาะบริเวณนั้นความละเอียดสูง (ไม่ใช่ขยายภาพ -> ไม่แตก)
  3) เก็บส่วนที่อยู่นอกกรอบประ เช่น "T1" แต่ตัดเส้นตาราง/หัวข้อรอบ ๆ ออก
  4) ลบเส้นประสีแดงออก -> ภาพมาร์คขาวดำ ใช้เป็น Master ได้ทันที

ใช้เดี่ยว ๆ:  python pdf_master.py "CAD.pdf" --out master_mark.png
"""
import argparse
import os
import shutil
import subprocess
import tempfile

import numpy as np
import cv2

try:
    import pymupdf as fitz
    HAVE_FITZ = True
except Exception:
    try:
        import fitz
        HAVE_FITZ = True
    except Exception:
        HAVE_FITZ = False


def _is_red(col):
    return col is not None and len(col) >= 3 and col[0] > 0.7 and col[1] < 0.4 and col[2] < 0.4


def find_red_dashed_boxes_vector(page, min_side=15.0, min_segments=8, gap=6.0):
    """รวมท่อนเส้น vector สีแดงที่อยู่ติดกันเป็นกรอบ คืน list fitz.Rect (ใหญ่ -> เล็ก)"""
    segs, whole = [], []
    for d in page.get_drawings():
        if _is_red(d.get("color")) and d.get("fill") is None:
            r = d["rect"]
            if max(r.width, r.height) < 60:
                segs.append([r.x0, r.y0, r.x1, r.y1])          # เส้นประที่วาดเป็นท่อน ๆ
            elif d.get("dashes") not in (None, "", "[] 0") and min(r.width, r.height) >= min_side:
                whole.append(fitz.Rect(r))                      # สี่เหลี่ยมเส้นประชิ้นเดียว
    if not segs:
        whole.sort(key=lambda r: r.width * r.height, reverse=True)
        return whole

    def near(a, b):
        return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])

    def merge(a, b):
        a[0], a[1] = min(a[0], b[0]), min(a[1], b[1])
        a[2], a[3] = max(a[2], b[2]), max(a[3], b[3])
        a[4] += b[4]

    groups = []
    for s in segs:
        hit = [g for g in groups if near(g, s)]
        if not hit:
            groups.append(s + [1]); continue
        merge(hit[0], s + [1])
        for g in hit[1:]:
            merge(hit[0], g); groups.remove(g)
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if near(groups[i], groups[j]):
                    merge(groups[i], groups[j]); groups.pop(j); changed = True; break
            if changed:
                break
    boxes = [fitz.Rect(g[:4]) for g in groups
             if g[4] >= min_segments and g[2] - g[0] >= min_side and g[3] - g[1] >= min_side]
    boxes += whole
    boxes.sort(key=lambda r: r.width * r.height, reverse=True)
    return boxes


def red_mask(bgr, diff=35):
    b, g, r = [c.astype(np.int16) for c in cv2.split(bgr)]
    return ((r - np.maximum(g, b)) > diff).astype(np.uint8) * 255


def find_red_boxes_raster(bgr, min_side=40):
    m = red_mask(bgr)
    k = max(5, int(min(bgr.shape[:2]) * 0.006)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_side or h < min_side:
            continue
        inside = m[y + h // 5:y + 4 * h // 5, x + w // 5:x + 4 * w // 5]
        b = 4
        sides = [m[y:y + h, x:x + b].max(axis=1), m[y:y + h, x + w - b:x + w].max(axis=1),
                 m[y:y + b, x:x + w].max(axis=0), m[y + h - b:y + h, x:x + w].max(axis=0)]
        cover = min(float((s > 0).mean()) for s in sides)
        if inside.size and inside.mean() < 40 and cover > 0.6:   # กรอบกลวงครบ 4 ด้าน
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b_: b_[2] * b_[3], reverse=True)
    return boxes


def _remove_long_lines(ink, frac=0.45):
    h, w = ink.shape
    hl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(w * frac)), 1)))
    vl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(h * frac)))))
    return cv2.subtract(ink, cv2.dilate(cv2.bitwise_or(hl, vl), np.ones((3, 3), np.uint8)))


def tight_crop_mark(bgr, box_px, side_margin=0.30, pad=0.04):
    """เก็บหมึกในกรอบ + นอกกรอบไม่เกิน side_margin (เช่น T1) แล้วครอบตัดพอดี"""
    x, y, w, h = box_px
    clean = bgr.copy()
    clean[red_mask(clean) > 0] = (255, 255, 255)
    ink = (cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY) < 140).astype(np.uint8) * 255
    ink = _remove_long_lines(ink)
    H, W = ink.shape
    zx0, zx1 = max(0, int(x - w * side_margin)), min(W, int(x + w * (1 + side_margin)))
    zy0, zy1 = max(0, int(y - h * 0.03)), min(H, int(y + h * 1.03))
    n, _, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    xs0, ys0, xs1, ys1 = [], [], [], []
    for i in range(1, n):
        bx, by, bw, bh, a = st[i]
        if a >= 6 and zx0 <= bx + bw / 2 <= zx1 and zy0 <= by + bh / 2 <= zy1:
            xs0.append(bx); ys0.append(by); xs1.append(bx + bw); ys1.append(by + bh)
    if xs0:
        cx0, cy0, cx1, cy1 = min(xs0), min(ys0), max(xs1), max(ys1)
    else:
        cx0, cy0, cx1, cy1 = x, y, x + w, y + h
    pw, ph = int((cx1 - cx0) * pad), int((cy1 - cy0) * pad)
    cx0, cy0, cx1, cy1 = max(0, cx0 - pw), max(0, cy0 - ph), min(W, cx1 + pw), min(H, cy1 + ph)
    return clean[cy0:cy1, cx0:cx1].copy(), (cx0, cy0, cx1 - cx0, cy1 - cy0)


def _render_fitz(page, clip, long_side_px):
    scale = long_side_px / max(clip.width, clip.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_GRAY2BGR), scale


def _render_pdftoppm(path, dpi=200, page=1):
    exe = shutil.which("pdftoppm")
    if not exe:
        raise RuntimeError("ไม่พบ PyMuPDF และ pdftoppm — ติดตั้ง:  python -m pip install pymupdf")
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([exe, "-r", str(dpi), "-f", str(page), "-l", str(page), "-png", path,
                        os.path.join(td, "p")], check=True, capture_output=True)
        f = [x for x in os.listdir(td) if x.endswith(".png")][0]
        return cv2.imread(os.path.join(td, f), cv2.IMREAD_COLOR)


def _resize_long(img, long_side):
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def load_pdf_mark(path, page_no=0, target_long_side=1000):
    """คืน (mark_bgr, info) — raise RuntimeError ถ้าหากรอบไม่เจอ"""
    if HAVE_FITZ:
        page = fitz.open(path)[page_no]
        boxes = find_red_dashed_boxes_vector(page)
        if boxes:
            box = boxes[0]
            mx, my = box.width * 0.45, box.height * 0.10
            clip = fitz.Rect(box.x0 - mx, box.y0 - my, box.x1 + mx, box.y1 + my) & page.rect
            img, s = _render_fitz(page, clip, target_long_side * 1.9)
            bx = (int((box.x0 - clip.x0) * s), int((box.y0 - clip.y0) * s), int(box.width * s), int(box.height * s))
            mark, _ = tight_crop_mark(img, bx)
            return _resize_long(mark, target_long_side), dict(method="PDF-vector", n_boxes=len(boxes), page=page_no)
        img, _ = _render_fitz(page, page.rect, 6000)
    else:
        img = _render_pdftoppm(path, 200, page_no + 1)
    boxes = find_red_boxes_raster(img)
    if not boxes:
        raise RuntimeError("ไม่พบกรอบเส้นประสีแดงรอบมาร์คใน PDF — ใช้การลากกรอบครอบตัดเองแทน")
    x, y, w, h = boxes[0]
    H, W = img.shape[:2]
    ex0, ey0 = max(0, int(x - w * 0.45)), max(0, int(y - h * 0.10))
    ex1, ey1 = min(W, int(x + w * 1.45)), min(H, int(y + h * 1.10))
    mark, _ = tight_crop_mark(img[ey0:ey1, ex0:ex1], (x - ex0, y - ey0, w, h))
    return _resize_long(mark, target_long_side), dict(method="PDF-raster", n_boxes=len(boxes), page=page_no)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf"); ap.add_argument("--out", default="master_mark.png")
    a = ap.parse_args()
    mark, info = load_pdf_mark(a.pdf)
    cv2.imencode(".png", mark)[1].tofile(a.out)
    print("OK", info, "->", a.out, mark.shape)

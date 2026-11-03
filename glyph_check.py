#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""glyph_check.py — ตรวจ "องค์ประกอบจางขาดหาย" แบบอัตโนมัติ
v8: แก้เส้นที่หายถูกข้ามเป็น "ภาพเบลอ"
v9: แก้ "ขีดของตัวอักษรหาย แต่ไม่เจอ" (เช่น ขาตั้งของตัว L ใน TEMPERLITE)
    แต่ละตัวอักษรใช้หมึกได้เฉพาะ "เขตของตัวเอง"
v10: แก้ false alarm สีส้ม/แดง/ม่วง เมื่อถ่ายชิ้นงานเดียวกัน (แสง เงา ความคม กล้อง ต่างกัน)
    สาเหตุ: ตัดสินจาก "หน้ากากหมึก (binary)" อย่างเดียว -> เส้นบาง (6, 0, 8, เส้นประมุม)
            ขาดหายบางส่วนตอน threshold ทั้งที่ตัวอักษรอยู่ครบ
    แก้:   ชิ้นที่ judge ได้ PARTIAL/MISSING -> "ยืนยันซ้ำด้วยภาพเทา" (verify_gray)
            ลบพื้นหลัง/เงา + เบลอให้คมเท่ากัน + หาตำแหน่งรอบ ๆ + ลองหมึกเข้ม/สว่าง
            ผ่านถ้า รูปร่างเหมือน (NCC >= v_ncc) และ หมึก Master เจอใน Test (>= v_cover)
            ตัวที่หายจริง -> NCC/cover ต่ำ -> ยังเป็น NG
          + เศษเส้นตรงเล็ก (LINE) ที่ขอบกรอบมาร์ค (มุมเส้นประ) -> SKIP
          + หมึกเกิน (สีม่วง) ไม่ใช้ตัดสินและไม่วาด (check_extra=False)
"""
import argparse, json, os
from datetime import datetime
import numpy as np
import cv2
from mark_match import to_ink_mask, align


def segment_elements(mask, min_area=40, merge_gap=0, dash_params=None):
    work = mask
    if merge_gap > 0:
        work = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * merge_gap + 1,) * 2))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(work, 8)
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    els = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, w, h = (int(v) for v in stats[i, :4])
        sub = np.where(lab[y:y + h, x:x + w] == i, 255, 0).astype(np.uint8)
        sub = cv2.bitwise_and(sub, mask[y:y + h, x:x + w])
        a = int((sub > 0).sum())
        if a < min_area:
            continue
        d = dist[y:y + h, x:x + w][sub > 0]
        stroke = float(2 * np.percentile(d, 90)) if d.size else 2.0
        els.append(dict(id=0, x=x, y=y, w=w, h=h, area=a, stroke=stroke,
                        cx=x + w / 2, cy=y + h / 2, mask=sub, kind=None))
    if els:
        base_h = float(np.median([e["h"] for e in els]))
        for e in els:
            e["kind"] = classify(e["mask"], e["w"], e["h"], e["area"], base_h)
    H, W = mask.shape[:2]
    mark_dash_groups(els, W, H, **(dash_params or {}))
    els.sort(key=lambda e: (e["y"] // 20, e["x"]))
    for i, e in enumerate(els, 1):
        e["id"] = i
    return els


def classify(sub, w, h, area, base_h):
    ar = w / max(h, 1)
    fill = area / max(w * h, 1)
    thin = max(3, base_h * 0.30)
    if (ar >= 3.5 and h <= thin) or (ar <= 0.28 and w <= thin):
        return "LINE"
    if 0.72 <= ar <= 1.40 and min(w, h) >= base_h * 1.6 and fill < 0.55:
        ys, xs = np.nonzero(sub)
        if len(xs) > 30:
            rad = np.hypot(xs - xs.mean(), ys - ys.mean())
            if rad.mean() > 0 and rad.std() / rad.mean() < 0.22 and rad.mean() > 0.30 * max(w, h):
                return "RING"
    return "TEXT"


def _cluster_1d(items, key, tol):
    groups, cur = [], []
    for it in sorted(items, key=key):
        if cur and abs(key(it) - key(cur[-1])) > tol:
            groups.append(cur); cur = []
        cur.append(it)
    if cur:
        groups.append(cur)
    return groups


def _is_regular(vals, cv_max):
    gaps = np.diff(sorted(vals))
    if len(gaps) == 0 or np.median(gaps) <= 0:
        return False
    med = np.median(gaps)
    return np.median(np.abs(gaps - med)) / med <= cv_max


def mark_dash_groups(els, img_w, img_h, min_count=4, gap_cv_max=0.6,
                     edge_ratio=0.07, area_percentile=70):
    if not els:
        return
    max_area = max(60.0, float(np.percentile([e["area"] for e in els], area_percentile)))
    ex, ey = max(15, int(edge_ratio * img_w)), max(15, int(edge_ratio * img_h))
    xt, yt = max(10, int(0.025 * img_w)), max(10, int(0.025 * img_h))
    sm = [e for e in els if e["area"] < max_area
          and min(e["w"], e["h"]) <= 0.35 * max(e["w"], e["h"])]
    for pool in ([e for e in sm if e["cx"] < ex], [e for e in sm if e["cx"] > img_w - ex]):
        for g in _cluster_1d(pool, lambda e: e["cx"], xt):
            if len(g) >= min_count and _is_regular([e["cy"] for e in g], gap_cv_max):
                for e in g:
                    e["kind"] = "DASH"
    for pool in ([e for e in sm if e["cy"] < ey], [e for e in sm if e["cy"] > img_h - ey]):
        pool = [e for e in pool if e["kind"] != "DASH"]
        for g in _cluster_1d(pool, lambda e: e["cy"], yt):
            if len(g) >= min_count and _is_regular([e["cx"] for e in g], gap_cv_max):
                for e in g:
                    e["kind"] = "DASH"


def element_zones(mmask, els):
    src = np.where(mmask > 0, 0, 255).astype(np.uint8)
    _, zl = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP)
    lab_of = {}
    for e in els:
        ys, xs = np.nonzero(e["mask"])
        if len(xs):
            lab_of[e["id"]] = int(zl[e["y"] + ys[0], e["x"] + xs[0]])
    return zl, lab_of


def extent_fractions(el, test_mask, tol_px=2, loss_density=0.60, local_search=0, stroke_tol=0.0,
                     zone_map=None, zone_label=None):
    x, y, w, h = el["x"], el["y"], el["w"], el["h"]
    if zone_map is not None and zone_label is not None:
        Hz, Wz = zone_map.shape
        g = max(2, int(round(0.25 * float(el.get("stroke", 4.0)))))
        p = g + 30
        zx0, zy0 = max(0, x - p), max(0, y - p)
        zx1, zy1 = min(Wz, x + el["w"] + p), min(Hz, y + el["h"] + p)
        own = (zone_map[zy0:zy1, zx0:zx1] == zone_label).astype(np.uint8)
        own = cv2.dilate(own, np.ones((2 * g + 1,) * 2, np.uint8))
        tm2 = np.zeros_like(test_mask)
        tm2[zy0:zy1, zx0:zx1] = np.where(own > 0, test_mask[zy0:zy1, zx0:zx1], 0)
        test_mask = tm2
    m = (el["mask"] > 0).astype(np.uint8)
    stroke = float(el.get("stroke", 4.0))
    tol = int(round(max(tol_px, stroke_tol * stroke)))
    s = int(min(15, max(3, local_search * min(w, h)))) if local_search else 0
    Ht, Wt = test_mask.shape
    pad = tol + s + 2
    X0, Y0 = x - pad, y - pad
    canvas = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    cx0, cy0 = max(0, X0), max(0, Y0)
    cx1, cy1 = min(Wt, x + w + pad), min(Ht, y + h + pad)
    if cx1 > cx0 and cy1 > cy0:
        canvas[cy0 - Y0:cy1 - Y0, cx0 - X0:cx1 - X0] = (test_mask[cy0:cy1, cx0:cx1] > 0)
    if tol > 0:
        canvas = cv2.dilate(canvas, np.ones((2 * tol + 1,) * 2, np.uint8))
    dx = dy = 0
    if s > 0:
        win = canvas[pad - s:pad + h + s, pad - s:pad + w + s].astype(np.float32)
        res = cv2.matchTemplate(win, m.astype(np.float32), cv2.TM_CCORR)
        yy, xx = np.mgrid[-s:s + 1, -s:s + 1]
        score = res - 1e-3 * (np.abs(xx) + np.abs(yy))
        iy, ix = np.unravel_index(np.argmax(score), score.shape)
        dx, dy = int(ix - s), int(iy - s)
    t = canvas[pad + dy:pad + dy + h, pad + dx:pad + dx + w]
    lost = (m & (1 - t)).astype(np.uint8) * 255
    k = max(3, int(round(stroke * 0.55)))
    lost = cv2.morphologyEx(lost, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    lost[m == 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(lost, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= max(6, 0.02 * el["area"])
    real = keep[lab]
    cov = 1.0 - real.sum() / max(int(m.sum()), 1)
    if not real.any():
        return float(cov), 0.0, 0.0, (dx, dy)
    ys, xs = np.nonzero(real)
    return float(cov), (ys.max() - ys.min() + 1) / h, (xs.max() - xs.min() + 1) / w, (dx, dy)


def _sharp(g):
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _crop(img, el, pad=4):
    H, W = img.shape[:2]
    x, y, w, h = el["x"], el["y"], el["w"], el["h"]
    return cv2.cvtColor(img[max(0, y - pad):min(H, y + h + pad),
                            max(0, x - pad):min(W, x + w + pad)], cv2.COLOR_BGR2GRAY)


def sharpness_baseline(els, master_bgr, aligned_bgr):
    r = []
    for e in els:
        if e["kind"] == "DASH":
            continue
        sm = _sharp(_crop(master_bgr, e))
        if sm >= 6.0:
            r.append(_sharp(_crop(aligned_bgr, e)) / sm)
    return float(np.median(r)) if r else 1.0


def background_noise_std(gray, patch=50):
    H, W = gray.shape
    v = [float(gray[y:y + patch, x:x + patch].std())
         for y, x in [(0, 0), (0, W - patch), (H - patch, 0), (H - patch, W - patch)]]
    return float(np.median(v))


def _ink_darkness(gray, el, pad=6, k=5):
    H, W = gray.shape
    x, y, w, h = el["x"], el["y"], el["w"], el["h"]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    dark = cv2.erode(crop, np.ones((k, k), np.uint8))
    m = np.zeros(crop.shape, bool)
    m[y - y0:y - y0 + h, x - x0:x - x0 + w] = el["mask"] > 0
    bg = float(np.percentile(crop, 90))
    return max(0.0, bg - float(dark[m].mean())) if m.any() else 0.0


def is_blurred(el, master_bgr, aligned_bgr, baseline, bg_std, blur_rel_thr=0.35, ink_thr=0.15):
    gm, gt = _crop(master_bgr, el), _crop(aligned_bgr, el)
    sm = _sharp(gm)
    if sm < 6.0 or baseline <= 0:
        return False, 1.0, 1.0
    rel = (_sharp(gt) / sm) / baseline
    if rel >= blur_rel_thr:
        return False, rel, 1.0
    g_m = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY)
    g_t = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    ink_ratio = _ink_darkness(g_t, el) / max(_ink_darkness(g_m, el), 1.0)
    return ink_ratio >= ink_thr, rel, ink_ratio


# ------------------------------------------------------------------ v10: ยืนยันด้วยภาพเทา
def _flat(g, k):
    """ลบพื้นหลัง/เงา -> 0..1 (หมึกเข้ม = ค่าต่ำ)"""
    g = g.astype(np.float32) + 1.0
    f = g / cv2.GaussianBlur(g, (0, 0), k)
    lo, hi = np.percentile(f, 2), np.percentile(f, 98)
    return np.clip((f - lo) / max(hi - lo, 1e-3), 0, 1)


def _otsu_ink(p):
    _, b = cv2.threshold((p * 255).astype(np.uint8), 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b


def verify_gray(el, g_m, g_t, search=0.45):
    """คืน (ncc, cover) — ncc = รูปร่างเหมือน, cover = หมึก Master ที่เจอใน Test"""
    H, W = g_m.shape
    x, y, w, h = el["x"], el["y"], el["w"], el["h"]
    sz = max(w, h)
    pad = max(4, int(0.25 * sz))
    sr = max(4, int(search * sz))
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    X0, Y0, X1, Y1 = max(0, x0 - sr), max(0, y0 - sr), min(W, x1 + sr), min(H, y1 + sr)
    k = max(3.0, sz * 0.9)
    mw, tw = _flat(g_m[Y0:Y1, X0:X1], k), _flat(g_t[Y0:Y1, X0:X1], k)
    tpl = mw[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
    m_ink = _otsu_ink(tpl)
    if m_ink.sum() < 6 or tpl.std() < 0.03:
        return 1.0, 1.0
    s = max(0.6, sz * 0.03)
    tpl_b = cv2.GaussianBlur(tpl, (0, 0), s)
    best = (-1.0, None, None)
    for inv in (False, True):                         # หมึกเข้ม / หมึกสว่าง (กระจกเขียว)
        t = 1.0 - tw if inv else tw
        r = cv2.matchTemplate(cv2.GaussianBlur(t, (0, 0), s), tpl_b, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        if mx > best[0]:
            best = (mx, loc, t)
    ncc, loc, t = best
    th, tw_ = tpl.shape
    t_ink = _otsu_ink(t[loc[1]:loc[1] + th, loc[0]:loc[0] + tw_])
    tol = max(1, int(round(sz * 0.06)))
    t_ink = cv2.dilate(t_ink, np.ones((2 * tol + 1,) * 2, np.uint8))
    return float(ncc), float((t_ink[m_ink > 0] > 0).mean())


def _frame_edge(el, bbox, frac=0.06):
    """ชิ้นอยู่ที่ขอบกรอบรวมของมาร์ค (เช่น มุมเส้นประ)"""
    bx0, by0, bx1, by1 = bbox
    mx, my = frac * (bx1 - bx0), frac * (by1 - by0)
    return (el["x"] <= bx0 + mx or el["x"] + el["w"] >= bx1 - mx or
            el["y"] <= by0 + my or el["y"] + el["h"] >= by1 - my)


def judge(cov, vfrac, hfrac, p):
    loss = 1.0 - cov
    if vfrac < p["min_extent"] or hfrac < p["min_extent"] or loss < p["min_loss"]:
        if vfrac == 0 and hfrac == 0:
            return "OK", f"ครบ ({cov:.0%})"
        return "OK", (f"หายเล็กน้อย ไม่ถึงเกณฑ์ (หาย {loss:.0%}, "
                      f"แนวตั้ง {vfrac:.0%}, แนวนอน {hfrac:.0%})")
    if cov < p["miss_full"]:
        return "MISSING", f"หายทั้งชิ้น (เหลือ {cov:.0%}) แนวตั้ง {vfrac:.0%} แนวนอน {hfrac:.0%}"
    return "PARTIAL", f"จางขาดหาย (หาย {loss:.0%}) แนวตั้ง {vfrac:.0%} แนวนอน {hfrac:.0%}"


def in_ignore(cx, cy, zones):
    for z in zones:
        if z["x"] <= cx <= z["x"] + z["w"] and z["y"] <= cy <= z["y"] + z["h"]:
            return z.get("name", "IGNORE")
    return None


def mask_out(mask, zones):
    m = mask.copy()
    for z in zones:
        m[max(0, z["y"]):z["y"] + z["h"], max(0, z["x"]):z["x"] + z["w"]] = 0
    return m


DEFAULT_P = dict(
    min_extent=0.30, min_loss=0.08, row_loss_thr=0.60, miss_full=0.25,
    local_search=0.35, stroke_tol=0.35,
    check_blur=True, blur_rel_thr=0.35, ink_thr=0.15,
    check_dash=True, dash_edge_ratio=0.07, dash_min_count=4,
    dash_gap_cv_max=0.6, dash_area_percentile=70,
    min_area=40, merge_gap=0, tol_px=2,
    check_extra=False, extra_area=120,      # v10: หมึกเกิน (สีม่วง) ปิด
    own_zone=True,
    verify=True,        # v10: ยืนยันชิ้น NG ด้วยภาพเทา
    v_ncc=0.55,         # v10: รูปร่างเหมือนขั้นต่ำ   (ลด = alarm น้อยลง)
    v_cover=0.70,       # v10: หมึก Master เจอใน Test ขั้นต่ำ (ลด = alarm น้อยลง, ห้าม < 0.60)
    v_search=0.45,      # v10: ค้นหาตำแหน่งรอบชิ้น ± สัดส่วนขนาดชิ้น
    skip_frame_line=True,  # v10: เศษเส้นตรงที่ขอบกรอบมาร์ค (มุมเส้นประ) -> SKIP
)


def inspect_glyph(master_bgr, test_bgr, ignore_zones=None, params=None):
    p = dict(DEFAULT_P); p.update(params or {})
    zones = ignore_zones or []
    aligned, method, inl, valid = align(test_bgr, master_bgr, return_valid=True)
    mmask = to_ink_mask(master_bgr)
    tmask = to_ink_mask(aligned, valid=valid)
    dp = dict(min_count=p["dash_min_count"], gap_cv_max=p["dash_gap_cv_max"],
              edge_ratio=p["dash_edge_ratio"], area_percentile=p["dash_area_percentile"])
    els = segment_elements(mmask, p["min_area"], p["merge_gap"],
                           dp if p["check_dash"] else dict(min_count=10 ** 9))
    base = sharpness_baseline(els, master_bgr, aligned) if p["check_blur"] else 1.0
    zmap, zlab = element_zones(mmask, els) if p["own_zone"] else (None, {})
    bg_std = background_noise_std(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))
    g_m = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY)
    g_t = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    bbox = (min(e["x"] for e in els), min(e["y"] for e in els),
            max(e["x"] + e["w"] for e in els), max(e["y"] + e["h"] for e in els)) if els else (0, 0, 1, 1)
    rows, ng, n_blur, n_dash, n_ver = [], 0, 0, 0, 0
    for el in els:
        cov = vfrac = hfrac = None
        off = (0, 0)
        z = in_ignore(el["cx"], el["cy"], zones)
        outside = valid[min(valid.shape[0] - 1, int(el["cy"])), min(valid.shape[1] - 1, int(el["cx"]))] == 0
        if p["check_dash"] and el["kind"] == "DASH":
            status, note = "SKIP", "เส้นประ/สเกล — ไม่ตรวจ"; n_dash += 1
        elif z:
            status, note = "SKIP", f"อยู่ในโซน {z}"
        elif outside:
            status, note = "SKIP", "อยู่นอกภาพถ่าย — ไม่ตรวจ"
        elif p["skip_frame_line"] and el["kind"] == "LINE" and _frame_edge(el, bbox):
            status, note = "SKIP", "เส้นกรอบ/มุมเส้นประ — ไม่ตรวจ"; n_dash += 1
        else:
            blur, rel, con = (False, 1, 1)
            if p["check_blur"]:
                blur, rel, con = is_blurred(el, master_bgr, aligned, base, bg_std,
                                            p["blur_rel_thr"], p["ink_thr"])
            if blur:
                status, note = "SKIP", f"ภาพเบลอ (คมชัด {rel:.0%}, หมึกยังอยู่ {con:.0%}) — ข้าม"; n_blur += 1
            else:
                cov, vfrac, hfrac, off = extent_fractions(
                    el, tmask, p["tol_px"], p["row_loss_thr"], float(p["local_search"]), p["stroke_tol"],
                    zmap, zlab.get(el["id"]))
                status, note = judge(cov, vfrac, hfrac, p)
                if status != "OK" and p["verify"]:
                    ncc, vc = verify_gray(el, g_m, g_t, p["v_search"])
                    if ncc >= p["v_ncc"] and vc >= p["v_cover"]:
                        status = "OK"; n_ver += 1
                        note = f"ภาพเทายืนยันครบ (ncc {ncc:.2f}, cover {vc:.0%}) | เดิม: {note}"
                    else:
                        note = f"{note} | ncc {ncc:.2f} cover {vc:.0%}"
                ng += status != "OK"
        rows.append(dict(id=el["id"], kind=el["kind"], status=status, note=note,
                         x=el["x"], y=el["y"], w=el["w"], h=el["h"], area=el["area"],
                         dx=off[0], dy=off[1],
                         cov=None if cov is None else round(cov, 3),
                         vfrac=None if vfrac is None else round(vfrac, 3),
                         hfrac=None if hfrac is None else round(hfrac, 3)))
    extras = []
    if p["check_extra"]:
        tol = max(p["tol_px"], 6)
        ex = cv2.subtract(mask_out(tmask, zones), cv2.dilate(mmask, np.ones((2 * tol + 3,) * 2, np.uint8)))
        ex = cv2.morphologyEx(ex, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(ex, 8)
        extras = [dict(x=int(st[i, 0]), y=int(st[i, 1]), w=int(st[i, 2]), h=int(st[i, 3]),
                       area=int(st[i, 4])) for i in range(1, n) if st[i, 4] >= p["extra_area"]]
    rep = dict(timestamp=datetime.now().isoformat(timespec="seconds"),
               verdict="NG" if ng else "OK", align_method=f"{method} | v10 ยืนยัน {n_ver}",
               align_inliers=inl,
               n_elements=len(els),
               n_ok=sum(r["status"] == "OK" for r in rows),
               n_missing=sum(r["status"] == "MISSING" for r in rows),
               n_partial=sum(r["status"] == "PARTIAL" for r in rows),
               n_skip=sum(r["status"] == "SKIP" for r in rows),
               n_blur=n_blur, n_dash=n_dash, n_extra=len(extras), n_verified=n_ver,
               elements=rows, extras=extras, params=p)
    return rep, draw(master_bgr, aligned, rows, extras, zones), aligned, mmask, tmask, els


COLOR = {"OK": (0, 170, 0), "PARTIAL": (0, 165, 255), "MISSING": (0, 0, 255), "SKIP": (150, 150, 150)}


def draw_result(aligned, rows, extras, zones, show_ok=True):
    R = aligned.copy()
    for z in zones:
        cv2.rectangle(R, (z["x"], z["y"]), (z["x"] + z["w"], z["y"] + z["h"]), (160, 160, 160), 2)
    for r in rows:
        bad = r["status"] in ("MISSING", "PARTIAL")
        if not bad and not show_ok:
            continue
        c = COLOR[r["status"]]
        x, y = r["x"] + r.get("dx", 0), r["y"] + r.get("dy", 0)
        cv2.rectangle(R, (x - 2, y - 2), (x + r["w"] + 2, y + r["h"] + 2), c, 3 if bad else 1)
        if bad:
            cv2.putText(R, f'#{r["id"]}', (x, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    for e in extras:
        cv2.rectangle(R, (e["x"] - 2, e["y"] - 2), (e["x"] + e["w"] + 2, e["y"] + e["h"] + 2), (255, 0, 255), 2)
    return R


def draw(master_bgr, aligned, rows, extras, zones):
    L = master_bgr.copy()
    for r in rows:
        if r["status"] in ("MISSING", "PARTIAL"):
            cv2.rectangle(L, (r["x"] - 2, r["y"] - 2), (r["x"] + r["w"] + 2, r["y"] + r["h"] + 2),
                          COLOR[r["status"]], 2)
    R = draw_result(aligned, rows, extras, zones, show_ok=True)
    h = max(L.shape[0], R.shape[0])
    f = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    return np.hstack([f(L), np.full((h, 6, 3), 60, np.uint8), f(R)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--ignore"); ap.add_argument("--out", default="res_glyph")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    rd = lambda f: cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
    zones = json.load(open(a.ignore, encoding="utf-8")).get("ignore", []) if a.ignore else []
    params = {}
    for kv in a.set:
        k, v = kv.split("=")
        params[k] = (v in ("1", "True", "true")) if isinstance(DEFAULT_P[k], bool) else type(DEFAULT_P[k])(float(v))
    rep, ov, *_ = inspect_glyph(rd(a.master), rd(a.test), zones, params)
    os.makedirs(a.out, exist_ok=True)
    cv2.imencode(".png", ov)[1].tofile(os.path.join(a.out, "overlay.png"))
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f'RESULT = {rep["verdict"]}  align={rep["align_method"]}  ชิ้น={rep["n_elements"]} '
          f'MISSING={rep["n_missing"]} PARTIAL={rep["n_partial"]} SKIP={rep["n_skip"]} '
          f'(blur={rep["n_blur"]} dash={rep["n_dash"]} ยืนยันด้วยภาพเทา={rep["n_verified"]})')
    for r in rep["elements"]:
        if r["status"] != "OK":
            print(f'  [{r["status"]}] #{r["id"]} {r["kind"]} ({r["x"]},{r["y"]}) {r["w"]}x{r["h"]} {r["note"]}')


if __name__ == "__main__":
    main()

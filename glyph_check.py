#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""glyph_check.py — ตรวจ "องค์ประกอบจางขาดหาย" แบบอัตโนมัติ (v4)"""
import argparse, json, os
from datetime import datetime
import numpy as np
import cv2

from mark_match import to_ink_mask, align


def segment_elements(mask, min_area=40, merge_gap=0, dash_params=None):
    work = mask
    if merge_gap > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * merge_gap + 1, 2 * merge_gap + 1))
        work = cv2.dilate(mask, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(work, 8)
    els = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        sub = np.zeros((h, w), np.uint8)
        sub[lab[y:y + h, x:x + w] == i] = 255
        sub = cv2.bitwise_and(sub, mask[y:y + h, x:x + w])
        a = int(sub.sum() // 255)
        if a < min_area:
            continue
        els.append(dict(id=0, x=int(x), y=int(y), w=int(w), h=int(h), area=a,
                        cx=float(x + w / 2), cy=float(y + h / 2), mask=sub, kind=None))
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
    sm = [e for e in els if e["area"] < max_area]
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


def extent_fractions(el, test_mask, tol_px=2, loss_density=0.60):
    x, y, w, h = el["x"], el["y"], el["w"], el["h"]
    H, W = test_mask.shape
    pad = tol_px + 2
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    troi = test_mask[y0:y1, x0:x1]
    if tol_px > 0:
        troi = cv2.dilate(troi, np.ones((2 * tol_px + 1, 2 * tol_px + 1), np.uint8))
    ox, oy = x - x0, y - y0
    m = el["mask"]
    t = troi[oy:oy + h, ox:ox + w]
    kept = cv2.bitwise_and(m, t)
    cov = (kept > 0).sum() / max((m > 0).sum(), 1)
    lost = cv2.subtract(m, kept)
    if lost.max() == 0:
        return float(cov), 0.0, 0.0
    k = max(3, int(round(min(w, h) * 0.12)) | 1)
    lost_c = cv2.morphologyEx(lost, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    lost_c = cv2.bitwise_and(cv2.dilate(lost_c, np.ones((3, 3), np.uint8)),
                             cv2.dilate(m, np.ones((3, 3), np.uint8)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(lost_c, 8)
    bx0, by0, bx1, by1 = w, h, 0, 0
    found = False
    for i in range(1, n):
        bx, by = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
        bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        ink_in = (m[by:by + bh, bx:bx + bw] > 0).sum()
        lost_in = (lost[by:by + bh, bx:bx + bw] > 0).sum()
        if ink_in < 6:
            continue
        if lost_in / ink_in >= loss_density:
            ys, xs = np.nonzero((lab == i) & (lost > 0))
            if len(xs) == 0:
                continue
            found = True
            bx0, by0 = min(bx0, xs.min()), min(by0, ys.min())
            bx1, by1 = max(bx1, xs.max() + 1), max(by1, ys.max() + 1)
    if not found:
        return float(cov), 0.0, 0.0
    return float(cov), (by1 - by0) / h, (bx1 - bx0) / w


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


def is_blurred(el, master_bgr, aligned_bgr, baseline, bg_std,
               blur_rel_thr=0.35, content_factor=3.0):
    gm, gt = _crop(master_bgr, el), _crop(aligned_bgr, el)
    sm = _sharp(gm)
    if sm < 6.0 or baseline <= 0:
        return False, 1.0, 1.0
    rel = (_sharp(gt) / sm) / baseline
    content = float(gt.std()) / max(bg_std, 1e-6)
    return (rel < blur_rel_thr and content > content_factor), rel, content


def judge(cov, vfrac, hfrac, p):
    if vfrac < p["min_extent"] or hfrac < p["min_extent"]:
        if vfrac == 0 and hfrac == 0:
            return "OK", f"ครบ ({cov:.0%})"
        return "OK", (f"หายเล็กน้อย ไม่ถึงเกณฑ์ (แนวตั้ง {vfrac:.0%}, "
                      f"แนวนอน {hfrac:.0%} < {p['min_extent']:.0%})")
    if cov < p["miss_full"]:
        return "MISSING", f"หายทั้งชิ้น (เหลือ {cov:.0%}) แนวตั้ง {vfrac:.0%} แนวนอน {hfrac:.0%}"
    return "PARTIAL", f"จางขาดหาย (เหลือ {cov:.0%}) แนวตั้ง {vfrac:.0%} แนวนอน {hfrac:.0%}"


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
    min_extent=0.30, row_loss_thr=0.60, miss_full=0.25,
    check_blur=True, blur_rel_thr=0.35, content_factor=3.0,
    check_dash=True, dash_edge_ratio=0.07, dash_min_count=4,
    dash_gap_cv_max=0.6, dash_area_percentile=70,
    min_area=40, merge_gap=0, tol_px=2,
    check_extra=True, extra_area=120,
)


def inspect_glyph(master_bgr, test_bgr, ignore_zones=None, params=None):
    p = dict(DEFAULT_P); p.update(params or {})
    zones = ignore_zones or []
    aligned, method, inl = align(test_bgr, master_bgr)
    mmask, tmask = to_ink_mask(master_bgr), to_ink_mask(aligned)
    dp = dict(min_count=p["dash_min_count"], gap_cv_max=p["dash_gap_cv_max"],
              edge_ratio=p["dash_edge_ratio"], area_percentile=p["dash_area_percentile"])
    els = segment_elements(mmask, p["min_area"], p["merge_gap"],
                           dp if p["check_dash"] else dict(min_count=10 ** 9))
    base = sharpness_baseline(els, master_bgr, aligned) if p["check_blur"] else 1.0
    bg_std = background_noise_std(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))

    rows, ng, n_blur, n_dash = [], 0, 0, 0
    for el in els:
        cov = vfrac = hfrac = None
        z = in_ignore(el["cx"], el["cy"], zones)
        if p["check_dash"] and el["kind"] == "DASH":
            status, note = "SKIP", "เส้นประ/สเกล — ไม่ตรวจ"; n_dash += 1
        elif z:
            status, note = "SKIP", f"อยู่ในโซน {z}"
        else:
            blur, rel, con = (False, 1, 1)
            if p["check_blur"]:
                blur, rel, con = is_blurred(el, master_bgr, aligned, base, bg_std,
                                            p["blur_rel_thr"], p["content_factor"])
            if blur:
                status, note = "SKIP", f"ภาพเบลอ (คมชัด {rel:.0%} ของปกติ) — ข้าม"; n_blur += 1
            else:
                cov, vfrac, hfrac = extent_fractions(el, tmask, p["tol_px"], p["row_loss_thr"])
                status, note = judge(cov, vfrac, hfrac, p)
                ng += status != "OK"
        rows.append(dict(id=el["id"], kind=el["kind"], status=status, note=note,
                         x=el["x"], y=el["y"], w=el["w"], h=el["h"], area=el["area"],
                         cov=None if cov is None else round(cov, 3),
                         vfrac=None if vfrac is None else round(vfrac, 3),
                         hfrac=None if hfrac is None else round(hfrac, 3)))

    extras = []
    if p["check_extra"]:
        k = np.ones((2 * p["tol_px"] + 3,) * 2, np.uint8)
        ex = cv2.subtract(mask_out(tmask, zones), cv2.dilate(mmask, k))
        ex = cv2.morphologyEx(ex, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(ex, 8)
        extras = [dict(x=int(st[i, 0]), y=int(st[i, 1]), w=int(st[i, 2]), h=int(st[i, 3]),
                       area=int(st[i, 4])) for i in range(1, n) if st[i, 4] >= p["extra_area"]]

    rep = dict(timestamp=datetime.now().isoformat(timespec="seconds"),
               verdict="NG" if ng else "OK", align_method=method, align_inliers=inl,
               n_elements=len(els),
               n_ok=sum(r["status"] == "OK" for r in rows),
               n_missing=sum(r["status"] == "MISSING" for r in rows),
               n_partial=sum(r["status"] == "PARTIAL" for r in rows),
               n_skip=sum(r["status"] == "SKIP" for r in rows),
               n_blur=n_blur, n_dash=n_dash, n_extra=len(extras),
               elements=rows, extras=extras, params=p)
    return rep, draw(master_bgr, aligned, rows, extras, zones), aligned, mmask, tmask, els


COLOR = {"OK": (0, 170, 0), "PARTIAL": (0, 165, 255), "MISSING": (0, 0, 255), "SKIP": (150, 150, 150)}


def draw(master_bgr, aligned, rows, extras, zones):
    L, R = master_bgr.copy(), aligned.copy()
    for z in zones:
        for im in (L, R):
            cv2.rectangle(im, (z["x"], z["y"]), (z["x"] + z["w"], z["y"] + z["h"]), (160, 160, 160), 2)
    for r in rows:
        c = COLOR[r["status"]]
        bad = r["status"] in ("MISSING", "PARTIAL")
        cv2.rectangle(R, (r["x"] - 2, r["y"] - 2), (r["x"] + r["w"] + 2, r["y"] + r["h"] + 2), c, 3 if bad else 1)
        if bad:
            cv2.rectangle(L, (r["x"] - 2, r["y"] - 2), (r["x"] + r["w"] + 2, r["y"] + r["h"] + 2), c, 2)
            cv2.putText(R, f'#{r["id"]}', (r["x"], max(12, r["y"] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    for e in extras:
        cv2.rectangle(R, (e["x"] - 2, e["y"] - 2), (e["x"] + e["w"] + 2, e["y"] + e["h"] + 2), (255, 0, 255), 2)
    h = max(L.shape[0], R.shape[0])
    f = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    return np.hstack([f(L), np.full((h, 6, 3), 60, np.uint8), f(R)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--ignore"); ap.add_argument("--out", default="res_glyph")
    ap.add_argument("--min-extent", type=float, default=0.30)
    a = ap.parse_args()
    rd = lambda f: cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
    zones = json.load(open(a.ignore, encoding="utf-8")).get("ignore", []) if a.ignore else []
    rep, ov, *_ = inspect_glyph(rd(a.master), rd(a.test), zones, dict(min_extent=a.min_extent))
    os.makedirs(a.out, exist_ok=True)
    cv2.imencode(".png", ov)[1].tofile(os.path.join(a.out, "overlay.png"))
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f'RESULT = {rep["verdict"]}  align={rep["align_method"]}  ชิ้น={rep["n_elements"]} '
          f'MISSING={rep["n_missing"]} PARTIAL={rep["n_partial"]} SKIP={rep["n_skip"]}')


if __name__ == "__main__":
    main()

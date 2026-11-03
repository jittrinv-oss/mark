#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sharp_loose.py v3 — แพตช์ตัวตรวจคมชัด (sharp_check + sharp_fix)  *ไม่แก้ไฟล์เดิม*
ถูกเรียกจาก mark_ui_sharp.py / web_sharp.py (บรรทัด import sharp_loose)

v3 (แก้ "loose v2 error: OpenCV ... Invalid number of channels ... scn is 3"):
  สาเหตุ: v2 เรียกฟังก์ชันภายในของ sharp_check บน Pi (preprocess / draw) ซึ่งเป็นคนละเวอร์ชัน
          -> แปลงสีภาพ 3 ช่องผิด -> error -> ตกไปใช้ตัวตรวจเดิม -> ยังส้ม/แดงเต็ม
  แก้:   v3 "ไม่เรียกฟังก์ชันภายในของ sharp_check เลย" ใช้แค่ผลลัพธ์ (report + ภาพ aligned)
          แปลงภาพ / คิด %match / วาดผล เองทั้งหมด (รับภาพ 1 หรือ 3 ช่องได้)
  spec หลวมลงมาก (ค่าเริ่ม):
    - match ขั้นต่ำ 40%  (ตัวที่หายทั้งตัว ~0%, ตัวครบที่ถ่ายเบลอ/จาง ≥ 60%)
    - ชิ้นเล็ก (ขีด "-", จุด ".", ชิ้นในโลโก้) ขั้นต่ำ 25%
    - ชิ้นใหญ่ (วงกลม / โลโก้) ใช้ "รูปร่าง" อย่างเดียวขั้นต่ำ 35%
    - ไม่มีกรอบม่วง / ไม่มีถ่ายใหม่
    - กรอบ OK สีเขียวแสดงอัตโนมัติ
"""
import math
import numpy as np
import cv2
import sharp_check as SC
try:
    import sharp_fix  # noqa: F401
except Exception as _ex:
    print("[sharp_loose] ไม่มี sharp_fix:", _ex)

LOOSE_VERSION = "loose v3"

LOOSE_DEFAULT = {
    "match_min": 0.40,         # %match ขั้นต่ำ (0-1)  ลด = alarm น้อยลง
    "match_min_small": 0.25,   # ชิ้นเล็ก ("-", ".", ชิ้นในโลโก้)
    "match_min_big": 0.35,     # ชิ้นใหญ่ (วงกลม/โลโก้) — ใช้รูปร่าง
    "match_search_frac": 0.25, # ค้นหาตำแหน่งรอบชิ้น ± สัดส่วนขนาดชิ้น
    "no_retake": True,
    "hide_extra": True,
    "show_ok_boxes": True,
    "test_crop": None,
}
LOOSE_SPEC = {
    "cover_min": 0.50, "shape_ncc_min": 0.30, "missing_blob_letter_frac": 0.50,
    "tol_px": 3, "local_shift_frac": 0.30, "min_inliers": 6, "min_inliers_force": 4,
    "check_extra": False, "continue_if_blur": True,
}
COLOR = {"OK": (0, 170, 0), "PARTIAL": (0, 165, 255), "MISSING": (0, 0, 255), "SKIP": (150, 150, 150)}


def _bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _chans(img):
    """ภาพ 1/3 ช่อง -> list ภาพเทา (L, B, G, R)"""
    if img.ndim == 2:
        return [img]
    b = _bgr(img)
    lab = cv2.cvtColor(b, cv2.COLOR_BGR2LAB)
    return [lab[:, :, 0]] + list(cv2.split(b))


def _flat_ink(g, k):
    """ลบพื้นหลัง/เงา -> 0..1 (ค่าสูง = เข้ม)"""
    g = g.astype(np.float32) + 1.0
    f = g / cv2.GaussianBlur(g, (0, 0), k)
    lo, hi = np.percentile(f, 2), np.percentile(f, 98)
    return 1.0 - np.clip((f - lo) / max(hi - lo, 1e-3), 0, 1)


def match_score(mg, tchans, box, cfg):
    """mg = Master เทา, tchans = Test (หลายช่อง) ขนาดเดียวกัน -> (match, หมึก, รูปร่าง)"""
    H, W = mg.shape
    x, y, w, h = box
    sz = max(w, h)
    tol = max(1, int(cfg.get("tol_px", 3)))
    s = max(3, int(cfg.get("match_search_frac", 0.25) * sz))
    p = tol + 2
    x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
    X0, Y0, X1, Y1 = max(0, x0 - s), max(0, y0 - s), min(W, x1 + s), min(H, y1 + s)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 1.0, 1.0, 1.0
    k = max(4.0, sz * 1.2)
    mi = _flat_ink(mg[Y0:Y1, X0:X1], k)
    ty0, tx0 = y0 - Y0, x0 - X0
    tpl = mi[ty0:ty0 + (y1 - y0), tx0:tx0 + (x1 - x0)]
    u = (tpl * 255).astype(np.uint8)
    _, own = cv2.threshold(u, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    own = own.astype(bool)
    if own.sum() < 4 or tpl.std() < 0.02:
        return 1.0, 1.0, 1.0
    sm = max(0.6, sz * 0.03)
    tpl_b = cv2.GaussianBlur(tpl, (0, 0), sm)
    best = None
    for tg in tchans:
        ti0 = _flat_ink(tg[Y0:Y1, X0:X1], k)
        for ti in (ti0, 1.0 - ti0):                       # polarity: หมึกเข้ม / หมึกสว่าง
            if ti.std() < 0.02:
                continue
            res = cv2.matchTemplate(cv2.GaussianBlur(ti, (0, 0), sm), tpl_b, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if best is None or mx > best[0]:
                best = (mx, loc, ti)
    if best is None:
        return 0.0, 0.0, 0.0
    shape, loc, ti = best
    shape = max(0.0, float(shape))
    patch = ti[loc[1]:loc[1] + tpl.shape[0], loc[0]:loc[0] + tpl.shape[1]]
    pu = (patch * 255).astype(np.uint8)
    thr, _ = cv2.threshold(pu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thr - float(np.percentile(pu, 30)) < 20:            # ไม่มีหมึกชัด = หาย
        return 0.0, 0.0, shape
    tink = cv2.dilate((pu > thr).astype(np.uint8), np.ones((2 * tol + 1,) * 2, np.uint8))
    cover = float(tink[own].mean())
    return math.sqrt(cover * shape), cover, shape


def draw_result(photo, els, show_ok=True):
    R = _bgr(photo).copy()
    for e in els:
        st = e.get("status", "OK")
        bad = st in ("MISSING", "PARTIAL")
        if not bad and (not show_ok or st == "SKIP"):
            continue
        x, y = int(e["x"] + e.get("dx", 0)), int(e["y"] + e.get("dy", 0))
        cv2.rectangle(R, (x - 3, y - 3), (x + int(e["w"]) + 3, y + int(e["h"]) + 3),
                      COLOR.get(st, (0, 170, 0)), 3 if bad else 1)
    return R


def _fail_result(master_bgr, test_bgr, notes):
    m, t = _bgr(master_bgr), _bgr(test_bgr)
    mh, mw = m.shape[:2]
    th, tw = t.shape[:2]
    s = min(mw / tw, mh / th)
    view = cv2.resize(t, (max(1, int(tw * s)), max(1, int(th * s))))
    canvas = np.full((mh, mw, 3), 255, np.uint8)
    canvas[:view.shape[0], :view.shape[1]] = view
    cv2.rectangle(canvas, (0, 0), (mw - 1, 40), (40, 40, 200), -1)
    cv2.putText(canvas, "ALIGN FAIL - check model / focus / crop", (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    rep = dict(verdict="NG", align_method="จับคู่ไม่ได้", align_inliers=0, n_elements=0, n_ok=0,
               n_missing=0, n_partial=0, n_skip=0, n_blur=0, n_dash=0, n_extra=0,
               elements=[], extras=[], engine=LOOSE_VERSION, notes=notes)
    return dict(verdict="NG", report=rep, overlay=canvas, aligned=canvas, notes=notes,
                debug_dir=None, test=test_bgr)


if not getattr(SC, "_LOOSE3", False):
    SC._LOOSE3 = True
    _base_load = getattr(SC, "_loose_base_load", SC.load_config)
    _base_inspect = getattr(SC, "_loose_base_inspect", SC.inspect)
    SC._loose_base_load, SC._loose_base_inspect = _base_load, _base_inspect

    def load_config():
        cfg = _base_load()
        if cfg.get("_loose") != LOOSE_VERSION:
            cfg.update(LOOSE_SPEC)
            cfg.update(LOOSE_DEFAULT)          # v3: ใส่ค่าหลวมใหม่ทับ (รวม match_min 40%)
            cfg["test_crop"] = cfg.get("test_crop")
            cfg["_loose"] = LOOSE_VERSION
            try:
                SC.save_config(cfg)
            except Exception:
                pass
        for k, v in LOOSE_DEFAULT.items():
            cfg.setdefault(k, v)
        for k in ("match_min", "match_min_small", "match_min_big"):
            try:
                cfg[k] = float(min(1.0, max(0.0, float(cfg[k]))))
            except (TypeError, ValueError):
                cfg[k] = LOOSE_DEFAULT[k]
        cfg["check_extra"] = False
        cfg["continue_if_blur"] = True
        return cfg

    def _apply_match(r, master_bgr, cfg):
        rep = r["report"]
        photo = r.get("aligned")
        if photo is None:
            return 0, None
        ph, pw = photo.shape[:2]
        mg = cv2.cvtColor(cv2.resize(_bgr(master_bgr), (pw, ph), interpolation=cv2.INTER_AREA),
                          cv2.COLOR_BGR2GRAY)
        tchans = _chans(photo)
        els = rep.get("elements", [])
        hs = [e["h"] for e in els if e.get("h", 0) > 3]
        med_h = float(np.median(hs)) if hs else 20.0
        n_fix, low = 0, None
        for e in els:
            box = (int(e["x"]), int(e["y"]), int(e["w"]), int(e["h"]))
            try:
                m, cov, shp = match_score(mg, tchans, box, cfg)
            except Exception as ex:
                e["note"] = f'{e.get("note", "")} | match error {ex}'
                continue
            big = max(box[2], box[3]) > 2.2 * med_h
            small = max(box[2], box[3]) < 0.6 * med_h or min(box[2], box[3]) < 0.25 * med_h
            if big:
                score = shp if cov >= 0.6 else m          # หมึกต้องอยู่จริงก่อน ถึงใช้รูปร่างอย่างเดียว
                mm, kind = cfg["match_min_big"], "ใหญ่"
            elif small:
                score, mm, kind = m, cfg["match_min_small"], "เล็ก"
            else:
                score, mm, kind = m, cfg["match_min"], ""
            e["match"] = round(score, 3)
            low = score if low is None else min(low, score)
            if e.get("status") not in ("MISSING", "PARTIAL"):
                continue
            tag = f"match {score:.0%} (หมึก {cov:.0%} รูปร่าง {shp:.2f}{' ' + kind if kind else ''})"
            if score >= mm:
                e["status"] = "OK"
                e["note"] = f'{e.get("note", "")} | {tag} ≥ {mm:.0%} -> OK'
                n_fix += 1
            else:
                e["note"] = f'{e.get("note", "")} | {tag} < {mm:.0%}'
        return n_fix, low

    def inspect(master_bgr, test_bgr, cfg=None, ignore_zones=None, debug_dir=None, force_continue=None):
        c = load_config()
        if cfg:
            c.update({k: v for k, v in cfg.items() if k not in LOOSE_SPEC and k not in LOOSE_DEFAULT})
            for k in ("match_min", "test_crop"):
                if k in cfg:
                    c[k] = cfg[k]
        cfg = c
        try:
            r = _base_inspect(master_bgr, test_bgr, cfg, ignore_zones, debug_dir, True)
        except TypeError:
            r = _base_inspect(master_bgr, test_bgr, cfg, ignore_zones, debug_dir)
        if r.get("verdict") == "RETAKE" or r.get("report") is None:
            return _fail_result(master_bgr, test_bgr,
                                [LOOSE_VERSION] + list(r.get("notes", [])) +
                                ["⚠ จับคู่ไม่ได้ -> NG (ไม่ใช่ถ่ายใหม่) — ตรวจโมเดล / โฟกัส / ครอบรูปให้เห็นมาร์ค"])
        rep = r["report"]
        try:
            n_fix, low = _apply_match(r, master_bgr, cfg)
            mnote = f"ล้าง alarm {n_fix} ชิ้น" + (f" | match ต่ำสุด {low:.0%}" if low is not None else "")
        except Exception as ex:
            mnote = f"คิด %match ไม่ได้ ({type(ex).__name__}: {ex})"
        rep["extras"] = []
        rep["n_extra"] = 0
        els = rep.get("elements", [])
        bad = [e for e in els if e.get("status") in ("MISSING", "PARTIAL")]
        rep["n_missing"] = sum(e.get("status") == "MISSING" for e in els)
        rep["n_partial"] = sum(e.get("status") == "PARTIAL" for e in els)
        rep["n_ok"] = len(els) - len(bad)
        rep["verdict"] = r["verdict"] = "NG" if bad else "OK"
        note = f"{LOOSE_VERSION} | match ขั้นต่ำ {cfg['match_min']:.0%} | {mnote}"
        r["notes"] = [note] + [n for n in r.get("notes", []) if "ถ่ายใหม่" not in str(n)]
        rep["notes"] = r["notes"]
        rep["align_method"] = f"{rep.get('align_method', '')} | {LOOSE_VERSION}"
        if r.get("aligned") is not None:
            r["aligned"] = _bgr(r["aligned"])
            r["overlay"] = draw_result(r["aligned"], els, bool(cfg.get("show_ok_boxes", True)))
        return r

    def inspect_safe(master_bgr, test_bgr, cfg=None, ignore_zones=None, debug_dir=None, force_continue=None):
        try:
            return inspect(master_bgr, test_bgr, cfg, ignore_zones, debug_dir, force_continue)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            return _fail_result(master_bgr, test_bgr,
                                [f"⚠ {LOOSE_VERSION} error: {type(ex).__name__}: {ex}"])

    SC.load_config = load_config
    SC.inspect = inspect_safe
    SC.match_score = match_score
    SC.LOOSE_VERSION = LOOSE_VERSION
    SC.VERSION = f"{getattr(SC, 'VERSION', 'sharp_check')} + {LOOSE_VERSION}"
    print(f"[sharp_loose] {LOOSE_VERSION} ติดตั้งแล้ว (ไม่เรียกฟังก์ชันภายในของ sharp_check)")

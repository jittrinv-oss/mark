#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sharp_loose.py v1 — แพตช์ตัวตรวจคมชัด (sharp_check v1.2 + sharp_fix v1.3)  *ไม่แก้ไฟล์เดิม*
ถูกเรียกจาก mark_ui_sharp.py v1.5 / web_sharp.py v1.4 (บรรทัด import sharp_loose)

 1) ไม่มี "ถ่ายใหม่" (สีส้ม) อีก — ตรวจต่อเสมอ (ภาพเบลอ / แสงสะท้อน / จุดอ้างอิงน้อย)
      จับคู่ไม่ได้ -> ลองจัดภาพสำรอง (prealign) -> ยังไม่ได้ -> NG พร้อมเหตุผล (ไม่ค้างเป็นส้ม)
 3) ปลด spec ให้หลวม + ไม่มีกรอบม่วง (หมึกเกิน) ทั้งโปรแกรมและเว็บ
 4) ภาพผลตรวจแสดงกรอบ OK (เขียว) อัตโนมัติ
 5) %match ต่อชิ้น (คำนวณบนภาพหลังแปลง polarity / sharpen ของ Master และ Test แล้ว)
      ชิ้นที่ตัวตรวจเดิมตีส้ม/แดง แต่ %match >= match_min (ค่าเริ่ม 50%) -> OK
      %match = sqrt( cover x shape )
        cover = หมึกของ "ตัวอักษรนั้นเอง" ใน Master ที่เจอใน Test (เผื่อตำแหน่ง)
        shape = ความเหมือนรูปร่าง (NCC) หลังค้นหาตำแหน่งรอบ ๆ
      ตัวที่หายจริง -> cover และ shape ต่ำทั้งคู่ -> ยัง NG

ค่าปรับได้ใน sharp_config.json (ปุ่ม "ตั้งค่าคมชัด") หรือช่อง "match ขั้นต่ำ %" บนหน้าจอ / เว็บ
"""
import math
import numpy as np
import cv2
import sharp_check as SC
try:
    import sharp_fix  # noqa: F401  ต้องแพตช์ fix v1.3 ก่อน
except Exception as _ex:
    print("sharp_fix:", _ex)

LOOSE_VERSION = "loose v1"

# ค่าที่ "ใส่ทับครั้งเดียว" ตอนติดตั้ง (หลังจากนั้นแก้ใน sharp_config.json ได้)
LOOSE_DEFAULT = {
    "loose_on": True,
    "match_min": 0.50,            # %match ขั้นต่ำ (0-1) — ลด = alarm น้อยลง
    "match_search_frac": 0.15,    # ค้นหาตำแหน่งรอบชิ้น ± สัดส่วนของขนาดชิ้น (ภาพจัดตำแหน่งแล้ว)
    "no_retake": True,            # ไม่ขึ้น "ถ่ายใหม่"
    "hide_extra": True,           # ไม่ตรวจ/ไม่วาด หมึกเกิน (ม่วง)
    "show_ok_boxes": True,        # วาดกรอบ OK สีเขียว
    "test_crop": None,            # กรอบครอบรูปกล้อง [x0,y0,x1,y1] สัดส่วน 0-1 (โปรแกรมหน้าจอ)
}
# spec เดิม -> หลวมลง (ใส่ทับครั้งเดียว)
LOOSE_SPEC = {
    "cover_min": 0.50,                # เดิม 0.70
    "shape_ncc_min": 0.30,            # เดิม 0.50
    "missing_blob_letter_frac": 0.50, # เดิม 0.30
    "tol_px": 3,                      # เดิม 2
    "local_shift_frac": 0.30,         # เดิม 0.20
    "min_inliers": 6,                 # เดิม 15
    "min_inliers_force": 4,           # เดิม 10
    "check_extra": False,             # เดิม True
    "continue_if_blur": True,         # เดิม False
}

if not getattr(SC, "_LOOSE_APPLIED", False):
    SC._LOOSE_APPLIED = True
    _prev_load = SC.load_config
    _prev_align = SC.align
    _prev_compare = SC.compare
    _prev_inspect = SC.inspect
    _prev_draw = SC.draw
    _CTX = {}

    # ---------------------------------------------------------------- config
    def load_config():
        cfg = _prev_load()
        if cfg.get("_loose") != LOOSE_VERSION:
            cfg.update(LOOSE_SPEC)
            for k, v in LOOSE_DEFAULT.items():
                cfg.setdefault(k, v)
            cfg["_loose"] = LOOSE_VERSION
            SC.save_config(cfg)
        for k, v in LOOSE_DEFAULT.items():
            cfg.setdefault(k, v)
        if cfg.get("hide_extra", True):
            cfg["check_extra"] = False
        if cfg.get("no_retake", True):
            cfg["continue_if_blur"] = True
        cfg["match_min"] = float(min(1.0, max(0.0, cfg.get("match_min", 0.5))))
        return cfg

    def _cfg(cfg):
        base = load_config()
        if cfg:
            base.update(cfg)
        if base.get("hide_extra", True):
            base["check_extra"] = False
        return base

    # ---------------------------------------------------------------- เก็บภาพหลังแปลงไว้ใช้คิด %match
    def align(mpre, tpre, mink, cfg):
        w, v, info = _prev_align(mpre, tpre, mink, cfg)
        _CTX["m"], _CTX["t"] = mpre, w
        return w, v, info

    # ---------------------------------------------------------------- %match
    def _flat_ink(g, k):
        """ภาพเทา (หมึกดำ) -> ค่าหมึก 0..1 (หมึก = สูง) หลังลบพื้นหลัง"""
        g = g.astype(np.float32) + 1.0
        f = g / cv2.GaussianBlur(g, (0, 0), k)
        lo, hi = np.percentile(f, 2), np.percentile(f, 98)
        return 1.0 - np.clip((f - lo) / max(hi - lo, 1e-3), 0, 1)

    def match_score(mg, tg, lab, i, r, cfg):
        """คืน (match, cover, shape) ของชิ้นที่ i"""
        H, W = mg.shape
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        sz = max(w, h)
        tol = max(1, int(cfg.get("tol_px", 3)))
        s = max(3, int(cfg.get("match_search_frac", 0.15) * sz))
        p = tol + 2
        x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
        X0, Y0, X1, Y1 = max(0, x0 - s), max(0, y0 - s), min(W, x1 + s), min(H, y1 + s)
        k = max(4.0, sz * 1.2)
        mi = _flat_ink(mg[Y0:Y1, X0:X1], k)
        own = (lab[Y0:Y1, X0:X1] == i)
        ty0, tx0 = y0 - Y0, x0 - X0
        tpl = mi[ty0:ty0 + (y1 - y0), tx0:tx0 + (x1 - x0)] * cv2.dilate(
            own[ty0:ty0 + (y1 - y0), tx0:tx0 + (x1 - x0)].astype(np.uint8), np.ones((3, 3), np.uint8))
        own_c = own[ty0:ty0 + (y1 - y0), tx0:tx0 + (x1 - x0)]
        if own_c.sum() < 6 or tpl.std() < 1e-3:
            return 1.0, 1.0, 1.0
        sm = max(0.6, sz * 0.03)
        tpl_b = cv2.GaussianBlur(tpl, (0, 0), sm)
        best = None
        ti0 = _flat_ink(tg[Y0:Y1, X0:X1], k)
        for ti in (ti0, 1.0 - ti0):                 # polarity: หมึกเข้ม / หมึกสว่าง
            if ti.std() < 0.02:
                continue
            res = cv2.matchTemplate(cv2.GaussianBlur(ti, (0, 0), sm), tpl_b, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if best is None or mx > best[0]:
                best = (mx, loc, ti)
        if best is None:
            return 0.0, 0.0, 0.0
        shape, loc, ti = best
        patch = ti[loc[1]:loc[1] + tpl.shape[0], loc[0]:loc[0] + tpl.shape[1]]
        u = (patch * 255).astype(np.uint8)
        thr, _ = cv2.threshold(u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bg = float(np.percentile(u, 30))
        if thr - bg < 25:                             # ไม่มีหมึกชัดในบริเวณ -> หาย
            return 0.0, 0.0, max(0.0, shape)
        tink = (u > thr).astype(np.uint8)
        tink = cv2.dilate(tink, np.ones((2 * tol + 1,) * 2, np.uint8))
        cover = float(tink[own_c].mean())
        shape = max(0.0, float(shape))
        return math.sqrt(cover * shape), cover, shape

    def compare(mink, tink, check_mask, cfg):
        rows, extras, miss, extra = _prev_compare(mink, tink, check_mask, cfg)
        if cfg.get("hide_extra", True):
            extras = []
        mg, tg = _CTX.get("m"), _CTX.get("t")
        if not cfg.get("loose_on", True) or mg is None or tg is None or tg.shape != mg.shape:
            return rows, extras, miss, extra
        mchk = cv2.bitwise_and(mink, check_mask)
        n, lab, st, _ = cv2.connectedComponentsWithStats(mchk, 8)
        key = {(int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3])): i for i in range(1, n)}
        mm = cfg["match_min"]
        for r in rows:
            i = key.get((r["x"], r["y"], r["w"], r["h"]))
            if i is None:
                continue
            m, cov, shp = match_score(mg, tg, lab, i, r, cfg)
            r["match"] = round(m, 3)
            if r["status"] == "OK":
                continue
            if m >= mm:
                r["status"] = "OK"
                r["note"] += f" | match {m:.0%} (หมึก {cov:.0%} รูปร่าง {shp:.2f}) ≥ {mm:.0%} -> OK"
            else:
                r["note"] += f" | match {m:.0%} (หมึก {cov:.0%} รูปร่าง {shp:.2f}) < {mm:.0%}"
        return rows, extras, miss, extra

    # ---------------------------------------------------------------- วาด
    def draw(base, rows, extras, skips, show_ok=None, glare=None, labels=None):
        cfg = load_config()
        if show_ok is None or show_ok is False:
            show_ok = bool(cfg.get("show_ok_boxes", True))
        return _prev_draw(base, rows, [] if cfg.get("hide_extra", True) else extras,
                          skips, show_ok, glare, labels)

    # ---------------------------------------------------------------- ตรวจ (ไม่มีถ่ายใหม่)
    def _fail_result(master_bgr, test_bgr, notes):
        mh, mw = master_bgr.shape[:2]
        th, tw = test_bgr.shape[:2]
        s = min(mw / tw, mh / th)
        view = cv2.resize(test_bgr, (max(1, int(tw * s)), max(1, int(th * s))))
        canvas = np.full((mh, mw, 3), 255, np.uint8)
        canvas[:view.shape[0], :view.shape[1]] = view
        cv2.rectangle(canvas, (0, 0), (mw - 1, 44), (40, 40, 200), -1)
        cv2.putText(canvas, "ALIGN FAIL - check model / focus", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        rep = dict(verdict="NG", align_method="จับคู่ไม่ได้", align_inliers=0, n_elements=0, n_ok=0,
                   n_missing=0, n_partial=0, n_skip=0, n_blur=0, n_dash=0, n_extra=0,
                   elements=[], extras=[], engine=SC.VERSION + " + " + LOOSE_VERSION, notes=notes)
        return dict(verdict="NG", report=rep, overlay=canvas, aligned=canvas, notes=notes,
                    debug_dir=None, test=test_bgr, test_view=test_bgr)

    def inspect(master_bgr, test_bgr, cfg=None, ignore_zones=None, debug_dir=None, force_continue=None):
        cfg = _cfg(cfg)
        cont = True if cfg.get("no_retake", True) else force_continue
        r = _prev_inspect(master_bgr, test_bgr, cfg, ignore_zones, debug_dir, cont)
        if r["verdict"] == "RETAKE" and cfg.get("no_retake", True):
            try:                                         # จัดภาพสำรอง: prealign (แผนที่หมึก + SIFT)
                import prealign as PA
                old = PA.MIN_INLIERS
                PA.MIN_INLIERS = 6
                try:
                    p = PA.prealign(master_bgr, test_bgr)
                finally:
                    PA.MIN_INLIERS = old
                if p["ok"]:
                    mh, mw = master_bgr.shape[:2]
                    mx, my = int(round(mw * PA.MARGIN)), int(round(mh * PA.MARGIN))
                    t2 = p["test"][my:my + mh, mx:mx + mw]
                    r2 = _prev_inspect(master_bgr, t2, cfg, ignore_zones, debug_dir, True)
                    if r2["verdict"] != "RETAKE":
                        r2["notes"] = [f"จัดภาพสำรอง (prealign จุดอ้างอิง {p['inliers']})"] + r2["notes"]
                        r = r2
            except Exception as ex:
                r["notes"] = list(r.get("notes", [])) + [f"จัดภาพสำรองไม่สำเร็จ ({ex})"]
        if r["verdict"] == "RETAKE":
            return _fail_result(master_bgr, test_bgr,
                                [LOOSE_VERSION] + list(r.get("notes", [])) +
                                ["⚠ จับคู่ไม่ได้ -> NG (ไม่ใช่ถ่ายใหม่) — ตรวจโมเดล / โฟกัส / ครอบรูปให้เห็นมาร์ค"])
        rep = r.get("report")
        if rep:
            if cfg.get("hide_extra", True):
                rep["extras"] = []
                rep["n_extra"] = 0
            bad = [e for e in rep["elements"] if e["status"] in ("MISSING", "PARTIAL")]
            rep["n_missing"] = sum(e["status"] == "MISSING" for e in rep["elements"])
            rep["n_partial"] = sum(e["status"] == "PARTIAL" for e in rep["elements"])
            rep["n_ok"] = rep["n_elements"] - len(bad)
            rep["verdict"] = r["verdict"] = "NG" if bad else "OK"
            ms = [e["match"] for e in rep["elements"] if e.get("match") is not None]
            note = f"{LOOSE_VERSION} | match ขั้นต่ำ {cfg['match_min']:.0%}"
            if ms:
                note += f" | match ต่ำสุด {min(ms):.0%}"
            r["notes"] = [note] + [n for n in r["notes"] if "ถ่ายใหม่" not in n]
            rep["notes"] = r["notes"]
            rep["align_method"] = f"{rep.get('align_method', '')} | {LOOSE_VERSION}"
            # วาดใหม่: กรอบ OK เขียว ไม่มีม่วง
            if r.get("aligned") is not None:
                r["overlay"] = draw(r["aligned"], rep["elements"], [], [], True)
        return r

    SC.load_config = load_config
    SC.align = align
    SC.compare = compare
    SC.draw = draw
    SC.inspect = inspect
    SC.match_score = match_score
    SC.VERSION = SC.VERSION + " + " + LOOSE_VERSION
    print(f"[sharp_loose] {LOOSE_VERSION} ติดตั้งแล้ว")

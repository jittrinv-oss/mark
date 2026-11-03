#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sharp_loose.py v2 — แพตช์ตัวตรวจคมชัด (sharp_check v1.2 + sharp_fix v1.3)  *ไม่แก้ไฟล์เดิม*
ถูกเรียกจาก mark_ui_sharp.py v1.6 / web_sharp.py v1.5 (บรรทัด import sharp_loose)

v2 (แก้ ERR / เว็บไม่ตรวจ):
  - แก้ error "align() takes 4 positional arguments but 5 were given"
  - ไม่แตะฟังก์ชันภายในของ sharp_check (align / compare / draw) อีก
    ห่อเฉพาะ inspect() ชั้นนอกสุด -> ไม่ชนกับ sharp_fix / ไฟล์เก่าบน Pi
  - ถ้า sharp_fix ไม่มี / import ไม่ได้ ก็ยังทำงาน
  - ถ้า inspect ของตัวใหม่ error -> ตกกลับไปใช้ inspect เดิมอัตโนมัติ (ไม่ขึ้น ERR / เว็บไม่หาย)

ฟังก์ชัน (เหมือน v1):
 1) ไม่มี "ถ่ายใหม่" (สีส้ม) — ตรวจต่อเสมอ, จับคู่ไม่ได้ -> ลอง prealign -> ยังไม่ได้ -> NG พร้อมเหตุผล
 3) spec หลวม + ไม่มีกรอบม่วง (หมึกเกิน)
 4) ภาพผลแสดงกรอบ OK (เขียว)
 5) %match ต่อชิ้น บนภาพหลังแปลง polarity + sharpen (ภาพเทาที่ sharp_check ใช้จริง)
      ชิ้นที่ขึ้นส้ม/แดง แต่ %match >= match_min (เริ่ม 50%) -> OK
      %match = sqrt(หมึก x รูปร่าง)
"""
import math
import numpy as np
import cv2
import sharp_check as SC
try:
    import sharp_fix  # noqa: F401
except Exception as _ex:
    print("[sharp_loose] ไม่มี sharp_fix:", _ex)

LOOSE_VERSION = "loose v2"

LOOSE_DEFAULT = {
    "match_min": 0.50,            # %match ขั้นต่ำ (0-1)  ลด = alarm น้อยลง
    "match_search_frac": 0.15,    # ค้นหาตำแหน่งรอบชิ้น ± สัดส่วนขนาดชิ้น
    "no_retake": True,            # ไม่ขึ้น "ถ่ายใหม่"
    "hide_extra": True,           # ไม่ตรวจ/ไม่วาด หมึกเกิน (ม่วง)
    "show_ok_boxes": True,        # วาดกรอบ OK สีเขียว
    "test_crop": None,            # กรอบครอบรูปกล้อง [x0,y0,x1,y1] (สัดส่วน 0-1)
}
LOOSE_SPEC = {                    # ใส่ทับครั้งเดียวตอนติดตั้ง (แก้ต่อใน sharp_config.json ได้)
    "cover_min": 0.50,
    "shape_ncc_min": 0.30,
    "missing_blob_letter_frac": 0.50,
    "tol_px": 3,
    "local_shift_frac": 0.30,
    "min_inliers": 6,
    "min_inliers_force": 4,
    "check_extra": False,
    "continue_if_blur": True,
}

if not getattr(SC, "_LOOSE2", False):
    SC._LOOSE2 = True
    _base_load = SC.load_config
    _base_inspect = SC.inspect

    # ------------------------------------------------------------ config
    def load_config():
        cfg = _base_load()
        if cfg.get("_loose") != LOOSE_VERSION:
            cfg.update(LOOSE_SPEC)
            for k, v in LOOSE_DEFAULT.items():
                cfg.setdefault(k, v)
            cfg["_loose"] = LOOSE_VERSION
            SC.save_config(cfg)
        for k, v in LOOSE_DEFAULT.items():
            cfg.setdefault(k, v)
        try:
            cfg["match_min"] = float(min(1.0, max(0.0, float(cfg.get("match_min", 0.5)))))
        except (TypeError, ValueError):
            cfg["match_min"] = 0.5
        if cfg.get("hide_extra", True):
            cfg["check_extra"] = False
        if cfg.get("no_retake", True):
            cfg["continue_if_blur"] = True
        return cfg

    # ------------------------------------------------------------ %match
    def _flat_ink(g, k):
        g = g.astype(np.float32) + 1.0
        f = g / cv2.GaussianBlur(g, (0, 0), k)
        lo, hi = np.percentile(f, 2), np.percentile(f, 98)
        return 1.0 - np.clip((f - lo) / max(hi - lo, 1e-3), 0, 1)     # หมึก = สูง

    def match_score(mg, tg, own_full, r, cfg):
        """mg/tg = ภาพเทา (หมึกดำ) ขนาดเดียวกัน, own_full = mask ของชิ้นนี้ใน Master -> (match, หมึก, รูปร่าง)"""
        H, W = mg.shape
        x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
        sz = max(w, h)
        tol = max(1, int(cfg.get("tol_px", 3)))
        s = max(3, int(cfg.get("match_search_frac", 0.15) * sz))
        p = tol + 2
        x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
        X0, Y0, X1, Y1 = max(0, x0 - s), max(0, y0 - s), min(W, x1 + s), min(H, y1 + s)
        k = max(4.0, sz * 1.2)
        mi = _flat_ink(mg[Y0:Y1, X0:X1], k)
        ti0 = _flat_ink(tg[Y0:Y1, X0:X1], k)
        ty0, tx0 = y0 - Y0, x0 - X0
        own = own_full[y0:y1, x0:x1] > 0
        if own.sum() < 6:
            return 1.0, 1.0, 1.0
        tpl = mi[ty0:ty0 + (y1 - y0), tx0:tx0 + (x1 - x0)] * cv2.dilate(own.astype(np.uint8), np.ones((3, 3), np.uint8))
        if tpl.std() < 1e-3:
            return 1.0, 1.0, 1.0
        sm = max(0.6, sz * 0.03)
        tpl_b = cv2.GaussianBlur(tpl, (0, 0), sm)
        best = None
        for ti in (ti0, 1.0 - ti0):                         # polarity หมึกเข้ม / หมึกสว่าง
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
        if thr - float(np.percentile(u, 30)) < 25:              # ไม่มีหมึกชัด = หาย
            return 0.0, 0.0, max(0.0, float(shape))
        tink = cv2.dilate((u > thr).astype(np.uint8), np.ones((2 * tol + 1,) * 2, np.uint8))
        cover = float(tink[own].mean())
        shape = max(0.0, float(shape))
        return math.sqrt(cover * shape), cover, shape

    def _master_gray(master_bgr, cfg):
        m, _ = SC._resize_long(master_bgr, cfg["proc_long_side"])
        mpre, _ = SC.preprocess(m, cfg)
        mbin = SC.binarize(mpre, cfg)
        mink = SC.mark_roi(mbin, cfg)[1] if cfg.get("roi_auto") else mbin
        return mpre, mink

    def _apply_match(r, master_bgr, cfg):
        """ตัดสินซ้ำด้วย %match บนภาพเทาหลังแปลง polarity + sharpen"""
        rep = r.get("report")
        tg = r.get("after_aligned")
        if rep is None:
            return
        mpre, mink = _master_gray(master_bgr, cfg)
        if tg is None:
            ph = r.get("aligned")
            if ph is None:
                return
            ph = cv2.resize(ph, (mpre.shape[1], mpre.shape[0])) if ph.shape[:2] != mpre.shape else ph
            tg, _ = SC.preprocess(ph, cfg)
        if tg.shape != mpre.shape:
            tg = cv2.resize(tg, (mpre.shape[1], mpre.shape[0]))
        n, lab, st, _ = cv2.connectedComponentsWithStats(mink, 8)
        mm = cfg["match_min"]
        for e in rep["elements"]:
            x, y, w, h = e["x"], e["y"], e["w"], e["h"]
            sub = lab[y:y + h, x:x + w]
            ids, cnt = np.unique(sub[sub > 0], return_counts=True)
            if len(ids) == 0:
                continue
            i = int(ids[np.argmax(cnt)])
            own = (lab == i).astype(np.uint8)
            m, cov, shp = match_score(mpre, tg, own, e, cfg)
            e["match"] = round(m, 3)
            if e["status"] == "OK":
                continue
            if m >= mm:
                e["status"] = "OK"
                e["note"] = f'{e.get("note", "")} | match {m:.0%} (หมึก {cov:.0%} รูปร่าง {shp:.2f}) ≥ {mm:.0%} -> OK'
            else:
                e["note"] = f'{e.get("note", "")} | match {m:.0%} (หมึก {cov:.0%} รูปร่าง {shp:.2f}) < {mm:.0%}'

    # ------------------------------------------------------------ ผลเมื่อจับคู่ไม่ได้ (แทน "ถ่ายใหม่")
    def _fail_result(master_bgr, test_bgr, notes):
        mh, mw = master_bgr.shape[:2]
        th, tw = test_bgr.shape[:2]
        s = min(mw / tw, mh / th)
        view = cv2.resize(test_bgr, (max(1, int(tw * s)), max(1, int(th * s))))
        canvas = np.full((mh, mw, 3), 255, np.uint8)
        canvas[:view.shape[0], :view.shape[1]] = view
        cv2.rectangle(canvas, (0, 0), (mw - 1, 40), (40, 40, 200), -1)
        cv2.putText(canvas, "ALIGN FAIL - check model / focus / crop", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        rep = dict(verdict="NG", align_method="จับคู่ไม่ได้", align_inliers=0, n_elements=0, n_ok=0,
                   n_missing=0, n_partial=0, n_skip=0, n_blur=0, n_dash=0, n_extra=0,
                   elements=[], extras=[], engine=f"{SC.VERSION} + {LOOSE_VERSION}", notes=notes)
        return dict(verdict="NG", report=rep, overlay=canvas, aligned=canvas, notes=notes,
                    debug_dir=None, test=test_bgr)

    def _prealign_retry(master_bgr, test_bgr, cfg, ignore_zones, debug_dir):
        import prealign as PA
        old = PA.MIN_INLIERS
        PA.MIN_INLIERS = 6
        try:
            p = PA.prealign(master_bgr, test_bgr)
        finally:
            PA.MIN_INLIERS = old
        if not p.get("ok"):
            return None
        mh, mw = master_bgr.shape[:2]
        mx, my = int(round(mw * PA.MARGIN)), int(round(mh * PA.MARGIN))
        t2 = p["test"][my:my + mh, mx:mx + mw]
        r2 = _base_inspect(master_bgr, t2, cfg, ignore_zones, debug_dir, True)
        if r2.get("verdict") == "RETAKE":
            return None
        r2["notes"] = [f"จัดภาพสำรอง (prealign จุดอ้างอิง {p['inliers']})"] + list(r2.get("notes", []))
        return r2

    # ------------------------------------------------------------ inspect (ห่อชั้นนอกสุดเท่านั้น)
    def inspect(master_bgr, test_bgr, cfg=None, ignore_zones=None, debug_dir=None, force_continue=None):
        base_cfg = load_config()
        if cfg:
            base_cfg.update({k: v for k, v in cfg.items() if k not in LOOSE_SPEC})
        cfg = base_cfg
        r = _base_inspect(master_bgr, test_bgr, cfg, ignore_zones, debug_dir, True)
        if r.get("verdict") == "RETAKE" and cfg.get("no_retake", True):
            try:
                r2 = _prealign_retry(master_bgr, test_bgr, cfg, ignore_zones, debug_dir)
                if r2 is not None:
                    r = r2
            except Exception as ex:
                r["notes"] = list(r.get("notes", [])) + [f"จัดภาพสำรองไม่สำเร็จ ({ex})"]
        if r.get("verdict") == "RETAKE":
            return _fail_result(master_bgr, test_bgr,
                                [LOOSE_VERSION] + list(r.get("notes", [])) +
                                ["⚠ จับคู่ไม่ได้ -> NG (ไม่ใช่ถ่ายใหม่) — ตรวจโมเดล / โฟกัส / ครอบรูปให้เห็นมาร์ค"])
        rep = r.get("report")
        if rep is None:
            return r
        try:
            _apply_match(r, master_bgr, cfg)
        except Exception as ex:
            r["notes"] = list(r.get("notes", [])) + [f"คิด %match ไม่ได้ ({ex})"]
        rep["extras"] = [] if cfg.get("hide_extra", True) else rep.get("extras", [])
        rep["n_extra"] = len(rep["extras"])
        els = rep["elements"]
        bad = [e for e in els if e["status"] in ("MISSING", "PARTIAL")]
        rep["n_missing"] = sum(e["status"] == "MISSING" for e in els)
        rep["n_partial"] = sum(e["status"] == "PARTIAL" for e in els)
        rep["n_ok"] = len(els) - len(bad)
        rep["verdict"] = r["verdict"] = "NG" if (bad or rep["extras"]) else "OK"
        ms = [e["match"] for e in els if e.get("match") is not None]
        note = f"{LOOSE_VERSION} | match ขั้นต่ำ {cfg['match_min']:.0%}" + (f" | match ต่ำสุด {min(ms):.0%}" if ms else "")
        r["notes"] = [note] + [n for n in r.get("notes", []) if "ถ่ายใหม่" not in n]
        rep["notes"] = r["notes"]
        rep["align_method"] = f"{rep.get('align_method', '')} | {LOOSE_VERSION}"
        if r.get("aligned") is not None:
            r["overlay"] = SC.draw(r["aligned"], els, rep["extras"], [], bool(cfg.get("show_ok_boxes", True)))
        return r

    def inspect_safe(master_bgr, test_bgr, cfg=None, ignore_zones=None, debug_dir=None, force_continue=None):
        try:
            return inspect(master_bgr, test_bgr, cfg, ignore_zones, debug_dir, force_continue)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            r = _base_inspect(master_bgr, test_bgr, cfg, ignore_zones, debug_dir, True)
            r["notes"] = [f"⚠ {LOOSE_VERSION} error: {ex} — ใช้ตัวตรวจเดิม"] + list(r.get("notes", []))
            if r.get("verdict") == "RETAKE":
                return _fail_result(master_bgr, test_bgr, r["notes"])
            return r

    SC.load_config = load_config
    SC.inspect = inspect_safe
    SC.match_score = match_score
    SC.LOOSE_VERSION = LOOSE_VERSION
    SC.VERSION = f"{SC.VERSION} + {LOOSE_VERSION}"
    print(f"[sharp_loose] {LOOSE_VERSION} ติดตั้งแล้ว")

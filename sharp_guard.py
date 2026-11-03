#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sharp_guard.py v1 — ด่านตรวจ "เส้น/ขีดหายจริง" ต่อจาก sharp_loose v3  *ไม่แก้ sharp_loose / sharp_check / sharp_fix*
ถูกเรียกจาก mark_ui_sharp.py v1.7 / web_sharp.py v1.6 (บรรทัด import sharp_guard)

ปัญหาที่แก้:
 1) SNI ขีดล่างหาย / ขาตั้งตัว L หาย / ขา M หาย -> ผ่านเป็น OK
    สาเหตุ: loose v3 ค้นหาตำแหน่งรอบชิ้น ±25% ของขนาดชิ้น -> ขีดล่างที่หายไป "เลื่อนไปจับ"
            ขีดบน / ฐานตัวอักษรข้าง ๆ ได้ และเกณฑ์ match 40% ยอมให้ตัวที่ขาดครึ่งตัวผ่าน
    แก้:   - หาตำแหน่งจาก "ตัวอักษรรอบข้าง" (หน้าต่างกว้าง ~1.2 เท่าความสูงตัวอักษร) แล้วล็อกตำแหน่ง
              ขยับเองได้แค่ ±2 px -> ขีดที่หายไม่สามารถไปจับเส้นอื่นแทนได้
            - ดูทีละ "ส่วนของเส้น": ส่วนที่หาย ต้องเป็นก้อนต่อเนื่อง ≥15% ของชิ้น และยาว ≥40% ของ
              ความกว้างหรือสูงของชิ้น (เส้นตรงบาง ≥30% ของความยาว) -> ส้ม, หายเกือบทั้งชิ้น -> แดง
            - ความสว่างหมึก/พื้น เทียบในบริเวณนั้นเอง (ทนภาพจาง/เบลอ/เงา/กระจกเขียว)
 2) กรอบเขียวไม่ตรงตัวอักษร
    สาเหตุ: กรอบวาดที่ตำแหน่ง Master + การเลื่อน dx,dy ของตัวตรวจเดิม (ค้นได้ไกล ±30% ความสูงตัวอักษร
            -> ตัวที่ขาดถูกเลื่อนไปจับตัวข้าง ๆ -> กรอบเบี้ยว)
    แก้:   วาดกรอบที่ตำแหน่งจากข้อ 1 (ล็อกจากตัวอักษรรอบข้าง) ทุกชิ้น

ด่านนี้ "เพิ่ม" NG ได้อย่างเดียว ไม่ล้างผลของ loose v3 และไม่แตะการลดส้ม/แดงของ loose v3
ปิด/ปรับได้ใน sharp_config.json: guard_on, guard_present, guard_part_area, guard_part_extent, guard_line_extent
"""
import numpy as np
import cv2
import sharp_check as SC
import sharp_loose  # noqa: F401  ต้องติดตั้ง loose v3 ก่อน (guard ห่อชั้นนอก)

GUARD_VERSION = "guard v1"
GUARD_DEFAULT = {
    "guard_on": True,
    "guard_present": 0.35,     # ความเข้มหมึก (เทียบหมึกรอบข้าง) ที่ถือว่า "มีหมึก"
    "guard_part_area": 0.15,   # ส่วนที่หายต่อเนื่อง ≥ สัดส่วนนี้ของชิ้น
    "guard_part_extent": 0.40, # และยาว ≥ สัดส่วนนี้ของความกว้างหรือสูงของชิ้น
    "guard_line_extent": 0.30, # เส้นตรงบาง (ขีดล่าง/ขีดบน): ส่วนที่หายยาว ≥ สัดส่วนนี้ของเส้น
    "guard_missing": 0.70,     # หาย ≥ สัดส่วนนี้ของชิ้น = หายทั้งชิ้น (แดง)
    "guard_shift_px": 2,       # ขยับเองได้ ±px หลังล็อกตำแหน่งจากตัวอักษรรอบข้าง
}


def _cfg():
    try:
        c = SC.load_config()
    except Exception:
        c = {}
    out = dict(GUARD_DEFAULT)
    for k in GUARD_DEFAULT:
        if k in c:
            out[k] = c[k]
    return out


def _bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img[:, :, :3]


def _flat(ch, k):
    f = ch.astype(np.float32) + 1.0
    return f / cv2.GaussianBlur(f, (0, 0), k) - 1.0           # >0 สว่างกว่ารอบข้าง, <0 มืดกว่า


def _master_ink(master_bgr, shape):
    """Master -> (หมึก 0..1, หมึก binary) ในพิกัดภาพผล  (ลบเส้นสี เช่น กรอบประแดง)"""
    m = _bgr(master_bgr)
    b = m.astype(np.int16)
    col = ((b.max(axis=2) - b.min(axis=2)) >= 40) & (b.max(axis=2) >= 120)
    g = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    H, W = shape
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
    col = cv2.resize(col.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    _, ob = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (ob == 0).mean() < 0.5                          # หมึก = กลุ่มที่น้อยกว่า
    k = max(15.0, 0.03 * max(H, W))
    f = _flat(g, k)
    ink = -f if dark else f
    ink[col] = 0
    ink = np.clip(ink / max(float(np.percentile(ink, 99.5)), 1e-3), 0, 1)
    _, mb = cv2.threshold((ink * 255).astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mb[col] = 0
    return ink, mb.astype(np.uint8)


def _test_ink(photo, mb, k):
    """ภาพผล (Test จัดตำแหน่งแล้ว) -> หมึก (ค่าสูง = หมึก) เลือกช่องสี/ขั้วที่ตรงกับ Master มากที่สุด"""
    p = _bgr(photo)
    lab = cv2.cvtColor(p, cv2.COLOR_BGR2LAB)
    chans = [lab[:, :, 0]] + list(cv2.split(p))
    valid = cv2.erode((p.min(axis=2) < 254).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    sel = valid & (cv2.dilate(mb, np.ones((25, 25), np.uint8)) > 0)
    if sel.sum() < 100:
        sel = valid
    mv = mb[sel].astype(np.float32)
    mv -= mv.mean()
    best = None
    for ch in chans:
        f = _flat(ch, k)
        v = f[sel] - f[sel].mean()
        r = float((v * mv).sum() / (np.sqrt((v * v).sum() * (mv * mv).sum()) + 1e-9))
        if best is None or abs(r) > abs(best[0]):
            best = (r, f)
    r, f = best
    return (f if r > 0 else -f), valid, abs(r)


def _judge(e, mink, mb_full, tink, valid, text_h, gcon, cfg):
    H, W = mb_full.shape
    x, y, w, h = int(e["x"]), int(e["y"]), int(e["w"]), int(e["h"])
    # ชิ้นของตัวเอง (หมึก Master ในกรอบ ที่จุดกลางอยู่ในกรอบชิ้น)
    p = 3
    X0, Y0, X1, Y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
    sub = mb_full[Y0:Y1, X0:X1]
    n, lab, st, cen = cv2.connectedComponentsWithStats(sub, 8)
    own = np.zeros_like(sub, bool)
    for i in range(1, n):
        cx, cy = cen[i][0] + X0, cen[i][1] + Y0
        if x - 1 <= cx <= x + w + 1 and y - 1 <= cy <= y + h + 1 and st[i, 4] >= 4:
            own |= lab == i
    if own.sum() < 8:
        return None
    # หน้าต่างบริบท: ตัวอักษรรอบข้าง -> ล็อกตำแหน่ง
    c = int(1.2 * text_h)
    R = int(max(4, 0.30 * text_h))
    wx0, wy0, wx1, wy1 = max(0, x - c), max(0, y - c), min(W, x + w + c), min(H, y + h + c)
    sx0, sy0, sx1, sy1 = max(0, wx0 - R), max(0, wy0 - R), min(W, wx1 + R), min(H, wy1 + R)
    tpl = cv2.GaussianBlur(mink[wy0:wy1, wx0:wx1], (0, 0), 1.5)
    srch = cv2.GaussianBlur(tink[sy0:sy1, sx0:sx1], (0, 0), 1.5)
    if tpl.std() < 1e-3 or srch.shape[0] < tpl.shape[0] or srch.shape[1] < tpl.shape[1]:
        return None
    res = cv2.matchTemplate(srch, tpl, cv2.TM_CCOEFF_NORMED)
    _, ctx, _, loc = cv2.minMaxLoc(res)
    if ctx < 0.30:
        return None                                        # บริบทจับคู่ไม่ได้ -> ไม่ตัดสิน
    dx, dy = loc[0] - (wx0 - sx0), loc[1] - (wy0 - sy0)
    # ขยับละเอียด ±guard_shift_px ด้วยชิ้นตัวเอง
    s = int(cfg["guard_shift_px"])
    own_full = np.zeros((H, W), np.uint8)
    own_full[Y0:Y1, X0:X1] = own
    best = None
    for ddy in range(-s, s + 1):
        for ddx in range(-s, s + 1):
            M = np.float32([[1, 0, dx + ddx], [0, 1, dy + ddy]])
            ox0, oy0 = X0 + dx + ddx, Y0 + dy + ddy
            if ox0 < 0 or oy0 < 0 or ox0 + (X1 - X0) > W or oy0 + (Y1 - Y0) > H:
                continue
            v = float(tink[oy0:oy0 + (Y1 - Y0), ox0:ox0 + (X1 - X0)][own].mean())
            if best is None or v > best[0]:
                best = (v, dx + ddx, dy + ddy)
    if best is None:
        return None
    _, dx, dy = best
    ox0, oy0 = X0 + dx, Y0 + dy
    # ระดับหมึก / พื้น ในหน้าต่างบริบท (ตำแหน่งที่ล็อกแล้ว)
    tw = tink[wy0 + dy:wy1 + dy, wx0 + dx:wx1 + dx] if (wy0 + dy >= 0 and wx0 + dx >= 0 and wy1 + dy <= H and wx1 + dx <= W) else None
    if tw is None or tw.shape != (wy1 - wy0, wx1 - wx0):
        return None
    vw = valid[wy0 + dy:wy1 + dy, wx0 + dx:wx1 + dx]
    mbw = mb_full[wy0:wy1, wx0:wx1] > 0
    bgm = (cv2.dilate(mbw.astype(np.uint8), np.ones((7, 7), np.uint8)) == 0) & vw
    inkm = (cv2.erode(mbw.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & vw
    if bgm.sum() < 30:
        return None
    bg = float(np.median(tw[bgm]))
    ink_ref = float(np.percentile(tw[inkm], 85)) if inkm.sum() >= 20 else bg
    con = max(ink_ref - bg, 0.6 * gcon)
    if con <= 1e-3:
        return None
    noise = float(np.std(tw[bgm])) / con
    thr = float(min(0.6, max(cfg["guard_present"], 2.5 * noise)))
    tp = tink[oy0:oy0 + (Y1 - Y0), ox0:ox0 + (X1 - X0)]
    vp = valid[oy0:oy0 + (Y1 - Y0), ox0:ox0 + (X1 - X0)]
    cmap = (tp - bg) / con
    cmax = cv2.dilate(cmap.astype(np.float32), np.ones((5, 5), np.uint8))   # เผื่อ ±2 px
    miss = own & (cmax < thr) & vp
    miss = cv2.morphologyEx(miss.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    area = int(own.sum())
    tot = float(miss.sum()) / area
    ys, xs = np.nonzero(own)
    ew, eh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    line = max(ew, eh) >= 3 * min(ew, eh)
    status, why = None, ""
    if tot >= cfg["guard_missing"]:
        status, why = "MISSING", f"หาย {tot:.0%}"
    else:
        nm, lm, sm, _ = cv2.connectedComponentsWithStats(miss.astype(np.uint8), 8)
        for j in range(1, nm):
            a = sm[j, 4] / area
            fx, fy = sm[j, 2] / ew, sm[j, 3] / eh
            if line:
                along = fx if ew >= eh else fy
                if along >= cfg["guard_line_extent"] and a >= 0.10:
                    status, why = "PARTIAL", f"เส้นขาด {along:.0%} ของความยาว"
                    break
            elif a >= cfg["guard_part_area"] and max(fx, fy) >= cfg["guard_part_extent"]:
                status, why = "PARTIAL", f"ส่วนที่หาย {a:.0%} ยาว {max(fx, fy):.0%}"
                break
    return dict(dx=int(dx), dy=int(dy), status=status, why=why, miss=tot, ctx=ctx)


def apply_guard(r, master_bgr):
    cfg = _cfg()
    rep = r.get("report")
    photo = r.get("aligned")
    if not cfg["guard_on"] or rep is None or photo is None or not rep.get("elements"):
        return r, ""
    H, W = photo.shape[:2]
    mink, mb = _master_ink(master_bgr, (H, W))
    els = rep["elements"]
    hs = [e["h"] for e in els if e.get("h", 0) > 3]
    text_h = float(np.median(hs)) if hs else 20.0
    tink, valid, corr = _test_ink(photo, mb, max(15.0, 2.0 * text_h))
    if corr < 0.15:
        return r, f"{GUARD_VERSION}: ภาพไม่ตรงกับ Master พอ — ไม่ตรวจเพิ่ม"
    sel = valid & (mb > 0)
    selb = valid & (cv2.dilate(mb, np.ones((7, 7), np.uint8)) == 0)
    gcon = float(np.percentile(tink[sel], 85) - np.median(tink[selb])) if sel.sum() > 50 and selb.sum() > 50 else 0.0
    added, moved = 0, 0
    for e in els:
        if e.get("status") == "SKIP":
            continue
        try:
            g = _judge(e, mink, mb, tink, valid, text_h, gcon, cfg)
        except Exception as ex:
            e["note"] = f'{e.get("note", "")} | guard error {ex}'
            continue
        if g is None:
            continue
        e["dx"], e["dy"] = g["dx"], g["dy"]                 # วาดกรอบตรงตำแหน่งจริง
        moved += 1
        if g["status"] and e.get("status") == "OK":
            e["status"] = g["status"]
            e["note"] = f'{e.get("note", "")} | {GUARD_VERSION}: {g["why"]}'
            added += 1
    bad = [e for e in els if e.get("status") in ("MISSING", "PARTIAL")]
    rep["n_missing"] = sum(e.get("status") == "MISSING" for e in els)
    rep["n_partial"] = sum(e.get("status") == "PARTIAL" for e in els)
    rep["n_ok"] = len(els) - len(bad)
    rep["verdict"] = r["verdict"] = "NG" if bad else "OK"
    try:
        show_ok = bool(SC.load_config().get("show_ok_boxes", True))
    except Exception:
        show_ok = True
    r["overlay"] = sharp_loose.draw_result(photo, els, show_ok)
    return r, f"{GUARD_VERSION}: พบเส้นขาด/หายเพิ่ม {added} ชิ้น"


if not getattr(SC, "_GUARD1", False):
    SC._GUARD1 = True
    _loose_inspect = SC.inspect

    def inspect(master_bgr, test_bgr, cfg=None, ignore_zones=None, debug_dir=None, force_continue=None):
        r = _loose_inspect(master_bgr, test_bgr, cfg, ignore_zones, debug_dir, force_continue)
        try:
            r, note = apply_guard(r, master_bgr)
        except Exception as ex:
            note = f"{GUARD_VERSION} error: {type(ex).__name__}: {ex} — ใช้ผล loose v3"
        if note:
            r["notes"] = [note] + list(r.get("notes", []))
            if r.get("report") is not None:
                r["report"]["notes"] = r["notes"]
        return r

    SC.inspect = inspect
    SC.GUARD_VERSION = GUARD_VERSION
    SC.VERSION = f"{getattr(SC, 'VERSION', 'sharp_check')} + {GUARD_VERSION}"
    print(f"[sharp_guard] {GUARD_VERSION} ติดตั้งแล้ว (ต่อจาก {getattr(SC, 'LOOSE_VERSION', 'loose')})")

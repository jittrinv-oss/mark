#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sharp_final.py v1 — ตัวตัดสินสุดท้ายของตัวตรวจคมชัด  *ไม่แก้ sharp_check / sharp_fix / sharp_loose / sharp_guard*
ถูกเรียกจาก mark_ui_sharp.py v1.8 / web_sharp.py v1.7 (บรรทัด import sharp_final)

แก้ 3 เรื่อง (ทดสอบกับภาพจริงชุด 56-62):
 1) ชิ้นงานดียังขึ้นส้ม/แดง
    สาเหตุหลัก (จากภาพ 58): กรอบที่ขึ้นส้ม/แดง = "ขีดเส้นประของกรอบ" ใน Master ที่เป็นรูปถ่ายกระดาษ
      (ขีดเทาซ้าย "A", ซ้าย "6", ขวา "5.0") — Test บนกระจกไม่มีกรอบประ -> หาไม่เจอ -> แดง
      + ภาพผลที่ Pi ส่งกลับมาเป็นภาพขาวดำที่ผ่าน CLAHE (มีลาย) -> guard v1 จับคู่ไม่ได้ ("ภาพไม่ตรงกับ Master พอ")
    แก้: - จัดภาพเอง (SIFT บน "แผนที่หมึก" ที่กลับขั้วแล้ว -> ตัวขาวบนเขียว / ตัวดำบนขาว จับคู่กันได้)
         - ตัดสินทุกชิ้นใหม่ด้วยการเทียบ "แผนที่หมึก" แบบล็อกตำแหน่งจากตัวอักษรรอบข้าง (ไม่ใช้ผลส้ม/แดงเดิม)
           ขึ้น NG เฉพาะส่วนที่หาย "เป็นก้อนใหญ่" (≥20% ของชิ้น และยาว ≥45%) / เส้นขาด ≥35% / หายทั้งชิ้น ≥70%
         - ขีดเส้นตรงบาง ๆ ที่อยู่ "นอกขอบเขตตัวอักษร" (กรอบประของ Master) -> ข้าม
         - ชิ้นเล็ก (จุด "." ขีด "-") ต้องหาย ≥90% ถึงจะนับ, จุดฝุ่นเล็กมาก / QR / รหัสวันที่ 1 2 4 8 -> ไม่ตรวจ
         - บริเวณที่ภาพรบกวนสูง (noise ≥ 0.35 ของคอนทราสต์ เช่น ภาพจาง/ลายจอ) -> ไม่ตัดสินชิ้นนั้น + แจ้งเตือน
         - แยกชิ้นส่วนจาก Master เอง (ไม่พึ่งรายการของตัวตรวจเดิม ที่บางครั้งไม่มีขีดล่าง SNI)
 2) ภาพผลมีแต่ลาย (polarity)
    แก้: ภาพผลสร้างใหม่จากภาพ Test ที่จัดตำแหน่งเอง: พื้นขาวสะอาด + ตัวดำ (ตัดลาย/จุดรบกวนออก)
 3) ครอบรูปแล้ว ALIGN FAIL + ภาพผลเล็ก
    แก้: จัดภาพเอง (ขยายภาพเล็ก / ลดภาพใหญ่ก่อนจับคู่ ทนภาพจาง) แล้วส่งภาพที่จัดแล้วให้ตัวตรวจเดิม
         ถ้าตัวตรวจเดิมยังจับคู่ไม่ได้ -> ใช้การจัดภาพของ final ตัดสินต่อเอง (ไม่ ALIGN FAIL)
         ถ้าจับคู่ไม่ได้จริง ๆ -> ภาพผลแสดงภาพ Test เต็มกรอบ + แถบแดง (ไม่ใช่ภาพเล็ก)

ปรับได้ใน sharp_config.json: final_on, f_present, f_part_area (= ช่อง "NG เมื่อหาย ≥ %"), f_part_extent,
                              f_line_extent, f_missing, f_small_missing, f_noise_max, f_clean_view
ปิด final (กลับไปใช้ loose v3 + guard v1): "final_on": false
"""
import time
import numpy as np
import cv2
import sharp_check as SC
try:
    import sharp_fix  # noqa: F401
except Exception:
    pass
try:
    import sharp_loose  # noqa: F401
except Exception:
    pass
try:
    import sharp_guard  # noqa: F401
except Exception:
    pass

FINAL_VERSION = "final v1"
FINAL_DEFAULT = {
    "final_on": True,
    "f_present": 0.30,        # ความเข้มหมึก (เทียบหมึกรอบข้าง) ที่ถือว่า "มีหมึก"
    "f_part_area": 0.20,      # ส่วนที่หายต่อเนื่อง ≥ สัดส่วนนี้ของชิ้น
    "f_part_extent": 0.45,    # และยาว ≥ สัดส่วนนี้ของความกว้าง/สูงของชิ้น
    "f_line_extent": 0.35,    # เส้นตรงบาง (ขีดล่าง SNI ฯลฯ): ส่วนที่หายยาว ≥ สัดส่วนนี้
    "f_line_area": 0.12,
    "f_missing": 0.70,        # หาย ≥ = หายทั้งชิ้น (แดง)
    "f_small_missing": 0.90,  # ชิ้นเล็ก: ต้องหาย ≥ นี้ถึงนับ
    "f_shift_px": 2,
    "f_noise_max": 0.35,      # บริเวณที่ภาพรบกวนสูง (สัญญาณ/noise ต่ำ) -> ไม่ตัดสินชิ้นนั้น (ไม่ขึ้นส้มผิด)
    "f_clean_view": True,     # ภาพผลพื้นขาวสะอาด (ตัวอ่อน/พื้นสี)
}
COLOR = {"OK": (0, 170, 0), "PARTIAL": (0, 140, 255), "MISSING": (0, 0, 255), "SKIP": (150, 150, 150)}


def _cfg():
    try:
        c = SC.load_config()
    except Exception:
        c = {}
    out = dict(c)
    for k, v in FINAL_DEFAULT.items():
        out.setdefault(k, v)
    return out


def _bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _chans(bgr):
    b = _bgr(bgr)
    lab = cv2.cvtColor(b, cv2.COLOR_BGR2LAB)
    return [lab[:, :, 0]] + list(cv2.split(b))


def _bg(c, k, close):
    H, W = c.shape
    s = 4
    small = cv2.resize(c, (max(8, W // s), max(8, H // s)), interpolation=cv2.INTER_AREA)
    ks = max(3, (k // s) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE if close else cv2.MORPH_OPEN, se)
    bg = cv2.GaussianBlur(bg, (0, 0), ks / 4.0)
    return cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)


def ink_map(bgr, k=None, polarity=None):
    """ภาพสี -> แผนที่หมึก 0..1 (ค่าสูง = หมึก)  เลือกช่องสี+ขั้วอัตโนมัติ (ตัวเข้มบนพื้นอ่อน / ตัวอ่อนบนพื้นสี)"""
    H, W = bgr.shape[:2]
    k = k or max(31, int(0.06 * max(H, W)))
    best = None
    for ch in _chans(bgr):
        c = ch.astype(np.float32) / 255.0
        cands = []
        if polarity in (None, "dark"):
            cands.append(("dark", _bg(c, k, True) - c))
        if polarity in (None, "light"):
            cands.append(("light", c - _bg(c, k, False)))
        for pol, d in cands:
            sc = float(np.percentile(d, 99.5)) / (float(np.std(d)) + 1e-6)
            if best is None or sc > best[0]:
                best = (sc, pol, d)
    _, pol, d = best
    d = np.clip(d, 0, None)
    d = d / max(float(np.percentile(d, 99.7)), 1e-4)
    return np.clip(d, 0, 1).astype(np.float32), pol


def _sift():
    try:
        return cv2.SIFT_create(nfeatures=8000), cv2.NORM_L2
    except Exception:
        return cv2.ORB_create(8000), cv2.NORM_HAMMING


def _sane(H, test_shape, frame_shape):
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return False
    fh, fw = frame_shape[:2]
    c = cv2.perspectiveTransform(np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]]).reshape(-1, 1, 2), Hi).reshape(-1, 2)
    if not np.isfinite(c).all() or not cv2.isContourConvex(c.astype(np.float32).reshape(-1, 1, 2)):
        return False
    th, tw = test_shape[:2]
    a = abs(cv2.contourArea(c.astype(np.float32)))
    return 0.01 * th * tw < a < 6.0 * th * tw


def register(master_frame, test_bgr):
    """จัดภาพ Test ให้ทับ Master (ขนาด master_frame)  คืน (H: test->frame, จุดอ้างอิง, กลับด้าน) หรือ (None, n, False)"""
    fh, fw = master_frame.shape[:2]
    mi, _ = ink_map(master_frame)
    mg = (255 * (1 - mi)).astype(np.uint8)
    th, tw = test_bgr.shape[:2]
    s = 1.0
    if max(th, tw) > 1800:
        s = 1800.0 / max(th, tw)
    elif max(th, tw) < 1000:
        s = min(3.0, 1200.0 / max(th, tw))
    ts = cv2.resize(test_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC) if s != 1 else test_bgr
    ti, _ = ink_map(ts)
    tg = (255 * (1 - ti)).astype(np.uint8)
    det, norm = _sift()
    km, dm = det.detectAndCompute(mg, None)
    if dm is None or len(km) < 8:
        return None, 0, False
    best = None
    for mir in (False, True):
        t = cv2.flip(tg, 1) if mir else tg
        kt, dt = det.detectAndCompute(t, None)
        if dt is None or len(kt) < 8:
            continue
        good = [p[0] for p in cv2.BFMatcher(norm).knnMatch(dt, dm, k=2)
                if len(p) == 2 and p[0].distance < 0.80 * p[1].distance]
        if len(good) < 8:
            continue
        pt = np.float32([kt[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
        pm = np.float32([km[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
        Hh, inl = cv2.findHomography(pt, pm, cv2.RANSAC, 5.0)
        if Hh is None or inl is None:
            continue
        n = int(inl.sum())
        if best is None or n > best[1]:
            best = (Hh, n, mir)
    if best is None or best[1] < 10:
        return None, 0 if best is None else best[1], False
    Hh, n, mir = best
    S = np.diag([s, s, 1.0])
    if mir:
        S = np.array([[-1, 0, tg.shape[1] - 1], [0, 1, 0], [0, 0, 1]], np.float64) @ S
    Ht = Hh @ S
    if not _sane(Ht, test_bgr.shape, master_frame.shape):
        return None, n, mir
    # ECC ละเอียด (บนแผนที่หมึก ครึ่งขนาด)
    try:
        ti_full, _ = ink_map(test_bgr)
        w = cv2.warpPerspective(ti_full, Ht, (fw, fh), flags=cv2.INTER_LINEAR)
        a = cv2.GaussianBlur(cv2.resize(mi, (fw // 2, fh // 2)), (0, 0), 1.2)
        b = cv2.GaussianBlur(cv2.resize(w, (fw // 2, fh // 2)), (0, 0), 1.2)
        _, E = cv2.findTransformECC(a, b, np.eye(3, dtype=np.float32), cv2.MOTION_HOMOGRAPHY,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 3)
        S2 = np.diag([0.5, 0.5, 1.0])
        E = np.linalg.inv(S2) @ E.astype(np.float64) @ S2
        if np.abs(E - np.eye(3)).max() < 0.1:
            Ht = np.linalg.inv(E) @ Ht
    except Exception:
        pass
    return Ht, n, mir


# ------------------------------------------------------------------ แผนที่หมึก Test ในกรอบ Master
def _signed_flat(ch, k):
    f = ch.astype(np.float32) + 1.0
    return f / cv2.GaussianBlur(f, (0, 0), k) - 1.0


def test_ink(photo, mb, k):
    """photo = Test ที่จัดตำแหน่งแล้ว -> หมึก (สูง = หมึก) เลือกช่อง/ขั้วที่ตรงกับ Master"""
    valid = np.ones(mb.shape, bool)
    sel = cv2.dilate(mb, np.ones((25, 25), np.uint8)) > 0
    if sel.sum() < 100:
        sel = valid
    mv = mb[sel].astype(np.float32)
    mv -= mv.mean()
    best = None
    for ch in _chans(photo):
        f = _signed_flat(ch, k)
        v = f[sel] - f[sel].mean()
        r = float((v * mv).sum() / (np.sqrt((v * v).sum() * (mv * mv).sum()) + 1e-9))
        if best is None or abs(r) > abs(best[0]):
            best = (r, f)
    r, f = best
    return (f if r > 0 else -f), abs(r), r < 0


def master_ink(master_frame):
    m = _bgr(master_frame)
    b = m.astype(np.int16)
    col = ((b.max(axis=2) - b.min(axis=2)) >= 40) & (b.max(axis=2) >= 120)    # เส้นสี (กรอบประแดง)
    mi, _ = ink_map(m)
    mi[col] = 0
    _, mb = cv2.threshold((mi * 255).astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mb = cv2.morphologyEx(mb, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mi, mb.astype(np.uint8)


def frame_band(mb, text_h):
    """กรอบเส้นประของ Master (รูปถ่ายกระดาษ): ขีดบาง ๆ ≥3 ขีด เรียงแนวเดียวกัน และอยู่ "ด้านนอกสุด" -> แถบ mask"""
    H, W = mb.shape
    band = np.zeros_like(mb)
    vk = max(5, int(0.4 * text_h))
    tol = max(3.0, 0.25 * text_h)
    for vert in (True, False):
        m = mb if vert else mb.T
        b = band if vert else band.T
        v = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((vk, 1), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(v, 8)
        segs = [(cen[i][0], st[i, 1], st[i, 1] + st[i, 3], st[i, 2]) for i in range(1, n)
                if st[i, 2] <= 0.3 * text_h and st[i, 3] >= vk]
        segs.sort()
        grp = []
        for sgm in segs + [None]:
            if sgm is not None and (not grp or sgm[0] - grp[-1][0] <= tol):
                grp.append(sgm)
                continue
            if len(grp) >= 3:
                gx = float(np.median([g[0] for g in grp]))
                th = float(np.median([g[3] for g in grp]))
                y0, y1 = min(g[1] for g in grp), max(g[2] for g in grp)
                if y1 - y0 >= 3 * text_h:
                    a0 = int(gx - th / 2 - 3)
                    a1 = int(gx + th / 2 + 4)
                    out_r = int(m[y0:y1, max(0, a1 + 2):].sum())
                    out_l = int(m[y0:y1, :max(0, a0 - 2)].sum())
                    if min(out_r, out_l) <= 3 * text_h:
                        b[max(0, y0 - 2):y1 + 2, max(0, a0):a1] = 1
            grp = [sgm] if sgm is not None else []
    return band


def segment(mb, min_area=10):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mb, 8)
    els = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if a >= min_area:
            els.append(dict(id=len(els) + 1, kind="TEXT", status="OK", note="", x=x, y=y, w=w, h=h, area=a,
                            cov=None, vfrac=None, hfrac=None, dx=0, dy=0))
    return els


# ------------------------------------------------------------------ ตัดสินรายชิ้น
def _judge(e, mi, mb, tink, text_h, gcon, cfg, small, pad=0):
    H, W = mb.shape
    x, y, w, h = int(e["x"]) + pad, int(e["y"]) + pad, int(e["w"]), int(e["h"])
    p = 3
    X0, Y0, X1, Y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
    sub = mb[Y0:Y1, X0:X1]
    n, lab, st, cen = cv2.connectedComponentsWithStats(sub, 8)
    own = np.zeros_like(sub, bool)
    for i in range(1, n):
        cx, cy = cen[i][0] + X0, cen[i][1] + Y0
        if x - 1 <= cx <= x + w + 1 and y - 1 <= cy <= y + h + 1 and st[i, 4] >= 3:
            own |= lab == i
    if own.sum() < 6:
        return None
    c = int(1.2 * text_h)
    R = int(max(4, 0.30 * text_h))
    wx0, wy0, wx1, wy1 = max(0, x - c), max(0, y - c), min(W, x + w + c), min(H, y + h + c)
    sx0, sy0, sx1, sy1 = max(0, wx0 - R), max(0, wy0 - R), min(W, wx1 + R), min(H, wy1 + R)
    tpl = cv2.GaussianBlur(mi[wy0:wy1, wx0:wx1], (0, 0), 1.5)
    srch = cv2.GaussianBlur(tink[sy0:sy1, sx0:sx1], (0, 0), 1.5)
    if tpl.std() < 1e-3 or srch.shape[0] < tpl.shape[0] or srch.shape[1] < tpl.shape[1]:
        return None
    res = cv2.matchTemplate(srch, tpl, cv2.TM_CCOEFF_NORMED)
    _, ctx, _, loc = cv2.minMaxLoc(res)
    if ctx < 0.25:
        return dict(dx=0, dy=0, status=None, why=f"บริบทไม่ชัด ({ctx:.2f}) — ไม่ตัดสิน", miss=0.0)
    dx, dy = loc[0] - (wx0 - sx0), loc[1] - (wy0 - sy0)
    s = int(cfg["f_shift_px"])
    hh, ww = Y1 - Y0, X1 - X0
    best = None
    for ddy in range(-s, s + 1):
        for ddx in range(-s, s + 1):
            ox0, oy0 = X0 + dx + ddx, Y0 + dy + ddy
            if ox0 < 0 or oy0 < 0 or ox0 + ww > W or oy0 + hh > H:
                continue
            v = float(tink[oy0:oy0 + hh, ox0:ox0 + ww][own].mean())
            if best is None or v > best[0]:
                best = (v, dx + ddx, dy + ddy)
    if best is None:
        return None
    _, dx, dy = best
    ox0, oy0 = X0 + dx, Y0 + dy
    if wy0 + dy < 0 or wx0 + dx < 0 or wy1 + dy > H or wx1 + dx > W:
        return dict(dx=int(dx), dy=int(dy), status=None, why="ชิดขอบภาพ — ไม่ตัดสิน", miss=0.0)
    tw = tink[wy0 + dy:wy1 + dy, wx0 + dx:wx1 + dx]
    mbw = mb[wy0:wy1, wx0:wx1] > 0
    bgm = cv2.dilate(mbw.astype(np.uint8), np.ones((7, 7), np.uint8)) == 0
    inkm = cv2.erode(mbw.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if bgm.sum() < 30:
        return None
    bg = float(np.median(tw[bgm]))
    ink_ref = float(np.percentile(tw[inkm], 85)) if inkm.sum() >= 20 else bg
    con = max(ink_ref - bg, 0.6 * gcon)
    if con <= 1e-4:
        return None
    noise = float(np.std(tw[bgm])) / con
    if noise >= cfg["f_noise_max"]:
        return dict(dx=int(dx), dy=int(dy), status=None, unclear=True,
                    why=f"ภาพไม่ชัดบริเวณนี้ (noise {noise:.2f}) — ไม่ตัดสิน", miss=0.0)
    thr = float(min(0.6, max(cfg["f_present"], 2.5 * noise)))
    tp = tink[oy0:oy0 + hh, ox0:ox0 + ww]
    cmap = (tp - bg) / con
    cmax = cv2.dilate(cmap.astype(np.float32), np.ones((5, 5), np.uint8))
    miss = own & (cmax < thr)
    miss = cv2.morphologyEx(miss.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    area = int(own.sum())
    tot = float(miss.sum()) / area
    zone = cv2.dilate(own.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    a_ = cv2.GaussianBlur(own.astype(np.float32), (0, 0), 1.5)[zone]
    b_ = cv2.GaussianBlur(np.clip(cmap, 0, 1.5).astype(np.float32), (0, 0), 1.5)[zone]
    a_ = a_ - a_.mean()
    b_ = b_ - b_.mean()
    ncc = float((a_ * b_).sum() / (np.sqrt((a_ * a_).sum() * (b_ * b_).sum()) + 1e-6))
    ys, xs = np.nonzero(own)
    ew, eh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    line = max(ew, eh) >= 3 * min(ew, eh)
    status, why = None, f"ครบ (หาย {tot:.0%})"
    if small:
        if tot >= cfg["f_small_missing"]:
            status, why = "PARTIAL", f"ชิ้นเล็กหาย {tot:.0%}"
    elif tot >= cfg["f_missing"]:
        status, why = "MISSING", f"หาย {tot:.0%}"
    else:
        nm, _, sm, _ = cv2.connectedComponentsWithStats(miss.astype(np.uint8), 8)
        for j in range(1, nm):
            a = sm[j, 4] / area
            fx, fy = sm[j, 2] / ew, sm[j, 3] / eh
            if line:
                along = fx if ew >= eh else fy
                if along >= cfg["f_line_extent"] and a >= cfg["f_line_area"]:
                    status, why = "PARTIAL", f"เส้นขาด {along:.0%} ของความยาว"
                    break
            elif a >= cfg["f_part_area"] and max(fx, fy) >= cfg["f_part_extent"]:
                status, why = "PARTIAL", f"ส่วนที่หาย {a:.0%} ยาว {max(fx, fy):.0%}"
                break
    return dict(dx=int(dx), dy=int(dy), status=status, why=f"{why} (noise {noise:.2f})", miss=tot, ncc=ncc)


def _frame_parts(els, text_h, mb):
    """ขีดบาง ๆ ที่อยู่นอกขอบเขตตัวอักษร และเรียงแนวเดียวกัน ≥2 ขีด (กรอบประของ Master) / จุดเล็กนอกขอบเขต"""
    def thin(w, h):
        return min(w, h) <= 0.35 * text_h and max(w, h) >= 2.5 * min(w, h)
    n, _, st, _ = cv2.connectedComponentsWithStats(mb, 8)
    core = [st[i] for i in range(1, n) if not thin(st[i, 2], st[i, 3]) and max(st[i, 2], st[i, 3]) >= 0.5 * text_h]
    if not core:
        return set()
    bx0 = min(c[0] for c in core); by0 = min(c[1] for c in core)
    bx1 = max(c[0] + c[2] for c in core); by1 = max(c[1] + c[3] for c in core)
    m = 0.15 * text_h
    outside = []
    for e in els:
        cx, cy = e["x"] + e["w"] / 2, e["y"] + e["h"] / 2
        if cx < bx0 - m or cx > bx1 + m or cy < by0 - m or cy > by1 + m:
            outside.append(e)
    out = set()
    tol = 0.3 * text_h
    for e in outside:
        if max(e["w"], e["h"]) < 0.5 * text_h:
            out.add(id(e))
            continue
        if not thin(e["w"], e["h"]):
            continue
        vert = e["h"] > e["w"]
        key = (e["x"] + e["w"] / 2) if vert else (e["y"] + e["h"] / 2)
        mates = [o for o in outside if o is not e and thin(o["w"], o["h"]) and (o["h"] > o["w"]) == vert
                 and abs(((o["x"] + o["w"] / 2) if vert else (o["y"] + o["h"] / 2)) - key) <= tol]
        if mates:
            out.add(id(e))
    return out


def clean_view(tink, mb, light, text_h=30.0):
    """ภาพผลพื้นขาวสะอาด ตัวดำ (นอกบริเวณมาร์คเป็นขาว)"""
    sel = cv2.dilate(mb, np.ones((5, 5), np.uint8)) > 0
    bgz = cv2.dilate(mb, np.ones((15, 15), np.uint8)) == 0
    bg = float(np.median(tink[bgz])) if bgz.sum() > 100 else float(np.median(tink))
    full = float(np.percentile(tink[sel], 90)) if sel.sum() > 100 else float(np.percentile(tink, 99))
    con = max(full - bg, 1e-4)
    noise = float(np.std(tink[bgz])) if bgz.sum() > 100 else 0.0
    lo = bg + max(0.25 * con, 2.0 * noise)
    hi = bg + 0.75 * con
    v = np.clip((cv2.GaussianBlur(tink, (0, 0), 0.8) - lo) / max(hi - lo, 1e-4), 0, 1)
    out = (255 * (1 - v)).astype(np.uint8)
    dark = (out < 170).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    a_min = max(8, int(0.00002 * out.size))
    small = np.zeros(n, bool)
    small[1:] = st[1:, 4] < a_min
    out[small[lab]] = 255
    k = int(1.5 * text_h) | 1
    near = cv2.dilate(mb, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    out[~near] = 255
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def draw(view, els, show_ok=True):
    R = _bgr(view).copy()
    for e in els:
        st = e.get("status", "OK")
        bad = st in ("MISSING", "PARTIAL")
        if st == "SKIP" or (not bad and not show_ok):
            continue
        x, y = int(e["x"] + e.get("dx", 0)), int(e["y"] + e.get("dy", 0))
        cv2.rectangle(R, (x - 3, y - 3), (x + int(e["w"]) + 3, y + int(e["h"]) + 3), COLOR[st], 3 if bad else 1)
    return R


def fail_view(test_bgr, text="ALIGN FAIL - check model / focus / crop"):
    t = _bgr(test_bgr).copy()
    h, w = t.shape[:2]
    bh = max(30, h // 18)
    cv2.rectangle(t, (0, 0), (w - 1, bh), (40, 40, 200), -1)
    cv2.putText(t, text, (8, int(bh * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, bh / 45.0, (255, 255, 255),
                max(1, bh // 20), cv2.LINE_AA)
    return t


def _fail(test_bgr, notes):
    v = fail_view(test_bgr)
    rep = dict(verdict="NG", align_method="จับคู่ไม่ได้", align_inliers=0, n_elements=0, n_ok=0, n_missing=0,
               n_partial=0, n_skip=0, n_blur=0, n_dash=0, n_extra=0, elements=[], extras=[],
               engine=FINAL_VERSION, notes=notes)
    return dict(verdict="NG", report=rep, overlay=v, aligned=v, notes=notes, debug_dir=None, test=test_bgr)


# ------------------------------------------------------------------ หลัก
def final_inspect(master_bgr, test_bgr, cfg=None, ignore_zones=None, debug_dir=None, force_continue=None,
                  base=None):
    t0 = time.time()
    c = _cfg()
    if cfg:
        c.update({k: v for k, v in cfg.items() if k not in FINAL_DEFAULT})
    cfg = c
    master_bgr, test_bgr = _bgr(master_bgr), _bgr(test_bgr)
    m0 = master_bgr
    try:
        m0, _ = SC.remove_master_color_frame(master_bgr)
    except Exception:
        pass
    L = int(cfg.get("proc_long_side", 1400))
    frame, _ = SC._resize_long(m0, L)
    notes = []
    Ht, n_reg, mir = register(frame, test_bgr)
    t_al = None
    valid = np.full(frame.shape[:2], 255, np.uint8)
    if Ht is not None:
        t_al = cv2.warpPerspective(test_bgr, Ht, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpPerspective(np.full(test_bgr.shape[:2], 255, np.uint8), Ht,
                                    (frame.shape[1], frame.shape[0]), flags=cv2.INTER_NEAREST)
        valid = cv2.erode(valid, np.ones((7, 7), np.uint8))
        notes.append(f"{FINAL_VERSION}: จัดภาพเอง จุดอ้างอิง {n_reg}{' กลับด้าน' if mir else ''}")
    r = None
    if t_al is None:                                   # จัดภาพเองไม่ได้ -> ให้ตัวตรวจเดิมจัดภาพ
        base = base or getattr(SC, "_loose_base_inspect", None) or SC.inspect
        try:
            r = base(master_bgr, test_bgr, cfg, ignore_zones, debug_dir, True)
        except Exception as ex:
            notes.append(f"ตัวตรวจเดิม error: {type(ex).__name__}: {ex}")
        if not r or r.get("report") is None or r.get("aligned") is None:
            return _fail(test_bgr, notes + list((r or {}).get("notes", [])) +
                         [f"⚠ จับคู่ไม่ได้ (จุดอ้างอิง {n_reg}) — ตรวจโมเดล / โฟกัส / แสง / ครอบรูปให้เห็นมาร์คทั้งหมด"])
        t_al = cv2.resize(_bgr(r["aligned"]), (frame.shape[1], frame.shape[0]))
        notes.append(f"{FINAL_VERSION}: ใช้ภาพจัดของตัวตรวจเดิม")
    mi, mb = master_ink(frame)
    n0, _, st0, _ = cv2.connectedComponentsWithStats(mb, 8)
    hs0 = np.array([st0[i, 3] for i in range(1, n0) if st0[i, 4] >= 20])
    text_h = float(np.percentile(hs0, 60)) if len(hs0) else 20.0
    fb = frame_band(mb, text_h)
    if fb.any():
        mb = mb.copy()
        mi = mi.copy()
        mb[fb > 0] = 0
        mi[fb > 0] = 0
    els = segment(mb, max(10, int(0.004 * text_h * text_h)))
    areas = sorted(e["area"] for e in els if e["h"] >= 0.6 * text_h)
    med_a = float(np.median(areas)) if areas else 100.0
    # โซนที่ไม่ตรวจ: QR / รหัสวันที่ (1 2 4 8) / IGNORE / จุดฝุ่นเล็กมาก
    skips = []
    try:
        if cfg.get("skip_qr", True):
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for bx in SC.find_qr(g, mb * 255, cfg):
                skips.append((bx, "QR"))
    except Exception:
        pass
    try:
        if cfg.get("skip_date_code", True):
            dc = SC.find_date_code(mb * 255)
            if dc:
                skips.append((dc, "รหัสวันที่"))
    except Exception:
        pass
    scz = frame.shape[1] / float(master_bgr.shape[1])
    for z in ignore_zones or []:
        skips.append(((z["x"] * scz, z["y"] * scz, (z["x"] + z["w"]) * scz, (z["y"] + z["h"]) * scz),
                      z.get("name", "IGNORE")))
    for e in els:
        cx, cy = e["x"] + e["w"] / 2, e["y"] + e["h"] / 2
        for (x0, y0, x1, y1), nm in skips:
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                e["status"], e["note"] = "SKIP", f"{nm} — ไม่ตรวจ"
                break
        if e["status"] != "SKIP" and e["area"] < 0.06 * med_a and max(e["w"], e["h"]) < 0.35 * text_h:
            e["status"], e["note"] = "SKIP", "จุดเล็กมาก (ฝุ่น) — ไม่ตรวจ"
    rep = dict(align_method="final", align_inliers=n_reg, n_blur=0, n_dash=0, engine=FINAL_VERSION)
    base_notes = list((r or {}).get("notes", []))
    if skips:
        base_notes.append("ไม่ตรวจ: " + ", ".join(sorted(set(nm for _, nm in skips))))
    tink, corr, _ = test_ink(t_al, mb, max(15.0, 2.0 * text_h))
    if corr < 0.25:                                    # จัดภาพผิด -> ไม่ตัดสินรายชิ้น (กัน NG มั่ว)
        return _fail(test_bgr, notes + base_notes +
                     [f"⚠ จับคู่ไม่ได้ (ภาพไม่ตรงกับ Master: {corr:.2f}) — ตรวจโมเดล / โฟกัส / แสง / ครอบรูปให้เห็นมาร์คทั้งหมด"])
    tmed = float(np.median(tink[valid > 0])) if (valid > 0).any() else 0.0
    if (valid == 0).any():
        tink = tink.copy()
        tink[valid == 0] = tmed
    PAD = int(1.6 * text_h) + 10                      # ขอบภาพ: ค้นหาตำแหน่งได้ทุกทิศ
    tinkP = cv2.copyMakeBorder(tink, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=tmed)
    miP = cv2.copyMakeBorder(mi, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)
    mbP = cv2.copyMakeBorder(mb, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)
    sel = mb > 0
    selb = cv2.dilate(mb, np.ones((7, 7), np.uint8)) == 0
    gcon = float(np.percentile(tink[sel], 85) - np.median(tink[selb])) if sel.sum() > 50 and selb.sum() > 50 else 0.0
    frame_ids = _frame_parts(els, text_h, mb)
    n_bad, n_frame, n_clear, n_out, n_unclear = 0, 0, 0, 0, 0
    for e in els:
        prev = e.get("status")
        if prev == "SKIP":
            continue
        vb = valid[max(0, e["y"] - 4):e["y"] + e["h"] + 4, max(0, e["x"] - 4):e["x"] + e["w"] + 4]
        if vb.size and (vb == 0).any():
            e["status"], e["note"] = "SKIP", f'{e.get("note", "")} | {FINAL_VERSION}: อยู่นอก/ชิดขอบภาพถ่าย — ไม่ตรวจ'
            n_out += 1
            continue
        ob = mb[e["y"]:e["y"] + e["h"], e["x"]:e["x"] + e["w"]]
        if fb.any() and ob.size and ob.sum() < 0.3 * max(1, e.get("area", ob.size)):
            e["status"], e["note"] = "SKIP", f'{e.get("note", "")} | {FINAL_VERSION}: เส้นกรอบประของ Master — ไม่ตรวจ'
            n_frame += 1
            continue
        if id(e) in frame_ids:
            e["status"], e["note"] = "SKIP", f'{e.get("note", "")} | {FINAL_VERSION}: เส้นกรอบประของ Master — ไม่ตรวจ'
            n_frame += 1
            continue
        small = max(e["w"], e["h"]) < 0.6 * text_h or e.get("area", e["w"] * e["h"]) < 0.25 * med_a
        try:
            g = _judge(e, miP, mbP, tinkP, text_h, gcon, cfg, small, PAD)
        except Exception as ex:
            g = dict(dx=0, dy=0, status=None, why=f"error {ex}")
        if g is None:
            e["status"] = "OK"
            continue
        e["dx"], e["dy"] = g["dx"], g["dy"]
        n_unclear += bool(g.get("unclear"))
        e["status"] = g["status"] or "OK"
        e["note"] = f'{e.get("note", "")} | {FINAL_VERSION}: {g["why"]}'
        n_bad += e["status"] != "OK"
    bad = [e for e in els if e["status"] in ("MISSING", "PARTIAL")]
    verdict = "NG" if bad else "OK"
    light = _test_is_light(test_bgr)
    view = clean_view(tink, mb, light, text_h) if (cfg.get("f_clean_view", True) and light) else t_al
    show_ok = bool(cfg.get("show_ok_boxes", True))
    overlay = draw(view, els, show_ok)
    rep.update(verdict=verdict, elements=els, extras=[], n_elements=len(els), n_ok=len(els) - len(bad),
               n_missing=sum(e["status"] == "MISSING" for e in els),
               n_partial=sum(e["status"] == "PARTIAL" for e in els),
               n_skip=sum(e["status"] == "SKIP" for e in els), n_extra=0)
    rep["align_method"] = f'{rep.get("align_method", "")} | {FINAL_VERSION}'
    note = (f"{FINAL_VERSION}: NG {len(bad)} ชิ้น จาก {len(els)} | "
            f"ข้ามกรอบประ {n_frame} | นอกภาพ {n_out} | {time.time() - t0:.1f}s")
    judged = max(1, sum(e["status"] != "SKIP" for e in els))
    if n_unclear:
        note += f" | ภาพไม่ชัด ไม่ตัดสิน {n_unclear} ชิ้น"
        if n_unclear > 0.3 * judged:
            notes.insert(0, f"⚠ ภาพไม่ชัด {n_unclear}/{judged} ชิ้น — ผล OK อาจไม่ครบ: ปรับโฟกัส / ลดแสง / ขยับกล้องเข้าใกล้")
    all_notes = [note] + notes + base_notes
    rep["notes"] = all_notes
    return dict(verdict=verdict, report=rep, overlay=overlay, aligned=view, notes=all_notes,
                debug_dir=(r or {}).get("debug_dir"), test=test_bgr, view=view)


def _test_is_light(test_bgr):
    try:
        _, pol = ink_map(cv2.resize(test_bgr, None, fx=0.5, fy=0.5) if max(test_bgr.shape[:2]) > 1200 else test_bgr)
        hsv = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2HSV)
        colored = float(((hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 40)).mean()) > 0.25
        return pol == "light" or colored
    except Exception:
        return False


if not getattr(SC, "_FINAL1", False):
    SC._FINAL1 = True
    _chain = SC.inspect

    def inspect(master_bgr, test_bgr, cfg=None, ignore_zones=None, debug_dir=None, force_continue=None):
        if not _cfg().get("final_on", True):
            return _chain(master_bgr, test_bgr, cfg, ignore_zones, debug_dir, force_continue)
        try:
            return final_inspect(master_bgr, test_bgr, cfg, ignore_zones, debug_dir, force_continue)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            r = _chain(master_bgr, test_bgr, cfg, ignore_zones, debug_dir, force_continue)
            r["notes"] = [f"⚠ {FINAL_VERSION} error: {type(ex).__name__}: {ex} — ใช้ผลเดิม"] + list(r.get("notes", []))
            return r

    SC.inspect = inspect
    SC.FINAL_VERSION = FINAL_VERSION
    SC.final_draw = draw
    SC.final_fail_view = fail_view
    SC.VERSION = f"{getattr(SC, 'VERSION', 'sharp_check')} + {FINAL_VERSION}"
    print(f"[sharp_final] {FINAL_VERSION} ติดตั้งแล้ว")

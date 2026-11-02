#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sharp_fix.py v1.3 — แพตช์ของตัวตรวจคมชัด (sharp_check.py v1.2)  *ไม่แก้ไฟล์ sharp_check.py เดิม*
ถูกเรียกอัตโนมัติจาก mark_ui_sharp.py / web_sharp.py (บรรทัด import sharp_fix)

แก้ 3 เรื่อง:
 1) กรอบเส้นประแดงของ Master ขึ้นแดงที่มุม / แถว 1 2 4 8 ขึ้นแดง
    - ลบเส้นสี (แดง/ชมพู) บาง ๆ ออกจาก Master ก่อนตรวจ  -> มุมกรอบไม่ถูกตรวจ
    - เส้นประที่ "ติด" ตัวอักษร (E ของ AUTOMOTIVE, 4, 8, ตัวเลขแถวล่าง) ไม่รวมเป็นชิ้นเดียวกับตัวอักษรอีก
      -> หาแถวรหัสวันที่ (1 2 4 8) เจอ และข้ามได้อัตโนมัติ
    - Test ที่มีเส้นประแดง (เช่น Test จาก PDF) ลบเส้นแดงด้วยการเติมสีรอบข้าง (inpaint)
 2) ตัว E / ตัวอักษรอื่นขึ้นส้ม (จางขาดหาย) ผิด
    - สาเหตุหลัก = เส้นประติดตัวอักษร (ข้อ 1)
    - ตัวอักษรที่อยู่ "ชิดขอบพื้นที่ตรวจ" (ขอบภาพหลังจัดภาพ / รอยปากกา / แสงสะท้อน)
      และหมึกหายจริงน้อยกว่า 10% -> ไม่ตัดสินจาง (เดิมโดนเกณฑ์รูปร่างตัดสินผิด)
 3) Test จาก PDF แล้วผลรวน (ภาพผลบิดเป็นเส้นดำ)
    - พื้นขาวของ PDF ถูกมองเป็น "แสงสะท้อน" 77% -> หมึกถูกลบก่อนจับคู่ -> จัดภาพผิด (กลับด้าน, จุดอ้างอิง 11)
      แก้: พื้นกระดาษ/PDF ขาวทั้งภาพ ไม่ใช่แสงสะท้อน
    - Test ที่ขนาดเท่า Master อยู่แล้ว (Test จาก PDF) ลอง "จัดตรง" เทียบกับ SIFT แล้วเลือกแบบที่ซ้อนตรงกว่า
    - ตรวจความสมเหตุสมผลของการจัดภาพ (ภาพไม่บิดกลับด้าน/ยุบ)
 อื่น ๆ: QR code ปลอม — วงกลม E6 / TIS ถูกมองเป็น QR แล้วข้ามพื้นที่ครึ่งมาร์ค
         ตอนนี้ต้องอ่านรหัส QR ได้ หรือเห็นจุดมุมสี่เหลี่ยม 3 จุดจริง ถึงจะข้าม

ปิดแต่ละข้อได้ใน sharp_config.json (กด "ตั้งค่าคมชัด"):
  fix_master_color_frame, fix_test_red_lines, fix_paper_not_glare, fix_qr_strict,
  fix_align_check, fix_edge_partial
"""
import numpy as np
import cv2
import sharp_check as SC

FIX_VERSION = "1.3"
FIX_DEFAULT = {
    "fix_master_color_frame": True,   # ลบกรอบ/เส้นสีบาง ๆ ใน Master
    "fix_test_red_lines": True,       # ลบเส้นแดงบาง ๆ ใน Test (inpaint)
    "fix_paper_not_glare": True,      # พื้นขาวทั้งภาพ (PDF/กระดาษ) ไม่ใช่แสงสะท้อน
    "fix_qr_strict": True,            # ข้าม QR เฉพาะ QR จริง
    "fix_align_check": True,          # เทียบ SIFT กับจัดตรง + ตรวจภาพบิด
    "fix_edge_partial": True,         # ตัวอักษรชิดขอบพื้นที่ตรวจ หายจริง < 10% -> ไม่ส้ม
    "fix_edge_max_loss": 0.10,
}

if getattr(SC, "_FIX_APPLIED", False):          # import ซ้ำ ไม่แพตช์ซ้อน
    pass
else:
    SC._FIX_APPLIED = True
    SC.DEFAULT.update(FIX_DEFAULT)
    SC.VERSION = f"sharp_check v1.2 + fix v{FIX_VERSION}"

    _orig_load_config = SC.load_config
    _orig_inspect = SC.inspect
    _orig_align = SC.align
    _orig_glare = SC.glare_mask
    _orig_find_qr = SC.find_qr
    _orig_compare = SC.compare

    # ------------------------------------------------------------------ config
    def load_config():
        cfg = _orig_load_config()
        if any(k not in cfg for k in FIX_DEFAULT):
            for k, v in FIX_DEFAULT.items():
                cfg.setdefault(k, v)
            SC.save_config(cfg)
        return cfg

    def _cfg(cfg):
        cfg = dict(cfg or load_config())
        for k, v in FIX_DEFAULT.items():
            cfg.setdefault(k, v)
        return cfg

    # ------------------------------------------------------------------ 1) เส้นสี
    def _thin(mask, k_erode=7, k_dilate=9):
        """เฉพาะเส้นบาง: ส่วนที่หายไปเมื่อ erode (ก้อนสีหนา เช่นโลโก้สี ไม่นับ)"""
        thick = cv2.dilate(cv2.erode(mask, np.ones((k_erode, k_erode), np.uint8)),
                           np.ones((k_dilate, k_dilate), np.uint8))
        return mask & (1 - thick)

    def remove_master_color_frame(bgr, max_frac=0.06):
        """Master: ลบเส้นสี (แดง/ชมพู/เขียว/ฟ้า) บาง ๆ เป็นสีขาว — ไม่ทำกับ Master ที่เป็นภาพสี (สกรีนเขียว)"""
        b = bgr.astype(np.int16)
        chroma = b.max(axis=2) - b.min(axis=2)
        col = ((chroma >= 40) & (b.max(axis=2) >= 120)).astype(np.uint8)
        frac = float(col.mean())
        if frac == 0 or frac > max_frac:
            return bgr, 0
        lines = _thin(col)
        if lines.sum() == 0:
            return bgr, 0
        # ขอบเส้นที่ antialias (สีจาง) — ขยาย 1 px แต่ไม่ทับหมึกดำ
        grow = cv2.dilate(lines, np.ones((3, 3), np.uint8))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        kill = (grow > 0) & ((col > 0) | (gray > 150))
        out = bgr.copy()
        out[kill] = (255, 255, 255)
        return out, int(kill.sum())

    def clean_test_red_lines(bgr, max_frac=0.08):
        """Test: เส้นแดง/ชมพูบาง ๆ -> เติมสีรอบข้าง (ไม่แตะสกรีนเขียว / ปากกาน้ำเงิน)"""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        red = (((h <= 10) | (h >= 165)) & (s >= 60) & (v >= 80)).astype(np.uint8)
        frac = float(red.mean())
        if frac == 0 or frac > max_frac:
            return bgr, 0.0
        lines = _thin(red)
        if lines.sum() == 0:
            return bgr, 0.0
        lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
        return cv2.inpaint(bgr, lines * 255, 3, cv2.INPAINT_TELEA), float(lines.mean())

    # ------------------------------------------------------------------ 3) แสงสะท้อน
    def glare_mask(bgr, cfg):
        if cfg.get("fix_paper_not_glare", True):
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            if float(np.median(hsv[:, :, 2])) >= 225 and float(np.median(hsv[:, :, 1])) <= 40:
                # พื้นขาวทั้งภาพ (PDF / กระดาษขาว) — แยกแสงสะท้อนจากพื้นไม่ได้ -> ไม่ตัดพื้นที่
                return np.zeros(bgr.shape[:2], np.uint8)
        return _orig_glare(bgr, cfg)

    # ------------------------------------------------------------------ QR
    def _finder_count(binm):
        cnts, hier = cv2.findContours(binm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            return 0
        hier = hier[0]

        def square(c):
            x, y, w, h = cv2.boundingRect(c)
            if w < 6 or h < 6 or not (0.8 < w / h < 1.25):
                return False
            if cv2.contourArea(c) < 0.88 * w * h:            # วงกลม ~0.79 -> ไม่ผ่าน
                return False
            ap = cv2.approxPolyDP(c, 0.05 * cv2.arcLength(c, True), True)
            return len(ap) == 4

        n = 0
        for i, c in enumerate(cnts):
            if not square(c):
                continue
            k, depth, ok = hier[i][2], 0, True
            while k != -1 and depth < 2:
                if not square(cnts[k]):
                    ok = False
                    break
                depth += 1
                k = hier[k][2]
            if ok and depth >= 2:
                n += 1
        return n

    def find_qr(gray, binm, cfg):
        boxes = _orig_find_qr(gray, binm, cfg)
        if not cfg.get("fix_qr_strict", True) or not boxes:
            return boxes
        H, W = binm.shape
        keep = []
        det = cv2.QRCodeDetector()
        for (x0, y0, x1, y1) in boxes:
            if (x1 - x0) * (y1 - y0) > 0.30 * H * W:            # QR ไม่ใหญ่ขนาดครึ่งมาร์ค
                continue
            crop = gray[y0:y1, x0:x1]
            txt = ""
            try:
                txt, _, _ = det.detectAndDecode(crop)
            except Exception:
                pass
            if txt or _finder_count(binm[y0:y1, x0:x1]) >= 3:
                keep.append((x0, y0, x1, y1))
        return keep

    # ------------------------------------------------------------------ 3) จัดภาพ
    def _sane(Htot, tshape, mshape):
        if Htot is None:
            return False
        th, tw = tshape[:2]
        mh, mw = mshape[:2]
        c = cv2.perspectiveTransform(np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2),
                                     np.asarray(Htot, np.float64)).reshape(-1, 2)
        if not np.isfinite(c).all():
            return False
        area = abs(cv2.contourArea(c.astype(np.float32)))
        if not cv2.isContourConvex(c.astype(np.float32).reshape(-1, 1, 2)):
            return False
        r = area / float(mw * mh)
        if not 0.15 < r < 15:
            return False
        sides = [np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]
        if min(sides) < 0.25 * max(sides) * (min(tw, th) / max(tw, th)):
            return False
        return True

    def _fit_score(mpre, warped, valid, mink):
        zone = cv2.dilate(mink, np.ones((9, 9), np.uint8)) > 0
        if valid is not None:
            cov = float((valid[mink > 0] > 0).mean()) if (mink > 0).any() else 0.0
            zone &= valid > 0
        else:
            cov = 1.0
        if zone.sum() < 200:
            return -1.0
        a = cv2.GaussianBlur(mpre, (0, 0), 1.5).astype(np.float32)[zone]
        b = cv2.GaussianBlur(warped, (0, 0), 1.5).astype(np.float32)[zone]
        a -= a.mean()
        b -= b.mean()
        ncc = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))
        return ncc * min(1.0, cov / 0.8)

    def align(mpre, tpre, mink, cfg):
        if not cfg.get("fix_align_check", True):
            return _orig_align(mpre, tpre, mink, cfg)
        Hm, Wm = mpre.shape
        Ht, Wt = tpre.shape
        cands = []
        w1, v1, i1 = _orig_align(mpre, tpre, mink, cfg)
        if w1 is not None:
            ok = _sane(i1.get("Htot"), tpre.shape, mpre.shape)
            cands.append((_fit_score(mpre, w1, v1, mink) if ok else -2.0, w1, v1, i1))
        # จัดตรง: Test สัดส่วนเท่า Master (เช่น Test จาก PDF ที่จัดให้ทับ Master แล้ว)
        if abs((Wt / float(Ht)) / (Wm / float(Hm)) - 1) < 0.06:
            S = np.diag([Wm / float(Wt), Hm / float(Ht), 1.0])
            real = SC._match

            def fake(a, b, det, norm, subpix):
                return (S.copy(), 10 ** 4) if b is tpre else (None, 0)
            SC._match = fake
            try:
                w2, v2, i2 = _orig_align(mpre, tpre, mink, cfg)
            finally:
                SC._match = real
            if w2 is not None:
                i2["direct"] = True
                i2["inliers"] = i1.get("inliers", 0) if w1 is not None else 0
                i2["method"] = i2["method"].replace("SIFT+subpix", "จัดตรง").replace("SIFT", "จัดตรง")
                cands.append((_fit_score(mpre, w2, v2, mink), w2, v2, i2))
        if not cands:
            return w1, v1, i1
        cands.sort(key=lambda c: -c[0])
        score, w, v, info = cands[0]
        if score <= -2.0:                                  # SIFT บิดผิดรูป และไม่มีทางเลือก
            return None, None, dict(ok=False, inliers=i1.get("inliers", 0))
        info["fit"] = round(score, 3)
        if info.get("direct"):
            need = cfg["min_inliers_force"] if cfg.get("continue_if_blur") else cfg["min_inliers"]
            info["inliers"] = max(int(info.get("inliers", 0)), int(need))
        return w, v, info

    # ------------------------------------------------------------------ 2) ส้มที่ขอบ
    def compare(mink, tink, check_mask, cfg):
        rows, extras, miss, extra = _orig_compare(mink, tink, check_mask, cfg)
        if not cfg.get("fix_edge_partial", True):
            return rows, extras, miss, extra
        H, W = check_mask.shape
        lim = float(cfg.get("fix_edge_max_loss", 0.10))
        for r in rows:
            if r["status"] != "PARTIAL" or r.get("cov") is None or 1 - r["cov"] > lim:
                continue
            p = max(4, int(0.15 * max(r["w"], r["h"])))
            x0, y0 = max(0, r["x"] - p), max(0, r["y"] - p)
            x1, y1 = min(W, r["x"] + r["w"] + p), min(H, r["y"] + r["h"] + p)
            m = mink[y0:y1, x0:x1] > 0
            near = cv2.dilate(m.astype(np.uint8), np.ones((2 * p + 1,) * 2, np.uint8)) > 0
            if (check_mask[y0:y1, x0:x1][near] == 0).mean() > 0.02:
                r["status"] = "OK"
                r["note"] += " | ชิดขอบพื้นที่ตรวจ หายจริงน้อย -> ไม่นับจาง"
        return rows, extras, miss, extra

    # ------------------------------------------------------------------ inspect
    def inspect(master_bgr, test_bgr, cfg=None, ignore_zones=None, debug_dir=None, force_continue=None):
        cfg = _cfg(cfg)
        pre = []
        if cfg["fix_master_color_frame"]:
            master_bgr, n = remove_master_color_frame(master_bgr)
            if n:
                pre.append("ลบกรอบเส้นประสีของ Master แล้ว")
        if cfg["fix_test_red_lines"]:
            test_bgr, f = clean_test_red_lines(test_bgr)
            if f:
                pre.append("ลบเส้นแดงบน Test แล้ว")
        r = _orig_inspect(master_bgr, test_bgr, cfg, ignore_zones, debug_dir, force_continue)
        notes = []
        for n in r.get("notes", []):
            if n.startswith("จัดภาพ:") and "จัดตรง" in n:
                n = n.split(" จุดอ้างอิง")[0] + " (ภาพขนาดเท่า Master)"
            notes.append(n)
        r["notes"] = [f"fix v{FIX_VERSION}"] + pre + notes
        if r.get("report"):
            r["report"]["notes"] = r["notes"]
            r["report"]["engine"] = SC.VERSION
        return r

    SC.load_config = load_config
    SC.glare_mask = glare_mask
    SC.find_qr = find_qr
    SC.align = align
    SC.compare = compare
    SC.inspect = inspect
    SC.remove_master_color_frame = remove_master_color_frame
    SC.clean_test_red_lines = clean_test_red_lines

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mark_ui_plus.py — เปิดโปรแกรมตรวจมาร์คเดิม (mark_ui.py) + ส่วนเสริม  [plus v3]

ไม่แก้ไขไฟล์เดิม: สืบทอดหน้าจอเดิม แล้วเพิ่ม
  1) แถบ "สีภาพ Test" — ภาพตัวหนังสือสีอ่อนบนพื้นเขียว/พื้นเข้ม (test_polarity.py)
  2) ช่อง "โมเดล" — พิมพ์ชื่อโมเดล หรือเลือกจากรายการโมเดลที่เคยใช้
     ผลจะถูกเก็บแยกโฟลเดอร์ตามชื่อโมเดล (โปรแกรมจำโมเดลล่าสุดไว้ให้)
  3) บันทึกผลอัตโนมัติทุกครั้งที่ตรวจ (ติ๊กปิดได้)
       results/<โมเดล>/YYYYMMDD/HHMMSS_OK/   หรือ   ..._NG/
           1_master.png   2_test.png   3_result.png   report.json
       results/<โมเดล>/log.csv   ← สรุปของโมเดลนั้น
       results/log_all.csv       ← สรุปทุกโมเดลรวมกัน
     ภาพทุกภาพประทับ วันเวลา + ชื่อโมเดล (+ OK/NG ที่ภาพผล)
  4) ปุ่ม "เปิดโฟลเดอร์ผล" — เปิดโฟลเดอร์ของโมเดลที่เลือกอยู่

รัน:  python mark_ui_plus.py   (Windows: RUN_UI_PLUS.bat / Pi: bash run_plus.sh)
"""
import csv
import json
import os
import re
import subprocess
import sys
import time
import tkinter as tk
from tkinter import ttk

import cv2

import mark_ui
from test_polarity import MODES, normalize_test

try:
    from glyph_check import draw_result
except Exception:          # glyph_check รุ่นเก่าไม่มี draw_result -> ใช้ภาพ overlay แทน
    draw_result = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLUS_SETTINGS = os.path.join(BASE_DIR, "plus_settings.json")
NO_MODEL = "_ไม่ระบุโมเดล"


def safe_model_name(name):
    """แปลงชื่อโมเดลให้ใช้เป็นชื่อโฟลเดอร์ได้ (ตัดอักขระต้องห้ามของ Windows/Linux)"""
    name = (name or "").strip()
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:60] or NO_MODEL


def _imwrite(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)          # รองรับ path ภาษาไทยบน Windows


def _stamp(img, text, color):
    """ประทับข้อความ (วันเวลา/โมเดล/ผล) ที่มุมล่างซ้ายของภาพ  — ใช้ตัวอักษรอังกฤษ/ตัวเลข"""
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    scale = max(0.45, w / 1000)
    th = max(1, int(round(scale * 2)))
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = 8, h - 10
    cv2.rectangle(out, (x - 6, y - tht - 10), (x + tw + 6, y + base + 2), (0, 0, 0), -1)
    cv2.putText(out, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)
    return out


def _ascii(s):
    """OpenCV เขียนภาษาไทยลงภาพไม่ได้ -> แสดงเฉพาะอักขระอังกฤษ/ตัวเลข (ชื่อโฟลเดอร์ยังเป็นไทยได้)"""
    a = "".join(c if 32 <= ord(c) < 127 else "?" for c in s)
    return a if a.strip("? ") else ""


class AppPlus(mark_ui.App):
    def __init__(self):
        self.pol_mode = None
        self.auto_save = None
        self.model = None
        self._pol_msg = ""
        self._raw_test = None
        super().__init__()
        self.title(mark_ui.APP + "  (+ โมเดล, บันทึกผลอัตโนมัติ, ภาพตัวหนังสือสีอ่อน)")
        saved = self._load_plus()
        self.pol_mode = tk.StringVar(value=MODES[0])
        self.auto_save = tk.BooleanVar(value=saved.get("auto_save", True))
        self.model = tk.StringVar(value=saved.get("last_model", ""))

        bar = tk.Frame(self, bg=mark_ui.CARD)
        tk.Label(bar, text="โมเดล", bg=mark_ui.CARD, font=mark_ui.F(10, "bold"),
                 fg="#1f3864").pack(side="left", padx=(10, 4))
        self.model_box = ttk.Combobox(bar, textvariable=self.model, values=self._known_models(),
                                      width=22, font=mark_ui.F(10))
        self.model_box.pack(side="left", pady=2)
        self.model_box.bind("<<ComboboxSelected>>", lambda e: self._on_model_change())
        self.model_box.bind("<FocusOut>", lambda e: self._on_model_change())
        self.model_box.bind("<Return>", lambda e: self._on_model_change())

        tk.Label(bar, text="สีภาพ Test", bg=mark_ui.CARD, font=mark_ui.F(9, "bold")).pack(side="left", padx=(16, 4))
        ttk.Combobox(bar, textvariable=self.pol_mode, values=list(MODES), state="readonly",
                     width=20, font=mark_ui.F(9)).pack(side="left", pady=2)
        ttk.Checkbutton(bar, text="บันทึกผลอัตโนมัติ", variable=self.auto_save,
                        command=self._save_plus).pack(side="left", padx=(16, 4))
        ttk.Button(bar, text="เปิดโฟลเดอร์ผล", command=self.open_results).pack(side="left", padx=4)
        nbs = [w for w in self.winfo_children() if isinstance(w, ttk.Notebook)]
        if nbs:
            bar.pack(fill="x", padx=6, before=nbs[0])
        else:
            bar.pack(fill="x", padx=6)

    # ---------------------------------------------------------- โมเดล
    def _known_models(self):
        """รายชื่อโมเดล = โฟลเดอร์ที่มีอยู่ใน results/"""
        if not os.path.isdir(RESULTS_DIR):
            return []
        return sorted(d for d in os.listdir(RESULTS_DIR)
                      if os.path.isdir(os.path.join(RESULTS_DIR, d)))

    def _model_dir(self):
        return os.path.join(RESULTS_DIR, safe_model_name(self.model.get() if self.model else ""))

    def _on_model_change(self):
        name = safe_model_name(self.model.get())
        if name != NO_MODEL:
            os.makedirs(os.path.join(RESULTS_DIR, name), exist_ok=True)   # สร้างโฟลเดอร์โมเดลทันที
            self.model_box["values"] = self._known_models()
            self.status.set(f"โมเดล: {name}  —  ผลจะเก็บที่ results/{name}/")
        self._save_plus()

    def _load_plus(self):
        try:
            return json.load(open(PLUS_SETTINGS, encoding="utf-8"))
        except Exception:
            return {}

    def _save_plus(self):
        try:
            json.dump({"last_model": self.model.get() if self.model else "",
                       "auto_save": bool(self.auto_save.get()) if self.auto_save else True},
                      open(PLUS_SETTINGS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ---------------------------------------------------------- ตรวจ
    def run_inspect(self, frame=None):
        test = frame if frame is not None else self.test_img
        self._raw_test = None if test is None else test.copy()     # เก็บภาพก่อนกลับสีไว้บันทึก
        if test is not None and self.pol_mode is not None and not self.busy:
            try:
                test, self._pol_msg = normalize_test(test, self.pol_mode.get())
            except Exception as ex:
                self._pol_msg = f"กลับสีไม่สำเร็จ ({ex}) — ใช้ภาพเดิม"
        return super().run_inspect(test)

    def _on_result(self, res):
        super()._on_result(res)
        extra = []
        if self._pol_msg:
            extra.append(self._pol_msg)
        if self.auto_save is not None and self.auto_save.get():
            try:
                folder = self._save_result()
                extra.append("บันทึกที่ " + os.path.relpath(folder, BASE_DIR))
            except Exception as ex:          # บันทึกพัง ห้ามทำให้การตรวจพัง
                extra.append(f"บันทึกผลไม่สำเร็จ: {ex}")
        if extra:
            self.status.set(self.status.get() + "   |   " + "   |   ".join(extra))

    # ---------------------------------------------------------- บันทึก
    def _save_result(self):
        rep = self.report
        verdict = rep.get("verdict", "")
        model = safe_model_name(self.model.get())
        mdir = os.path.join(RESULTS_DIR, model)
        now = time.localtime()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", now)
        folder = os.path.join(mdir, time.strftime("%Y%m%d", now), f"{time.strftime('%H%M%S', now)}_{verdict}")
        base, n = folder, 2
        while os.path.exists(folder):                     # กดตรวจซ้ำในวินาทีเดียวกัน
            folder = f"{base}_{n}"; n += 1
        os.makedirs(folder)

        mtxt = _ascii(model) if model != NO_MODEL else ""
        head = f"{stamp}  {mtxt}".strip()
        white, green, red = (235, 235, 235), (60, 220, 60), (60, 60, 255)
        if self.master_img is not None:
            _imwrite(os.path.join(folder, "1_master.png"), _stamp(self.master_img, head + "  MASTER", white))
        if self._raw_test is not None:
            _imwrite(os.path.join(folder, "2_test.png"), _stamp(self._raw_test, head + "  TEST", white))
        if draw_result is not None and self.aligned is not None:
            res_img = draw_result(self.aligned, rep.get("elements", []), rep.get("extras", []),
                                  self.ignores, True)
        else:
            res_img = self.overlay
        if res_img is not None:
            _imwrite(os.path.join(folder, "3_result.png"),
                     _stamp(res_img, f"{head}  {verdict}", green if verdict == "OK" else red))

        bad = [r for r in rep.get("elements", []) if r.get("status") in ("MISSING", "PARTIAL")]
        info = dict(timestamp=stamp, model=model, verdict=verdict, align=rep.get("align_method"),
                    master_file=self.master_path.get(), test_file=self.test_path.get(),
                    polarity=self._pol_msg, n_missing=rep.get("n_missing"), n_partial=rep.get("n_partial"),
                    defects=[{k: r.get(k) for k in ("id", "kind", "status", "x", "y", "w", "h", "note")}
                             for r in bad])
        with open(os.path.join(folder, "report.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        row = [stamp, model, verdict, rep.get("n_missing"), rep.get("n_partial"),
               " ".join(f"#{r.get('id')}" for r in bad),
               os.path.basename(self.master_path.get()), os.path.basename(self.test_path.get()),
               os.path.relpath(folder, RESULTS_DIR)]
        header = ["วันเวลา", "โมเดล", "ผล", "หายทั้งชิ้น", "จางขาดหาย", "ตำแหน่ง NG (id)",
                  "Master", "Test", "โฟลเดอร์"]
        for log in (os.path.join(mdir, "log.csv"), os.path.join(RESULTS_DIR, "log_all.csv")):
            new = not os.path.exists(log)
            with open(log, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(header)
                w.writerow(row)
        self.model_box["values"] = self._known_models()
        return folder

    def open_results(self):
        target = self._model_dir() if safe_model_name(self.model.get()) != NO_MODEL else RESULTS_DIR
        os.makedirs(target, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(target)
            else:
                subprocess.Popen(["xdg-open", target])
        except Exception as ex:
            self.status.set(f"เปิดโฟลเดอร์ไม่ได้: {ex}  —  ผลอยู่ที่ {target}")


if __name__ == "__main__":
    AppPlus().mainloop()

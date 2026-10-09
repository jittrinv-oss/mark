#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mark_ui_plus.py — โปรแกรมตรวจมาร์คเดิม (mark_ui.py) + ส่วนเสริม  [plus v6]
ไม่แก้ไขไฟล์เดิม:
  1) มาร์คบนสกรีนสี (test_polarity.py v2) — แปลง Master+Test, ตัดแผ่นสกรีน, ขยาย Test
  2) ช่อง "โมเดล" — เก็บผลแยกโฟลเดอร์ตามโมเดล (จำโมเดลล่าสุด)
  3) บันทึกผลอัตโนมัติ: results/<โมเดล>/YYYYMMDD/HHMMSS_OK|NG/
         1_master.png 2_test.png 3_result.png report.json + log.csv, log_all.csv
  4) ปุ่ม "เปิดโฟลเดอร์ผล"
  5) ปุ่ม "📷 ถ่ายเป็น Master" — ใช้ภาพกล้องเป็น Master (บันทึกที่ results/<โมเดล>/masters/)
  6) v6: ปุ่ม "📄 Test จาก PDF" — เลือก PDF (Order screen) แล้วโปรแกรมหามาร์คเอง
         + zoom + กลับด้าน (mirror) + หมุน ให้ตรงกับ Master อัตโนมัติ (pdf_autofind.py)
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
from test_polarity import MODES, prepare_pair

try:
    from glyph_check import draw_result
except Exception:
    draw_result = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLUS_SETTINGS = os.path.join(BASE_DIR, "plus_settings.json")
NO_MODEL = "_ไม่ระบุโมเดล"


def safe_model_name(name):
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", (name or "").strip())
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:60] or NO_MODEL


def _imwrite(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)


def _stamp(img, text, color):
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
    a = "".join(c if 32 <= ord(c) < 127 else "?" for c in s)
    return a if a.strip("? ") else ""


class AppPlus(mark_ui.App):
    def __init__(self):
        self.pol_mode = None
        self.auto_save = None
        self.model = None
        self._pol_msg = ""
        self._raw_test = None
        self._raw_master = None
        super().__init__()
        self.title(mark_ui.APP + "  (+ สกรีนสี, โมเดล, บันทึกผล, Test จาก PDF)")
        saved = self._load_plus()
        self.pol_mode = tk.StringVar(value=MODES[0])
        self.auto_save = tk.BooleanVar(value=saved.get("auto_save", True))
        self.model = tk.StringVar(value=saved.get("last_model", ""))

        bar = tk.Frame(self, bg=mark_ui.CARD)
        tk.Label(bar, text="โมเดล", bg=mark_ui.CARD, font=mark_ui.F(10, "bold"),
                 fg="#1f3864").pack(side="left", padx=(10, 4))
        self.model_box = ttk.Combobox(bar, textvariable=self.model, values=self._known_models(),
                                      width=18, font=mark_ui.F(10))
        self.model_box.pack(side="left", pady=2)
        for ev in ("<<ComboboxSelected>>", "<FocusOut>", "<Return>"):
            self.model_box.bind(ev, lambda e: self._on_model_change())
        tk.Label(bar, text="สีภาพ", bg=mark_ui.CARD, font=mark_ui.F(9, "bold")).pack(side="left", padx=(12, 4))
        ttk.Combobox(bar, textvariable=self.pol_mode, values=list(MODES), state="readonly",
                     width=18, font=mark_ui.F(9)).pack(side="left", pady=2)
        ttk.Checkbutton(bar, text="บันทึกผลอัตโนมัติ", variable=self.auto_save,
                        command=self._save_plus).pack(side="left", padx=(12, 4))
        ttk.Button(bar, text="เปิดโฟลเดอร์ผล", command=self.open_results).pack(side="left", padx=4)
        ttk.Button(bar, text="📷 ถ่ายเป็น Master", command=self.capture_master).pack(side="left", padx=(12, 4))
        ttk.Button(bar, text="📄 Test จาก PDF", command=self.test_from_pdf).pack(side="left", padx=4)
        nbs = [w for w in self.winfo_children() if isinstance(w, ttk.Notebook)]
        if nbs:
            bar.pack(fill="x", padx=6, before=nbs[0])
        else:
            bar.pack(fill="x", padx=6)

    # ---------------------------------------------------------- โมเดล
    def _known_models(self):
        if not os.path.isdir(RESULTS_DIR):
            return []
        return sorted(d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d)))

    def _model_dir(self):
        return os.path.join(RESULTS_DIR, safe_model_name(self.model.get() if self.model else ""))

    def _on_model_change(self):
        name = safe_model_name(self.model.get())
        if name != NO_MODEL:
            os.makedirs(os.path.join(RESULTS_DIR, name), exist_ok=True)
            self.model_box["values"] = self._known_models()
            self.status.set(f"โมเดล: {name}  —  ผลจะเก็บที่ results/{name}/")
        self._save_plus()

    def _load_plus(self):
        try:
            with open(PLUS_SETTINGS, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_plus(self):
        try:
            with open(PLUS_SETTINGS, "w", encoding="utf-8") as f:
                json.dump({"last_model": self.model.get() if self.model else "",
                           "auto_save": bool(self.auto_save.get()) if self.auto_save else True},
                          f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ---------------------------------------------------------- ถ่าย Master จากกล้อง
    def capture_master(self):
        from tkinter import messagebox
        frame = getattr(self, "live_frame", None)
        if not getattr(self, "live", False) or frame is None:
            messagebox.showwarning(mark_ui.APP, "ยังไม่ได้เปิดกล้อง\n"
                                   "กด 'ค้นหากล้อง' → เลือกกล้อง → 'เปิดกล้อง' ก่อน แล้ววางชิ้นงาน Master ใต้กล้อง")
            return
        if not messagebox.askyesno(mark_ui.APP, "ใช้ภาพจากกล้องตอนนี้เป็น Master หรือไม่?\n\n"
                                   "ใช้ชิ้นงานที่ QA อนุมัติแล้วเท่านั้น\n"
                                   "(โซน IGNORE / การครอบตัดเดิมจะถูกล้าง ต้องลากใหม่)"):
            return
        img = frame.copy()
        folder = os.path.join(self._model_dir(), "masters")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, time.strftime("master_%Y%m%d_%H%M%S.png"))
        _imwrite(path, img)
        self.master_path.set(path)
        if hasattr(self, "_set_master"):
            self._set_master(img, keep_full=True)
        else:
            self.master_img = img
        self.model_box["values"] = self._known_models()
        self.status.set(f"ตั้ง Master จากกล้องแล้ว — บันทึกที่ {os.path.relpath(path, BASE_DIR)}"
                        "   |   ลากโซน IGNORE ใหม่ แล้วกด 'บันทึก' config")

    # ---------------------------------------------------------- Test จาก PDF
    def test_from_pdf(self):
        """เลือก PDF -> หามาร์ค + zoom + mirror + หมุน ให้ตรงกับ Master -> ใช้เป็นภาพ Test"""
        from tkinter import filedialog, messagebox
        if self.master_img is None:
            messagebox.showwarning(mark_ui.APP, "เลือก Master ก่อน (ภาพมาร์คที่อ่านได้ปกติ)\n"
                                   "โปรแกรมใช้ Master เป็นแบบในการหามาร์คใน PDF")
            return
        p = filedialog.askopenfilename(title="เลือก PDF (Order screen) สำหรับ Test",
                                       filetypes=[("PDF", "*.pdf")])
        if not p:
            return
        self.status.set("กำลังหามาร์คใน PDF (zoom / mirror / หมุน อัตโนมัติ) … บน Pi อาจใช้ 10-30 วินาที")
        self.update_idletasks()
        try:
            from pdf_autofind import find_mark_in_pdf
            r = find_mark_in_pdf(p, self.master_img)
        except Exception as ex:
            messagebox.showerror(mark_ui.APP, f"หามาร์คใน PDF ไม่สำเร็จ:\n{ex}")
            self.status.set("หามาร์คใน PDF ไม่สำเร็จ")
            return
        if getattr(self, "live", False) and hasattr(self, "toggle_camera"):
            self.toggle_camera()
        self.test_img = r["test"]
        self.test_path.set(p)
        pn = self.__dict__.get("pn_test")
        if pn is not None and hasattr(pn, "title_var"):
            pn.title_var.set("Test (จาก PDF — ปรับทิศแล้ว)")
            pn.set(self.test_img)
        self.status.set(f"พบมาร์คใน PDF: กลับด้าน={'ใช่' if r['mirrored'] else 'ไม่'}  หมุน {r['angle']}°  "
                        f"จุดอ้างอิง {r['inliers']}  — กด ▶ ตรวจสอบ")

    # ---------------------------------------------------------- ตรวจ
    def run_inspect(self, frame=None):
        test = frame if frame is not None else self.test_img
        if self.busy or test is None or self.master_img is None or self.pol_mode is None:
            return super().run_inspect(frame)
        self._raw_test = test.copy()
        self._raw_master = self.master_img
        try:
            m2, t2, self._pol_msg = prepare_pair(self.master_img, test, self.pol_mode.get())
        except Exception as ex:
            m2, t2, self._pol_msg = self.master_img, test, f"แปลงสีไม่สำเร็จ ({ex}) — ใช้ภาพเดิม"
        self.master_img = m2
        try:
            return super().run_inspect(t2)
        finally:
            self.master_img = self._raw_master

    def _on_result(self, res):
        super()._on_result(res)
        extra = [self._pol_msg] if self._pol_msg else []
        if self.auto_save is not None and self.auto_save.get():
            try:
                folder = self._save_result()
                extra.append("บันทึกที่ " + os.path.relpath(folder, BASE_DIR))
            except Exception as ex:
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
        while os.path.exists(folder):
            folder = f"{base}_{n}"
            n += 1
        os.makedirs(folder)

        mtxt = _ascii(model) if model != NO_MODEL else ""
        head = f"{stamp}  {mtxt}".strip()
        white, green, red = (235, 235, 235), (60, 220, 60), (60, 60, 255)
        master = self._raw_master if self._raw_master is not None else self.master_img
        if master is not None:
            _imwrite(os.path.join(folder, "1_master.png"), _stamp(master, head + "  MASTER", white))
        if self._raw_test is not None:
            _imwrite(os.path.join(folder, "2_test.png"), _stamp(self._raw_test, head + "  TEST", white))
        if draw_result is not None and self.aligned is not None:
            res_img = draw_result(self.aligned, rep.get("elements", []), rep.get("extras", []), self.ignores, True)
        else:
            res_img = self.overlay
        if res_img is not None:
            _imwrite(os.path.join(folder, "3_result.png"),
                     _stamp(res_img, f"{head}  {verdict}", green if verdict == "OK" else red))

        bad = [r for r in rep.get("elements", []) if r.get("status") in ("MISSING", "PARTIAL")]
        info = dict(timestamp=stamp, model=model, verdict=verdict, align=rep.get("align_method"),
                    master_file=self.master_path.get(), test_file=self.test_path.get(),
                    color_conversion=self._pol_msg, n_missing=rep.get("n_missing"), n_partial=rep.get("n_partial"),
                    defects=[{k: r.get(k) for k in ("id", "kind", "status", "x", "y", "w", "h", "note")} for r in bad])
        with open(os.path.join(folder, "report.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        row = [stamp, model, verdict, rep.get("n_missing"), rep.get("n_partial"),
               " ".join(f"#{r.get('id')}" for r in bad),
               os.path.basename(self.master_path.get()), os.path.basename(self.test_path.get()),
               os.path.relpath(folder, RESULTS_DIR)]
        header = ["วันเวลา", "โมเดล", "ผล", "หายทั้งชิ้น", "จางขาดหาย", "ตำแหน่ง NG (id)", "Master", "Test", "โฟลเดอร์"]
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mark_ui_plus.py — เปิดโปรแกรมตรวจมาร์คเดิม (mark_ui.py) + ส่วนเสริม  [plus v2]

ไม่แก้ไขไฟล์เดิม: สืบทอดหน้าจอเดิม แล้วเพิ่ม
  1) แถบ "สีภาพ Test" — ภาพตัวหนังสือสีอ่อนบนพื้นเขียว/พื้นเข้ม (test_polarity.py)
  2) บันทึกผลอัตโนมัติทุกครั้งที่ตรวจ (ติ๊กปิดได้)
       results/YYYYMMDD/HHMMSS_OK/   หรือ   results/YYYYMMDD/HHMMSS_NG/
           1_master.png   2_test.png   3_result.png   report.json
       results/log.csv   ← สรุปทุกครั้ง (วันเวลา, ผล, ไฟล์ Master/Test, จำนวนจุด NG)
     มุมขวาล่างของภาพผลจะมีวันเวลาและผล OK/NG ประทับไว้
  3) ปุ่ม "เปิดโฟลเดอร์ผล"

รัน:  python mark_ui_plus.py   (Windows: RUN_UI_PLUS.bat / Pi: bash run_plus.sh)
"""
import csv
import json
import os
import subprocess
import sys
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import cv2

import mark_ui
from test_polarity import MODES, normalize_test

try:
    from glyph_check import draw_result
except Exception:          # glyph_check รุ่นเก่าไม่มี draw_result -> ใช้ภาพ overlay แทน
    draw_result = None

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _imwrite(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)          # รองรับ path ภาษาไทยบน Windows


def _stamp(img, text, ok):
    """ประทับวันเวลา + ผล ที่มุมล่างของภาพ"""
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    scale = max(0.5, w / 900)
    th = max(1, int(round(scale * 2)))
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = 8, h - 10
    cv2.rectangle(out, (x - 6, y - tht - 10), (x + tw + 6, y + base + 2), (0, 0, 0), -1)
    cv2.putText(out, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (60, 220, 60) if ok else (60, 60, 255), th, cv2.LINE_AA)
    return out


class AppPlus(mark_ui.App):
    def __init__(self):
        self.pol_mode = None
        self.auto_save = None
        self._pol_msg = ""
        self._raw_test = None
        super().__init__()
        self.title(mark_ui.APP + "  (+ ภาพตัวหนังสือสีอ่อน, บันทึกผลอัตโนมัติ)")
        self.pol_mode = tk.StringVar(value=MODES[0])
        self.auto_save = tk.BooleanVar(value=True)

        bar = tk.Frame(self, bg=mark_ui.CARD)
        tk.Label(bar, text="สีภาพ Test", bg=mark_ui.CARD, font=mark_ui.F(9, "bold")).pack(side="left", padx=(10, 4))
        ttk.Combobox(bar, textvariable=self.pol_mode, values=list(MODES), state="readonly",
                     width=22, font=mark_ui.F(9)).pack(side="left", pady=2)
        ttk.Checkbutton(bar, text="บันทึกผลอัตโนมัติ (Master/Test/ผล + วันเวลา)",
                        variable=self.auto_save).pack(side="left", padx=(16, 4))
        ttk.Button(bar, text="เปิดโฟลเดอร์ผล", command=self.open_results).pack(side="left", padx=4)
        nbs = [w for w in self.winfo_children() if isinstance(w, ttk.Notebook)]
        if nbs:
            bar.pack(fill="x", padx=6, before=nbs[0])
        else:
            bar.pack(fill="x", padx=6)

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
                extra.append("บันทึกที่ " + os.path.relpath(folder, os.path.dirname(RESULTS_DIR)))
            except Exception as ex:          # บันทึกพัง ห้ามทำให้การตรวจพัง
                extra.append(f"บันทึกผลไม่สำเร็จ: {ex}")
        if extra:
            self.status.set(self.status.get() + "   |   " + "   |   ".join(extra))

    # ---------------------------------------------------------- บันทึก
    def _save_result(self):
        rep = self.report
        ok = rep.get("verdict") == "OK"
        now = time.localtime()
        day = time.strftime("%Y%m%d", now)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", now)
        folder = os.path.join(RESULTS_DIR, day, f"{time.strftime('%H%M%S', now)}_{rep.get('verdict', '')}")
        n = 2
        base = folder
        while os.path.exists(folder):                     # กดตรวจซ้ำในวินาทีเดียวกัน
            folder = f"{base}_{n}"; n += 1
        os.makedirs(folder)

        label = f"{stamp}  {rep.get('verdict', '')}"
        if self.master_img is not None:
            _imwrite(os.path.join(folder, "1_master.png"), _stamp(self.master_img, stamp + "  MASTER", True))
        if self._raw_test is not None:
            _imwrite(os.path.join(folder, "2_test.png"), _stamp(self._raw_test, stamp + "  TEST", True))
        if draw_result is not None and self.aligned is not None:
            res_img = draw_result(self.aligned, rep.get("elements", []), rep.get("extras", []),
                                  self.ignores, True)
        else:
            res_img = self.overlay
        if res_img is not None:
            _imwrite(os.path.join(folder, "3_result.png"), _stamp(res_img, label, ok))

        bad = [r for r in rep.get("elements", []) if r.get("status") in ("MISSING", "PARTIAL")]
        info = dict(timestamp=stamp, verdict=rep.get("verdict"), align=rep.get("align_method"),
                    master_file=self.master_path.get(), test_file=self.test_path.get(),
                    polarity=self._pol_msg, n_missing=rep.get("n_missing"), n_partial=rep.get("n_partial"),
                    defects=[{k: r.get(k) for k in ("id", "kind", "status", "x", "y", "w", "h", "note")}
                             for r in bad])
        with open(os.path.join(folder, "report.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        log = os.path.join(RESULTS_DIR, "log.csv")
        new = not os.path.exists(log)
        with open(log, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["วันเวลา", "ผล", "หายทั้งชิ้น", "จางขาดหาย", "ตำแหน่ง NG (id)",
                            "Master", "Test", "โฟลเดอร์"])
            w.writerow([stamp, rep.get("verdict"), rep.get("n_missing"), rep.get("n_partial"),
                        " ".join(f"#{r.get('id')}" for r in bad),
                        os.path.basename(self.master_path.get()), os.path.basename(self.test_path.get()),
                        os.path.relpath(folder, RESULTS_DIR)])
        return folder

    def open_results(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(RESULTS_DIR)
            else:
                subprocess.Popen(["xdg-open", RESULTS_DIR])
        except Exception as ex:
            self.status.set(f"เปิดโฟลเดอร์ไม่ได้: {ex}  —  ผลอยู่ที่ {RESULTS_DIR}")


if __name__ == "__main__":
    AppPlus().mainloop()

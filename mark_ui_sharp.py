#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mark_ui_sharp.py v1 — โปรแกรมตรวจมาร์ค + "ตัวตรวจคมชัด" (sharp_check.py) สำหรับภาพเบลอ / กล้องความละเอียดต่ำ
ไม่แก้ไฟล์เดิม: ต่อยอดจาก mark_ui_kiosk (เต็มจอ / ตัวหนังสือใหญ่ / prealign) -> mark_ui_plus -> mark_ui

เพิ่ม:
  ☑ ใช้ตัวตรวจคมชัด        เปิด = ใช้ sharp_check / ปิด = ตัวตรวจเดิม (glyph_check v9) ทุกอย่างเหมือนเดิม
  ☐ ตรวจต่อแม้ภาพเบลอ      ปิด = ภาพเบลอขึ้น "ถ่ายใหม่" (ไม่ตัด NG) / เปิด = ตรวจต่อ
  [ตั้งค่าคมชัด]            เปิดไฟล์ sharp_config.json (ค่าทุกตัวอยู่ที่นี่)
  กล้อง: เปิดที่ความละเอียดสูงสุดที่กล้องรองรับ (เดิมกล้อง 2K เปิดได้แค่ 640x480)
  📷 จับภาพ+ตรวจ: ถ่ายหลายเฟรม เลือกเฟรมคมสุด / เฉลี่ย (ตั้งใน config)
  ภาพ debug ทุกขั้น: results/<โมเดล>/<วัน>/<เวลา>_<ผล>_debug/
"""
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2

import mark_ui
import sharp_check as SC

try:
    from mark_ui_kiosk import KioskApp as Base, maximize
except Exception:
    maximize = None
    try:
        from mark_ui_plus import AppPlus as Base
    except Exception:
        from mark_ui import App as Base

ORANGE = "#ef8a00"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ กล้องความละเอียดสูง
def open_camera_hi(index, backend_id=None, width=1920, height=1080):
    """ลองความละเอียดสูง -> ต่ำ (ตั้ง MJPG ก่อนตั้งขนาด) คืน (cap, (w,h)) ที่ได้ภาพจริงขนาดใหญ่สุด"""
    backs = [backend_id] if backend_id is not None else (
        [cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform == "win32" else [cv2.CAP_V4L2])
    sizes = [(2560, 1440), (1920, 1080), (1600, 1200), (1280, 960), (1280, 720)]
    for bid in backs:
        for (w, h) in sizes:
            cap = cv2.VideoCapture(index, bid)
            if not cap.isOpened():
                cap.release()
                break
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            fr = None
            for _ in range(10):
                ok, f = cap.read()
                if ok and f is not None and f.size and f.mean() > 1:
                    fr = f
                    break
            if fr is not None and fr.shape[1] >= min(w, 1280):
                return cap, (fr.shape[1], fr.shape[0])
            cap.release()
    return mark_ui._orig_open_camera(index, backend_id)       # สำรอง = แบบเดิม


if not hasattr(mark_ui, "_orig_open_camera"):
    mark_ui._orig_open_camera = mark_ui.open_camera
    mark_ui.open_camera = open_camera_hi


class SharpApp(Base):
    def __init__(self):
        self.use_sharp = None
        super().__init__()
        self.title(mark_ui.APP + "  (+ ตัวตรวจคมชัด)")
        cfg = SC.load_config()
        self.use_sharp = tk.BooleanVar(value=True)
        self.cont_blur = tk.BooleanVar(value=bool(cfg.get("continue_if_blur", False)))
        bar = tk.Frame(self, bg=mark_ui.CARD)
        ttk.Checkbutton(bar, text="ใช้ตัวตรวจคมชัด (ภาพเบลอ / กล้อง res ต่ำ)",
                        variable=self.use_sharp).pack(side="left", padx=(10, 6))
        ttk.Checkbutton(bar, text="ตรวจต่อแม้ภาพเบลอ", variable=self.cont_blur).pack(side="left", padx=6)
        ttk.Button(bar, text="ตั้งค่าคมชัด", command=self.open_cfg).pack(side="left", padx=6)
        nbs = [w for w in self.winfo_children() if isinstance(w, ttk.Notebook)]
        if nbs:
            bar.pack(fill="x", padx=6, before=nbs[0])
        else:
            bar.pack(fill="x", padx=6)

    def open_cfg(self):
        SC.load_config()                                    # สร้างไฟล์ถ้ายังไม่มี
        try:
            if sys.platform == "win32":
                os.startfile(SC.CONFIG_FILE)
            else:
                subprocess.Popen(["xdg-open", SC.CONFIG_FILE])
        except Exception as ex:
            self.status.set(f"เปิดไฟล์ไม่ได้: {ex} — ไฟล์อยู่ที่ {SC.CONFIG_FILE}")

    def _big(self, text, color):
        if hasattr(super(), "_big"):
            return super()._big(text, color)
        lbl = self.__dict__.get("big_verdict")
        if lbl is not None:
            lbl.config(text=text, bg=color)

    # -------------------------------------------------------------- จับภาพหลายเฟรม
    def cam_grab(self):
        cap = getattr(self, "cap", None)
        if not getattr(self, "live", False) or cap is None or self.use_sharp is None or not self.use_sharp.get():
            return super().cam_grab()
        cfg = SC.load_config()
        frames = []
        for _ in range(max(1, int(cfg["multi_frames"]))):
            ok, f = cap.read()
            if ok and f is not None:
                frames.append(f)
        img, scores = SC.best_frame(frames, cfg["frame_mode"])
        if img is None:
            return super().cam_grab()
        self.test_img = img
        self.test_path.set(f"<จากกล้อง {len(frames)} เฟรม>")
        self._grab_note = f"เลือกเฟรมคมสุด {max(scores):.0f} จาก {len(frames)} เฟรม ({cfg['frame_mode']})"
        self.run_inspect(self.test_img)

    # -------------------------------------------------------------- ตรวจ
    def run_inspect(self, frame=None):
        if self.use_sharp is None or not self.use_sharp.get():
            return super().run_inspect(frame)
        test = frame if frame is not None else self.test_img
        if self.busy or test is None or self.master_img is None:
            return super().run_inspect(frame)
        cfg = SC.load_config()
        master = self.master_img.copy()
        ign = list(getattr(self, "ignores", []) or [])
        model = self.__dict__.get("model")
        mname = mark_ui_plus_safe(model.get() if model is not None else "")
        dbg = os.path.join(BASE_DIR, "results", mname, time.strftime("%Y%m%d"),
                           time.strftime("%H%M%S") + "_debug") if cfg["debug_save"] else None
        cont = bool(self.cont_blur.get())
        self.busy = True
        self._big("…", "#9aa5b1")
        self.status.set("ตัวตรวจคมชัด กำลังตรวจ …")
        self._raw_test = test.copy()
        self._raw_master = master
        self._pol_msg = ""

        def work():
            try:
                r = SC.inspect(master, test, cfg, ign, dbg, cont)
                self.after(0, lambda rr=r: self._sharp_done(rr))
            except Exception as ex:
                self.after(0, lambda e=ex: self._sharp_err(e))
        threading.Thread(target=work, daemon=True).start()

    def _sharp_err(self, ex):
        self.busy = False
        self._big("ERR", "#cc2b2b")
        self.status.set(f"ตัวตรวจคมชัดผิดพลาด: {ex} — ปิดติ๊ก 'ใช้ตัวตรวจคมชัด' เพื่อใช้ตัวตรวจเดิม")

    def _sharp_done(self, r):
        self.busy = False
        notes = list(r["notes"])
        gn = getattr(self, "_grab_note", "")
        if gn:
            notes.insert(0, gn)
            self._grab_note = ""
        if r["debug_dir"]:
            notes.append("debug: " + os.path.relpath(r["debug_dir"], BASE_DIR))
        if r["verdict"] == "RETAKE":
            self._big("ถ่ายใหม่", ORANGE)
            if r.get("aligned") is not None and self.__dict__.get("pn_result") is not None:
                self.pn_result.set(r["aligned"])
            self.status.set("⚠ " + "   |   ".join(notes))
            return
        self._pre_msgs = ["ตัวตรวจคมชัด"] + notes
        super()._on_result((r["report"], r["overlay"], r["aligned"]))


def mark_ui_plus_safe(name):
    try:
        from mark_ui_plus import safe_model_name
        return safe_model_name(name)
    except Exception:
        return name or "_ไม่ระบุโมเดล"


def main():
    app = SharpApp()
    if maximize is not None:
        fs = {"on": "--fullscreen" in sys.argv}
        app.bind("<F11>", lambda e: (fs.__setitem__("on", not fs["on"]), app.attributes("-fullscreen", fs["on"])))
        app.bind("<Escape>", lambda e: (fs.__setitem__("on", False), app.attributes("-fullscreen", False)))
        app.after(300, lambda: maximize(app))
        app.after(2500, lambda: maximize(app))
    app.mainloop()


if __name__ == "__main__":
    main()

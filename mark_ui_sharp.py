#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mark_ui_sharp.py v1.4 — โปรแกรมตรวจมาร์ค + "ตัวตรวจคมชัด" (sharp_check.py)
ไม่แก้ไฟล์เดิม: ต่อยอดจาก mark_ui_kiosk -> mark_ui_plus -> mark_ui
v1.4: - Test จาก PDF: ใช้ pdf_autofind v3 -> ภาพ Test เห็น "มาร์คเต็มก้อน" (เดิมตัดเท่ากรอบ Master)
      - เตือนเมื่อมาร์คใน PDF เรียง/มีข้อความต่างจาก Master (เช่น มี QR / เลขพาร์ท / S27413 เพิ่ม)
      - โหลด Test ใหม่ -> ล้างช่องผลตรวจ/ป้าย OK-NG ของครั้งก่อน (ไม่ค้างภาพผลเก่า)
v1.3: เรียก sharp_fix.py (แพตช์ตัวตรวจ) อัตโนมัติ — ไม่ต้องเปลี่ยนไฟล์ .bat
v1.2: - ขึ้น "ถ่ายใหม่" แล้วช่องผลตรวจแสดงภาพ Test ปัจจุบัน (เดิมค้างภาพผลของครั้งก่อน)
      - ข้อความแถบล่างบอกเหตุผลถ่ายใหม่ชัดเจน
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
    import sharp_fix  # noqa: F401  v1.3 แพตช์ (กรอบเส้นประ / ส้มผิด / Test จาก PDF)
    SHARP_VER = "v1.4"
except Exception as _ex:
    print("sharp_fix:", _ex)
    SHARP_VER = "v1.2"
try:
    from mark_ui_kiosk import KioskApp as Base, maximize
except Exception:
    maximize = None
    try:
        from mark_ui_plus import AppPlus as Base
    except Exception:
        from mark_ui import App as Base

ORANGE = "#ef8a00"
IDLE_GREY = "#9aa5b1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def open_camera_hi(index, backend_id=None, width=1920, height=1080):
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
    return mark_ui._orig_open_camera(index, backend_id)


if not hasattr(mark_ui, "_orig_open_camera"):
    mark_ui._orig_open_camera = mark_ui.open_camera
    mark_ui.open_camera = open_camera_hi


def mark_ui_plus_safe(name):
    try:
        from mark_ui_plus import safe_model_name
        return safe_model_name(name)
    except Exception:
        return name or "_ไม่ระบุโมเดล"


class SharpApp(Base):
    def __init__(self):
        self.use_sharp = None
        super().__init__()
        self.title(mark_ui.APP + f"  (+ ตัวตรวจคมชัด {SHARP_VER})")
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
        SC.load_config()
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

    def _show_result_img(self, img, title):
        pn = self.__dict__.get("pn_result")
        if pn is None or img is None:
            return
        try:
            if hasattr(pn, "title_var"):
                pn.title_var.set(title)
            pn.set(img)
        except Exception:
            pass

    # ---------------------------------------------------------- v1.4: ล้างผลเก่าเมื่อเปลี่ยน Test
    def _clear_result(self, img=None):
        try:
            self._big("—", IDLE_GREY)
        except Exception:
            pass
        if hasattr(self, "_show_verdict"):
            try:
                self._show_verdict("")
            except Exception:
                pass
        pn = self.__dict__.get("pn_result")
        if pn is not None:
            try:
                pn.config(highlightbackground=IDLE_GREY, highlightcolor=IDLE_GREY, highlightthickness=8)
            except Exception:
                pass
        if img is not None:
            self._show_result_img(img, "ผลตรวจ : ยังไม่ได้ตรวจ — กด ▶ ตรวจสอบ")
        # ป้ายเล็กข้างปุ่มตรวจสอบ (mark_ui) — ถ้ามี
        for name in ("verdict", "verdict_lbl", "lbl_verdict"):
            w = self.__dict__.get(name)
            if isinstance(w, tk.Label):
                try:
                    w.config(text="—", bg=IDLE_GREY)
                except Exception:
                    pass

    def _show_test(self, img, path, title):
        super()._show_test(img, path, title)
        self._clear_result(img)

    def test_from_pdf(self, path=None):
        """v1.4: หามาร์คใน PDF แบบเต็มก้อน + เตือน layout ต่างจาก Master"""
        from tkinter import filedialog, messagebox
        if self.master_img is None:
            messagebox.showwarning(mark_ui.APP, "เลือก Master ก่อน (ภาพมาร์คที่อ่านได้ปกติ)\n"
                                   "โปรแกรมใช้ Master เป็นแบบในการหามาร์คใน PDF")
            return
        p = path or filedialog.askopenfilename(title="เลือก PDF (Order screen) สำหรับ Test",
                                               filetypes=[("PDF", "*.pdf")])
        if not p:
            return
        self.status.set("กำลังหามาร์คใน PDF (zoom / mirror / หมุน อัตโนมัติ) … บน Pi อาจใช้ 1-2 นาที")
        self._big("…", IDLE_GREY)
        self.update_idletasks()
        try:
            from pdf_autofind import find_mark_in_pdf
            r = find_mark_in_pdf(p, self.master_img)
        except Exception as ex:
            self._clear_result()
            messagebox.showerror(mark_ui.APP, f"หามาร์คใน PDF ไม่สำเร็จ:\n{ex}")
            self.status.set("หามาร์คใน PDF ไม่สำเร็จ")
            return
        self._show_test(r["test"], p, "Test (จาก PDF — มาร์คเต็ม ปรับทิศแล้ว)")
        msg = (f"พบมาร์คใน PDF: กลับด้าน={'ใช่' if r['mirrored'] else 'ไม่'}  หมุน {r['angle']}°  "
               f"จุดอ้างอิง {r['inliers']} ({r.get('quality', '')})")
        if r.get("layout_note"):
            msg += "  |  " + r["layout_note"]
        self.status.set(msg + "  — กด ▶ ตรวจสอบ")
        if r.get("layout_diff"):
            messagebox.showwarning(mark_ui.APP, r["layout_note"].replace(" | ", "\n").replace(" — ", "\n") +
                                   "\n\nดูภาพ Test เทียบกับ Master ก่อนกดตรวจสอบ\n"
                                   "(ส่วนที่อยู่นอกกรอบ Master เช่น QR / เลขพาร์ท จะไม่ถูกตรวจ)")

    # ---------------------------------------------------------- กล้อง / ตรวจ
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
        self._big("…", IDLE_GREY)
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
        if r.get("debug_dir"):
            notes.append("debug: " + os.path.relpath(r["debug_dir"], BASE_DIR))
        if r["verdict"] == "RETAKE":
            self._big("ถ่ายใหม่", ORANGE)
            # v1.2: ไม่ค้างภาพผลครั้งก่อน — แสดงภาพ Test ปัจจุบัน
            self._show_result_img(r.get("test_view") if r.get("test_view") is not None else self._raw_test,
                                  "ผลตรวจ : ถ่ายใหม่ (ยังไม่ตัดสิน)")
            pn = self.__dict__.get("pn_result")
            if pn is not None:
                try:
                    pn.config(highlightbackground=ORANGE, highlightcolor=ORANGE, highlightthickness=8)
                except Exception:
                    pass
            self.status.set("⚠ ถ่ายใหม่ | " + "   |   ".join(notes))
            return
        self._pre_msgs = [f"ตัวตรวจคมชัด {SHARP_VER}"] + notes
        super()._on_result((r["report"], r["overlay"], r["aligned"]))


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

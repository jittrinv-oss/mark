#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mark_ui_sharp.py v1.5 — โปรแกรมตรวจมาร์ค + "ตัวตรวจคมชัด" (sharp_check.py)
ไม่แก้ไฟล์เดิม: ต่อยอดจาก mark_ui_kiosk -> mark_ui_plus -> mark_ui
v1.5: - ไม่มี "ถ่ายใหม่" (ส้ม) แล้ว — ตรวจต่อเสมอ (sharp_loose.py)
      - ปุ่ม "✂ ครอบรูป Test" ลากกรอบครอบภาพที่ถ่าย + "จำกรอบ" ใช้กับภาพกล้องถัดไปอัตโนมัติ
      - ช่อง "match ขั้นต่ำ %" (ค่าเริ่ม 50) — ชิ้นที่ match ถึงเกณฑ์ไม่ขึ้นส้ม/แดง
      - แสดงกรอบ OK (เขียว) ในช่องผลตรวจอัตโนมัติ, ไม่มีกรอบม่วง
v1.4: Test จาก PDF มาร์คเต็มก้อน / เตือน layout ต่าง / ล้างผลเก่าเมื่อเปลี่ยน Test
v1.3: เรียก sharp_fix.py อัตโนมัติ
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
    SHARP_VER = "v1.5"
except Exception as _ex:
    print("sharp_fix:", _ex)
    SHARP_VER = "v1.2"
try:
    import sharp_loose  # noqa: F401  v1.5 ไม่มีถ่ายใหม่ / spec หลวม / %match / กรอบ OK
except Exception as _ex:
    print("sharp_loose:", _ex)
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
    try:                                    # camera.py v2 (2K จริง)
        import camera
        if hasattr(camera, "RES_CANDIDATES"):
            return camera.open_camera(index, backend_id)
    except Exception:
        pass
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
            for _ in range(30):
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


def apply_crop(img, box):
    """box = [x0,y0,x1,y1] สัดส่วน 0-1"""
    if img is None or not box:
        return img
    h, w = img.shape[:2]
    x0, x1 = sorted((int(box[0] * w), int(box[2] * w)))
    y0, y1 = sorted((int(box[1] * h), int(box[3] * h)))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return img
    return img[y0:y1, x0:x1].copy()


class CropDialog(tk.Toplevel):
    """ลากกรอบครอบรูป -> on_done(box สัดส่วน, จำกรอบ, ตรวจเลย)"""

    def __init__(self, app, img, on_done, keep=False):
        super().__init__(app)
        self.title("ครอบรูป Test — ลากกรอบครอบเฉพาะมาร์ค")
        sw, sh = app.winfo_screenwidth(), app.winfo_screenheight()
        self.geometry(f"{int(sw * 0.8)}x{int(sh * 0.8)}+{int(sw * 0.1)}+{int(sh * 0.05)}")
        self.img, self.on_done, self.box = img, on_done, None
        self.keep = tk.BooleanVar(value=keep)
        bar = tk.Frame(self, bg=mark_ui.CARD)
        bar.pack(fill="x")
        ttk.Button(bar, text="✔ ใช้กรอบ + ตรวจ", command=lambda: self._done(True)).pack(side="left", padx=4, pady=4)
        ttk.Button(bar, text="ใช้กรอบ", command=lambda: self._done(False)).pack(side="left", padx=4)
        ttk.Button(bar, text="ล้างกรอบ", command=self._clear).pack(side="left", padx=4)
        ttk.Checkbutton(bar, text="จำกรอบ ใช้กับภาพกล้องถัดไปอัตโนมัติ", variable=self.keep).pack(side="left", padx=10)
        ttk.Button(bar, text="ยกเลิก", command=self.destroy).pack(side="right", padx=4)
        self.cv = tk.Canvas(self, bg=mark_ui.DARK, cursor="cross", highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        self.cv.bind("<Configure>", lambda e: self._draw())
        self.cv.bind("<ButtonPress-1>", self._down)
        self.cv.bind("<B1-Motion>", self._move)
        self.cv.bind("<ButtonRelease-1>", self._up)
        self.p0 = None
        self.transient(app)
        self.grab_set()

    def _draw(self):
        self.cv.delete("all")
        self._ph, self.s = mark_ui.to_photo(self.img, self.cv.winfo_width(), self.cv.winfo_height())
        if self._ph:
            self.cv.create_image(0, 0, anchor="nw", image=self._ph)
        if self.box:
            h, w = self.img.shape[:2]
            x0, y0, x1, y1 = (self.box[0] * w * self.s, self.box[1] * h * self.s,
                              self.box[2] * w * self.s, self.box[3] * h * self.s)
            self.cv.create_rectangle(x0, y0, x1, y1, outline="#ffd60a", width=4)

    def _down(self, e):
        self.p0 = (e.x, e.y)

    def _move(self, e):
        if self.p0:
            self._set(self.p0, (e.x, e.y))
            self._draw()

    def _up(self, e):
        if self.p0:
            self._set(self.p0, (e.x, e.y))
            self.p0 = None
            self._draw()

    def _set(self, a, b):
        h, w = self.img.shape[:2]
        fx = lambda v: min(1.0, max(0.0, v / (w * self.s)))
        fy = lambda v: min(1.0, max(0.0, v / (h * self.s)))
        x0, x1 = sorted((fx(a[0]), fx(b[0])))
        y0, y1 = sorted((fy(a[1]), fy(b[1])))
        self.box = [x0, y0, x1, y1] if (x1 - x0) * w > 20 and (y1 - y0) * h > 20 else None

    def _clear(self):
        self.box = None
        self._draw()

    def _done(self, run):
        box, keep = self.box, self.keep.get()
        self.destroy()
        self.on_done(box, keep, run)


class SharpApp(Base):
    def __init__(self):
        self.use_sharp = None
        self._crop_keep = None
        self._crop_src_shape = None
        super().__init__()
        self.title(mark_ui.APP + f"  (+ ตัวตรวจคมชัด {SHARP_VER})")
        cfg = SC.load_config()
        self.use_sharp = tk.BooleanVar(value=True)
        self.cont_blur = tk.BooleanVar(value=True)          # v1.5: ตรวจต่อเสมอ (ไม่มีถ่ายใหม่)
        self.match_pct = tk.IntVar(value=int(round(100 * float(cfg.get("match_min", 0.5)))))
        self._crop_keep = cfg.get("test_crop")
        try:
            self.show_ok.set(True)                          # v1.5: กรอบ OK อัตโนมัติ
            self.p["check_extra"].set(False)                # ไม่วาดกรอบม่วง
        except Exception:
            pass
        bar = tk.Frame(self, bg=mark_ui.CARD)
        ttk.Checkbutton(bar, text="ใช้ตัวตรวจคมชัด", variable=self.use_sharp).pack(side="left", padx=(10, 6))
        tk.Label(bar, text="match ขั้นต่ำ %", bg=mark_ui.CARD, font=mark_ui.F(9, "bold")).pack(side="left", padx=(8, 2))
        ttk.Spinbox(bar, from_=0, to=100, increment=5, textvariable=self.match_pct, width=5,
                    command=self._save_match).pack(side="left")
        ttk.Button(bar, text="✂ ครอบรูป Test", command=self.open_crop).pack(side="left", padx=6)
        self.crop_lbl = tk.Label(bar, text="", bg=mark_ui.CARD, fg="#1d5a93", font=mark_ui.F(9))
        self.crop_lbl.pack(side="left", padx=2)
        ttk.Button(bar, text="ตั้งค่าคมชัด", command=self.open_cfg).pack(side="left", padx=6)
        nbs = [w for w in self.winfo_children() if isinstance(w, ttk.Notebook)]
        if nbs:
            bar.pack(fill="x", padx=6, before=nbs[0])
        else:
            bar.pack(fill="x", padx=6)
        self._crop_label()

    # ---------------------------------------------------------- ค่า match / ครอบรูป
    def _save_match(self):
        try:
            v = max(0, min(100, int(self.match_pct.get())))
        except (tk.TclError, ValueError):
            return
        cfg = SC.load_config()
        cfg["match_min"] = v / 100.0
        SC.save_config(cfg)

    def _crop_label(self):
        self.crop_lbl.config(text="(จำกรอบครอบรูปอยู่)" if self._crop_keep else "")

    def open_crop(self):
        from tkinter import messagebox
        img = self.__dict__.get("live_frame") if getattr(self, "live", False) else None
        if img is None:
            img = self.test_img
        if img is None:
            messagebox.showwarning(mark_ui.APP, "ยังไม่มีภาพ Test — เปิดกล้อง หรือเลือกไฟล์ก่อน")
            return
        img = img.copy()

        def done(box, keep, run):
            cfg = SC.load_config()
            self._crop_keep = box if (keep and box) else None
            self._crop_src_shape = img.shape[:2] if self._crop_keep else None
            cfg["test_crop"] = self._crop_keep
            SC.save_config(cfg)
            self._crop_label()
            if not box:
                return
            out = apply_crop(img, box)
            self.test_img = out
            self.test_path.set("<ครอบรูปแล้ว>")
            pn = self.__dict__.get("pn_test")
            if pn is not None and not getattr(self, "live", False):
                pn.set(out)
            self.status.set(f"ครอบรูปแล้ว {out.shape[1]}x{out.shape[0]}"
                            + (" — จำกรอบไว้ใช้กับภาพกล้องถัดไป" if self._crop_keep else ""))
            if run:
                self.run_inspect(out)
        CropDialog(self, img, done, keep=bool(self._crop_keep))

    def _auto_crop(self, test):
        """จำกรอบไว้ -> ครอบเฉพาะภาพเต็มจากกล้อง (ภาพที่ครอบแล้ว ขนาดไม่เท่า -> ไม่ครอบซ้ำ)"""
        if not self._crop_keep or test is None:
            return test
        src = self._crop_src_shape
        if src is None:
            lf = self.__dict__.get("live_frame")
            src = lf.shape[:2] if lf is not None else None
        if src is not None and tuple(test.shape[:2]) != tuple(src):
            return test
        return apply_crop(test, self._crop_keep)

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

    # ---------------------------------------------------------- ล้างผลเก่าเมื่อเปลี่ยน Test
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
        if self._crop_keep:
            self._crop_src_shape = img.shape[:2]
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
        crop_note = ""
        t2 = self._auto_crop(test)
        if t2 is not test:
            crop_note = f"ครอบรูปตามกรอบที่จำไว้ {t2.shape[1]}x{t2.shape[0]}"
            test = t2
        self._save_match()
        cfg = SC.load_config()
        master = self.master_img.copy()
        ign = list(getattr(self, "ignores", []) or [])
        model = self.__dict__.get("model")
        mname = mark_ui_plus_safe(model.get() if model is not None else "")
        dbg = os.path.join(BASE_DIR, "results", mname, time.strftime("%Y%m%d"),
                           time.strftime("%H%M%S") + "_debug") if cfg["debug_save"] else None
        self.busy = True
        self._big("…", IDLE_GREY)
        self.status.set("ตัวตรวจคมชัด กำลังตรวจ …")
        self._raw_test = test.copy()
        self._raw_master = master
        self._pol_msg = ""
        self._crop_note = crop_note

        def work():
            try:
                r = SC.inspect(master, test, cfg, ign, dbg, True)
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
        for n in (getattr(self, "_crop_note", ""), getattr(self, "_grab_note", "")):
            if n:
                notes.insert(0, n)
        self._grab_note = self._crop_note = ""
        if r.get("debug_dir"):
            notes.append("debug: " + os.path.relpath(r["debug_dir"], BASE_DIR))
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

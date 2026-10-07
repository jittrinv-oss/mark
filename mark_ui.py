#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mark_ui.py — ระบบตรวจมาร์คกระจก (AATH)   [v7 — จอ Argon Industria 10" / 1180x800]

v7 (แก้ "ตัวอักษรปกติขึ้นสีส้ม/NG ผิด" กับภาพถ่ายจริง):
  - หมึกดำโทนน้ำตาลในภาพถ่ายไม่ถูกลบเป็น "รอยปากกา" อีกต่อไป
  - ตัดหมึกด้วยการปรับพื้นหลังให้เรียบ + Otsu (ตัวอักษรเส้นบางไม่แตก)
  - จัดตำแหน่ง 2 รอบ (ORB + ECC) + จัดตำแหน่งรายตัวอักษร
  - นับเฉพาะ "เนื้อเส้นหายทั้งความหนา" ไม่นับเศษขอบจากภาพเลื่อน
  - หน้าตั้งค่ามีปุ่ม "คืนค่าเริ่มต้น"

ไฟล์ที่ต้องอยู่โฟลเดอร์เดียวกัน: mark_match.py, glyph_check.py, pdf_master.py, camera.py
"""
import csv
import json
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import numpy as np
import cv2

from glyph_check import inspect_glyph, draw_result, DEFAULT_P
from mark_match import auto_detect_crop
from pdf_master import load_pdf_mark
from camera import list_cameras, open_camera

APP = "ระบบตรวจมาร์คกระจก — AATH"
WIN_W, WIN_H = 1180, 800
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")


def _fix_windows_dpi():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


_fix_windows_dpi()

NAVY, BLUE, BLUE_D = "#1f3864", "#2e75b6", "#1d5a93"
GREY_BG, CARD = "#eef1f5", "#ffffff"
GREEN, RED = "#1a9e3c", "#cc2b2b"
DARK = "#1e242c"
FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"


def F(size, weight="normal"):
    return (FONT, size, weight)


def imread_u(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def to_photo(img, max_w, max_h):
    if img is None or max_w < 10 or max_h < 10:
        return None, 1.0
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 2.0)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".ppm", img)
    return (tk.PhotoImage(data=buf.tobytes()) if ok else None), s


class ImagePanel(tk.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, bg=CARD, highlightbackground="#d6dde6", highlightthickness=1)
        self.title_var = tk.StringVar(value=title)
        tk.Label(self, textvariable=self.title_var, bg=CARD, fg=NAVY, font=F(9, "bold"),
                 anchor="w").pack(fill="x", padx=6, pady=(3, 0))
        self.cv = tk.Canvas(self, bg=DARK, highlightthickness=0)
        self.cv.pack(fill="both", expand=True, padx=4, pady=4)
        self.img = None
        self._photo = None
        self.cv.bind("<Configure>", lambda e: self.redraw())

    def set(self, img):
        self.img = img
        self.redraw()

    def redraw(self):
        self.cv.delete("all")
        w, h = self.cv.winfo_width(), self.cv.winfo_height()
        if self.img is None:
            self.cv.create_text(w // 2, h // 2, text="—", fill="#6b7785", font=F(14))
            return
        self._photo, _ = to_photo(self.img, w, h)
        if self._photo:
            self.cv.create_image(w // 2, h // 2, anchor="center", image=self._photo)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP)
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.minsize(1000, 680)
        self.configure(bg=GREY_BG)
        self.tk.call("tk", "scaling", 96 / 72.0)
        self._style()

        self.master_path, self.test_path, self.cfg_path = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.master_img = self.master_full = self.test_img = None
        self.ignores = []
        self.report = self.overlay = self.aligned = None
        self._pending_crop = None
        self._crop_box = None
        self.edit_scale = 1.0
        self.drag0 = self.rect_id = None
        self.cap = None
        self.live = False
        self.live_frame = None
        self.cams = []
        self.busy = False
        self._last_auto = 0.0

        self.draw_kind = tk.StringVar(value="ignore")
        self.cam_choice = tk.StringVar()
        self.auto_run = tk.BooleanVar(value=False)
        self.show_ok = tk.BooleanVar(value=False)
        self.filt = tk.StringVar(value="เฉพาะที่ผิดปกติ")
        self.p = {k: (tk.BooleanVar(value=v) if isinstance(v, bool)
                      else tk.DoubleVar(value=v) if isinstance(v, float)
                      else tk.IntVar(value=v)) for k, v in DEFAULT_P.items()}
        self._load_settings()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------------------------------------------------------- settings file
    def _load_settings(self):
        try:
            d = json.load(open(SETTINGS_FILE, encoding="utf-8"))
            if d.get("_version") != 7:      # ค่าจากเวอร์ชันเก่าไม่เข้ากับวิธีตรวจใหม่ -> ไม่ใช้
                return
            for k, v in d.items():
                if k in self.p:
                    self.p[k].set(v)
        except Exception:
            pass

    def _save_settings(self):
        try:
            d = {k: v.get() for k, v in self.p.items()}
            d["_version"] = 7
            json.dump(d, open(SETTINGS_FILE, "w", encoding="utf-8"), indent=2)
        except Exception:
            pass

    def reset_settings(self):
        for k, v in DEFAULT_P.items():
            self.p[k].set(v)
        self._save_settings()
        self.status.set("คืนค่าเริ่มต้นแล้ว")

    # ---------------------------------------------------------------- style
    def _style(self):
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", font=F(9), background=GREY_BG)
        st.configure("TButton", font=F(9, "bold"), padding=(8, 4), background=BLUE,
                     foreground="white", borderwidth=0)
        st.map("TButton", background=[("active", BLUE_D), ("disabled", "#9aa5b1")])
        st.configure("Run.TButton", font=F(13, "bold"), padding=(16, 8), background=GREEN, foreground="white")
        st.map("Run.TButton", background=[("active", "#157a2e")])
        st.configure("TCheckbutton", background=CARD, font=F(9))
        st.configure("TRadiobutton", background=CARD, font=F(9))
        st.configure("TNotebook", background=GREY_BG, borderwidth=0)
        st.configure("TNotebook.Tab", font=F(9, "bold"), padding=(12, 4), background="#dce3ee", foreground=NAVY)
        st.map("TNotebook.Tab", background=[("selected", CARD)])
        st.configure("Treeview", font=F(9), rowheight=22)
        st.configure("Treeview.Heading", font=F(9, "bold"), background=BLUE, foreground="white")

    # ---------------------------------------------------------------- layout
    def _build(self):
        hdr = tk.Frame(self, bg=NAVY, height=30)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="ระบบตรวจมาร์คกระจก — AATH  (v7)", bg=NAVY, fg="white",
                 font=F(11, "bold")).pack(side="left", padx=10)

        fr = tk.Frame(self, bg=CARD); fr.pack(fill="x", padx=6, pady=(4, 2))

        def file_box(label, var, cmd, btn, w):
            tk.Label(fr, text=label, bg=CARD, font=F(9, "bold")).pack(side="left", padx=(6, 2))
            ttk.Entry(fr, textvariable=var, width=w, font=F(8)).pack(side="left")
            ttk.Button(fr, text=btn, command=cmd).pack(side="left", padx=(2, 8))
        file_box("Master", self.master_path, self.pick_master, "PDF/ภาพ…", 24)
        file_box("Test", self.test_path, self.pick_test, "เลือก…", 22)
        file_box("Config", self.cfg_path, self.load_cfg, "โหลด", 14)
        ttk.Button(fr, text="บันทึก", command=self.save_cfg).pack(side="left")

        ar = tk.Frame(self, bg=CARD); ar.pack(fill="x", padx=6, pady=2)
        ttk.Button(ar, text="▶ ตรวจสอบ", style="Run.TButton", command=self.run_inspect).pack(side="left", padx=(4, 6), pady=3)
        self.verdict = tk.Label(ar, text="--", font=F(18, "bold"), fg="white", bg="#9aa5b1", width=5)
        self.verdict.pack(side="left", padx=(0, 14), ipady=2)
        tk.Label(ar, text="กล้อง", bg=CARD, font=F(9, "bold")).pack(side="left")
        self.cam_box = ttk.Combobox(ar, textvariable=self.cam_choice, width=34, state="readonly", font=F(8))
        self.cam_box.pack(side="left", padx=3)
        ttk.Button(ar, text="ค้นหากล้อง", command=self.scan_cameras).pack(side="left", padx=2)
        self.btn_cam = ttk.Button(ar, text="เปิดกล้อง", command=self.toggle_camera)
        self.btn_cam.pack(side="left", padx=2)
        ttk.Button(ar, text="📷 จับภาพ+ตรวจ", command=self.cam_grab).pack(side="left", padx=2)
        ttk.Checkbutton(ar, text="ตรวจอัตโนมัติ", variable=self.auto_run).pack(side="left", padx=4)
        ttk.Checkbutton(ar, text="แสดงกรอบ OK", variable=self.show_ok, command=self.refresh_result).pack(side="left", padx=4)

        nb = ttk.Notebook(self); nb.pack(fill="both", expand=True, padx=6, pady=(2, 2))

        t1 = tk.Frame(nb, bg=GREY_BG); nb.add(t1, text="ตรวจ")
        row = tk.Frame(t1, bg=GREY_BG); row.pack(fill="both", expand=True)
        for i in range(3):
            row.columnconfigure(i, weight=1, uniform="p")
        row.rowconfigure(0, weight=1)
        self.pn_master = ImagePanel(row, "Master (product spec.)")
        self.pn_test = ImagePanel(row, "Test (ชิ้นงาน)")
        self.pn_result = ImagePanel(row, "ผลตรวจ")
        for i, pn in enumerate((self.pn_master, self.pn_test, self.pn_result)):
            pn.grid(row=0, column=i, sticky="nsew", padx=2, pady=2)
        self.summary = tk.Label(t1, text="🟩 OK  🟧 จางขาดหาย  🟥 หายทั้งชิ้น  🟪 หมึกเกิน  ⬜ ข้าม",
                                bg=GREY_BG, font=F(9), anchor="w")
        self.summary.pack(fill="x", padx=4)

        t2 = tk.Frame(nb, bg=CARD); nb.add(t2, text="กำหนดกรอบ Master")
        b2 = tk.Frame(t2, bg=CARD); b2.pack(fill="x", padx=6, pady=4)
        ttk.Radiobutton(b2, text="โซนไม่ต้องตรวจ (IGNORE)", value="ignore", variable=self.draw_kind).pack(side="left")
        ttk.Radiobutton(b2, text="ลากกรอบครอบตัด Master", value="crop", variable=self.draw_kind).pack(side="left", padx=8)
        ttk.Button(b2, text="ใช้กรอบครอบตัด", command=self.apply_crop).pack(side="left", padx=2)
        ttk.Button(b2, text="Auto-zoom จาก Test", command=self.auto_crop_from_test).pack(side="left", padx=2)
        ttk.Button(b2, text="คืนค่า Master", command=self.revert_crop).pack(side="left", padx=2)
        ttk.Button(b2, text="ลบกรอบล่าสุด", command=self.undo_box).pack(side="left", padx=(12, 2))
        ttk.Button(b2, text="ลบทั้งหมด", command=self.clear_box).pack(side="left", padx=2)
        self.cv_edit = tk.Canvas(t2, bg=DARK, cursor="cross", highlightthickness=0)
        self.cv_edit.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.cv_edit.bind("<ButtonPress-1>", self.on_down)
        self.cv_edit.bind("<B1-Motion>", self.on_move)
        self.cv_edit.bind("<ButtonRelease-1>", self.on_up)
        self.cv_edit.bind("<Configure>", lambda e: self.draw_edit())

        t3 = tk.Frame(nb, bg=CARD); nb.add(t3, text="ตารางรายชิ้น")
        f3 = tk.Frame(t3, bg=CARD); f3.pack(fill="x", padx=6, pady=4)
        cb = ttk.Combobox(f3, textvariable=self.filt, width=16, state="readonly", values=["ทั้งหมด", "เฉพาะที่ผิดปกติ"])
        cb.pack(side="left"); cb.bind("<<ComboboxSelected>>", lambda e: self.fill_table())
        ttk.Button(f3, text="บันทึกผล (ภาพ+json+csv)", command=self.save_result).pack(side="right")
        cols = ("id", "kind", "status", "หาย", "แนวตั้ง", "แนวนอน", "pos", "note")
        self.tree = ttk.Treeview(t3, columns=cols, show="headings", height=12)
        for c, w in zip(cols, (40, 55, 80, 55, 60, 60, 130, 520)):
            self.tree.heading(c, text=c); self.tree.column(c, width=w)
        self.tree.pack(fill="both", expand=True, padx=6)
        self.tree.tag_configure("MISSING", background="#ffd6d6")
        self.tree.tag_configure("PARTIAL", background="#ffeccc")
        self.tree.tag_configure("SKIP", background="#eeeeee", foreground="#888")

        t4 = tk.Frame(nb, bg=CARD); nb.add(t4, text="ตั้งค่า")
        items = [
            ("ส่วนที่หาย กว้าง และ สูง อย่างน้อย (สัดส่วนของตัวอักษร)", "min_extent", 0.05, 0.9, 0.05),
            ("เนื้อหมึกหายอย่างน้อย (สัดส่วน)", "min_loss", 0.02, 0.6, 0.01),
            ("Tolerance ขั้นต่ำ px  (แนะนำ 2)", "tol_px", 0, 10, 1),
            ("Tolerance ตามความหนาเส้น (สัดส่วน)", "stroke_tol", 0.0, 1.0, 0.05),
            ("ค้นหาตำแหน่งรายตัวอักษร ± (สัดส่วนของขนาด)", "local_search", 0.0, 0.6, 0.05),
            ("ชิ้นเล็กสุด px²", "min_area", 10, 500, 10),
            ("รวมชิ้นติดกัน px", "merge_gap", 0, 20, 1),
            ("แถบขอบหาเส้นประ (สัดส่วน)", "dash_edge_ratio", 0.03, 0.25, 0.01),
            ("จำนวนขีดเส้นประขั้นต่ำ", "dash_min_count", 3, 15, 1),
        ]
        g = tk.Frame(t4, bg=CARD); g.pack(anchor="nw", padx=16, pady=10)
        for r, (lbl, key, lo, hi, inc) in enumerate(items):
            tk.Label(g, text=lbl, bg=CARD, font=F(10)).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Spinbox(g, from_=lo, to=hi, increment=inc, textvariable=self.p[key], width=7,
                        font=F(10)).grid(row=r, column=1, padx=10)
        r = len(items)
        for key, lbl in (("check_dash", "ไม่ตรวจเส้นประ/สเกล"), ("check_blur", "ข้ามบริเวณภาพเบลอ"),
                         ("check_extra", "แสดงหมึกเกิน (สีม่วง — ไม่มีผลต่อ OK/NG)")):
            ttk.Checkbutton(g, text=lbl, variable=self.p[key]).grid(row=r, column=0, sticky="w", pady=3)
            r += 1
        ttk.Button(g, text="คืนค่าเริ่มต้น", command=self.reset_settings).grid(row=r, column=0, sticky="w", pady=10)

        self.status = tk.StringVar(value="เริ่มจาก: Master → เลือกไฟล์ PDF (CAD FOR ORDER SCREEN) หรือภาพ")
        tk.Label(self, textvariable=self.status, bg="#dde3ea", fg="#333", font=F(8), anchor="w").pack(fill="x", side="bottom")

    # ---------------------------------------------------------------- master / test
    def _read_master(self, p):
        if p.lower().endswith(".pdf"):
            img, info = load_pdf_mark(p)
            self.p["check_dash"].set(False)
            return img, f"PDF → Auto-zoom MARK PATTERN สำเร็จ ({info['method']}, พบกรอบประ {info['n_boxes']} กรอบ)"
        img = imread_u(p)
        if img is None:
            raise RuntimeError("อ่านภาพไม่ได้")
        return img, f"Master ภาพ {img.shape[1]}x{img.shape[0]}"

    def pick_master(self):
        p = filedialog.askopenfilename(title="เลือก Master (PDF หรือภาพ)",
                                       filetypes=[("PDF / Image", "*.pdf *.png *.jpg *.jpeg *.bmp")])
        if not p:
            return
        self.master_path.set(p)
        self.status.set("กำลังโหลด Master …"); self.update_idletasks()
        try:
            img, msg = self._read_master(p)
        except Exception as ex:
            messagebox.showerror(APP, f"โหลด Master ไม่สำเร็จ:\n{ex}\n\nใช้แท็บ 'กำหนดกรอบ Master' ลากกรอบครอบตัดเองแทน")
            self.status.set("โหลด Master ไม่สำเร็จ"); return
        self._set_master(img, keep_full=True)
        self.status.set(msg)

    def _set_master(self, img, keep_full=False):
        self.master_img = img
        if keep_full:
            self.master_full = img.copy()
            self._crop_box = None
        self.ignores = []
        self._pending_crop = None
        self.pn_master.set(img)
        self.draw_edit()

    def pick_test(self):
        p = filedialog.askopenfilename(filetypes=[("Image", "*.png *.jpg *.jpeg *.bmp")])
        if p:
            if self.live:
                self.toggle_camera()
            self.test_path.set(p)
            self.test_img = imread_u(p)
            self.pn_test.title_var.set("Test (ชิ้นงาน)")
            self.pn_test.set(self.test_img)
            self.status.set("โหลดภาพ Test แล้ว — กด ▶ ตรวจสอบ")

    def load_cfg(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not p:
            return
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception as ex:
            messagebox.showerror(APP, f"อ่านไฟล์ไม่ได้:\n{ex}"); return
        self.cfg_path.set(p)
        src = d.get("master")
        if src and os.path.exists(src) and src != self.master_path.get():
            if messagebox.askyesno(APP, f"Config นี้ผูกกับ Master:\n{src}\n\nโหลด Master นี้ด้วยหรือไม่?"):
                try:
                    img, _ = self._read_master(src)
                    self.master_path.set(src)
                    self._set_master(img, keep_full=True)
                except Exception as ex:
                    messagebox.showerror(APP, f"โหลด Master ไม่สำเร็จ:\n{ex}")
        crop = d.get("crop")
        if crop and self.master_full is not None:
            x, y, w, h = crop
            self._set_master(self.master_full[y:y + h, x:x + w].copy())
            self._crop_box = list(crop)
        self.ignores = d.get("ignore", [])
        if d.get("params_version") == 7:
            for k, v in d.get("params", {}).items():
                if k in self.p:
                    self.p[k].set(v)
        self.draw_edit()
        self.status.set(f"โหลด config แล้ว — IGNORE {len(self.ignores)} กรอบ")

    def save_cfg(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", initialfile="config.json", filetypes=[("JSON", "*.json")])
        if not p:
            return
        d = dict(master=self.master_path.get(), ignore=self.ignores, crop=self._crop_box,
                 params=self._params(), params_version=7)
        json.dump(d, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        self.cfg_path.set(p)
        self.status.set("บันทึก config แล้ว")

    # ---------------------------------------------------------------- edit tab
    def draw_edit(self):
        cv = self.cv_edit
        cv.delete("all")
        if self.master_img is None:
            return
        self._edit_photo, s = to_photo(self.master_img, cv.winfo_width(), cv.winfo_height())
        self.edit_scale = s
        cv.create_image(0, 0, anchor="nw", image=self._edit_photo)
        for z in self.ignores:
            cv.create_rectangle(z["x"] * s, z["y"] * s, (z["x"] + z["w"]) * s, (z["y"] + z["h"]) * s,
                                outline="#b0b6bd", width=2, dash=(4, 3))
            cv.create_text(z["x"] * s + 3, z["y"] * s - 8, anchor="w", text=z["name"], fill="#b0b6bd", font=F(8, "bold"))
        if self._pending_crop:
            x, y, bw, bh = self._pending_crop
            cv.create_rectangle(x * s, y * s, (x + bw) * s, (y + bh) * s, outline="#ff2e2e", width=3)

    def on_down(self, e):
        if self.master_img is None:
            return
        self.drag0 = (e.x, e.y)
        self.rect_id = self.cv_edit.create_rectangle(e.x, e.y, e.x, e.y, outline="#4da6ff", width=2)

    def on_move(self, e):
        if self.rect_id:
            self.cv_edit.coords(self.rect_id, self.drag0[0], self.drag0[1], e.x, e.y)

    def on_up(self, e):
        if not self.rect_id:
            return
        self.cv_edit.delete(self.rect_id)
        self.rect_id = None
        x0, y0 = min(self.drag0[0], e.x), min(self.drag0[1], e.y)
        w, h = abs(e.x - self.drag0[0]), abs(e.y - self.drag0[1])
        if w < 5 or h < 5:
            return
        s = self.edit_scale
        box = (int(x0 / s), int(y0 / s), int(w / s), int(h / s))
        if self.draw_kind.get() == "crop":
            self._pending_crop = box
            self.status.set("กด 'ใช้กรอบครอบตัด' เพื่อยืนยัน")
        else:
            name = simpledialog.askstring(APP, "ชื่อโซน IGNORE", initialvalue="DOT_1248", parent=self)
            if not name:
                return
            self.ignores.append(dict(name=name, x=box[0], y=box[1], w=box[2], h=box[3]))
        self.draw_edit()

    def apply_crop(self):
        if not self._pending_crop or self.master_img is None:
            messagebox.showwarning(APP, "เลือก 'ลากกรอบครอบตัด Master' แล้วลากกรอบก่อน"); return
        x, y, w, h = self._pending_crop
        bx, by = (self._crop_box[0], self._crop_box[1]) if self._crop_box else (0, 0)
        new_box = [bx + x, by + y, w, h]
        self._set_master(self.master_img[y:y + h, x:x + w].copy())
        self._crop_box = new_box
        self.status.set("ครอบตัด Master แล้ว")

    def auto_crop_from_test(self):
        if self.master_full is None or self.test_img is None:
            messagebox.showwarning(APP, "ต้องมีทั้ง Master และ Test ก่อน"); return
        box, n = auto_detect_crop(self.master_full, self.test_img)
        if box is None:
            messagebox.showinfo(APP, f"หาไม่เจอ (จุดตรงกัน {n}) — ลากกรอบครอบตัดเองแทน"); return
        x, y, w, h = box
        self._set_master(self.master_full[y:y + h, x:x + w].copy())
        self._crop_box = list(box)
        self.status.set(f"Auto-zoom จาก Test สำเร็จ (จุดอ้างอิง {n})")

    def revert_crop(self):
        if self.master_full is not None:
            self._set_master(self.master_full.copy(), keep_full=True)

    def undo_box(self):
        if self.draw_kind.get() == "crop":
            self._pending_crop = None
        elif self.ignores:
            self.ignores.pop()
        self.draw_edit()

    def clear_box(self):
        self.ignores = []
        self._pending_crop = None
        self.draw_edit()

    # ---------------------------------------------------------------- inspect
    def _params(self):
        return {k: v.get() for k, v in self.p.items()}

    def run_inspect(self, frame=None):
        if self.busy:
            return
        if self.master_img is None:
            messagebox.showwarning(APP, "ยังไม่ได้เลือก Master"); return
        test = frame if frame is not None else self.test_img
        if test is None:
            messagebox.showwarning(APP, "ยังไม่มีภาพ Test (เลือกไฟล์ หรือเปิดกล้องแล้วกดจับภาพ)"); return
        try:
            params = self._params()
        except (tk.TclError, ValueError):
            messagebox.showerror(APP, "ค่าในหน้าตั้งค่าไม่ถูกต้อง"); return
        self._save_settings()
        self.busy = True
        self.verdict.config(text="…", bg="#9aa5b1")
        master, ign = self.master_img.copy(), list(self.ignores)

        def work():
            try:
                res = inspect_glyph(master, test, ign, params)
                self.after(0, lambda r=res: self._on_result(r))
            except Exception as ex:
                self.after(0, lambda e=ex: self._on_error(e))
        threading.Thread(target=work, daemon=True).start()

    def _on_error(self, ex):
        self.busy = False
        self.verdict.config(text="ERR", bg=RED)
        messagebox.showerror(APP, f"ตรวจไม่สำเร็จ:\n{ex}")

    def _on_result(self, res):
        self.busy = False
        rep, ov, aligned, *_ = res
        self.report, self.overlay, self.aligned = rep, ov, aligned
        self.verdict.config(text=rep["verdict"], bg=GREEN if rep["verdict"] == "OK" else RED)
        self.refresh_result()
        warn = "  ⚠ align ไม่สำเร็จ ผลไม่น่าเชื่อถือ" if "RESIZE" in rep["align_method"] else ""
        self.summary.config(text=(f"ผล {rep['verdict']}  |  ชิ้นส่วน {rep['n_elements']}  หายทั้งชิ้น {rep['n_missing']}  "
                                  f"จางขาดหาย {rep['n_partial']}  ข้าม {rep['n_skip']}  |  align {rep['align_method']}{warn}"))
        self.fill_table()
        self.status.set(f"ตรวจเสร็จ {time.strftime('%H:%M:%S')}")

    def refresh_result(self):
        if self.report is None or self.aligned is None:
            return
        extras = self.report["extras"] if self.p["check_extra"].get() else []
        self.pn_result.set(draw_result(self.aligned, self.report["elements"], extras, self.ignores, self.show_ok.get()))

    def fill_table(self):
        rep = self.report
        self.tree.delete(*self.tree.get_children())
        if rep is None:
            return
        bad_only = self.filt.get() == "เฉพาะที่ผิดปกติ"
        f = lambda v: "-" if v is None else f"{v:.0%}"
        for r in rep["elements"]:
            if bad_only and r["status"] in ("OK", "SKIP"):
                continue
            loss = None if r["cov"] is None else 1 - r["cov"]
            self.tree.insert("", "end", tags=(r["status"],), values=(
                r["id"], r["kind"], r["status"], f(loss), f(r["vfrac"]), f(r["hfrac"]),
                f'({r["x"]},{r["y"]}) {r["w"]}x{r["h"]}', r["note"]))

    def save_result(self):
        if self.report is None:
            messagebox.showwarning(APP, "ยังไม่มีผลตรวจ"); return
        d = filedialog.askdirectory()
        if not d:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imencode(".png", self.overlay)[1].tofile(os.path.join(d, f"overlay_{ts}.png"))
        json.dump({k: v for k, v in self.report.items() if k != "params"},
                  open(os.path.join(d, f"report_{ts}.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        with open(os.path.join(d, f"elements_{ts}.csv"), "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self.report["elements"][0].keys()))
            w.writeheader(); w.writerows(self.report["elements"])
        self.status.set(f"บันทึกผลที่ {d}")

    # ---------------------------------------------------------------- camera
    def scan_cameras(self):
        if self.live:
            self.toggle_camera()
        self.status.set("กำลังค้นหากล้อง … (ทดลองอ่านภาพจากทุกตัว)"); self.update_idletasks()
        self.cams = list_cameras()
        labels = [c["label"] for c in self.cams]
        self.cam_box["values"] = labels
        if not labels:
            self.cam_choice.set("")
            messagebox.showwarning(APP, "ไม่พบกล้องที่อ่านภาพได้\n- เช็คสาย USB / ลองพอร์ตอื่น\n"
                                        "- ปิดโปรแกรมอื่นที่ใช้กล้องอยู่ (Teams, Camera)")
            self.status.set("ไม่พบกล้อง"); return
        self.cam_choice.set(labels[-1])
        self.status.set(f"พบกล้อง {len(labels)} ตัว — เลือกในรายการแล้วกด 'เปิดกล้อง'")

    def toggle_camera(self):
        if self.live:
            self.live = False
            if self.cap:
                self.cap.release()
            self.cap = None
            self.btn_cam.config(text="เปิดกล้อง")
            self.status.set("ปิดกล้องแล้ว"); return
        if not self.cams:
            self.scan_cameras()
            if not self.cams:
                return
        cam = next((c for c in self.cams if c["label"] == self.cam_choice.get()), self.cams[-1])
        self.cap, size = open_camera(cam["index"], cam["backend_id"])
        if self.cap is None:
            messagebox.showerror(APP, f"เปิด {cam['label']} ไม่ได้ — กด 'ค้นหากล้อง' ใหม่"); return
        self.live = True
        self.btn_cam.config(text="ปิดกล้อง")
        self.pn_test.title_var.set(f"กล้องสด — {cam['label']}")
        self.status.set(f"เปิดกล้องแล้ว {size[0]}x{size[1]}")
        self._loop()

    def _loop(self):
        if not self.live or self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self.live_frame = frame
            self.pn_test.set(frame)
            if self.auto_run.get() and not self.busy and time.time() - self._last_auto > 1.5:
                self._last_auto = time.time()
                self.test_img = frame.copy()
                self.run_inspect(self.test_img)
        self.after(60, self._loop)

    def cam_grab(self):
        if self.live_frame is None:
            messagebox.showwarning(APP, "ยังไม่ได้เปิดกล้อง"); return
        self.test_img = self.live_frame.copy()
        self.test_path.set("<จากกล้อง>")
        self.run_inspect(self.test_img)

    def on_close(self):
        self.live = False
        if self.cap:
            self.cap.release()
        self._save_settings()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()

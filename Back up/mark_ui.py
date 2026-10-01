#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mark_ui.py - หน้าจอ UI ตรวจมาร์คกระจก (AATH)   [v5 - ปรับหน้าตาให้คมชัดและใหญ่ขึ้น]
รัน:  python mark_ui.py     หรือ ดับเบิลคลิก RUN_UI.bat
ต้องมี mark_match.py และ glyph_check.py อยู่โฟลเดอร์เดียวกัน

สิ่งที่แก้ใน v5 (ตัวหนังสือไม่ชัด / หน้าตาไม่สวย):
  1) เปิด DPI awareness บน Windows ก่อนสร้างหน้าต่าง -> แก้ตัวหนังสือเบลอ/พร่า
     (สาเหตุหลักของปัญหานี้คือ Windows scale จอ แต่ Tkinter ไม่รู้ จึงวาดภาพ
      ความละเอียดต่ำแล้วขยาย ทำให้เบลอ)
  2) ใช้ฟอนต์ "Segoe UI" ชัดเจนทุกจุด ขนาดใหญ่ขึ้นทั้งหมด เพื่อให้อ่านง่ายบน
     จอสัมผัส 10 นิ้วและจอคอมทั่วไป
  3) ใช้ ttk theme 'clam' + กำหนดสีเอง ให้ปุ่ม/แท็บดูทันสมัยขึ้น ไม่ใช้ธีม
     ค่าเริ่มต้นของ Windows ที่ดูหยาบ
  4) เพิ่มระยะห่าง (padding) รอบปุ่มและองค์ประกอบ ให้กดง่ายขึ้นบนจอสัมผัส
  5) ปรับผลลัพธ์ OK/NG ให้เด่นชัด ตัวใหญ่ อ่านจากระยะไกลได้
"""
import os, sys, json, time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, font as tkfont
import numpy as np
import cv2

from glyph_check import inspect_glyph, DEFAULT_P

APP = "ระบบตรวจมาร์คกระจก — AATH"

# ----------------------------------------------------------------------
# 1) แก้ปัญหาตัวหนังสือเบลอบน Windows (DPI awareness)
#    ต้องเรียก "ก่อน" สร้างหน้าต่าง Tk ตัวแรกเท่านั้น
# ----------------------------------------------------------------------
def _fix_windows_dpi():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)   # Per-Monitor DPI aware (คมชัดสุด)
        except Exception:
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


_fix_windows_dpi()

# ---------------------------------------------------------------- สี/ฟอนต์หลัก
NAVY = "#1f3864"
BLUE = "#2e75b6"
BLUE_D = "#1d5a93"
GREY_BG = "#eef1f5"
CARD_BG = "#ffffff"
GREEN = "#1a9e3c"
RED = "#cc2b2b"
ORANGE = "#e08a00"
GREY_TXT = "#55606e"
FONT_FAMILY = "Segoe UI"


def F(size, weight="normal"):
    return (FONT_FAMILY, size, weight)


def cv2_to_photo(img_bgr, max_w, max_h):
    if img_bgr is None:
        return None, 1.0
    h, w = img_bgr.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    if s < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".ppm", img_bgr)
    return (tk.PhotoImage(data=buf.tobytes()) if ok else None), s


def imread_u(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP)
        self.geometry("1440x940")
        self.minsize(1180, 760)
        self.configure(bg=GREY_BG)
        self._set_scaling()
        self._setup_style()

        self.master_path = tk.StringVar()
        self.test_path = tk.StringVar()
        self.cfg_path = tk.StringVar()
        self.master_img = self.test_img = None
        self.rois, self.ignores = [], []
        self.report = self.overlay = None
        self.scale = 1.0
        self.drag0 = self.rect_id = None
        self.cap = None; self.live = False; self.live_frame = None

        self.mode = tk.StringVar(value="AUTO")
        self.draw_kind = tk.StringVar(value="ignore")
        self.min_extent = tk.DoubleVar(value=DEFAULT_P["min_extent"])
        self.row_loss_thr = tk.DoubleVar(value=DEFAULT_P["row_loss_thr"])
        self.miss_full = tk.DoubleVar(value=DEFAULT_P["miss_full"])
        self.tol = tk.IntVar(value=DEFAULT_P["tol_px"])
        self.min_area = tk.IntVar(value=DEFAULT_P["min_area"])
        self.merge_gap = tk.IntVar(value=DEFAULT_P["merge_gap"])
        self.check_extra = tk.BooleanVar(value=DEFAULT_P["check_extra"])
        self.check_blur = tk.BooleanVar(value=DEFAULT_P["check_blur"])
        self.check_dash = tk.BooleanVar(value=DEFAULT_P["check_dash"])
        self.blur_rel_thr = tk.DoubleVar(value=DEFAULT_P["blur_rel_thr"])
        self.dash_edge_ratio = tk.DoubleVar(value=DEFAULT_P["dash_edge_ratio"])
        self.dash_min_count = tk.IntVar(value=DEFAULT_P["dash_min_count"])
        self.cam_idx = tk.IntVar(value=0)
        self.auto_run = tk.BooleanVar(value=False)
        self.show_ok = tk.BooleanVar(value=True)
        self.filt = tk.StringVar(value="เฉพาะที่ผิดปกติ")

        self._build()

    # ---------------------------------------------------------- DPI / style
    def _set_scaling(self):
        """ปรับสเกลของ Tk ให้ตรงกับความละเอียดจอจริง เพื่อไม่ให้ font เบลอ"""
        try:
            dpi = self.winfo_fpixels("1i")
            self.tk.call("tk", "scaling", dpi / 72.0)
        except Exception:
            pass

    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", font=F(11), background=GREY_BG)
        style.configure("TFrame", background=GREY_BG)
        style.configure("Card.TFrame", background=CARD_BG)
        style.configure("TLabel", background=GREY_BG, font=F(11), foreground="#1b2430")
        style.configure("Card.TLabel", background=CARD_BG, font=F(11), foreground="#1b2430")
        style.configure("Hint.TLabel", background=CARD_BG, font=F(9), foreground=GREY_TXT)
        style.configure("Heading.TLabel", background=CARD_BG, font=F(13, "bold"), foreground=NAVY)

        style.configure("TButton", font=F(11, "bold"), padding=(14, 9),
                        background=BLUE, foreground="white", borderwidth=0)
        style.map("TButton", background=[("active", BLUE_D), ("pressed", BLUE_D)],
                  foreground=[("disabled", "#aaaaaa")])

        style.configure("Big.TButton", font=F(14, "bold"), padding=(22, 14),
                        background=GREEN, foreground="white", borderwidth=0)
        style.map("Big.TButton", background=[("active", "#157a2e")])

        style.configure("TCheckbutton", font=F(10), background=CARD_BG)
        style.configure("TRadiobutton", font=F(10), background=CARD_BG)
        style.configure("TSpinbox", font=F(11), padding=4)
        style.configure("TEntry", font=F(10), padding=4)
        style.configure("TCombobox", font=F(10), padding=4)

        style.configure("TNotebook", background=GREY_BG, borderwidth=0, tabmargins=(4, 6, 4, 0))
        style.configure("TNotebook.Tab", font=F(11, "bold"), padding=(18, 10),
                        background="#dce3ee", foreground=NAVY)
        style.map("TNotebook.Tab", background=[("selected", CARD_BG)],
                  foreground=[("selected", NAVY)])

        style.configure("Treeview", font=F(10), rowheight=30, background="white",
                        fieldbackground="white")
        style.configure("Treeview.Heading", font=F(10, "bold"), background=BLUE, foreground="white")
        style.map("Treeview.Heading", background=[("active", BLUE_D)])

        style.configure("TLabelframe", background=CARD_BG, font=F(11, "bold"), bordercolor="#d6dde6")
        style.configure("TLabelframe.Label", background=CARD_BG, font=F(12, "bold"), foreground=NAVY)

    # ---------------------------------------------------------- layout
    def _build(self):
        header = tk.Frame(self, bg=NAVY, height=56)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        tk.Label(header, text="🔍  ระบบตรวจมาร์คกระจก — AATH", bg=NAVY, fg="white",
                 font=F(16, "bold")).pack(side="left", padx=20)

        top = tk.Frame(self, bg=CARD_BG, highlightbackground="#d6dde6", highlightthickness=1)
        top.pack(fill="x", padx=10, pady=(10, 4))
        pad = dict(padx=8, pady=7)
        rows = [("Master (product spec.)", self.master_path, self.pick_master),
                ("Test (ชิ้นงาน)", self.test_path, self.pick_test)]
        for i, (lbl, var, cmd) in enumerate(rows):
            tk.Label(top, text=lbl, bg=CARD_BG, font=F(11), width=22, anchor="w").grid(
                row=i, column=0, sticky="w", **pad)
            ttk.Entry(top, textvariable=var, width=74, font=F(10)).grid(row=i, column=1, **pad)
            ttk.Button(top, text="เลือก...", command=cmd).grid(row=i, column=2, **pad)
        tk.Label(top, text="Config (roi/ignore)", bg=CARD_BG, font=F(11), width=22, anchor="w").grid(
            row=2, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.cfg_path, width=74, font=F(10)).grid(row=2, column=1, **pad)
        ttk.Button(top, text="โหลด...", command=self.load_cfg).grid(row=2, column=2, **pad)
        ttk.Button(top, text="บันทึก", command=self.save_cfg).grid(row=2, column=3, padx=(0, 8))

        pb = ttk.LabelFrame(self, text="  โหมดและเกณฑ์ตัดสิน  ", padding=(14, 10))
        pb.pack(fill="x", padx=10, pady=4)

        row0 = tk.Frame(pb, bg=CARD_BG); row0.pack(fill="x", pady=(0, 8))
        tk.Label(row0, text="โหมด:", bg=CARD_BG, font=F(11, "bold")).pack(side="left", padx=(0, 10))
        ttk.Radiobutton(row0, text="AUTO (แยกชิ้นส่วนเอง)", value="AUTO",
                        variable=self.mode).pack(side="left", padx=4)
        ttk.Radiobutton(row0, text="ROI (กรอบกำหนดเอง)", value="ROI",
                        variable=self.mode).pack(side="left", padx=(4, 30))

        self.verdict_lbl = tk.Label(row0, text="--", font=F(26, "bold"), fg="white",
                                    bg="#9aa5b1", width=6, relief="flat")
        self.verdict_lbl.pack(side="right", padx=(10, 0))
        ttk.Button(row0, text="▶  ตรวจสอบ (Run)", style="Big.TButton",
                   command=self.run_inspect).pack(side="right", padx=10)

        row1 = tk.Frame(pb, bg=CARD_BG); row1.pack(fill="x", pady=4)
        def spin(parent, var, lo, hi, inc, w=6):
            return ttk.Spinbox(parent, from_=lo, to=hi, increment=inc, textvariable=var,
                               width=w, font=F(11))

        tk.Label(row1, text="หายอย่างน้อยแนวตั้ง+แนวนอน ≥", bg=CARD_BG, font=F(10)).pack(side="left")
        spin(row1, self.min_extent, 0.05, 0.90, 0.05).pack(side="left", padx=(6, 18))
        tk.Label(row1, text="เกณฑ์แถว/คอลัมน์เสียหาย", bg=CARD_BG, font=F(10)).pack(side="left")
        spin(row1, self.row_loss_thr, 0.3, 0.95, 0.05).pack(side="left", padx=(6, 18))
        tk.Label(row1, text="Tolerance px", bg=CARD_BG, font=F(10)).pack(side="left")
        spin(row1, self.tol, 0, 10, 1, 5).pack(side="left", padx=(6, 18))
        tk.Label(row1, text="ชิ้นเล็กสุด px²", bg=CARD_BG, font=F(10)).pack(side="left")
        spin(row1, self.min_area, 10, 500, 10, 6).pack(side="left", padx=(6, 18))
        tk.Label(row1, text="รวมชิ้นติดกัน px", bg=CARD_BG, font=F(10)).pack(side="left")
        spin(row1, self.merge_gap, 0, 20, 1, 5).pack(side="left", padx=6)

        row2 = tk.Frame(pb, bg=CARD_BG); row2.pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(row2, text="ไม่ตรวจเส้นประ/สเกล", variable=self.check_dash).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(row2, text="ข้ามบริเวณภาพเบลอ", variable=self.check_blur).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(row2, text="ตรวจหมึกเกิน", variable=self.check_extra).pack(side="left", padx=(0, 24))
        tk.Label(row2, text="แถบขอบหาเส้นประ %", bg=CARD_BG, font=F(10)).pack(side="left")
        spin(row2, self.dash_edge_ratio, 0.03, 0.25, 0.01).pack(side="left", padx=(6, 18))
        tk.Label(row2, text="จำนวนขีดขั้นต่ำ", bg=CARD_BG, font=F(10)).pack(side="left")
        spin(row2, self.dash_min_count, 3, 15, 1, 5).pack(side="left", padx=6)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=(6, 10))

        t1 = tk.Frame(nb, bg=CARD_BG); nb.add(t1, text="1. กำหนดกรอบบน Master")
        b1 = tk.Frame(t1, bg=CARD_BG); b1.pack(fill="x", pady=8, padx=10)
        tk.Label(b1, text="ลากเมาส์เพื่อสร้างกรอบ →", bg=CARD_BG, font=F(10)).pack(side="left", padx=(0, 10))
        ttk.Radiobutton(b1, text="โซนไม่ต้องตรวจ (IGNORE)", value="ignore",
                        variable=self.draw_kind).pack(side="left")
        ttk.Radiobutton(b1, text="ROI ตรวจเฉพาะจุด", value="roi",
                        variable=self.draw_kind).pack(side="left", padx=12)
        ttk.Button(b1, text="ลบกรอบล่าสุด", command=self.undo_box).pack(side="left", padx=10)
        ttk.Button(b1, text="ลบทั้งหมด", command=self.clear_box).pack(side="left")
        self.cv_master = tk.Canvas(t1, bg="#1e242c", cursor="cross", highlightthickness=0)
        self.cv_master.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.cv_master.bind("<ButtonPress-1>", self.on_down)
        self.cv_master.bind("<B1-Motion>", self.on_move)
        self.cv_master.bind("<ButtonRelease-1>", self.on_up)

        t2 = tk.Frame(nb, bg=CARD_BG); nb.add(t2, text="2. ผลตรวจ")
        b2 = tk.Frame(t2, bg=CARD_BG); b2.pack(fill="x", pady=8, padx=10)
        ttk.Checkbutton(b2, text="แสดงกรอบชิ้นที่ OK ด้วย", variable=self.show_ok,
                        command=self.show_result).pack(side="left", padx=(0, 14))
        tk.Label(b2, text="🟩 OK   🟧 จางขาดหาย   🟥 หายทั้งชิ้น   🟪 หมึกเกิน   ⬜ ข้าม",
                 bg=CARD_BG, font=F(10)).pack(side="left")
        self.cv_result = tk.Canvas(t2, bg="#1e242c", highlightthickness=0)
        self.cv_result.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        t3 = tk.Frame(nb, bg=CARD_BG); nb.add(t3, text="3. ตารางรายชิ้น")
        f3 = tk.Frame(t3, bg=CARD_BG); f3.pack(fill="x", padx=10, pady=8)
        tk.Label(f3, text="กรอง:", bg=CARD_BG, font=F(10)).pack(side="left", padx=(0, 6))
        cb = ttk.Combobox(f3, textvariable=self.filt, width=20, state="readonly",
                          values=["ทั้งหมด", "เฉพาะที่ผิดปกติ"], font=F(10))
        cb.pack(side="left"); cb.bind("<<ComboboxSelected>>", lambda e: self.fill_table())
        ttk.Button(f3, text="บันทึกผล (ภาพ+json+csv)", command=self.save_result).pack(side="right", padx=8)
        cols = ("id", "kind", "status", "cov", "แนวตั้ง", "แนวนอน", "pos", "note")
        widths = (45, 65, 95, 65, 75, 75, 140, 360)
        self.tree = ttk.Treeview(t3, columns=cols, show="headings", height=14)
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c); self.tree.column(c, width=w)
        self.tree.pack(fill="both", expand=True, padx=10)
        self.tree.tag_configure("MISSING", background="#ffd6d6")
        self.tree.tag_configure("PARTIAL", background="#ffeccc")
        self.tree.tag_configure("SKIP", background="#eeeeee", foreground="#888888")
        self.txt = tk.Text(t3, height=8, font=("Consolas", 10), bg="#1e242c", fg="#d6e2f0",
                           insertbackground="white")
        self.txt.pack(fill="x", padx=10, pady=(8, 10))

        t4 = tk.Frame(nb, bg=CARD_BG); nb.add(t4, text="4. กล้องสด")
        b4 = tk.Frame(t4, bg=CARD_BG); b4.pack(fill="x", pady=8, padx=10)
        tk.Label(b4, text="กล้องหมายเลข", bg=CARD_BG, font=F(10)).pack(side="left", padx=(0, 6))
        ttk.Spinbox(b4, from_=0, to=5, textvariable=self.cam_idx, width=4, font=F(11)).pack(side="left")
        ttk.Button(b4, text="เปิดกล้อง", command=self.cam_start).pack(side="left", padx=8)
        ttk.Button(b4, text="ปิดกล้อง", command=self.cam_stop).pack(side="left")
        ttk.Button(b4, text="📷 จับภาพ + ตรวจ", command=self.cam_grab).pack(side="left", padx=14)
        ttk.Checkbutton(b4, text="ตรวจอัตโนมัติ", variable=self.auto_run).pack(side="left", padx=6)
        ttk.Button(b4, text="ใช้เฟรมนี้เป็น Master", command=self.cam_master).pack(side="left", padx=14)
        self.cv_live = tk.Canvas(t4, bg="#1e242c", highlightthickness=0)
        self.cv_live.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.status = tk.StringVar(value="พร้อมใช้งาน — เริ่มจากเลือกภาพ Master")
        status_bar = tk.Frame(self, bg="#dde3ea", height=30)
        status_bar.pack(fill="x", side="bottom")
        tk.Label(status_bar, textvariable=self.status, bg="#dde3ea", fg="#333",
                 font=F(9), anchor="w").pack(fill="x", padx=12, pady=4)

    # ---------------------------------------------------------- file io
    def pick_master(self):
        p = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp")])
        if p:
            self.master_path.set(p); self.master_img = imread_u(p); self.draw_master()
            self.status.set(f"Master {self.master_img.shape[1]}x{self.master_img.shape[0]} px")

    def pick_test(self):
        p = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp")])
        if p:
            self.test_path.set(p); self.test_img = imread_u(p)
            self.status.set("โหลดภาพ Test แล้ว — กด ▶ ตรวจสอบ")

    def load_cfg(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not p:
            return
        try:
            d = json.load(open(p, encoding="utf-8"))
            self.rois = d.get("rois", []); self.ignores = d.get("ignore", [])
            self.cfg_path.set(p); self.draw_master()
            self.status.set(f"โหลด ROI {len(self.rois)} / IGNORE {len(self.ignores)}")
        except Exception as e:
            messagebox.showerror(APP, f"อ่านไฟล์ไม่ได้:\n{e}")

    def save_cfg(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", initialfile="config.json",
                                         filetypes=[("JSON", "*.json")])
        if p:
            json.dump({"master": self.master_path.get(), "rois": self.rois, "ignore": self.ignores},
                      open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            self.cfg_path.set(p); self.status.set("บันทึกแล้ว")

    def save_result(self):
        if self.report is None:
            messagebox.showwarning(APP, "ยังไม่มีผลตรวจ"); return
        d = filedialog.askdirectory()
        if not d:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imencode(".png", self.overlay)[1].tofile(os.path.join(d, f"overlay_{ts}.png"))
        rep_clean = {k: v for k, v in self.report.items() if k != "params"}
        json.dump(rep_clean, open(os.path.join(d, f"report_{ts}.json"), "w",
                                  encoding="utf-8"), indent=2, ensure_ascii=False)
        if self.report.get("elements"):
            import csv
            keys = [k for k in self.report["elements"][0] if k != "mask"]
            with open(os.path.join(d, f"elements_{ts}.csv"), "w", newline="",
                      encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader(); w.writerows(self.report["elements"])
        self.status.set(f"บันทึกผลที่ {d}")

    # ---------------------------------------------------------- boxes
    def draw_master(self):
        if self.master_img is None:
            return
        self.cv_master.update_idletasks()
        photo, s = cv2_to_photo(self.master_img, max(self.cv_master.winfo_width(), 700),
                                max(self.cv_master.winfo_height(), 450))
        self.scale = s; self._pm = photo
        self.cv_master.delete("all")
        self.cv_master.create_image(0, 0, anchor="nw", image=photo)
        for lst, col in ((self.ignores, "#b0b6bd"), (self.rois, "#35c46a")):
            for r in lst:
                x, y = r["x"] * s, r["y"] * s
                self.cv_master.create_rectangle(x, y, x + r["w"] * s, y + r["h"] * s,
                                                outline=col, width=2,
                                                dash=(4, 3) if col == "#b0b6bd" else None)
                self.cv_master.create_text(x + 3, y - 10, anchor="w", text=r["name"],
                                           fill=col, font=F(9, "bold"))

    def on_down(self, e):
        if self.master_img is None:
            return
        self.drag0 = (e.x, e.y)
        self.rect_id = self.cv_master.create_rectangle(e.x, e.y, e.x, e.y,
                                                       outline="#4da6ff", width=2)

    def on_move(self, e):
        if self.rect_id:
            self.cv_master.coords(self.rect_id, self.drag0[0], self.drag0[1], e.x, e.y)

    def on_up(self, e):
        if not self.rect_id:
            return
        self.cv_master.delete(self.rect_id); self.rect_id = None
        x0, y0 = min(self.drag0[0], e.x), min(self.drag0[1], e.y)
        w, h = abs(e.x - self.drag0[0]), abs(e.y - self.drag0[1])
        if w < 5 or h < 5:
            return
        s = self.scale
        kind = self.draw_kind.get()
        default = "DOT_2468" if kind == "ignore" else ""
        name = simpledialog.askstring(APP, f"ชื่อกรอบ ({kind})", initialvalue=default, parent=self)
        if not name:
            return
        box = dict(name=name, x=int(x0 / s), y=int(y0 / s), w=int(w / s), h=int(h / s))
        if kind == "ignore":
            self.ignores.append(box)
        else:
            box["mode"] = "line" if messagebox.askyesno(APP, "ROI นี้เป็น 'เส้นขีด' ใช่หรือไม่?") else "ink"
            if box["mode"] == "line":
                box["line_min_len_ratio"] = 0.60
            else:
                box["min_ratio"] = 0.55; box["min_corr"] = 0.45
            self.rois.append(box)
        self.draw_master()
        self.status.set(f"เพิ่ม {kind} '{name}'  (IGNORE {len(self.ignores)} / ROI {len(self.rois)})")

    def undo_box(self):
        lst = self.ignores if self.draw_kind.get() == "ignore" else self.rois
        if lst:
            n = lst.pop()["name"]; self.draw_master(); self.status.set(f"ลบ '{n}'")

    def clear_box(self):
        if self.draw_kind.get() == "ignore":
            self.ignores = []
        else:
            self.rois = []
        self.draw_master()

    # ---------------------------------------------------------- inspect
    def run_inspect(self, frame=None):
        if self.master_img is None:
            messagebox.showwarning(APP, "ยังไม่ได้เลือกภาพ Master"); return
        test = frame if frame is not None else self.test_img
        if test is None:
            messagebox.showwarning(APP, "ยังไม่ได้เลือกภาพ Test"); return
        try:
            p = dict(min_extent=self.min_extent.get(), row_loss_thr=self.row_loss_thr.get(),
                     miss_full=self.miss_full.get(), tol_px=self.tol.get(),
                     min_area=self.min_area.get(), merge_gap=self.merge_gap.get(),
                     check_extra=self.check_extra.get(), check_blur=self.check_blur.get(),
                     check_dash=self.check_dash.get(), blur_rel_thr=self.blur_rel_thr.get(),
                     dash_edge_ratio=self.dash_edge_ratio.get(),
                     dash_min_count=self.dash_min_count.get())
            rep, ov, *_ = inspect_glyph(self.master_img, test, self.ignores, p)
        except Exception as ex:
            messagebox.showerror(APP, f"ตรวจไม่สำเร็จ:\n{ex}"); return
        self.report, self.overlay = rep, ov
        self.show_result()

    def show_result(self):
        rep = self.report
        if rep is None:
            return
        ok = rep["verdict"] == "OK"
        self.verdict_lbl.config(text=rep["verdict"], bg=GREEN if ok else RED)
        self.cv_result.update_idletasks()
        photo, _ = cv2_to_photo(self.overlay, max(self.cv_result.winfo_width(), 800),
                                max(self.cv_result.winfo_height(), 400))
        self._pr = photo
        self.cv_result.delete("all")
        self.cv_result.create_image(0, 0, anchor="nw", image=photo)
        self.fill_table()

    def fill_table(self):
        rep = self.report
        self.tree.delete(*self.tree.get_children())
        if rep is None:
            return
        only_bad = self.filt.get() == "เฉพาะที่ผิดปกติ"
        for r in rep.get("elements", []):
            if only_bad and r["status"] in ("OK", "SKIP"):
                continue
            f = lambda v: "-" if v is None else f"{v:.0%}"
            self.tree.insert("", "end", tags=(r["status"],), values=(
                r["id"], r["kind"], r["status"], f(r["cov"]),
                f(r["vfrac"]), f(r["hfrac"]),
                f'({r["x"]},{r["y"]}) {r["w"]}x{r["h"]}', r["note"]))

        self.txt.delete("1.0", "end")
        self.txt.insert("end", f'RESULT = {rep["verdict"]}   '
                               f'align: {rep["align_method"]} (inliers={rep["align_inliers"]})\n')
        if "RESIZE" in rep["align_method"]:
            self.txt.insert("end", "!! เตือน: align ไม่สำเร็จ ผลไม่น่าเชื่อถือ "
                                   "ให้ถ่ายภาพใหม่ให้มุม/ระยะใกล้เคียง Master\n")
        self.txt.insert("end",
            f'ชิ้นส่วนทั้งหมด {rep["n_elements"]}   OK={rep["n_ok"]}   '
            f'หายทั้งชิ้น={rep["n_missing"]}   จางขาดหาย={rep["n_partial"]}   '
            f'ข้าม={rep["n_skip"]} (เบลอ={rep.get("n_blur",0)} '
            f'เส้นประ={rep.get("n_dash",0)})   หมึกเกิน={rep["n_extra"]}\n\n')
        for r in rep["elements"]:
            if r["status"] not in ("OK", "SKIP"):
                self.txt.insert("end", f'  [{r["status"]:7s}] #{r["id"]:<3d} '
                                       f'{r["kind"]:5s} ที่ ({r["x"]},{r["y"]}) '
                                       f'{r["w"]}x{r["h"]}  {r["note"]}\n')
        self.status.set(f'ผลตรวจ: {rep["verdict"]}   โหมด {self.mode.get()}')

    # ---------------------------------------------------------- camera
    def cam_start(self):
        if self.live:
            return
        idx = self.cam_idx.get()
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if not self.cap.isOpened():
            messagebox.showerror(APP, f"เปิดกล้อง {idx} ไม่ได้ ลองเปลี่ยนเป็น 1 หรือ 2"); return
        self.live = True; self.status.set(f"กล้อง {idx} ทำงาน"); self._loop()

    def cam_stop(self):
        self.live = False
        if self.cap:
            self.cap.release(); self.cap = None

    def _loop(self):
        if not self.live or self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            self.live_frame = frame
            self.cv_live.update_idletasks()
            photo, _ = cv2_to_photo(frame, max(self.cv_live.winfo_width(), 800),
                                    max(self.cv_live.winfo_height(), 400))
            self._pl = photo
            self.cv_live.delete("all")
            self.cv_live.create_image(0, 0, anchor="nw", image=photo)
            if self.auto_run.get():
                self.run_inspect(frame)
        self.after(33, self._loop)

    def cam_grab(self):
        if self.live_frame is None:
            messagebox.showwarning(APP, "ยังไม่มีภาพจากกล้อง"); return
        self.run_inspect(self.live_frame)

    def cam_master(self):
        if self.live_frame is None:
            messagebox.showwarning(APP, "ยังไม่มีภาพจากกล้อง"); return
        self.master_img = self.live_frame.copy()
        self.master_path.set("<จากกล้อง>"); self.draw_master()
        self.status.set("ตั้งเฟรมนี้เป็น Master แล้ว — ไปแท็บ 1 เพื่อกำหนดโซน IGNORE")

    def destroy(self):
        self.cam_stop(); super().destroy()


if __name__ == "__main__":
    App().mainloop()

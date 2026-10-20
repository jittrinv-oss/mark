#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mark_ui_kiosk.py v4 — เปิดโปรแกรมตรวจมาร์ค (mark_ui_plus) แบบเต็มจอ + จัดภาพก่อนตรวจ (prealign)
ไม่แก้ไขไฟล์เดิม

v2: ด่านก่อนตรวจ (prealign.py) — จัดภาพ Test ให้ตรง Master, จับคู่ไม่ได้ -> "ถ่ายใหม่"
v3: ตัวหนังสือภาษาไทยใหญ่และชัดขึ้น
v4: ใช้ Master ที่ prealign v4 จำกัดเขตแล้ว (ตรวจเฉพาะในกรอบเส้นประแดง / พื้นที่มาร์ค)
    + ภาพกล้องถูกปรับตั้งฉาก -> หัวตาราง / ขอบกระดาษ / เงา ไม่ขึ้น alarm

  F11 = สลับเต็มจอ / Esc = ออกจากเต็มจอ
  ไม่ใช้ด่านก่อนตรวจ:  python mark_ui_kiosk.py --no-prealign
"""
import sys
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont

try:
    from mark_ui_plus import AppPlus as Base
except Exception:
    from mark_ui import App as Base

try:
    from prealign import prealign
except Exception:
    prealign = None

FULLSCREEN = "--fullscreen" in sys.argv
USE_PREALIGN = "--no-prealign" not in sys.argv and prealign is not None
ORANGE = "#ef8a00"

FONT_SIZE = 12
FONT_SIZE_RUN = 15
FONT_SIZE_TAB = 12
PAD = 4
KEEP_BIG = 18

THAI_FONTS = ("Noto Sans Thai", "Noto Sans Thai UI", "Sarabun", "Loma", "Garuda",
              "Leelawadee UI", "Leelawadee", "Tahoma", "Norasi", "Kinnari")


def pick_thai_font(root):
    have = set(tkfont.families(root))
    for f in THAI_FONTS:
        if f in have:
            return f
    return None


class KioskApp(Base):
    def __init__(self):
        self._pre_msgs = []
        self._orig_test = None
        super().__init__()
        try:
            self._bigger_ui()
        except Exception as ex:
            print("bigger_ui:", ex)

    # ------------------------------------------------------------ หน้าตา
    def _bigger_ui(self):
        fam = pick_thai_font(self)
        self._thai_font = fam
        if fam is None:
            fam = tkfont.nametofont("TkDefaultFont").actual("family")
        F = lambda s, w="normal": (fam, s, w)
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkCaptionFont"):
            try:
                tkfont.nametofont(name).configure(family=fam, size=FONT_SIZE)
            except Exception:
                pass
        self.option_add("*TCombobox*Listbox.font", F(FONT_SIZE))
        st = ttk.Style(self)
        st.configure(".", font=F(FONT_SIZE))
        st.configure("TButton", font=F(FONT_SIZE, "bold"), padding=(PAD, PAD))
        st.configure("Run.TButton", font=F(FONT_SIZE_RUN, "bold"), padding=(PAD * 2, PAD))
        st.configure("TCheckbutton", font=F(FONT_SIZE))
        st.configure("TRadiobutton", font=F(FONT_SIZE))
        st.configure("TCombobox", padding=(PAD, 1))
        st.configure("TEntry", padding=(PAD, 1))
        st.configure("TNotebook.Tab", font=F(FONT_SIZE_TAB, "bold"), padding=(PAD * 2, PAD))
        st.configure("Treeview", font=F(FONT_SIZE - 1), rowheight=int((FONT_SIZE + 10) * 1.4))
        st.configure("Treeview.Heading", font=F(FONT_SIZE - 1, "bold"))
        self._restyle(self, fam)

    def _restyle(self, w, fam):
        for c in w.winfo_children():
            cls = c.winfo_class()
            if cls in ("Label", "Entry", "Text", "TEntry", "TCombobox", "Spinbox", "TSpinbox"):
                try:
                    act = tkfont.Font(font=c.cget("font")).actual()
                    size = abs(int(act.get("size", 9))) or 9
                    if size < KEEP_BIG:
                        size = FONT_SIZE
                    c.configure(font=(fam, size, act.get("weight", "normal")))
                except Exception:
                    pass
            if cls in ("TButton", "TCheckbutton", "TRadiobutton", "TCombobox", "TEntry", "Label", "Entry"):
                for mgr in ("pack", "grid"):
                    try:
                        if c.winfo_manager() == mgr:
                            info = getattr(c, mgr + "_info")()
                            if "ipady" in info and int(str(info["ipady"]).split()[0]) > 4:
                                continue
                            getattr(c, mgr + "_configure")(padx=2, pady=1)
                    except Exception:
                        pass
            if cls == "Label":
                try:
                    if tkfont.Font(font=c.cget("font")).actual().get("size", 9) < KEEP_BIG:
                        c.configure(padx=PAD, pady=1)
                except Exception:
                    pass
            self._restyle(c, fam)

    # ------------------------------------------------------------ ด่านก่อนตรวจ
    def _big(self, text, color):
        lbl = self.__dict__.get("big_verdict")
        if lbl is not None:
            try:
                lbl.config(text=text, bg=color)
            except Exception:
                pass
        pn = self.__dict__.get("pn_result")
        if pn is not None:
            try:
                pn.config(highlightbackground=color, highlightcolor=color, highlightthickness=8)
            except Exception:
                pass

    def run_inspect(self, frame=None):
        test = frame if frame is not None else self.test_img
        self._pre_msgs = []
        if not USE_PREALIGN or self.busy or test is None or self.master_img is None:
            return super().run_inspect(frame)
        try:
            r = prealign(self.master_img, test)
        except Exception as ex:
            self._pre_msgs = [f"จัดภาพล่วงหน้าไม่สำเร็จ ({ex}) — ใช้วิธีเดิม"]
            return super().run_inspect(frame)
        if not r["ok"]:
            self._big("ถ่ายใหม่", ORANGE)
            self.status.set("⚠ " + " / ".join(r["problems"]) + f"   (จุดอ้างอิง {r['inliers']})")
            return None
        self._orig_test = test.copy()
        self._pre_msgs = [f"จัดภาพแล้ว: {r.get('warp', '')}{' กลับด้าน' if r['mirrored'] else ''}, "
                          f"จุดอ้างอิง {r['inliers']}"
                          + (f", ตรวจเฉพาะ{r['roi']}" if r.get("roi") else "")] \
            + ["⚠ " + p for p in r["problems"]]
        # v4: ใช้ Master ที่จำกัดเขตแล้ว ชั่วคราว (ขนาด/ตำแหน่งเท่าเดิม) แล้วคืนค่าเดิม
        orig_master = self.master_img
        self.master_img = r.get("master", orig_master)
        try:
            out = super().run_inspect(r["test"])
        finally:
            self.master_img = orig_master
        if self.__dict__.get("_raw_test") is not None:
            self._raw_test = self._orig_test
        if self.__dict__.get("_raw_master") is not None:
            self._raw_master = orig_master
        return out

    def _on_result(self, res):
        super()._on_result(res)
        msgs = list(self._pre_msgs)
        try:
            els = [e for e in self.report.get("elements", []) if e.get("status") != "SKIP"]
            bad = [e for e in els if e.get("status") in ("MISSING", "PARTIAL")]
            if els and len(bad) / len(els) > 0.30:
                msgs.append(f"⚠ NG {len(bad)}/{len(els)} ชิ้น — อาจเป็นคนละโมเดล หรือภาพไม่ชัด ตรวจ Master ก่อน")
        except Exception:
            pass
        if msgs:
            self.status.set(self.status.get() + "   |   " + "   |   ".join(msgs))


def maximize(app):
    app.update_idletasks()
    try:
        app.minsize(1, 1)
    except Exception:
        pass
    for how in (lambda: app.attributes("-zoomed", True), lambda: app.state("zoomed")):
        try:
            how()
            break
        except Exception:
            pass
    else:
        app.geometry(f"{app.winfo_screenwidth()}x{app.winfo_screenheight()}+0+0")
    if FULLSCREEN:
        app.attributes("-fullscreen", True)


def main():
    app = KioskApp()
    state = {"fs": FULLSCREEN}

    def toggle(_e=None):
        state["fs"] = not state["fs"]
        app.attributes("-fullscreen", state["fs"])

    def leave(_e=None):
        state["fs"] = False
        app.attributes("-fullscreen", False)

    app.bind("<F11>", toggle)
    app.bind("<Escape>", leave)
    app.after(300, lambda: maximize(app))
    app.after(2500, lambda: maximize(app))
    app.mainloop()


if __name__ == "__main__":
    main()

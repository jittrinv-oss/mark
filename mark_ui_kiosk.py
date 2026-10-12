#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mark_ui_kiosk.py — เปิดโปรแกรมตรวจมาร์ค (mark_ui_plus) แบบ "เต็มจอทันที"
ไม่แก้ไขไฟล์เดิม: แค่สร้างหน้าต่างจาก AppPlus แล้วขยายเต็มจอให้อัตโนมัติ
  F11  = สลับ เต็มจอ (ไม่มีแถบหัว) / ขยายเต็มหน้าต่าง
  Esc  = ออกจากโหมดเต็มจอ (กลับเป็นหน้าต่างขยายเต็ม)
ใช้กับ autostart บน Pi:  bash run_kiosk.sh
"""
import sys

try:
    from mark_ui_plus import AppPlus as App
except Exception:                      # ถ้าไม่มีไฟล์ plus ใช้โปรแกรมเดิม
    from mark_ui import App

FULLSCREEN = "--fullscreen" in sys.argv     # ไม่ใส่ = ขยายเต็มจอแบบยังมีแถบหัวหน้าต่าง


def maximize(app):
    app.update_idletasks()
    w, h = app.winfo_screenwidth(), app.winfo_screenheight()
    try:
        app.minsize(1, 1)
    except Exception:
        pass
    done = False
    for how in (lambda: app.attributes("-zoomed", True),   # Linux / Pi
                lambda: app.state("zoomed")):               # Windows
        try:
            how()
            done = True
            break
        except Exception:
            pass
    if not done:
        app.geometry(f"{w}x{h}+0+0")
    if FULLSCREEN:
        app.attributes("-fullscreen", True)


def main():
    app = App()
    state = {"fs": FULLSCREEN}

    def toggle(_e=None):
        state["fs"] = not state["fs"]
        app.attributes("-fullscreen", state["fs"])

    def leave(_e=None):
        state["fs"] = False
        app.attributes("-fullscreen", False)

    app.bind("<F11>", toggle)
    app.bind("<Escape>", leave)
    # รอให้หน้าจอ/ตัวจัดหน้าต่างพร้อมก่อน แล้วค่อยขยาย (ตอนบูตใหม่ ๆ สั่งเร็วไปจะไม่มีผล)
    app.after(300, lambda: maximize(app))
    app.after(2500, lambda: maximize(app))
    app.mainloop()


if __name__ == "__main__":
    main()

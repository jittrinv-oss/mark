#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mark_ui_plus.py — เปิดโปรแกรมตรวจมาร์คเดิม (mark_ui.py) + ส่วนเสริม "ภาพ Test ตัวหนังสือสีอ่อน"

ไม่แก้ไขไฟล์เดิมเลย: สืบทอด (subclass) หน้าจอเดิม แล้วเพิ่ม
  - แถบเลือก "สีภาพ Test": อัตโนมัติ / ตัวเข้ม พื้นอ่อน (เดิม) / ตัวอ่อน พื้นเข้ม/พื้นสี
  - ก่อนตรวจ แปลงภาพ Test เป็นตัวดำพื้นขาว (เฉพาะเมื่อจำเป็น) แล้วส่งเข้าการตรวจเดิม
ถ้าภาพ Test เป็นตัวเข้มบนพื้นอ่อนอยู่แล้ว โหมดอัตโนมัติจะไม่แตะภาพ -> ผลเหมือนโปรแกรมเดิมทุกอย่าง

รัน:  python mark_ui_plus.py   (Windows: RUN_UI_PLUS.bat / Pi: bash run_plus.sh)
"""
import tkinter as tk
from tkinter import ttk

import mark_ui
from test_polarity import MODES, normalize_test


class AppPlus(mark_ui.App):
    def __init__(self):
        self.pol_mode = None
        self._pol_msg = ""
        super().__init__()
        self.title(mark_ui.APP + "  (+ ภาพตัวหนังสือสีอ่อน)")
        self.pol_mode = tk.StringVar(value=MODES[0])
        bar = tk.Frame(self, bg=mark_ui.CARD)
        tk.Label(bar, text="สีภาพ Test", bg=mark_ui.CARD, font=mark_ui.F(9, "bold")).pack(side="left", padx=(10, 4))
        ttk.Combobox(bar, textvariable=self.pol_mode, values=list(MODES), state="readonly",
                     width=24, font=mark_ui.F(9)).pack(side="left", pady=2)
        tk.Label(bar, text="ตัวหนังสือขาว/เหลืองบนกระจกเขียว ใช้ 'อัตโนมัติ' หรือ 'ตัวอ่อน พื้นเข้ม/พื้นสี'",
                 bg=mark_ui.CARD, fg="#55606e", font=mark_ui.F(8)).pack(side="left", padx=8)
        nbs = [w for w in self.winfo_children() if isinstance(w, ttk.Notebook)]
        if nbs:
            bar.pack(fill="x", padx=6, before=nbs[0])
        else:
            bar.pack(fill="x", padx=6)

    def run_inspect(self, frame=None):
        test = frame if frame is not None else self.test_img
        if test is not None and self.pol_mode is not None and not self.busy:
            try:
                test, self._pol_msg = normalize_test(test, self.pol_mode.get())
            except Exception as ex:          # ส่วนเสริมพัง -> ใช้ภาพเดิม ไม่ให้การตรวจหลักหยุด
                self._pol_msg = f"กลับสีไม่สำเร็จ ({ex}) — ใช้ภาพเดิม"
        return super().run_inspect(test)

    def _on_result(self, res):
        super()._on_result(res)
        if self._pol_msg:
            self.status.set(self.status.get() + "   |   " + self._pol_msg)


if __name__ == "__main__":
    AppPlus().mainloop()

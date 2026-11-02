แพตช์ตัวตรวจคมชัด v1.3  (ต่อจาก v1.2 — ไม่แก้ไฟล์เดิมอื่น)

ไฟล์ในชุด
  sharp_fix.py      ใหม่  — แพตช์ตัวตรวจ (กรอบเส้นประ/รหัสวันที่, ส้มผิด, Test จาก PDF, QR ปลอม)
  mark_ui_sharp.py  ทับ   — เพิ่มบรรทัด import sharp_fix (หัวโปรแกรมขึ้น "ตัวตรวจคมชัด v1.3")
  web_sharp.py      ทับ   — เพิ่มบรรทัด import sharp_fix
  before_after_test.png  ภาพผลทดสอบ (ซ้าย→ขวา: เดิม NG / ใหม่ OK / Test จาก PDF ใหม่ OK / ลบขีดใต้ SNI = NG)

วิธีอัปเดต
 1) ปิดโปรแกรม (หน้าจอ และ web)
 2) ก๊อป 3 ไฟล์ .py วางทับในโฟลเดอร์โปรแกรม
 3) ลบโฟลเดอร์ __pycache__
 4) เปิดด้วยไฟล์ .bat / shortcut เดิม  (ไม่ต้องทำไฟล์ run ใหม่ ไม่ต้องติดตั้งเพิ่ม)
 5) แถบล่างต้องขึ้น "fix v1.3"
 Pi: วางไฟล์เหมือนกัน หรือ git pull แล้วรันสคริปต์เดิม

ปิดแต่ละข้อ (ถ้าต้องการ): กด "ตั้งค่าคมชัด" → sharp_config.json → ตั้ง false
  fix_master_color_frame / fix_test_red_lines / fix_paper_not_glare /
  fix_qr_strict / fix_align_check / fix_edge_partial / fix_edge_max_loss (0.10)
ย้อนกลับ v1.2: ลบ sharp_fix.py (โปรแกรมยังเปิดได้ ใช้ตัวตรวจ v1.2)

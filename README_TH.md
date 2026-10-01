# Mark Matching Inspection — ตรวจมาร์คกระจกด้วยกล้อง (Prototype)

เคสตั้งต้น: กระดาษเทส (ซ้าย) มี `5NI` **มีขีดด้านบน/ล่าง** ตาม product spec. แต่ของจริงขีดหาย
→ OCR อย่างเดียว **จับไม่ได้** เพราะ OCR อ่าน "5NI" ได้เหมือนเดิม
→ ต้องใช้ **pixel/geometry comparison** ควบคู่ไปด้วย ซึ่งเป็นสิ่งที่โปรแกรมชุดนี้ทำ

---

## 1. ไฟล์ในชุด

| ไฟล์ | หน้าที่ |
|---|---|
| `mark_match.py` | เอนจินหลัก: preprocess → align → compare → report |
| `roi_tool.py` | ลากกรอบ ROI บนภาพ Master ด้วยเมาส์ → เซฟ `roi.json` |
| `run_live.py` | ตรวจ real-time จากกล้อง USB พร้อม log NG อัตโนมัติ |
| `make_demo.py` | สร้างภาพตัวอย่าง (master / test OK / test NG ขีดหาย) ไว้ลองเล่นก่อนมีกล้อง |
| `roi.json` | ตัวอย่างนิยาม ROI รวมถึง `5NI_UNDERLINE` แบบ `mode:"line"` |

---

## 2. ติดตั้ง

```bash
pip install opencv-python numpy
pip install pytesseract          # เฉพาะถ้าจะใช้ OCR (ต้องลง Tesseract engine ด้วย)
```

---

## 3. ลองใช้ทันที (ยังไม่ต้องมีกล้อง)

```bash
python make_demo.py                                   # สร้าง master.jpg / test_ok.jpg / test_ng.jpg
python mark_match.py --master master.jpg --test test_ng.jpg --roi roi.json --out result_ng
```

ผลที่ได้จริงจากการรัน:

```
RESULT = NG   (align: ORB+RANSAC)
  missing blobs=1  extra blobs=0
  [NG] 5NI_UNDERLINE   master_line_px=65  test_line_px=0  score=0.0  threshold=0.6
  [OK] 5NI_TEXT        ink_ratio=1.139  corr=0.924
  [OK] TEMPERLITE      ink_ratio=1.151  corr=0.919
  * MISSING area=321 at (103,392) 65x5      <-- คือเส้นขีดที่หายไปพอดี
```

ไฟล์ออก: `overlay.png` (แดง = หมึกหาย, ม่วง = หมึกเกิน), `aligned.png`, `report.json`, `roi_report.csv`

---

## 4. ขั้นตอนการทำงานของโปรแกรม (อธิบายทีละสเต็ป)

### STEP 1 — ลบรอยปากกา (`remove_pen_ink`)
กระดาษเทสมีปากกาน้ำเงินเขียนกำกับ ถ้าไม่ลบจะกลายเป็น "หมึกเกิน" ทุกใบ
หลักการ: มาร์คพิมพ์เป็น **สีดำ → Saturation ต่ำ**, ปากกาเป็น **สี → Saturation สูง**
แปลงเป็น HSV แล้วลบพิกเซลที่ `S > 60` ทิ้ง (ปรับค่าได้ถ้าปากกาสีอื่น)

### STEP 2 — ทำ Ink Mask (`to_ink_mask`)
Gray → CLAHE (ดันคอนทราสต์เฉพาะที่ สู้แสงไม่สม่ำเสมอ) → Bilateral filter (ลด noise รักษาขอบ)
→ **Adaptive Threshold** (ไม่ใช้ Otsu ค่าเดียวทั้งภาพ เพราะกระจก/กระดาษสะท้อนแสงไม่เท่ากัน)
→ ตัด blob เล็กกว่า 8 px ทิ้ง

### STEP 3 — จัดภาพให้ซ้อนทับ Master (`align`)
1. **ORB + RANSAC Homography** — หาจุดเด่นในภาพทั้งสอง จับคู่ แล้วคำนวณเมทริกซ์แปลง
   ทนการหมุน/เอียง/ระยะกล้องต่างกันได้ ถ้าได้ inlier ≥ 12 ถือว่าใช้ได้
2. ถ้าไม่ผ่าน → fallback **ECC affine** (จูนจากความเข้มภาพตรง ๆ)
3. ถ้ายังไม่ได้ → resize อย่างเดียว + เตือนใน report (`RESIZE-ONLY(!)`) = อย่าเชื่อผล

> จุดนี้สำคัญที่สุด ถ้า align เพี้ยน 2–3 px ตัวอักษรบาง ๆ จะฟ้อง NG หมด

### STEP 4 — เทียบหา MISSING / EXTRA (`diff_regions`)
- `MISSING = master − dilate(test, tol)` → master มี แต่ test ไม่มี = **หมึกขาด**
- `EXTRA   = test − dilate(master, tol)` → test มี แต่ master ไม่มี = **หมึกเกิน/เลอะ**
- `tol_px` (ค่าเริ่มต้น 3) คือระยะยอมผิดพลาดจากการ align — **ปุ่มหลักในการลด false alarm**
- จับกลุ่มด้วย connected components แล้วกรองพื้นที่ < `min_area`

### STEP 5 — ตรวจราย ROI (`check_rois`) ← ตัวจับเคส 5NI
สองโหมด:

| mode | ใช้กับ | ตัวชี้วัด | ตัดสิน NG เมื่อ |
|---|---|---|---|
| `ink` | ตัวอักษร โลโก้ E6 TIS | `ink_ratio` = หมึกใน test ÷ หมึกใน master และ `corr` (template match) | ratio < 0.55 หรือ > 1.8 หรือ corr < 0.45 |
| `line` | **ขีดใต้/บน 5NI**, เส้นขีดต่าง ๆ | `longest_hline()` วัดความยาวเส้นแนวนอนที่ยาวที่สุด | `test_len / master_len < 0.60` |

`longest_hline()` ใช้ morphological opening ด้วย kernel แนวนอน `(w/6, 1)` → เหลือเฉพาะสิ่งที่
"ยาวในแนวนอนและบางในแนวตั้ง" = เส้นขีด แล้ววัดความกว้าง
ตัวอักษร `5NI` จะถูกกรองทิ้ง ไม่รบกวนการวัด

### STEP 6 — OCR (เสริม, `--ocr`)
เทียบข้อความ เช่น `43R-008574`, `TIS 2602-2556`, `M1H3S` ว่าเป็นรุ่น/มาร์คถูกใบไหม
ใช้เป็น **Pokayoke ป้องกันพิมพ์ผิดรุ่น** ซึ่งตรงกับแนวทาง Logo Mark Inspection ที่ AL12 ทำอยู่
(ใช้ webcam ถ่ายมาร์ค → OCR → เทียบ Master text → Match / Partial / Not match)

### STEP 7 — ตัดสินและออกรายงาน
`NG` เมื่อ มี ROI ใด NG **หรือ** มี blob MISSING/EXTRA ที่พื้นที่ ≥ `--fail-area`
ออก `report.json` + `roi_report.csv` + `overlay.png` เก็บเป็นหลักฐาน traceability

---

## 5. การตั้ง ROI ของงานจริง

```bash
python roi_tool.py --master master.jpg --out roi.json
```
ลากกรอบ → พิมพ์ชื่อใน terminal → เลือก `line` สำหรับขีดใต้ 5NI, `ink` สำหรับตัวอักษร
กด `s` เพื่อเซฟ, `u` undo, `q` ออก

ROI ที่แนะนำสำหรับมาร์คนี้:
`HONDA_LOGO`, `AGC_AUTOMOTIVE`, `E6_43R-008574`, `M1H3S`, `TIS_2602-2556`,
`T1`, `5NI_TEXT`, **`5NI_UNDERLINE` (mode=line)**, `TEMPERLITE`, `11T`, `DOT_1248`

---

## 6. ต่อกล้องจริง

```bash
python run_live.py --master master.jpg --roi roi.json --cam 0 --auto
```
SPACE = ตรวจเฟรมปัจจุบัน | s = เซฟ | q = ออก — NG จะถูกเซฟลงโฟลเดอร์ `log/` อัตโนมัติ

### เงื่อนไขฮาร์ดแวร์ที่ต้องทำให้ได้ก่อน (สำคัญกว่าโค้ด)
| หัวข้อ | ข้อกำหนด |
|---|---|
| กล้อง | ≥ 5 MP, **ปิด autofocus + ปิด auto exposure** (โค้ดสั่งปิดให้แล้ว) |
| Resolution | ต้องได้ ≥ 5 pixel ต่อความหนาเส้นที่บางที่สุด — ขีดใต้ 5NI หนาราว 0.3 mm → ต้อง ≤ 0.06 mm/px |
| ระยะ/มุม | ยึดกล้องตายตัว ตั้งฉากกับชิ้นงาน ระนาบเดียวกับ Master |
| แสง | **Diffuse / dome หรือ bar light 2 ข้าง** — ห้ามใช้ไฟจุดเดียว จะเกิดแสงสะท้อนบนกระจกแล้วหมึกหายเป็นหย่อม |
| กระจกโปร่ง | แนะนำ **backlight + พื้นหลังขาว** หรือถ่ายฝั่งพิมพ์โดยมีแผ่นขาวรองหลัง |
| Master | ถ่ายจาก first piece ที่ QA เซ็นอนุมัติ ด้วยกล้อง/แสง/ระยะ **ชุดเดียวกัน** แล้วล็อกไฟล์ไว้ต่อรุ่น |

---

## 7. วิธีจูนค่า (Tuning) — ลำดับที่ควรทำ

1. เก็บภาพ OK จริง 30–50 ใบ + NG จริงเท่าที่มี
2. รัน OK ทั้งหมด → ดูว่ามี false NG ไหม
   - ถ้าฟ้องเยอะ: เพิ่ม `--tol` จาก 3 → 4–5, เพิ่ม `--fail-area`, ลด `min_ratio` เป็น 0.45
3. รัน NG → ต้องจับได้ 100%
   - ถ้าหลุด: ลด `--fail-area`, ขยับ `line_min_len_ratio` ขึ้นเป็น 0.7
4. บันทึกค่าสุดท้ายลง `roi.json` แล้วทำ **Pokayoke test (ใบ NG จงใจ)** ก่อนเริ่มไลน์ทุกกะ

เป้าหมายที่ยอมรับได้: **จับ NG จริง 100%, false NG ≤ 1%** — ถ้า false เยอะ operator จะ bypass ระบบ

---

## 8. แผนทดลอง (แนะนำ 5 สัปดาห์)

| สัปดาห์ | งาน | ผลลัพธ์ |
|---|---|---|
| 1 | จัดชุดกล้อง+ไฟ, ถ่าย Master, สร้าง `roi.json` | ภาพนิ่ง คมชัด ทำซ้ำได้ |
| 2 | Offline test กับภาพ 50 ใบ จูนพารามิเตอร์ | ตาราง OK/NG + false rate |
| 3 | ต่อ live กับกล้อง รันคู่ขนาน (ไม่ stop line) | log NG ประจำวัน |
| 4 | เทียบผลระบบ vs. ตรวจคน | ยืนยันความแม่น |
| 5 | สรุป + ตั้งเป็น Pokayoke / เขียน WI | เอกสารส่งมอบ |

---

## 9. ข้อจำกัดที่ต้องรู้
- ต้องมี **Master ต่อรุ่น** ถ้าเปลี่ยนรุ่นแล้วไม่เปลี่ยน Master จะ NG หมดใบ
- ตรวจ "มาร์คครบ/ไม่ครบ" ได้ดี แต่ **ตำแหน่งมาร์คเทียบขอบกระจก (X/Y จากขอบ)** ต้องใช้ ROI
  อ้างอิงขอบกระจกเพิ่ม — เป็นคนละโจทย์กับเคส 640A ที่ตำแหน่ง logo เพี้ยน
- ถ้าใช้จริงบนไลน์ ควรผูกผล NG เข้ากับ PLC/alarm ไม่ใช่แค่โชว์บนจอ

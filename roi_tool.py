#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
roi_tool.py - ลากกรอบ ROI บนภาพ Master แล้วเซฟเป็น roi.json
วิธีใช้:
    python roi_tool.py --master master.jpg --out roi.json
คีย์:
    ลากเมาส์      = สร้างกรอบ
    พิมพ์ชื่อ + Enter (ในหน้าต่าง terminal) = ตั้งชื่อ ROI และเลือก mode
    u  = undo กรอบล่าสุด
    s  = save
    q  = ออก
"""
import argparse, json
import cv2

rois, drawing, p0, p1 = [], False, None, None
img_disp = None


def on_mouse(event, x, y, flags, param):
    global drawing, p0, p1, img_disp
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing, p0, p1 = True, (x, y), (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        p1 = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing, p1 = False, (x, y)
        x0, y0 = min(p0[0], x), min(p0[1], y)
        w, h = abs(x - p0[0]), abs(y - p0[1])
        if w > 4 and h > 4:
            name = input("ชื่อ ROI (เช่น 5NI_UNDERLINE): ").strip() or f"ROI{len(rois)+1}"
            mode = input("mode [ink/line] (Enter=ink): ").strip().lower() or "ink"
            r = dict(name=name, x=int(x0), y=int(y0), w=int(w), h=int(h), mode=mode)
            r["line_min_len_ratio"] = 0.60 if mode == "line" else None
            r["min_ratio"] = None if mode == "line" else 0.55
            r["min_corr"] = None if mode == "line" else 0.45
            rois.append({k: v for k, v in r.items() if v is not None})
            print("  + เพิ่มแล้ว:", rois[-1])


def redraw(base):
    im = base.copy()
    for r in rois:
        c = (0, 140, 255) if r["mode"] == "line" else (0, 170, 0)
        cv2.rectangle(im, (r["x"], r["y"]), (r["x"] + r["w"], r["y"] + r["h"]), c, 2)
        cv2.putText(im, r["name"], (r["x"], max(14, r["y"] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    if drawing and p0 and p1:
        cv2.rectangle(im, p0, p1, (255, 0, 0), 1)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--out", default="roi.json")
    a = ap.parse_args()
    base = cv2.imread(a.master)
    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI", on_mouse)
    while True:
        cv2.imshow("ROI", redraw(base))
        k = cv2.waitKey(20) & 0xFF
        if k == ord('u') and rois:
            print("  - ลบ:", rois.pop()["name"])
        elif k == ord('s'):
            json.dump({"master": a.master, "rois": rois},
                      open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            print("saved ->", a.out)
        elif k == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

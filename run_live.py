#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_live.py - ตรวจแบบ real-time จากกล้อง USB/webcam
    python run_live.py --master master.jpg --roi roi.json --cam 0
คีย์:  SPACE = ตรวจเฟรมปัจจุบัน | s = เซฟภาพ+รายงาน | q = ออก
"""
import argparse, os, json, time
import cv2
from mark_match import inspect

ap = argparse.ArgumentParser()
ap.add_argument("--master", required=True)
ap.add_argument("--roi", default=None)
ap.add_argument("--cam", type=int, default=0)
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--auto", action="store_true", help="ตรวจอัตโนมัติทุกเฟรม")
ap.add_argument("--log", default="log")
a = ap.parse_args()

master = cv2.imread(a.master)
rois = json.load(open(a.roi, encoding="utf-8"))["rois"] if a.roi else None
os.makedirs(a.log, exist_ok=True)

cap = cv2.VideoCapture(a.cam, cv2.CAP_DSHOW)          # Windows ใช้ CAP_DSHOW
cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)                    # ปิด AF -> โฟกัสคงที่ สำคัญมาก
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)             # ปิด auto exposure

cv2.namedWindow("LIVE", cv2.WINDOW_NORMAL)
last, overlay = None, None
while True:
    ok, frame = cap.read()
    if not ok:
        break
    view = frame.copy()
    key = cv2.waitKey(1) & 0xFF

    if a.auto or key == 32:
        rep, overlay, aligned, *_ = inspect(master, frame, rois)
        last = rep
        c = (0, 200, 0) if rep["verdict"] == "OK" else (0, 0, 255)
        cv2.putText(view, rep["verdict"], (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, c, 5, cv2.LINE_AA)
        if overlay is not None:
            cv2.imshow("RESULT", overlay)
        if rep["verdict"] == "NG":
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"{a.log}/NG_{ts}.png", overlay)
            json.dump(rep, open(f"{a.log}/NG_{ts}.json", "w", encoding="utf-8"),
                      indent=2, ensure_ascii=False)

    cv2.imshow("LIVE", view)
    if key == ord('s') and last:
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"{a.log}/{ts}.png", overlay if overlay is not None else frame)
        json.dump(last, open(f"{a.log}/{ts}.json", "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        print("saved", ts)
    if key == ord('q'):
        break
cap.release(); cv2.destroyAllWindows()

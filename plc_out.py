#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plc_out.py — ส่งผลตรวจ OK/NG ออก Digital Output ไปยัง PLC
             (Cytron Industrial IO HAT+ HAT-IND-AQ บน Raspberry Pi 5)
=============================================================================
ใช้กับงาน: First-piece mark check ที่สถานี PRINT INSPECTION O/P 1
           ถ้ากล้องยังไม่ยืนยันว่า OK -> PLC จะไม่ยอมให้กด CYCLE START
           แม้บิดสวิตช์ไปโหมด AUTO แล้วก็ตาม

-----------------------------------------------------------------------------
หลักการออกแบบ (สำคัญมาก อ่านก่อนใช้)
-----------------------------------------------------------------------------
ใช้ logic แบบ **FAIL-SAFE / ENERGIZE-TO-PERMIT**

    DO0 = READY-TO-RUN  ->  ON ก็ต่อเมื่อ "ผ่านการตรวจแล้ว OK"
    DO1 = HEARTBEAT     ->  กะพริบ 1 Hz ตลอดเวลาที่โปรแกรมยังมีชีวิต
    DO2 = NG PULSE      ->  ON 500 ms เมื่อเจอ NG (ไว้ให้ PLC นับ/เก็บ log)
    DO3 = SYSTEM FAULT  ->  ON เมื่อกล้องหลุด / align ไม่ผ่าน / โปรแกรม error

เหตุผลที่ห้ามใช้ "NG = ON":
    ถ้า Pi ดับ สายหลุด โปรแกรมค้าง -> DO จะเป็น OFF
    PLC จะเข้าใจว่า "ไม่มี NG = ดี" -> ของเสียหลุดออกไปทั้งล็อต
    แต่ถ้าใช้ OK = ON: Pi ดับ -> DO OFF -> PLC เข้าใจว่า "ยังไม่ยืนยัน" -> บล็อกไว้
    นี่คือหลัก Pokayoke ที่ถูกต้อง (ระบบล้มเหลว = ต้องหยุด ไม่ใช่ปล่อยผ่าน)

PLC ต้องเช็ค **DO0 AND DO1(heartbeat กะพริบ)** ทั้งคู่
ถ้าเช็คแค่ DO0 อย่างเดียว กรณี Pi ค้างแบบ output ค้าง ON จะจับไม่ได้

-----------------------------------------------------------------------------
การเดินสาย (PLC input แบบ SINK / PNP — กรณีทั่วไปของ Mitsubishi FX/Q)
-----------------------------------------------------------------------------
    PLC 24V+ ──────────► DOC        (common ฝั่ง output, ป้อนจาก PLC เท่านั้น)
    DO0      ──────────► X10        (READY)
    DO1      ──────────► X11        (HEARTBEAT)
    DO2      ──────────► X12        (NG PULSE)
    DO3      ──────────► X13        (FAULT)
    PLC 0V   ──────────► COM ของการ์ด input

⚠ ห้ามต่อ DOC เข้ากับ GND ของ Raspberry Pi เด็ดขาด — จะเสีย isolation 1500 Vrms
⚠ ไฟเลี้ยง HAT รับ DC 10–30V ต่อจาก 24V ในตู้ได้เลย และจ่ายไฟให้ Pi ในตัว

-----------------------------------------------------------------------------
ใช้งาน
-----------------------------------------------------------------------------
    # ทดสอบสายกับ PLC ก่อน (ไม่ใช้กล้อง)
    python3 plc_out.py --selftest

    # รันจริง: ตรวจ first piece ด้วยกล้อง
    python3 plc_out.py --master master.png --config config.json --cam 0

    # โหมดจำลองบน Windows (ไม่มี GPIO) ใช้ดู logic ก่อนขึ้น Pi
    python3 plc_out.py --master master.png --config config.json --sim
"""
import argparse, json, os, sys, threading, time
from datetime import datetime

# =====================================================================
# ตั้งค่าขา GPIO  <<<<<  แก้ตรงนี้หลังรัน find_do_pin.py แล้ว
# =====================================================================
DO_PINS = {
    "DO0": 5,    # READY-TO-RUN   (ยังไม่ยืนยัน! ต้องหาด้วย find_do_pin.py)
    "DO1": 6,    # HEARTBEAT
    "DO2": 16,   # NG PULSE
    "DO3": 26,   # SYSTEM FAULT
}

NG_PULSE_MS = 500        # ความกว้าง pulse แจ้ง NG (ต้อง > scan time ของ PLC x2)
HEARTBEAT_HZ = 1.0       # ความถี่ heartbeat
READY_TIMEOUT_S = 0      # 0 = READY ค้างจนกว่าจะตรวจใบใหม่
                         # >0 = READY จะดับเองหลัง n วินาที (บังคับตรวจซ้ำ)


# =====================================================================
# ชั้นควบคุม GPIO — มีโหมดจำลองสำหรับทดสอบบน PC
# =====================================================================
class DOController:
    def __init__(self, pins=None, simulate=False):
        self.pins = pins or DO_PINS
        self.sim = simulate
        self.state = {k: False for k in self.pins}
        self._hb_stop = threading.Event()
        self._hb_thread = None
        self._lock = threading.Lock()

        if not self.sim:
            try:
                from gpiozero import LED, Device
                from gpiozero.pins.lgpio import LGPIOFactory
                Device.pin_factory = LGPIOFactory()
                self.out = {k: LED(v) for k, v in self.pins.items()}
            except ImportError:
                print("!! ไม่พบ gpiozero — สลับเป็นโหมดจำลองอัตโนมัติ")
                self.sim = True
        if self.sim:
            self.out = None
            print(">> โหมดจำลอง (ไม่ส่งสัญญาณจริง)")

    def set(self, name, on):
        with self._lock:
            self.state[name] = on
            if not self.sim:
                (self.out[name].on if on else self.out[name].off)()
            print(f"   [{datetime.now():%H:%M:%S}] {name} = {'ON ' if on else 'OFF'}"
                  f"   ({self._meaning(name)})")

    @staticmethod
    def _meaning(name):
        return {"DO0": "READY-TO-RUN", "DO1": "HEARTBEAT",
                "DO2": "NG PULSE", "DO3": "SYSTEM FAULT"}.get(name, "")

    def pulse(self, name, ms=NG_PULSE_MS):
        self.set(name, True)
        threading.Timer(ms / 1000.0, lambda: self.set(name, False)).start()

    # ---------- heartbeat ----------
    def start_heartbeat(self):
        if self._hb_thread:
            return
        self._hb_stop.clear()

        def loop():
            half = 1.0 / (2 * HEARTBEAT_HZ)
            s = False
            while not self._hb_stop.wait(half):
                s = not s
                with self._lock:
                    self.state["DO1"] = s
                    if not self.sim:
                        (self.out["DO1"].on if s else self.out["DO1"].off)()
        self._hb_thread = threading.Thread(target=loop, daemon=True)
        self._hb_thread.start()
        print("   heartbeat DO1 เริ่มทำงาน (1 Hz)")

    def stop_heartbeat(self):
        self._hb_stop.set()
        if self._hb_thread:
            self._hb_thread.join(timeout=2)
        self._hb_thread = None
        self.set("DO1", False)

    # ---------- สถานะระดับงาน ----------
    def set_ready(self, ok):
        """OK -> อนุญาตให้กด CYCLE START ได้"""
        self.set("DO0", ok)
        if ok and READY_TIMEOUT_S > 0:
            threading.Timer(READY_TIMEOUT_S,
                            lambda: self.set("DO0", False)).start()

    def report_ng(self):
        self.set("DO0", False)      # ตัด READY ทันที
        self.pulse("DO2")

    def set_fault(self, on):
        self.set("DO3", on)
        if on:
            self.set("DO0", False)  # fault = ห้ามเดินเครื่องเด็ดขาด

    def all_off(self):
        self.stop_heartbeat()
        for k in self.pins:
            self.set(k, False)

    def close(self):
        self.all_off()
        if not self.sim and self.out:
            for o in self.out.values():
                o.close()


# =====================================================================
# SELF TEST — ทดสอบสายกับ PLC โดยไม่ต้องใช้กล้อง
# =====================================================================
def selftest(dc):
    print("\n" + "=" * 62)
    print(" SELF TEST — ให้ช่างไฟฟ้าดูที่ PLC monitor ว่า X10–X13 ติดตรงกัน")
    print("=" * 62)
    for name in ["DO0", "DO1", "DO2", "DO3"]:
        print(f"\n--- {name} ({DOController._meaning(name)}) GPIO{dc.pins[name]}")
        dc.set(name, True); time.sleep(2)
        dc.set(name, False); time.sleep(1)

    print("\n--- ทดสอบ NG pulse 500 ms (ดูว่า PLC จับทันไหม)")
    for i in range(3):
        print(f"  pulse {i+1}/3")
        dc.pulse("DO2"); time.sleep(1.5)

    print("\n--- ทดสอบ heartbeat 8 วินาที (X11 ต้องกะพริบ)")
    dc.start_heartbeat(); time.sleep(8); dc.stop_heartbeat()

    print("\n--- จำลองลำดับงานจริง")
    print("  1) เริ่มระบบ: ทุกอย่าง OFF = PLC ต้องบล็อก CYCLE START")
    dc.all_off(); time.sleep(2)
    print("  2) heartbeat ติด แต่ยังไม่ตรวจ = ยังบล็อกอยู่")
    dc.start_heartbeat(); time.sleep(3)
    print("  3) ตรวจแล้ว NG = บล็อก + pulse")
    dc.report_ng(); time.sleep(3)
    print("  4) ตรวจใหม่ OK = ปลดล็อก กด CYCLE START ได้")
    dc.set_ready(True); time.sleep(4)
    print("  5) จำลอง Pi ดับ = ทุกอย่างดับ PLC ต้องบล็อกทันที")
    dc.all_off()
    print("\n SELF TEST เสร็จสิ้น")


# =====================================================================
# โหมดรันจริง — ผูกกับ glyph_check.py
# =====================================================================
def run_inspection(dc, args):
    import cv2
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from glyph_check import inspect_glyph
    except ImportError:
        dc.set_fault(True)
        sys.exit("ไม่พบ glyph_check.py — ต้องวางไว้โฟลเดอร์เดียวกัน")

    master = cv2.imdecode(
        __import__("numpy").fromfile(args.master, __import__("numpy").uint8),
        cv2.IMREAD_COLOR)
    if master is None:
        dc.set_fault(True)
        sys.exit("อ่านภาพ Master ไม่ได้")

    ignores, params = [], {}
    if args.config and os.path.exists(args.config):
        cfg = json.load(open(args.config, encoding="utf-8"))
        ignores = cfg.get("ignore", [])
        params = cfg.get("params", {})

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if not cap.isOpened():
        dc.set_fault(True)
        sys.exit(f"เปิดกล้อง {args.cam} ไม่ได้")

    os.makedirs(args.log, exist_ok=True)
    dc.all_off()
    dc.start_heartbeat()
    dc.set_fault(False)

    print("\n" + "=" * 62)
    print(" FIRST-PIECE MARK CHECK — พร้อมทำงาน")
    print(" SPACE = ตรวจแผ่นปัจจุบัน   R = ล้างผล (บล็อกใหม่)   Q = ออก")
    print("=" * 62)

    cv2.namedWindow("FIRST PIECE CHECK", cv2.WINDOW_NORMAL)
    last_ov = None
    while True:
        ok, frame = cap.read()
        if not ok:
            dc.set_fault(True)
            print("!! กล้องหลุด")
            time.sleep(1)
            continue
        dc.set_fault(False)

        view = frame.copy()
        st = "READY - กด CYCLE START ได้" if dc.state["DO0"] else "ยังไม่ผ่าน - ล็อกอยู่"
        col = (0, 200, 0) if dc.state["DO0"] else (0, 0, 255)
        cv2.rectangle(view, (0, 0), (view.shape[1], 60), col, -1)
        cv2.putText(view, st, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imshow("FIRST PIECE CHECK", view)

        k = cv2.waitKey(30) & 0xFF
        if k == ord('q'):
            break
        if k == ord('r'):
            dc.set_ready(False)
            print("   ล้างผล -> ล็อกใหม่")
        if k == 32:                                    # SPACE
            try:
                rep, ov, *_ = inspect_glyph(master, frame, ignores, params)
            except Exception as e:
                dc.set_fault(True)
                print(f"!! ตรวจไม่สำเร็จ: {e}")
                continue
            last_ov = ov
            ts = time.strftime("%Y%m%d_%H%M%S")
            bad_align = "RESIZE" in rep["align_method"]

            print(f"\n>> ผล: {rep['verdict']}  align={rep['align_method']}  "
                  f"หายทั้งชิ้น={rep['n_missing']} หายบางส่วน={rep['n_partial']}")

            if bad_align:
                # align ไม่ผ่าน = เชื่อผลไม่ได้ ถือเป็น FAULT ไม่ใช่ OK
                dc.set_fault(True)
                print("!! align ไม่สำเร็จ — ถือเป็น FAULT ไม่ปลดล็อก")
            elif rep["verdict"] == "OK":
                dc.set_ready(True)
                print("   >>> ปลดล็อก: กด CYCLE START ได้แล้ว")
            else:
                dc.report_ng()
                print("   >>> NG: ล็อกไว้ ต้องแก้แล้วตรวจใหม่")
                for r in rep["elements"]:
                    if r["status"] not in ("OK", "SKIP"):
                        print(f"       [{r['status']}] #{r['id']} {r['kind']} "
                              f"({r['x']},{r['y']}) {r['note']}")

            cv2.imencode(".png", ov)[1].tofile(
                os.path.join(args.log, f"{rep['verdict']}_{ts}.png"))
            json.dump(rep, open(os.path.join(args.log, f"{rep['verdict']}_{ts}.json"),
                                "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            cv2.imshow("RESULT", ov)

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="ทดสอบสาย PLC")
    ap.add_argument("--master", help="ภาพ Master")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--log", default="log")
    ap.add_argument("--sim", action="store_true", help="โหมดจำลอง ไม่ใช้ GPIO จริง")
    a = ap.parse_args()

    dc = DOController(simulate=a.sim)
    try:
        if a.selftest:
            selftest(dc)
        elif a.master:
            run_inspection(dc, a)
        else:
            ap.print_help()
    except KeyboardInterrupt:
        print("\nหยุดโดยผู้ใช้")
    finally:
        print("\nปิดสัญญาณทั้งหมด (PLC จะกลับไปสถานะล็อก)")
        dc.close()


if __name__ == "__main__":
    main()

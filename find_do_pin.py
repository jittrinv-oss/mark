#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_do_pin.py — หาว่า DO0–DO3 ของ Cytron Industrial IO HAT+ (HAT-IND-AQ)
                 ผูกกับ GPIO ขาไหนบน Raspberry Pi 5
-----------------------------------------------------------------------------
ทำไมต้องหา:
  เอกสารสาธารณะระบุแค่ RS485=GPIO14/15, RS232=GPIO12/13
  ส่วน DO0–DO3 ต้องดูจาก Datasheet หรือหาเอาเองด้วยวิธีนี้

วิธีดู:
  บนบอร์ดมี LED สีแดง 4 ดวง ชื่อ DO0 DO1 DO2 DO3 (ตามสติกเกอร์ Cytron)
  สคริปต์จะ toggle GPIO ทีละขา ขาละ 3 วินาที -> ดูว่า LED ดวงไหนกะพริบ

รันบน Raspberry Pi 5 (ต้องใช้ gpiozero + lgpio ซึ่งติดมากับ Pi OS Bookworm):
    python3 find_do_pin.py                # ไล่ทุกขาที่ปลอดภัย
    python3 find_do_pin.py --pins 5 6 16 26   # ไล่เฉพาะขาที่สงสัย
    python3 find_do_pin.py --pin 26 --hold    # ค้าง ON ขาเดียว ไว้วัดไฟ

⚠ ก่อนรัน:
  1. ถอดสายที่ต่อไป PLC ออกก่อน (กันสั่งงานเครื่องจริงโดยไม่ตั้งใจ)
  2. ต่อไฟเลี้ยง HAT 24VDC เข้า Power terminal ให้เรียบร้อย
     (ถ้าไม่จ่ายไฟ LED ฝั่ง output จะไม่ติดแม้ GPIO ทำงานถูก)
"""
import argparse, sys, time

try:
    from gpiozero import LED
    from gpiozero.pins.lgpio import LGPIOFactory
    from gpiozero import Device
    Device.pin_factory = LGPIOFactory()
except ImportError:
    sys.exit("ไม่พบ gpiozero/lgpio\n"
             "ติดตั้งด้วย:  sudo apt install python3-gpiozero python3-lgpio")

# ขาที่ "ห้ามแตะ" เพราะ HAT ใช้ทำอย่างอื่นอยู่แล้ว หรือเป็นขาระบบ
RESERVED = {
    0: "ID_SD (EEPROM ของ HAT+)",
    1: "ID_SC (EEPROM ของ HAT+)",
    2: "I2C SDA",
    3: "I2C SCL",
    12: "RS232 TX (UART4)",
    13: "RS232 RX (UART4)",
    14: "RS485 TX (UART0)",
    15: "RS485 RX (UART0)",
}

# ขาที่น่าจะเป็น DO มากที่สุด (ไล่กลุ่มนี้ก่อนเพื่อประหยัดเวลา)
LIKELY = [5, 6, 16, 26, 17, 27, 22, 23, 24, 25, 4, 7, 8, 9, 10, 11, 18, 19, 20, 21]


def scan(pins, hold_sec, blink_hz):
    print("=" * 66)
    print(" กำลังไล่ทดสอบ GPIO — จ้องดู LED สีแดง DO0 DO1 DO2 DO3 บนบอร์ด")
    print(" กด Ctrl+C เพื่อหยุดทันทีเมื่อเห็น LED กะพริบ")
    print("=" * 66)
    found = []
    for gp in pins:
        if gp in RESERVED:
            print(f"  ข้าม GPIO{gp:<2}  ({RESERVED[gp]})")
            continue
        print(f"\n>>> GPIO{gp:<2}  กะพริบ {hold_sec} วินาที ... ", end="", flush=True)
        try:
            led = LED(gp)
        except Exception as e:
            print(f"เปิดไม่ได้ ({e})")
            continue
        try:
            led.blink(on_time=1 / (2 * blink_hz), off_time=1 / (2 * blink_hz))
            time.sleep(hold_sec)
            led.off()
        except KeyboardInterrupt:
            led.off(); led.close()
            ch = input(f"\n\n  LED ดวงไหนกะพริบ? (0/1/2/3 หรือ Enter=ไม่มี): ").strip()
            if ch in "0123" and ch:
                found.append((f"DO{ch}", gp))
                print(f"  ✓ บันทึก DO{ch} = GPIO{gp}")
            cont = input("  ไล่ต่อไหม? (y/n): ").strip().lower()
            if cont != "y":
                break
            continue
        led.close()
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pins", type=int, nargs="+", default=None,
                    help="ระบุขาที่จะไล่เอง เช่น --pins 5 6 16 26")
    ap.add_argument("--pin", type=int, default=None,
                    help="ทดสอบขาเดียว")
    ap.add_argument("--hold", action="store_true",
                    help="ใช้กับ --pin : ค้าง ON ไว้ (ไว้เอามัลติมิเตอร์วัด)")
    ap.add_argument("--sec", type=float, default=3.0, help="เวลาต่อขา (วินาที)")
    ap.add_argument("--hz", type=float, default=2.0, help="ความถี่กะพริบ")
    a = ap.parse_args()

    # --- โหมดขาเดียว
    if a.pin is not None:
        led = LED(a.pin)
        if a.hold:
            print(f"GPIO{a.pin} = ON ค้างไว้  (Ctrl+C เพื่อปิด)")
            print("ใช้มัลติมิเตอร์วัดระหว่าง DOC กับ DO ที่สงสัย — ถ้านำไฟ = ขาถูก")
            try:
                led.on()
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            print(f"GPIO{a.pin} กะพริบ {a.sec} วินาที")
            led.blink(on_time=0.25, off_time=0.25)
            time.sleep(a.sec)
        led.off(); led.close()
        print("\nปิดแล้ว")
        return

    pins = a.pins if a.pins else LIKELY
    found = scan(pins, a.sec, a.hz)

    print("\n" + "=" * 66)
    if found:
        print(" ผลที่บันทึกได้:")
        for name, gp in sorted(found):
            print(f"   {name} = GPIO{gp}")
        print("\n นำไปใส่ใน plc_out.py ตรงตัวแปร DO_PINS เช่น:")
        print("   DO_PINS = {" + ", ".join(f'"{n}": {g}' for n, g in sorted(found)) + "}")
    else:
        print(" ยังไม่พบ — ลองวิธีนี้ต่อ:")
        print("  1. ตรวจว่าจ่ายไฟ 24VDC เข้า Power terminal ของ HAT แล้วหรือยัง")
        print("  2. โหลด 'Industrial IO HAT+ Datasheet' จากหน้าสินค้า Cytron")
        print("  3. รัน:  gpioinfo | grep -i -E 'do|out'   ดูว่ามี label จาก overlay ไหม")
        print("  4. รัน:  dtoverlay -l   และ  cat /proc/device-tree/hat/product")
    print("=" * 66)


if __name__ == "__main__":
    main()

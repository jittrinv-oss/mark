#!/usr/bin/env bash
# tv_display.sh v2 — ตอนบูต เลือกจอแสดงผลอัตโนมัติ
#   ต่อทีวีอยู่      -> เปิดทีวี  ปิดจอ Argon
#   ไม่ได้ต่อทีวี   -> "เปิดจอ Argon กลับ" (v2: เดิมไม่สั่งเปิด -> ถ้าเคยปิดไว้ จอดำ)
ARGON="HDMI-A-1"     # MS Telematica 1280x800 (จอ Argon 10")
TV="HDMI-A-2"        # LG TV

sleep 3
if [ "$XDG_SESSION_TYPE" = "x11" ]; then
  if xrandr | grep -q "^$TV connected"; then
    xrandr --output "$TV" --auto --primary --output "$ARGON" --off
    echo "$(date) TV ON, Argon OFF (x11)"
  else
    xrandr --output "$ARGON" --auto --primary
    echo "$(date) no TV -> Argon ON (x11)"
  fi
else
  if wlr-randr | grep -q "^$TV "; then
    wlr-randr --output "$TV" --on --pos 0,0 --output "$ARGON" --off
    echo "$(date) TV ON, Argon OFF (wayland)"
  else
    wlr-randr --output "$ARGON" --on --pos 0,0 \
      || wlr-randr --output "$ARGON" --on --mode 1280x800 --pos 0,0
    echo "$(date) no TV -> Argon ON (wayland)"
  fi
fi

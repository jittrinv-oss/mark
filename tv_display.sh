#!/usr/bin/env bash
# tv_display.sh — ตอนบูต: ถ้าต่อทีวีอยู่ ให้ใช้ทีวีเป็นจอแสดงผล (ปิดจอ Argon)
#                  ถ้าไม่ได้ต่อทีวี -> ไม่ทำอะไร (ยังใช้จอ Argon ตามปกติ)
# แก้ชื่อจอ 2 บรรทัดนี้ให้ตรงกับเครื่อง (ดูได้จากคำสั่ง wlr-randr หรือ xrandr)
ARGON="HDMI-A-1"
TV="HDMI-A-2"

sleep 3
if [ "$XDG_SESSION_TYPE" = "x11" ]; then
  if xrandr | grep -q "^$TV connected"; then
    xrandr --output "$TV" --auto --primary --output "$ARGON" --off
    echo "$(date) TV ON (x11)"
  else
    echo "$(date) no TV - keep Argon"
  fi
else
  if wlr-randr | grep -q "^$TV "; then
    wlr-randr --output "$TV" --on --pos 0,0 --output "$ARGON" --off
    echo "$(date) TV ON (wayland)"
  else
    echo "$(date) no TV - keep Argon"
  fi
fi

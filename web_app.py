#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web_app.py — ใช้โปรแกรมตรวจมาร์คผ่าน "เว็บเบราว์เซอร์" (iPad / มือถือ / PC)
ไม่แก้ไฟล์เดิม: เรียกใช้ตัวตรวจเดิม (glyph_check) + ส่วนเสริมที่มีในโฟลเดอร์
  pdf_master (Master จาก PDF), pdf_autofind (Test จาก PDF), prealign, test_polarity

ทำไมต้องเป็นเว็บ: iPad รันโปรแกรมหน้าจอ Tkinter / OpenCV แบบ Windows-Pi ไม่ได้
  -> ให้ Pi (หรือ PC) เป็นเครื่องประมวลผล แล้ว iPad เปิดหน้าเว็บ ถ่ายรูป/เลือกไฟล์ ส่งไปตรวจ

ใช้แค่ Python มาตรฐาน (ไม่ต้องติดตั้งอะไรเพิ่ม)
รัน:  python web_app.py            (พอร์ต 8080)
      python web_app.py 8090       (เปลี่ยนพอร์ต)
iPad: เปิด Safari ไปที่  http://<IP ของ Pi>:8080   (iPad กับ Pi ต้องอยู่วง Wi-Fi/LAN เดียวกัน)
"""
import base64
import csv
import html
import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)
RESULTS = os.path.join(BASE, "results")
LAST_MASTER = os.path.join(RESULTS, "_web_last_master.png")
MAX_UPLOAD = 60 * 1024 * 1024
LOCK = threading.Lock()          # ตรวจทีละงาน (Pi หน่วยความจำจำกัด)


def _opt(name):
    try:
        return __import__(name)
    except Exception:
        return None


glyph_check = _opt("glyph_check")
pdf_master = _opt("pdf_master")
pdf_autofind = _opt("pdf_autofind")
prealign = _opt("prealign")
test_polarity = _opt("test_polarity")


# ------------------------------------------------------------------ helpers
def safe_name(s):
    s = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", (s or "").strip())
    return re.sub(r"\s+", " ", s).strip(" .")[:60] or "_ไม่ระบุโมเดล"


def decode_image(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def is_pdf(data, name):
    return data[:4] == b"%PDF" or (name or "").lower().endswith(".pdf")


def to_tmp_pdf(data):
    fd, p = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return p


def jpg_b64(img, long_side=1100):
    h, w = img.shape[:2]
    s = min(1.0, long_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def imwrite(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)


def load_ignores(model):
    """ใช้โซน IGNORE จาก config ของโมเดล (ถ้ามี results/<โมเดล>/config.json)"""
    p = os.path.join(RESULTS, safe_name(model), "config.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("ignore", [])
    except Exception:
        return []


# ------------------------------------------------------------------ inspection
def run_check(model, master_bytes, master_name, test_bytes, test_name, use_last):
    msgs = []
    # ---- Master
    if master_bytes:
        if is_pdf(master_bytes, master_name):
            if pdf_master is None:
                raise RuntimeError("ไม่มีไฟล์ pdf_master.py — ใช้ Master เป็นภาพแทน")
            p = to_tmp_pdf(master_bytes)
            try:
                master, info = pdf_master.load_pdf_mark(p)
            finally:
                os.remove(p)
            msgs.append(f"Master จาก PDF ({info.get('method', '')}, {info.get('orient', '')})")
        else:
            master = decode_image(master_bytes)
            if master is None:
                raise RuntimeError("อ่านไฟล์ Master ไม่ได้")
        os.makedirs(RESULTS, exist_ok=True)
        imwrite(LAST_MASTER, master)
    elif use_last and os.path.exists(LAST_MASTER):
        master = cv2.imdecode(np.fromfile(LAST_MASTER, np.uint8), cv2.IMREAD_COLOR)
        msgs.append("ใช้ Master เดิม")
    else:
        raise RuntimeError("ยังไม่ได้เลือก Master")

    # ---- Test
    if not test_bytes:
        raise RuntimeError("ยังไม่ได้ถ่ายรูป / เลือกไฟล์ Test")
    if is_pdf(test_bytes, test_name):
        if pdf_autofind is None:
            raise RuntimeError("ไม่มีไฟล์ pdf_autofind.py — ใช้ Test เป็นภาพแทน")
        p = to_tmp_pdf(test_bytes)
        try:
            r = pdf_autofind.find_mark_in_pdf(p, master)
        finally:
            os.remove(p)
        test = r["test"]
        msgs.append(f"Test จาก PDF (กลับด้าน={'ใช่' if r['mirrored'] else 'ไม่'}, หมุน {r['angle']}°)")
    else:
        test = decode_image(test_bytes)
        if test is None:
            raise RuntimeError("อ่านไฟล์ Test ไม่ได้")
    raw_test = test.copy()

    # ---- ส่วนเสริมก่อนตรวจ (เหมือน kiosk)
    if prealign is not None:
        pr = prealign.prealign(master, test)
        if not pr["ok"]:
            return dict(verdict="ถ่ายใหม่", msgs=msgs + pr["problems"], result=None,
                        master=master, test=raw_test, folder=None)
        test = pr["test"]
        msgs += [f"จัดภาพแล้ว (จุดอ้างอิง {pr['inliers']})"] + pr["problems"]
    m2 = master
    if test_polarity is not None:
        try:
            m2, test, pm = test_polarity.prepare_pair(master, test)
            if pm:
                msgs.append(pm)
        except Exception as ex:
            msgs.append(f"แปลงสีไม่สำเร็จ ({ex})")

    # ---- ตรวจด้วยตัวตรวจเดิม
    if glyph_check is None:
        raise RuntimeError("ไม่มีไฟล์ glyph_check.py (ตัวตรวจหลัก) ในโฟลเดอร์")
    ignores = load_ignores(model)
    rep, overlay, aligned, *_ = glyph_check.inspect_glyph(m2, test, ignores, None)
    verdict = rep.get("verdict", "?")
    if hasattr(glyph_check, "draw_result"):
        res_img = glyph_check.draw_result(aligned, rep.get("elements", []), rep.get("extras", []), ignores, False)
    else:
        res_img = overlay
    bad = [e for e in rep.get("elements", []) if e.get("status") in ("MISSING", "PARTIAL")]
    msgs.append(f"ชิ้นส่วน {rep.get('n_elements', '?')}  หายทั้งชิ้น {rep.get('n_missing', 0)}  "
                f"จางขาดหาย {rep.get('n_partial', 0)}  align {rep.get('align_method', '')}")

    # ---- บันทึกผล (โครงสร้างเดียวกับ mark_ui_plus)
    mdl = safe_name(model)
    now = time.localtime()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", now)
    folder = os.path.join(RESULTS, mdl, time.strftime("%Y%m%d", now), f"{time.strftime('%H%M%S', now)}_{verdict}_web")
    base, n = folder, 2
    while os.path.exists(folder):
        folder = f"{base}_{n}"
        n += 1
    os.makedirs(folder)
    imwrite(os.path.join(folder, "1_master.png"), master)
    imwrite(os.path.join(folder, "2_test.png"), raw_test)
    imwrite(os.path.join(folder, "3_result.png"), res_img)
    with open(os.path.join(folder, "report.json"), "w", encoding="utf-8") as f:
        json.dump(dict(timestamp=stamp, model=mdl, verdict=verdict, source="web",
                       master_file=master_name or "(เดิม)", test_file=test_name, notes=msgs,
                       defects=[{k: e.get(k) for k in ("id", "kind", "status", "x", "y", "w", "h", "note")}
                                for e in bad]), f, indent=2, ensure_ascii=False)
    row = [stamp, mdl, verdict, rep.get("n_missing"), rep.get("n_partial"),
           " ".join(f"#{e.get('id')}" for e in bad), master_name or "(เดิม)", test_name,
           os.path.relpath(folder, RESULTS)]
    hdr = ["วันเวลา", "โมเดล", "ผล", "หายทั้งชิ้น", "จางขาดหาย", "ตำแหน่ง NG (id)", "Master", "Test", "โฟลเดอร์"]
    for log in (os.path.join(RESULTS, mdl, "log.csv"), os.path.join(RESULTS, "log_all.csv")):
        new = not os.path.exists(log)
        with open(log, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new:
                w.writerow(hdr)
            w.writerow(row)
    return dict(verdict=verdict, msgs=msgs, result=res_img, master=master, test=raw_test,
                folder=os.path.relpath(folder, BASE))


# ------------------------------------------------------------------ HTML
CSS = """
body{font-family:-apple-system,'Noto Sans Thai',Tahoma,sans-serif;margin:0;background:#eef1f5;color:#1b1f24}
header{background:#0b2545;color:#fff;padding:14px 18px;font-size:22px;font-weight:700}
.wrap{padding:14px;max-width:1100px;margin:auto}
.card{background:#fff;border-radius:10px;padding:14px;margin-bottom:12px;box-shadow:0 1px 4px #0002}
label{font-weight:700;font-size:19px;display:block;margin:6px 0}
input[type=text]{font-size:20px;padding:10px;width:100%;box-sizing:border-box;border:1px solid #bbb;border-radius:8px}
input[type=file]{font-size:18px;margin:6px 0}
.btn{display:inline-block;background:#ffd60a;color:#000;font-size:20px;font-weight:700;border:0;border-radius:10px;
     padding:14px 22px;margin:6px 6px 6px 0;cursor:pointer}
.run{background:#1a9e3c;color:#fff;font-size:26px;width:100%;padding:18px}
.v{font-size:72px;font-weight:800;color:#fff;text-align:center;border-radius:12px;padding:10px;margin-bottom:12px}
.OK{background:#1a9e3c}.NG{background:#d62828}.RE{background:#ef8a00}
img{max-width:100%;border-radius:8px}
.row{display:flex;gap:10px;flex-wrap:wrap}.row>div{flex:1;min-width:260px}
.msg{font-size:16px;color:#333;margin:4px 0}.small{color:#666;font-size:15px}
"""


def page_form(model="", note=""):
    has_last = os.path.exists(LAST_MASTER)
    last = (f'<label><input type="checkbox" name="use_last" value="1" checked> ใช้ Master เดิม '
            f'<span class="small">(ไม่ต้องเลือกใหม่)</span></label>') if has_last else ""
    mods = ", ".join(n for n, m in (("glyph_check", glyph_check), ("pdf_master", pdf_master),
                                     ("pdf_autofind", pdf_autofind), ("prealign", prealign),
                                     ("test_polarity", test_polarity)) if m is None)
    warn = f'<p class="msg" style="color:#d62828">⚠ ไม่พบไฟล์: {html.escape(mods)}</p>' if mods else ""
    return f"""<!doctype html><html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ตรวจมาร์ค</title><style>{CSS}</style></head>
<body><header>ระบบตรวจมาร์คกระจก — AATH (เว็บ)</header><div class="wrap">{warn}
<form method="post" action="/inspect" enctype="multipart/form-data" onsubmit="document.getElementById('go').innerText='กำลังตรวจ…';">
<div class="card"><label>โมเดล</label><input type="text" name="model" value="{html.escape(model)}" placeholder="เช่น 3DA-J0"></div>
<div class="card"><label>Master (ภาพ หรือ PDF)</label>
 <input type="file" name="master" accept="image/*,application/pdf,.pdf">{last}</div>
<div class="card"><label>Test (ชิ้นงาน)</label>
 <div class="small">ถ่ายรูปจากกล้อง iPad:</div><input type="file" name="test_cam" accept="image/*" capture="environment">
 <div class="small">หรือเลือกไฟล์ภาพ / PDF:</div><input type="file" name="test" accept="image/*,application/pdf,.pdf"></div>
<button class="btn run" id="go" type="submit">▶ ตรวจสอบ</button></form>
<p class="msg">{html.escape(note)}</p></div></body></html>"""


def page_result(r, model):
    v = r["verdict"]
    cls = "OK" if v == "OK" else ("NG" if v == "NG" else "RE")
    imgs = []
    for title, im in (("Master", r.get("master")), ("Test", r.get("test")), ("ผลตรวจ", r.get("result"))):
        if im is not None:
            imgs.append(f'<div class="card"><b>{title}</b><br><img src="data:image/jpeg;base64,{jpg_b64(im)}"></div>')
    msgs = "".join(f'<p class="msg">• {html.escape(m)}</p>' for m in r["msgs"])
    saved = f'<p class="small">บันทึกที่ {html.escape(r["folder"])}</p>' if r.get("folder") else ""
    return f"""<!doctype html><html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ผล {html.escape(v)}</title><style>{CSS}</style></head>
<body><header>ผลตรวจ — {html.escape(safe_name(model))}</header><div class="wrap">
<div class="v {cls}">{html.escape(v)}</div>
<a class="btn" href="/?model={html.escape(model)}">↩ ตรวจชิ้นถัดไป</a>
<div class="card">{msgs}{saved}</div><div class="row">{''.join(imgs)}</div></div></body></html>"""


# ------------------------------------------------------------------ server
def parse_multipart(handler):
    n = int(handler.headers.get("Content-Length", 0))
    if n > MAX_UPLOAD:
        raise RuntimeError("ไฟล์ใหญ่เกิน 60 MB")
    body = handler.rfile.read(n)
    head = f"Content-Type: {handler.headers.get('Content-Type')}\r\nMIME-Version: 1.0\r\n\r\n".encode()
    msg = BytesParser(policy=policy.default).parsebytes(head + body)
    fields, files = {}, {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        fn = part.get_filename()
        data = part.get_payload(decode=True) or b""
        if fn is not None:
            if data:
                files[name] = (fn, data)
        else:
            fields[name] = data.decode("utf-8", "replace")
    return fields, files


class Handler(BaseHTTPRequestHandler):
    def _send(self, html_text, code=200):
        b = html_text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        m = re.search(r"model=([^&]*)", self.path)
        from urllib.parse import unquote_plus
        self._send(page_form(unquote_plus(m.group(1)) if m else ""))

    def do_POST(self):
        model = ""
        try:
            fields, files = parse_multipart(self)
            model = fields.get("model", "")
            mfn, mdata = files.get("master", (None, None))
            tfn, tdata = files.get("test_cam") or files.get("test") or (None, None)
            with LOCK:
                r = run_check(model, mdata, mfn, tdata, tfn, fields.get("use_last") == "1")
            self._send(page_result(r, model))
        except Exception as ex:
            self._send(page_form(model, f"⚠ {ex}"))

    def log_message(self, fmt, *args):
        sys.stderr.write("%s  %s\n" % (time.strftime("%H:%M:%S"), fmt % args))


def local_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return sorted(ips) or ["<IP ของเครื่องนี้>"]


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print("โปรแกรมตรวจมาร์ค (เว็บ) พร้อมแล้ว — เปิดบน iPad / มือถือ:")
    for ip in local_ips():
        print(f"   http://{ip}:{port}")
    print("ปิด: กด Ctrl+C")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

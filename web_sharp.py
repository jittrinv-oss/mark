#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web_sharp.py v1 — เว็บตรวจมาร์ค (มือถือ / iPad) + ตัวตรวจคมชัด (sharp_check.py)
ไม่แก้ไฟล์เดิม: ใช้หน้าเว็บ/ฟังก์ชันของ web_app.py แล้วเพิ่ม 2 ติ๊ก
  ☑ ใช้ตัวตรวจคมชัด    ☐ ตรวจต่อแม้ภาพเบลอ
รัน:  python web_sharp.py [พอร์ต]   (แทน web_app.py)
"""
import csv, json, os, sys, time
from http.server import ThreadingHTTPServer

import cv2
import web_app as W
import sharp_check as SC

EXTRA_HTML = ('<div class="card"><label><input type="checkbox" name="sharp" value="1" checked> '
              'ใช้ตัวตรวจคมชัด (ภาพเบลอ / กล้อง res ต่ำ)</label>'
              '<label><input type="checkbox" name="cont_blur" value="1"> ตรวจต่อแม้ภาพเบลอ</label></div>')


def page_form(model="", note=""):
    h = W.page_form(model, note)
    return h.replace('<button class="btn run"', EXTRA_HTML + '<button class="btn run"', 1).replace("(เว็บ", "(เว็บ + คมชัด", 1)


def _page_result(r, model, from_js):
    try:
        return W.page_result(r, model, from_js)
    except TypeError:
        return W.page_result(r, model)


def run_sharp(model, mdata, mname, tdata, tname, use_last, cont):
    t0 = time.time()
    msgs = []
    if mdata:
        if W.is_pdf(mdata, mname):
            p = W.to_tmp_pdf(mdata)
            try:
                master, info = W.pdf_master.load_pdf_mark(p)
            finally:
                os.remove(p)
            msgs.append(f"Master จาก PDF ({info.get('method', '')})")
        else:
            master = W.decode_image(mdata)
        os.makedirs(W.RESULTS, exist_ok=True)
        W.imwrite(W.LAST_MASTER, master)
    elif use_last and os.path.exists(W.LAST_MASTER):
        master = cv2.imdecode(__import__("numpy").fromfile(W.LAST_MASTER, "uint8"), cv2.IMREAD_COLOR)
        msgs.append("ใช้ Master เดิม")
    else:
        raise RuntimeError("ยังไม่ได้เลือก Master")
    if not tdata:
        raise RuntimeError("ยังไม่ได้ถ่ายรูป / เลือกไฟล์ Test")
    test = W.decode_image(tdata)
    if test is None:
        raise RuntimeError("อ่านไฟล์ Test ไม่ได้ (ตัวตรวจคมชัดรับเฉพาะรูปภาพ)")
    cfg = SC.load_config()
    mdl = W.safe_name(model)
    now = time.localtime()
    folder = os.path.join(W.RESULTS, mdl, time.strftime("%Y%m%d", now), time.strftime("%H%M%S", now) + "_web_sharp")
    r = SC.inspect(master, test, cfg, W.load_ignores(model), folder + "_debug" if cfg["debug_save"] else None, cont)
    verdict = {"RETAKE": "ถ่ายใหม่"}.get(r["verdict"], r["verdict"])
    msgs += ["ตัวตรวจคมชัด"] + r["notes"]
    out = dict(verdict=verdict, msgs=msgs, result=r["overlay"], master=master, test=test, folder=None,
               secs=time.time() - t0)
    if r["report"] is not None:
        folder = folder.replace("_web_sharp", f"_{r['verdict']}_web_sharp")
        os.makedirs(folder, exist_ok=True)
        W.imwrite(os.path.join(folder, "1_master.png"), master)
        W.imwrite(os.path.join(folder, "2_test.png"), test)
        W.imwrite(os.path.join(folder, "3_result.png"), r["overlay"])
        rep = r["report"]
        bad = [e for e in rep["elements"] if e["status"] != "OK"]
        with open(os.path.join(folder, "report.json"), "w", encoding="utf-8") as f:
            json.dump(dict(verdict=r["verdict"], model=mdl, notes=msgs, defects=bad, extras=rep["extras"]),
                      f, indent=2, ensure_ascii=False)
        row = [time.strftime("%Y-%m-%d %H:%M:%S", now), mdl, r["verdict"], rep["n_missing"], rep["n_partial"],
               " ".join(f"#{e['id']}" for e in bad), mname or "(เดิม)", tname, os.path.relpath(folder, W.RESULTS)]
        for log in (os.path.join(W.RESULTS, mdl, "log.csv"), os.path.join(W.RESULTS, "log_all.csv")):
            new = not os.path.exists(log)
            with open(log, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["วันเวลา", "โมเดล", "ผล", "หายทั้งชิ้น", "จางขาดหาย", "ตำแหน่ง NG (id)", "Master", "Test", "โฟลเดอร์"])
                w.writerow(row)
        out["folder"] = os.path.relpath(folder, W.BASE)
    return out


class Handler(W.Handler):
    def do_GET(self):
        import re
        from urllib.parse import unquote_plus
        m = re.search(r"model=([^&]*)", self.path)
        self._send(page_form(unquote_plus(m.group(1)) if m else ""))

    def do_POST(self):
        model = ""
        try:
            fields, files = W.parse_multipart(self)
            model = fields.get("model", "")
            if fields.get("sharp") != "1":
                return self._legacy(fields, files)
            mfn, mdata = files.get("master", (None, None))
            tfn, tdata = files.get("test_cam") or files.get("test") or (None, None)
            with W.LOCK:
                r = run_sharp(model, mdata, mfn, tdata, tfn, fields.get("use_last") == "1", fields.get("cont_blur") == "1")
            self._send(_page_result(r, model, fields.get("js") == "1"))
        except Exception as ex:
            self._send(page_form(model, f"⚠ {ex}"))

    def _legacy(self, fields, files):
        """ไม่ติ๊กคมชัด = ตรวจด้วยตัวตรวจเดิมของ web_app"""
        mfn, mdata = files.get("master", (None, None))
        tfn, tdata = files.get("test_cam") or files.get("test") or (None, None)
        with W.LOCK:
            r = W.run_check(fields.get("model", ""), mdata, mfn, tdata, tfn, fields.get("use_last") == "1")
        self._send(_page_result(r, fields.get("model", ""), fields.get("js") == "1"))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print("โปรแกรมตรวจมาร์ค (เว็บ + ตัวตรวจคมชัด) พร้อมแล้ว:")
    for ip in W.local_ips():
        print(f"   http://{ip}:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

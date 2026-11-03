#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web_sharp.py v1.6 — เว็บตรวจมาร์ค (มือถือ / iPad) + ตัวตรวจคมชัด (sharp_check.py)
ไม่แก้ไฟล์เดิม: ใช้หน้าเว็บ/ฟังก์ชันของ web_app.py
v1.6: เพิ่ม sharp_guard (SNI ขีดล่างหาย / ขาตัวอักษรหาย ต้องเจอ + กรอบตรงตำแหน่ง)
v1.5: แก้ "align() takes 4 positional arguments but 5 were given" (ใช้ sharp_loose v2)
      ตรวจพลาด -> แสดงข้อความ error พร้อมเหตุผล (ไม่ทิ้งหน้าเปล่า)
v1.4: ไม่มีถ่ายใหม่ / spec หลวม / ไม่มีม่วง / กรอบ OK / match ขั้นต่ำ % / ครอบรูป Test บนเว็บ
รัน:  python web_sharp.py [พอร์ต]   (แทน web_app.py)
"""
import csv, json, os, sys, time, traceback
from http.server import ThreadingHTTPServer
import numpy as np
import cv2
import web_app as W
import sharp_check as SC
try:
    import sharp_fix  # noqa: F401
except Exception as _ex:
    print("sharp_fix:", _ex)
import sharp_loose  # noqa: F401  loose v3 (ลดส้ม/แดง — ไม่แก้)
import sharp_guard  # noqa: F401  guard v1 (จับเส้น/ขีดหายจริง + กรอบตรงตำแหน่ง)

CROP_JS = r"""
<script>
(function(){
 const c=document.getElementById('cc'); if(!c)return; const g=c.getContext('2d'); let img=new Image(), box=null, st=null;
 function setv(){['cx0','cy0','cx1','cy1'].forEach((k,i)=>document.getElementById(k).value=box?box[i].toFixed(4):'');
   document.getElementById('cinfo').innerText=box?'ครอบแล้ว ✔ (ตรวจเฉพาะในกรอบ)':'ยังไม่ครอบ — ตรวจทั้งภาพ';}
 function draw(){ if(!img.naturalWidth)return; c.style.display='block';
   const W=Math.min(1400,img.naturalWidth); c.width=W; c.height=Math.round(W*img.naturalHeight/img.naturalWidth);
   g.drawImage(img,0,0,c.width,c.height);
   if(box){const x0=box[0]*c.width,y0=box[1]*c.height,x1=box[2]*c.width,y1=box[3]*c.height;
     g.fillStyle='rgba(0,0,0,0.45)'; g.fillRect(0,0,c.width,y0); g.fillRect(0,y1,c.width,c.height-y1);
     g.fillRect(0,y0,x0,y1-y0); g.fillRect(x1,y0,c.width-x1,y1-y0);
     g.strokeStyle='#ffd60a'; g.lineWidth=Math.max(3,c.width/250); g.strokeRect(x0,y0,x1-x0,y1-y0);}}
 function pos(e){const r=c.getBoundingClientRect();
   return [Math.min(1,Math.max(0,(e.clientX-r.left)/r.width)),Math.min(1,Math.max(0,(e.clientY-r.top)/r.height))];}
 c.addEventListener('pointerdown',e=>{e.preventDefault(); c.setPointerCapture(e.pointerId); st=pos(e); box=[st[0],st[1],st[0],st[1]];});
 c.addEventListener('pointermove',e=>{if(!st)return; e.preventDefault(); const p=pos(e);
   box=[Math.min(st[0],p[0]),Math.min(st[1],p[1]),Math.max(st[0],p[0]),Math.max(st[1],p[1])]; draw();});
 c.addEventListener('pointerup',e=>{st=null; if(box&&((box[2]-box[0])<0.03||(box[3]-box[1])<0.03))box=null; setv(); draw();});
 window.clrCrop=function(){box=null; setv(); draw();};
 function load(f){ if(!f||!(f.type||'').startsWith('image/'))return; box=null; setv();
   img=new Image(); img.onload=draw; img.src=URL.createObjectURL(f);}
 document.querySelectorAll('input[name=test_cam],input[name=test]').forEach(i=>i.addEventListener('change',e=>load(e.target.files[0])));
 setv();
})();
</script>"""


def extra_html():
    cfg = SC.load_config()
    mm = int(round(100 * float(cfg.get("match_min", 0.5))))
    return ('<div class="card"><label>✂ ครอบรูป Test (ไม่บังคับ)</label>'
            '<div class="small">เลือก/ถ่ายรูป Test ด้านบนก่อน แล้วลากนิ้วบนภาพครอบเฉพาะมาร์ค</div>'
            '<canvas id="cc" style="width:100%;display:none;touch-action:none;border-radius:8px;margin:6px 0"></canvas>'
            '<div class="small" id="cinfo"></div>'
            '<button type="button" class="btn" onclick="clrCrop()">ล้างกรอบ</button>'
            '<input type="hidden" name="cx0" id="cx0"><input type="hidden" name="cy0" id="cy0">'
            '<input type="hidden" name="cx1" id="cx1"><input type="hidden" name="cy1" id="cy1"></div>'
            '<div class="card"><label><input type="checkbox" name="sharp" value="1" checked> '
            'ใช้ตัวตรวจคมชัด</label>'
            f'<label>match ขั้นต่ำ % <input type="number" name="match_min" value="{mm}" min="0" max="100" step="5" '
            'style="font-size:20px;width:90px"></label>'
            '<div class="small">ชิ้นที่ match ถึงเกณฑ์ไม่ขึ้นส้ม/แดง — ลด = alarm น้อยลง (แนะนำ 40–60)</div></div>')


def page_form(model="", note=""):
    h = W.page_form(model, note)
    h = h.replace('<button class="btn run"', extra_html() + '<button class="btn run"', 1)
    h = h.replace("(เว็บ", "(เว็บ + คมชัด", 1)
    return h.replace("</body>", CROP_JS + "</body>", 1)


def _page_result(r, model, from_js):
    try:
        return W.page_result(r, model, from_js)
    except TypeError:
        return W.page_result(r, model)


def crop_frac(img, fields):
    try:
        x0, y0, x1, y1 = (float(fields.get(k, "")) for k in ("cx0", "cy0", "cx1", "cy1"))
    except ValueError:
        return img, ""
    h, w = img.shape[:2]
    X0, X1 = sorted((int(x0 * w), int(x1 * w)))
    Y0, Y1 = sorted((int(y0 * h), int(y1 * h)))
    if X1 - X0 < 20 or Y1 - Y0 < 20:
        return img, ""
    return img[Y0:Y1, X0:X1].copy(), f"ครอบรูป Test {X1 - X0}x{Y1 - Y0}"


def run_sharp(model, mdata, mname, tdata, tname, use_last, fields):
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
        master = cv2.imdecode(np.fromfile(W.LAST_MASTER, np.uint8), cv2.IMREAD_COLOR)
        msgs.append("ใช้ Master เดิม")
    else:
        raise RuntimeError("ยังไม่ได้เลือก Master")
    if not tdata:
        raise RuntimeError("ยังไม่ได้ถ่ายรูป / เลือกไฟล์ Test")
    test = W.decode_image(tdata)
    if test is None:
        raise RuntimeError("อ่านไฟล์ Test ไม่ได้ (ตัวตรวจคมชัดรับเฉพาะรูปภาพ)")
    test, cm = crop_frac(test, fields)
    if cm:
        msgs.append(cm)
    cfg = SC.load_config()
    try:
        mm = max(0, min(100, int(float(fields.get("match_min", "")))))
        if abs(mm / 100.0 - cfg.get("match_min", 0.5)) > 1e-6:
            cfg["match_min"] = mm / 100.0
            SC.save_config(cfg)
    except ValueError:
        pass
    mdl = W.safe_name(model)
    now = time.localtime()
    folder = os.path.join(W.RESULTS, mdl, time.strftime("%Y%m%d", now), time.strftime("%H%M%S", now) + "_web_sharp")
    r = SC.inspect(master, test, cfg, W.load_ignores(model), folder + "_debug" if cfg.get("debug_save") else None, True)
    verdict = "NG" if r["verdict"] == "RETAKE" else r["verdict"]
    msgs += ["ตัวตรวจคมชัด"] + list(r.get("notes", []))
    out = dict(verdict=verdict, msgs=msgs, result=r.get("overlay"), master=master, test=test, folder=None,
               secs=time.time() - t0)
    rep = r.get("report")
    if rep is not None:
        folder = folder.replace("_web_sharp", f"_{verdict}_web_sharp")
        os.makedirs(folder, exist_ok=True)
        W.imwrite(os.path.join(folder, "1_master.png"), master)
        W.imwrite(os.path.join(folder, "2_test.png"), test)
        if r.get("overlay") is not None:
            W.imwrite(os.path.join(folder, "3_result.png"), r["overlay"])
        bad = [e for e in rep["elements"] if e["status"] in ("MISSING", "PARTIAL")]
        with open(os.path.join(folder, "report.json"), "w", encoding="utf-8") as f:
            json.dump(dict(verdict=verdict, model=mdl, notes=msgs, defects=bad,
                           match_min=cfg.get("match_min")), f, indent=2, ensure_ascii=False, default=str)
        row = [time.strftime("%Y-%m-%d %H:%M:%S", now), mdl, verdict, rep.get("n_missing", 0), rep.get("n_partial", 0),
               " ".join(f"#{e['id']}" for e in bad), mname or "(เดิม)", tname, os.path.relpath(folder, W.RESULTS)]
        for log in (os.path.join(W.RESULTS, mdl, "log.csv"), os.path.join(W.RESULTS, "log_all.csv")):
            os.makedirs(os.path.dirname(log), exist_ok=True)
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
                r = run_sharp(model, mdata, mfn, tdata, tfn, fields.get("use_last") == "1", fields)
            self._send(_page_result(r, model, fields.get("js") == "1"))
        except Exception as ex:
            traceback.print_exc()
            self._send(page_form(model, f"⚠ ตรวจไม่สำเร็จ: {ex}  (ดูรายละเอียดในหน้าต่าง Terminal ของ Pi)"))

    def _legacy(self, fields, files):
        mfn, mdata = files.get("master", (None, None))
        tfn, tdata = files.get("test_cam") or files.get("test") or (None, None)
        with W.LOCK:
            r = W.run_check(fields.get("model", ""), mdata, mfn, tdata, tfn, fields.get("use_last") == "1")
        self._send(_page_result(r, fields.get("model", ""), fields.get("js") == "1"))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"โปรแกรมตรวจมาร์ค (เว็บ + ตัวตรวจคมชัด v1.6, {SC.VERSION}) พร้อมแล้ว:")
    for ip in W.local_ips():
        print(f"   http://{ip}:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

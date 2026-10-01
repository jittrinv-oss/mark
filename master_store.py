#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
master_store.py — ระบบจัดเก็บภาพ Master แยกตามโมเดล (Master Library)

โครงสร้างโฟลเดอร์ที่ใช้
    masters/
      index.json                     <- ทะเบียนรวมทุกโมเดล (ไฟล์เดียว)
      640A-3R0XL/
          current -> v003            <- ชี้ว่าเวอร์ชันไหนกำลังใช้งาน (เก็บใน meta.json)
          v001/  master.png  config.json  meta.json
          v002/  master.png  config.json  meta.json
          v003/  master.png  config.json  meta.json
      RS00-M-3RX-A3/
          v001/  ...

หลักการสำคัญ
  1) ชื่อโฟลเดอร์ = "Model Name" ที่แสดงบนจอ GOT1000 เป๊ะ ๆ
     -> operator เทียบด้วยตาได้ทันที ไม่ต้องแปลงรหัส
  2) ห้ามทับของเดิม — แก้ Master ใหม่ = สร้าง version ใหม่เสมอ
     -> ย้อนกลับได้ถ้าเวอร์ชันใหม่มีปัญหา และตรวจสอบย้อนหลังได้ (traceability)
  3) ทุกเวอร์ชันต้องบันทึกว่า "ใครอนุมัติ / เลขใบ first piece / เหตุผล"
  4) index.json ทำให้ค้นหาและแสดงรายการบนจอสัมผัสได้เร็ว ไม่ต้องไล่สแกนโฟลเดอร์

ใช้เป็นโมดูล
    from master_store import MasterStore
    ms = MasterStore("masters")
    ms.add("640A-3R0XL", img_bgr, cfg_dict,
           approved_by="QA: สมชาย", fp_doc="FP-2609-0142",
           reason="สร้างครั้งแรก")
    img, cfg, meta = ms.load("640A-3R0XL")      # ได้เวอร์ชันที่ตั้งเป็น current

ใช้จากคำสั่ง
    python3 master_store.py --list
    python3 master_store.py --info 640A-3R0XL
    python3 master_store.py --add 640A-3R0XL --image shot.png --config cfg.json \
            --by "QA: สมชาย" --fp FP-2609-0142 --reason "หน้าจอสกรีนใหม่"
    python3 master_store.py --rollback 640A-3R0XL --to v002
    python3 master_store.py --backup /mnt/usb/master_backup
"""
import argparse, json, os, shutil, sys
from datetime import datetime

import numpy as np
import cv2

INDEX = "index.json"


def _read(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def _write(path, img):
    cv2.imencode(".png", img)[1].tofile(path)


def safe_name(model: str) -> str:
    """กันอักขระที่ใช้เป็นชื่อโฟลเดอร์ไม่ได้ แต่ยังคงอ่านออกเหมือนเดิม"""
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in model.strip())
    return out.rstrip(". ")


class MasterStore:
    def __init__(self, root="masters"):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.ipath = os.path.join(self.root, INDEX)
        self.index = self._load_index()

    # ---------------------------------------------------------- index
    def _load_index(self):
        if os.path.exists(self.ipath):
            try:
                return json.load(open(self.ipath, encoding="utf-8"))
            except Exception:
                print("! index.json เสียหาย — สร้างใหม่จากโฟลเดอร์")
        return self._rebuild()

    def _save_index(self):
        json.dump(self.index, open(self.ipath, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)

    def _rebuild(self):
        """สร้าง index ใหม่จากไฟล์ meta.json ที่มีอยู่จริง (ใช้กู้เมื่อ index หาย)"""
        idx = {"updated": datetime.now().isoformat(timespec="seconds"), "models": {}}
        for d in sorted(os.listdir(self.root)):
            mdir = os.path.join(self.root, d)
            if not os.path.isdir(mdir):
                continue
            vers = sorted(v for v in os.listdir(mdir)
                          if v.startswith("v") and
                          os.path.isfile(os.path.join(mdir, v, "meta.json")))
            if not vers:
                continue
            cur = vers[-1]
            for v in vers:
                m = json.load(open(os.path.join(mdir, v, "meta.json"), encoding="utf-8"))
                if m.get("is_current"):
                    cur = v
            meta = json.load(open(os.path.join(mdir, cur, "meta.json"), encoding="utf-8"))
            idx["models"][d] = {"current": cur, "versions": vers,
                                "model_name": meta.get("model_name", d),
                                "approved_by": meta.get("approved_by", ""),
                                "created": meta.get("created", "")}
        return idx

    # ---------------------------------------------------------- write
    def add(self, model, image_bgr, config=None, approved_by="", fp_doc="",
            reason="", note="", set_current=True):
        """
        เพิ่ม Master เวอร์ชันใหม่ของโมเดลนี้ (ไม่ทับของเดิม)
        config = dict ที่มี key 'ignore' และ/หรือ 'rois'
        """
        folder = safe_name(model)
        mdir = os.path.join(self.root, folder)
        os.makedirs(mdir, exist_ok=True)

        exist = sorted(v for v in os.listdir(mdir) if v.startswith("v"))
        ver = f"v{len(exist) + 1:03d}"
        vdir = os.path.join(mdir, ver)
        os.makedirs(vdir, exist_ok=True)

        _write(os.path.join(vdir, "master.png"), image_bgr)
        cfg = config or {"ignore": [], "rois": []}
        json.dump(cfg, open(os.path.join(vdir, "config.json"), "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)

        h, w = image_bgr.shape[:2]
        meta = {
            "model_name": model,
            "version": ver,
            "created": datetime.now().isoformat(timespec="seconds"),
            "approved_by": approved_by,
            "first_piece_doc": fp_doc,
            "reason": reason,
            "note": note,
            "image_size": [int(w), int(h)],
            "n_ignore_zones": len(cfg.get("ignore", [])),
            "n_rois": len(cfg.get("rois", [])),
            "is_current": bool(set_current),
        }
        json.dump(meta, open(os.path.join(vdir, "meta.json"), "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)

        if set_current:
            self._mark_current(folder, ver)

        self.index = self._rebuild()
        self.index["updated"] = datetime.now().isoformat(timespec="seconds")
        self._save_index()
        return ver

    def _mark_current(self, folder, ver):
        mdir = os.path.join(self.root, folder)
        for v in os.listdir(mdir):
            mp = os.path.join(mdir, v, "meta.json")
            if os.path.isfile(mp):
                m = json.load(open(mp, encoding="utf-8"))
                m["is_current"] = (v == ver)
                json.dump(m, open(mp, "w", encoding="utf-8"),
                          indent=2, ensure_ascii=False)

    def rollback(self, model, to_version):
        folder = safe_name(model)
        vdir = os.path.join(self.root, folder, to_version)
        if not os.path.isdir(vdir):
            raise FileNotFoundError(f"ไม่พบเวอร์ชัน {to_version} ของ {model}")
        self._mark_current(folder, to_version)
        self.index = self._rebuild()
        self._save_index()
        return to_version

    # ---------------------------------------------------------- read
    def models(self):
        """รายชื่อโมเดลทั้งหมด (เรียงตามตัวอักษร) สำหรับแสดงบนจอสัมผัส"""
        return sorted(self.index.get("models", {}).keys())

    def search(self, keyword):
        k = keyword.strip().lower()
        return [m for m in self.models() if k in m.lower()]

    def load(self, model, version=None):
        """คืน (image_bgr, config_dict, meta_dict) ของเวอร์ชันที่ใช้งานอยู่"""
        folder = safe_name(model)
        info = self.index["models"].get(folder)
        if info is None:
            raise FileNotFoundError(f"ยังไม่มี Master ของรุ่น '{model}'")
        ver = version or info["current"]
        vdir = os.path.join(self.root, folder, ver)
        img = _read(os.path.join(vdir, "master.png"))
        cfg = json.load(open(os.path.join(vdir, "config.json"), encoding="utf-8"))
        meta = json.load(open(os.path.join(vdir, "meta.json"), encoding="utf-8"))
        return img, cfg, meta

    def info(self, model):
        folder = safe_name(model)
        info = self.index["models"].get(folder)
        if info is None:
            return None
        out = dict(info); out["history"] = []
        for v in info["versions"]:
            mp = os.path.join(self.root, folder, v, "meta.json")
            out["history"].append(json.load(open(mp, encoding="utf-8")))
        return out

    # ---------------------------------------------------------- backup
    def backup(self, dest):
        """คัดลอกทั้งคลังไปยังปลายทาง (USB / network drive) พร้อมประทับวันที่"""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = os.path.join(dest, f"masters_backup_{stamp}")
        shutil.copytree(self.root, target)
        return target


# ================================================================== CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="masters")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--info")
    ap.add_argument("--add")
    ap.add_argument("--image")
    ap.add_argument("--config")
    ap.add_argument("--by", default="")
    ap.add_argument("--fp", default="")
    ap.add_argument("--reason", default="")
    ap.add_argument("--rollback")
    ap.add_argument("--to")
    ap.add_argument("--backup")
    a = ap.parse_args()

    ms = MasterStore(a.root)

    if a.list:
        ms_all = ms.models()
        print(f"คลัง Master: {ms.root}   ทั้งหมด {len(ms_all)} รุ่น\n")
        print(f'{"Model Name":<26}{"ใช้อยู่":<8}{"เวอร์ชัน":<10}{"อนุมัติโดย"}')
        print("-" * 72)
        for m in ms_all:
            i = ms.index["models"][m]
            print(f'{m:<26}{i["current"]:<8}{len(i["versions"]):<10}{i["approved_by"]}')

    elif a.info:
        d = ms.info(a.info)
        if not d:
            sys.exit(f"ไม่พบรุ่น {a.info}")
        print(f'รุ่น: {a.info}   ใช้เวอร์ชัน: {d["current"]}\n')
        print(f'{"Ver":<7}{"วันที่สร้าง":<21}{"อนุมัติโดย":<20}{"ใบ FP":<16}เหตุผล')
        print("-" * 92)
        for h in d["history"]:
            mark = "★" if h.get("is_current") else " "
            print(f'{mark}{h["version"]:<6}{h["created"]:<21}'
                  f'{h["approved_by"]:<20}{h["first_piece_doc"]:<16}{h["reason"]}')

    elif a.add:
        if not a.image:
            sys.exit("ต้องระบุ --image")
        img = _read(a.image)
        if img is None:
            sys.exit(f"อ่านภาพไม่ได้: {a.image}")
        cfg = json.load(open(a.config, encoding="utf-8")) if a.config else None
        v = ms.add(a.add, img, cfg, a.by, a.fp, a.reason)
        print(f"เพิ่ม Master ของ '{a.add}' เป็นเวอร์ชัน {v} และตั้งเป็นเวอร์ชันใช้งาน")

    elif a.rollback:
        if not a.to:
            sys.exit("ต้องระบุ --to เช่น --to v002")
        print(f"ย้อนกลับ '{a.rollback}' ไปใช้ {ms.rollback(a.rollback, a.to)}")

    elif a.backup:
        print("สำรองข้อมูลไปที่:", ms.backup(a.backup))

    else:
        ap.print_help()


if __name__ == "__main__":
    main()

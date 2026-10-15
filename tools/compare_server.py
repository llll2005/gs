# -*- coding: utf-8 -*-
"""Local web tool: slider comparison of renders across training runs.

Scans outputs/**/{val,test}/epoch=*-step=*/ for saved eval images (the save_val
format: a WIDE png = GT | render side-by-side) and serves a page where 2-3
layers are stacked with draggable dividers (juxtapose style). Each pane selects
its source run / checkpoint / GT-or-render independently; the image id is
shared. Stdlib + PIL only.

用法:  python tools/compare_server.py [--port 8090] [--root outputs/lab] [--defaults run[:left|right] ...]
Then open http://localhost:8090

2026-10-15（使用者）：預設 root 改 outputs/lab（lab 跑次用 `lab.py renders <塊> <跑次...>` 抓回來）；
打開時預設 P1＝官方 release 的 GT、P2＝官方 release 渲染、P3＝我方 best1 渲染（官方圖由 scripts/lab/task_official_renders.sh 產生），
三格都可以再改；`--defaults` 可自訂（每項 `跑次/block_N[:left|right]`，left＝GT）。
"""
import argparse
import glob
import io
import json
import os
import re
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from PIL import Image

ROOT = "outputs"
DEFAULTS = None


def scan_catalog():
    """catalog[run][ckpt] = sorted list of image names"""
    cat = {}
    pats = [
        os.path.join(ROOT, "*", "val", "epoch=*"),
        os.path.join(ROOT, "*", "test", "epoch=*"),
        os.path.join(ROOT, "*", "blocks", "*", "val", "epoch=*"),
        os.path.join(ROOT, "*", "blocks", "*", "test", "epoch=*"),
    ]
    for pat in pats:
        for d in glob.glob(pat):
            imgs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "*.png")))
            if not imgs:
                continue
            rel = os.path.relpath(d, ROOT)
            parts = rel.split(os.sep)
            if parts[1] == "blocks":  # run/blocks/block_N/val/epoch...
                run = parts[0] + "/" + parts[2]
                ckpt = parts[3] + "/" + parts[4]
            else:                      # run/val/epoch...
                run = parts[0]
                ckpt = parts[1] + "/" + parts[2]
            cat.setdefault(run, {})[ckpt] = imgs
    return cat


def default_panes(cat):
    """[[run, half], ...]：--defaults 優先；否則挑第一個有官方 release 圖的塊（6、12、10、其他）：官方 GT／官方渲染／我方 best1（沒有就 full44）渲染。"""
    if DEFAULTS:
        out = []
        for d in DEFAULTS:
            run, _, half = d.partition(":")
            if run in cat:
                out.append([run, half or "right"])
        return out
    offs = [r for r in cat if r.startswith("official_release/")]
    if not offs:
        return []
    pref = {"block_6": 0, "block_12": 1, "block_10": 2}
    off = sorted(offs, key=lambda r: (pref.get(r.split("/", 1)[1], 9), r))[0]
    blk = off.split("/", 1)[1]
    out = [[off, "left"], [off, "right"]]
    for ours in ("g44_best1", "full44_best"):
        if f"{ours}/{blk}" in cat:
            out.append([f"{ours}/{blk}", "right"]); break
    return out


def img_path(run, ckpt, name):
    if "/" in run:  # block run
        base, block = run.split("/", 1)
        p = os.path.join(ROOT, base, "blocks", block, ckpt, name)
    else:
        p = os.path.join(ROOT, run, ckpt, name)
    p = os.path.normpath(p)
    if not p.startswith(os.path.normpath(ROOT) + os.sep):
        raise ValueError("bad path")
    return p


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Render Compare</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#ddd}
#bar{padding:8px 12px;background:#1c1c1c;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
select,button{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:4px;padding:3px 6px}
.pane-ctl{display:flex;gap:4px;align-items:center;border:1px solid #333;border-radius:6px;padding:4px 6px}
.tag{font-size:11px;color:#8ac}
#stage{position:relative;margin:10px auto;max-width:98vw;user-select:none}
#stage img{position:absolute;top:0;left:0;width:100%;display:block;pointer-events:none}
#stage img.base{position:relative}
.divider{position:absolute;top:0;bottom:0;width:3px;background:#fff;cursor:ew-resize;z-index:10;box-shadow:0 0 4px #000}
.divider::after{content:"";position:absolute;top:50%;left:-8px;width:19px;height:26px;border-radius:5px;background:#fff;opacity:.85}
.lab{position:absolute;top:6px;z-index:9;background:#000a;padding:2px 8px;border-radius:4px;font-size:12px;pointer-events:none}
</style></head><body>
<div id="bar">
  <label>層數 <select id="np"><option>2</option><option>3</option></select></label>
  <label>圖片 <select id="imgsel"></select></label>
  <span id="panes"></span>
  <button onclick="prevImg(-1)">◀</button><button onclick="prevImg(1)">▶</button>
</div>
<div id="stage"></div>
<script>
let CAT={}, DEF=[], layers=[], divs=[];
const q=s=>document.querySelector(s);
async function boot(){
  CAT=await (await fetch('/api/catalog')).json();
  DEF=await (await fetch('/api/defaults')).json();
  if(DEF.length>=2) q('#np').value=String(Math.min(3,DEF.length));
  q('#np').onchange=buildPanes; buildPanes();
}
function paneCtl(i){
  const runs=Object.keys(CAT).sort();
  const d=document.createElement('span'); d.className='pane-ctl';
  d.innerHTML=`<span class="tag">P${i+1}</span>
   <select class="run"></select><select class="ckpt"></select>
   <select class="half"><option value="right">render</option><option value="left">GT</option></select>`;
  const rs=d.querySelector('.run'), cs=d.querySelector('.ckpt'), hs=d.querySelector('.half');
  runs.forEach(r=>rs.add(new Option(r,r)));
  rs.selectedIndex=Math.min(i,runs.length-1);
  const df=DEF[i]; if(df && CAT[df[0]]) rs.value=df[0];      // 預設（可改）
  function fillCk(){ cs.innerHTML='';
    Object.keys(CAT[rs.value]).sort().forEach(c=>cs.add(new Option(c,c)));
    cs.selectedIndex=cs.length-1; fillImgs(); }
  rs.onchange=fillCk; cs.onchange=()=>{fillImgs()}; hs.onchange=render;
  if(i===0) hs.value='left';           // 最左預設 GT
  if(df && CAT[df[0]]) hs.value=df[1];
  fillCk();
  return d;
}
function fillImgs(){
  const p0=document.querySelectorAll('.pane-ctl')[0]; if(!p0) return;
  const r=p0.querySelector('.run').value, c=p0.querySelector('.ckpt').value;
  const sel=q('#imgsel'), cur=sel.value; sel.innerHTML='';
  (CAT[r][c]||[]).forEach(n=>sel.add(new Option(n,n)));
  if([...sel.options].some(o=>o.value===cur)) sel.value=cur;
  sel.onchange=render; render();
}
function buildPanes(){
  const n=+q('#np').value, host=q('#panes'); host.innerHTML='';
  for(let i=0;i<n;i++) host.appendChild(paneCtl(i));
  fillImgs();
}
function render(){
  const n=+q('#np').value, name=q('#imgsel').value; if(!name) return;
  const stage=q('#stage'); stage.innerHTML=''; layers=[]; divs=[];
  const ctls=[...document.querySelectorAll('.pane-ctl')];
  ctls.forEach((c,i)=>{
    const img=new Image();
    img.src=`/img?run=${encodeURIComponent(c.querySelector('.run').value)}&ckpt=${encodeURIComponent(c.querySelector('.ckpt').value)}&name=${encodeURIComponent(name)}&half=${c.querySelector('.half').value}`;
    if(i===0) img.className='base';
    // 2026-10-15：該跑次沒有這張圖時，原本 <img> 載入失敗＝透明 => 看到的是底下那層（使用者把 GT 誤認成 speed3）
    //   => 換成灰底＋說明，並在標籤上標「沒有這張」
    img.onerror=()=>{ img.onerror=null;
      img.src='data:image/svg+xml;utf8,'+encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900"><rect width="100%" height="100%" fill="#777"/><text x="50%" y="50%" font-size="48" fill="#fff" text-anchor="middle">${c.querySelector('.run').value}：沒有 ${name}</text></svg>`);
      const lb=divs[i]; if(lb) lb.textContent+='（沒有這張）'; };
    stage.appendChild(img); layers.push(img);
    const lab=document.createElement('div'); lab.className='lab';
    lab.textContent=`${c.querySelector('.run').value.split('/')[0]} · ${c.querySelector('.half').value==='left'?'GT':'render'}`;
    stage.appendChild(lab); divs.push(lab);
  });
  for(let i=1;i<n;i++){
    const dv=document.createElement('div'); dv.className='divider'; dv.dataset.i=i;
    stage.appendChild(dv); dragify(dv);
  }
  layout(Array.from({length:n-1},(_,k)=>(k+1)/n*100));
}
function layout(cuts){
  const n=layers.length, all=[0,...cuts,100];
  layers.forEach((im,i)=>{ if(i>0) im.style.clipPath=`inset(0 ${100-all[i+1]}% 0 ${all[i]}%)`; });
  document.querySelectorAll('.divider').forEach((dv,k)=>{ dv.style.left=`calc(${cuts[k]}% - 1px)`; });
  const labs=document.querySelectorAll('.lab');
  labs.forEach((l,i)=>{ l.style.left=`calc(${all[i]}% + 8px)`; });
  window._cuts=cuts;
}
function dragify(dv){
  dv.onpointerdown=e=>{
    dv.setPointerCapture(e.pointerId);
    dv.onpointermove=ev=>{
      const r=q('#stage').getBoundingClientRect();
      let pct=(ev.clientX-r.left)/r.width*100;
      const k=+dv.dataset.i-1, cuts=[...window._cuts];
      const lo=k>0?cuts[k-1]+2:2, hi=k<cuts.length-1?cuts[k+1]-2:98;
      cuts[k]=Math.min(hi,Math.max(lo,pct)); layout(cuts);
    };
    dv.onpointerup=()=>{dv.onpointermove=null};
  };
}
function prevImg(d){
  const s=q('#imgsel'); s.selectedIndex=Math.min(s.length-1,Math.max(0,s.selectedIndex+d)); render();
}
boot();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == "/":
                self._send(200, "text/html; charset=utf-8", PAGE.encode())
            elif u.path == "/api/catalog":
                self._send(200, "application/json", json.dumps(scan_catalog()).encode())
            elif u.path == "/api/defaults":
                self._send(200, "application/json", json.dumps(default_panes(scan_catalog())).encode())
            elif u.path == "/img":
                qs = parse_qs(u.query)
                p = img_path(qs["run"][0], qs["ckpt"][0], qs["name"][0])
                half = qs.get("half", ["full"])[0]
                im = Image.open(p)
                w, h = im.size
                if half == "left":
                    im = im.crop((0, 0, w // 2, h))
                elif half == "right":
                    im = im.crop((w // 2, 0, w, h))
                buf = io.BytesIO()
                im.save(buf, "PNG")
                self._send(200, "image/png", buf.getvalue())
            else:
                self._send(404, "text/plain", b"not found")
        except Exception as e:
            self._send(500, "text/plain", str(e).encode())


def main():
    global ROOT, DEFAULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--root", default="outputs/lab" if os.path.isdir("outputs/lab") else "outputs")
    ap.add_argument("--defaults", nargs="*", default=None, help="打開時各層的預設：跑次/block_N[:left|right]（left＝GT）")
    args = ap.parse_args()
    ROOT = args.root; DEFAULTS = args.defaults
    cat = scan_catalog()
    n_imgs = sum(len(v) for r in cat.values() for v in r.values())
    print("catalog: %d runs, %d image files" % (len(cat), n_imgs))
    print("root %s；預設層：%s" % (ROOT, default_panes(cat) or "（沒有官方圖 => 依名稱排序）"))
    print("open  http://localhost:%d" % args.port)
    HTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()

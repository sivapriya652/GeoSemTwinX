"""Digital-twin dashboard: one self-contained HTML file (Plotly from CDN, data embedded).

Sections: KPIs · interactive 3D twin (map, assets, trajectories) · asset inspector + spatial query ·
filterable asset table · change log + version history · comparative studies · system performance.
"""
import base64
import datetime as _dt
import json
from pathlib import Path

import numpy as np

CLASS_COLORS = ["#1F5BD8", "#D9731A", "#2A9D7F", "#A04FB0", "#C23B5A", "#5B7F1F", "#8A6D3B", "#3A8FB7"]


def _img(path):
    p = Path(path)
    if not p.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def _df_table(df, floatfmt=3):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(floatfmt)
    return dict(columns=[str(c) for c in d.columns], rows=d.astype(object).where(d.notna(), None).values.tolist())


def study(title, note="", df=None, image=None, highlight="GASF (ours)"):
    """Helper to describe one comparative-study card."""
    return dict(title=title, note=note, table=_df_table(df) if df is not None else None, image=_img(image) if image else None,
                highlight=highlight)


def build_dashboard(path, twin, map_points=None, map_colors=None, traj_est=None, traj_gt=None, studies=(),
                    perf=None, title="GeoSemTwin-X Digital Twin", subtitle="", max_points=40000, inline_plotly=True):
    """inline_plotly=True embeds plotly.js (~4.6 MB) so the file also works offline / after download."""
    tw = twin.to_dict() if hasattr(twin, "to_dict") else twin
    pts = None
    if map_points is not None and len(map_points):
        P = np.asarray(map_points); idx = np.random.default_rng(0).choice(len(P), min(max_points, len(P)), replace=False)
        P = P[idx]
        C = (np.asarray(map_colors)[idx] * 255).astype(int) if map_colors is not None else np.full((len(P), 3), 150)
        pts = dict(x=np.round(P[:, 0], 3).tolist(), y=np.round(P[:, 1], 3).tolist(), z=np.round(P[:, 2], 3).tolist(),
                   c=[f"rgb({r},{g},{b})" for r, g, b in C])
    labels = sorted({a["label"] for a in tw["assets"]})
    data = dict(title=title, subtitle=subtitle, generated=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"), twin=tw,
                map=pts, traj_est=np.round(np.asarray(traj_est), 3).tolist() if traj_est is not None else None,
                traj_gt=np.round(np.asarray(traj_gt), 3).tolist() if traj_gt is not None else None,
                colors={l: CLASS_COLORS[i % len(CLASS_COLORS)] for i, l in enumerate(labels)},
                studies=list(studies), perf=perf or {})
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data, default=float).replace("</", "<\\/"))
    if inline_plotly:
        try:
            import plotly
            js = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text(encoding="utf-8")
            html = html.replace('<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
                                "<script>" + js.replace("</script", "<\\/script") + "</script>")
        except Exception:
            pass
    Path(path).write_text(html, encoding="utf-8")
    return path


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GeoSemTwin-X Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#f4f6f9;--card:#ffffff;--ink:#1d2433;--muted:#5b6577;--line:#e3e7ee;--accent:#1F5BD8;--accent-soft:#eaf0fc;
--good:#1f7a3a;--warn:#b45309;--bad:#b4232a;--shadow:0 1px 2px rgba(16,24,40,.06),0 1px 3px rgba(16,24,40,.08)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0f141c;--card:#171e29;--ink:#e6eaf2;--muted:#9aa5b8;
--line:#263142;--accent:#6d9cff;--accent-soft:#1c2a45;--shadow:none}}
:root[data-theme="dark"]{--bg:#0f141c;--card:#171e29;--ink:#e6eaf2;--muted:#9aa5b8;--line:#263142;--accent:#6d9cff;--accent-soft:#1c2a45;--shadow:none}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}
header{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;padding:20px 24px 8px;flex-wrap:wrap}
h1{font-size:22px;margin:0}h2{font-size:15px;margin:0 0 10px}.sub{color:var(--muted);font-size:13px}
nav{display:flex;gap:6px;padding:0 24px 12px;flex-wrap:wrap}nav button{border:1px solid var(--line);background:var(--card);color:var(--ink);
padding:7px 14px;border-radius:999px;cursor:pointer;font:inherit}nav button.on{background:var(--accent);border-color:var(--accent);color:#fff}
main{padding:0 24px 32px}.grid{display:grid;gap:14px}.kpis{grid-template-columns:repeat(auto-fit,minmax(150px,1fr));margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;box-shadow:var(--shadow);min-width:0}
.kpi .v{font-size:26px;font-weight:700;letter-spacing:-.01em}.kpi .l{color:var(--muted);font-size:12px}
.two{grid-template-columns:minmax(0,2.1fr) minmax(280px,1fr)}@media(max-width:900px){.two{grid-template-columns:1fr}}
#plot3d{height:560px}.chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}.chip{display:inline-flex;align-items:center;gap:6px;
border:1px solid var(--line);border-radius:999px;padding:3px 10px;font-size:12px;cursor:pointer;user-select:none;background:var(--card)}
.chip.off{opacity:.4}.dot{width:9px;height:9px;border-radius:50%}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--muted);font-weight:600;cursor:pointer;position:sticky;top:0;background:var(--card)}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.sel td{background:var(--accent-soft)}tbody tr:hover td{background:var(--accent-soft);cursor:pointer}tr.hl td{font-weight:700}
.scroll{overflow:auto;max-height:420px}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
input,select{font:inherit;color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:6px 8px;min-width:0}
input[type=number]{width:84px}button.act{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:7px 12px;cursor:pointer;font:inherit}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:13px}.kv div:nth-child(odd){color:var(--muted)}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;border:1px solid var(--line)}
.moved{color:var(--warn)}.removed{color:var(--bad)}.added{color:var(--good)}.empty{color:var(--muted);padding:12px 0}
.study img{width:100%;height:auto;border-radius:8px;border:1px solid var(--line);background:#fff}.note{color:var(--muted);font-size:12px;margin:-4px 0 10px}
.studies{grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}@media(max-width:520px){.studies{grid-template-columns:1fr}#plot3d{height:420px}}
.tab{display:none}.tab.on{display:block}.bar{height:8px;border-radius:4px;background:var(--accent)}
</style></head><body>
<header><div><h1 id="title"></h1><div class="sub" id="subtitle"></div></div>
<div class="row"><span class="sub" id="gen"></span><button class="chip" id="theme" title="Toggle theme">◐ Theme</button></div></header>
<nav id="tabs"><button data-t="twin" class="on">Digital twin</button><button data-t="changes">Changes &amp; versions</button>
<button data-t="studies">Comparative studies</button><button data-t="perf">System performance</button></nav>
<main>
<section class="grid kpis" id="kpis"></section>
<div class="tab on" id="tab-twin">
 <div class="grid two">
  <div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">3D twin</h2>
   <label class="sub"><input type="checkbox" id="showmap" checked> map points</label></div>
   <div class="chips" id="classchips"></div><div id="plot3d"></div></div>
  <div class="grid" style="align-content:start">
   <div class="card"><h2>Asset inspector</h2><div id="inspector" class="empty">Click an asset in the table or the 3D view.</div></div>
   <div class="card"><h2>Spatial query</h2>
    <div class="row">x <input type="number" id="qx" step="0.5"> y <input type="number" id="qy" step="0.5"> r <input type="number" id="qr" value="5" step="0.5"></div>
    <div class="row"><select id="qc"></select><button class="act" id="qbtn">Find assets</button></div><div id="qres" class="sub"></div></div>
  </div></div>
 <div class="card" style="margin-top:14px"><div class="row" style="justify-content:space-between"><h2 style="margin:0">Asset registry</h2>
  <div class="row"><input id="search" placeholder="Search ID or class"><select id="fclass"></select>
  <label class="sub">min confidence <input type="number" id="fconf" value="0" min="0" max="1" step="0.05"></label></div></div>
  <div class="scroll"><table id="assets"></table></div></div>
</div>
<div class="tab" id="tab-changes"><div class="grid two">
 <div class="card"><h2>Change log</h2><div class="scroll"><table id="changes"></table></div></div>
 <div class="card"><h2>Version history</h2><div class="scroll"><table id="versions"></table></div></div></div></div>
<div class="tab" id="tab-studies"><div class="grid studies" id="studies"></div></div>
<div class="tab" id="tab-perf"><div class="grid two"><div class="card"><h2>Per-stage latency</h2><div id="perfbars"></div></div>
 <div class="card"><h2>Run details</h2><div class="kv" id="perfkv"></div></div></div></div>
</main>
<script>
const D=/*__DATA__*/;
const $=s=>document.querySelector(s), esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const A=D.twin.assets.filter(a=>a.status!=="removed"), COL=D.colors;
$("#title").textContent=D.title; $("#subtitle").textContent=D.subtitle||D.twin.name; $("#gen").textContent="Generated "+D.generated+" · twin v"+D.twin.version;
document.querySelectorAll("#tabs button").forEach(b=>b.onclick=()=>{document.querySelectorAll("#tabs button").forEach(x=>x.classList.toggle("on",x===b));
 document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("on",t.id==="tab-"+b.dataset.t)); if(b.dataset.t==="twin")Plotly.Plots.resize("plot3d");});
$("#theme").onclick=()=>{const r=document.documentElement;const dark=r.dataset.theme?r.dataset.theme==="dark":matchMedia("(prefers-color-scheme: dark)").matches;r.dataset.theme=dark?"light":"dark";draw();};
// KPIs
const classes=[...new Set(A.map(a=>a.label))].sort(), obs=A.reduce((s,a)=>s+(a.observations||0),0);
const meanConf=A.length?A.reduce((s,a)=>s+(a.confidence||0),0)/A.length:0;
const kp=[["Assets in twin",A.length],["Asset classes",classes.length],["Observations fused",obs],["Mean label confidence",meanConf.toFixed(2)],
 ["Changes logged",(D.twin.changes||[]).length],["Twin versions",(D.twin.history||[]).length]];
if(D.perf.ate_cm!=null)kp.push(["Trajectory error (ATE)",D.perf.ate_cm.toFixed(1)+" cm"]);
$("#kpis").innerHTML=kp.map(([l,v])=>`<div class="card kpi"><div class="v">${esc(v)}</div><div class="l">${esc(l)}</div></div>`).join("");
// class chips + filters
const hidden=new Set();
$("#classchips").innerHTML=classes.map(c=>`<span class="chip" data-c="${esc(c)}"><span class="dot" style="background:${COL[c]}"></span>${esc(c)} (${A.filter(a=>a.label===c).length})</span>`).join("");
document.querySelectorAll("#classchips .chip").forEach(ch=>ch.onclick=()=>{const c=ch.dataset.c;hidden.has(c)?hidden.delete(c):hidden.add(c);ch.classList.toggle("off");draw();});
$("#fclass").innerHTML=`<option value="">All classes</option>`+classes.map(c=>`<option>${esc(c)}</option>`).join("");
$("#qc").innerHTML=`<option value="">Any class</option>`+classes.map(c=>`<option>${esc(c)}</option>`).join("");
let selected=null, query=null;
function boxLines(a){const[x0,y0,z0]=a.bbox_min,[x1,y1,z1]=a.bbox_max;const c=[[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]];
 const e=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];const X=[],Y=[],Z=[];e.forEach(([p,q])=>{X.push(c[p][0],c[q][0],null);Y.push(c[p][1],c[q][1],null);Z.push(c[p][2],c[q][2],null)});return[X,Y,Z];}
function draw(){const dark=getComputedStyle(document.documentElement).getPropertyValue("--bg").trim()!=="#f4f6f9";const ink=getComputedStyle(document.documentElement).getPropertyValue("--ink").trim();
 const tr=[];
 if(D.map&&$("#showmap").checked)tr.push({type:"scatter3d",mode:"markers",x:D.map.x,y:D.map.y,z:D.map.z,marker:{size:1.4,color:D.map.c,opacity:.55},hoverinfo:"skip",name:"map",showlegend:false});
 classes.filter(c=>!hidden.has(c)).forEach(c=>{const L=A.filter(a=>a.label===c);const X=[],Y=[],Z=[];L.forEach(a=>{const[bx,by,bz]=boxLines(a);X.push(...bx);Y.push(...by);Z.push(...bz)});
  tr.push({type:"scatter3d",mode:"lines",x:X,y:Y,z:Z,line:{color:COL[c],width:4},hoverinfo:"skip",showlegend:false});
  tr.push({type:"scatter3d",mode:"markers",x:L.map(a=>a.centroid[0]),y:L.map(a=>a.centroid[1]),z:L.map(a=>a.bbox_max[2]+.1),customdata:L.map(a=>a.id),
   marker:{size:5,color:COL[c],line:{color:"#fff",width:1}},name:c,text:L.map(a=>`${a.id} · ${a.label}<br>conf ${(+a.confidence).toFixed(2)} · obs ${a.observations}<br>size ${a.size.map(v=>(+v).toFixed(2)).join(" × ")} m`),hoverinfo:"text"});});
 if(D.traj_gt)tr.push({type:"scatter3d",mode:"lines",x:D.traj_gt.map(p=>p[0]),y:D.traj_gt.map(p=>p[1]),z:D.traj_gt.map(p=>p[2]),line:{color:"#2a9d8f",width:3},name:"ground-truth path"});
 if(D.traj_est)tr.push({type:"scatter3d",mode:"lines",x:D.traj_est.map(p=>p[0]),y:D.traj_est.map(p=>p[1]),z:D.traj_est.map(p=>p[2]),line:{color:"#e76f51",width:3},name:"estimated path"});
 if(selected){const[bx,by,bz]=boxLines(selected);tr.push({type:"scatter3d",mode:"lines",x:bx,y:by,z:bz,line:{color:"#e11d48",width:9},hoverinfo:"skip",name:"selected "+selected.id});}
 if(query){const t=[...Array(65).keys()].map(i=>2*Math.PI*i/64);tr.push({type:"scatter3d",mode:"lines",x:t.map(a=>query.x+query.r*Math.cos(a)),y:t.map(a=>query.y+query.r*Math.sin(a)),z:t.map(()=>query.z),line:{color:"#f59e0b",width:5,dash:"dash"},name:"query radius",hoverinfo:"skip"});}
 Plotly.react("plot3d",tr,{margin:{l:0,r:0,t:0,b:0},paper_bgcolor:"rgba(0,0,0,0)",font:{color:ink},legend:{orientation:"h",y:1.02,x:0},
  scene:{aspectmode:"data",xaxis:{title:"x (m)",color:ink},yaxis:{title:"y (m)",color:ink},zaxis:{title:"z (m)",color:ink}}},{displaylogo:false,responsive:true});
 const el=$("#plot3d");el.removeAllListeners&&el.removeAllListeners("plotly_click");el.on("plotly_click",e=>{const id=e.points[0].customdata;if(id)select(id);});}
function select(id){selected=A.find(a=>a.id===id);const a=selected;if(!a)return;
 $("#inspector").className="";$("#inspector").innerHTML=`<div class="row"><span class="dot" style="background:${COL[a.label]}"></span><b>${esc(a.id)}</b> <span class="badge">${esc(a.label)}</span></div>
 <div class="kv"><div>Confidence</div><div>${(+a.confidence).toFixed(3)} (entropy ${(+a.label_entropy).toFixed(2)})</div><div>Centroid</div><div>${a.centroid.map(v=>(+v).toFixed(2)).join(", ")} m</div>
 <div>Size</div><div>${a.size.map(v=>(+v).toFixed(2)).join(" × ")} m</div><div>Observations</div><div>${a.observations} from ${a.views} viewpoints</div>
 <div>First / last seen</div><div>${esc(a.first_seen)}<br>${esc(a.last_seen)}</div><div>Versions</div><div>${(a.versions||[]).map(v=>"v"+v.version).join(", ")}</div><div>Status</div><div>${esc(a.status)}</div></div>`;
 renderTable();draw();}
// asset table
let sortKey="id",sortDir=1;
const cols=[["id","ID"],["label","Class"],["confidence","Conf.",1],["observations","Obs.",1],["views","Views",1],["cx","x (m)",1],["cy","y (m)",1],["cz","z (m)",1],["vol","Size (m³)",1]];
function renderTable(){const q=$("#search").value.toLowerCase(),fc=$("#fclass").value,mc=+$("#fconf").value||0;
 let R=A.filter(a=>(!fc||a.label===fc)&&(+a.confidence>=mc)&&(!q||a.id.toLowerCase().includes(q)||a.label.toLowerCase().includes(q)))
  .map(a=>({...a,cx:a.centroid[0],cy:a.centroid[1],cz:a.centroid[2],vol:a.size[0]*a.size[1]*a.size[2]}));
 R.sort((p,q2)=>(p[sortKey]>q2[sortKey]?1:p[sortKey]<q2[sortKey]?-1:0)*sortDir);
 $("#assets").innerHTML=`<thead><tr>${cols.map(([k,l,n])=>`<th data-k="${k}" class="${n?"num":""}">${l}${sortKey===k?(sortDir>0?" ▲":" ▼"):""}</th>`).join("")}</tr></thead><tbody>`+
  (R.length?R.map(a=>`<tr data-id="${esc(a.id)}" class="${selected&&selected.id===a.id?"sel":""}">${cols.map(([k,,n])=>`<td class="${n?"num":""}">${k==="label"?`<span class="dot" style="display:inline-block;background:${COL[a.label]}"></span> ${esc(a.label)}`:n?(+a[k]).toFixed(k==="observations"||k==="views"?0:2):esc(a[k])}</td>`).join("")}</tr>`).join(""):`<tr><td colspan="${cols.length}" class="empty">No assets match the filters.</td></tr>`)+"</tbody>";
 document.querySelectorAll("#assets th").forEach(th=>th.onclick=()=>{const k=th.dataset.k;sortDir=sortKey===k?-sortDir:1;sortKey=k;renderTable();});
 document.querySelectorAll("#assets tbody tr[data-id]").forEach(tr=>tr.onclick=()=>select(tr.dataset.id));}
["#search","#fclass","#fconf"].forEach(s=>$(s).addEventListener("input",renderTable));
$("#showmap").onchange=draw;
// spatial query
const cx=A.length?A.reduce((s,a)=>s+a.centroid[0],0)/A.length:0, cy=A.length?A.reduce((s,a)=>s+a.centroid[1],0)/A.length:0;
$("#qx").value=cx.toFixed(1);$("#qy").value=cy.toFixed(1);
$("#qbtn").onclick=()=>{const x=+$("#qx").value,y=+$("#qy").value,r=+$("#qr").value,c=$("#qc").value;
 const z=A.length?Math.min(...A.map(a=>a.bbox_min[2])):0;query={x,y,r,z};
 const hits=A.filter(a=>(!c||a.label===c)).map(a=>({a,d:Math.hypot(a.centroid[0]-x,a.centroid[1]-y)})).filter(h=>h.d<=r).sort((p,q)=>p.d-q.d);
 $("#qres").innerHTML=hits.length?`${hits.length} asset(s) within ${r} m:<br>`+hits.map(h=>`<a href="#" data-id="${esc(h.a.id)}">${esc(h.a.id)}</a> ${esc(h.a.label)} · ${h.d.toFixed(1)} m`).join("<br>"):"No assets in this radius.";
 document.querySelectorAll("#qres a").forEach(l=>l.onclick=e=>{e.preventDefault();select(l.dataset.id);});draw();};
// changes + versions
const ch=D.twin.changes||[];
$("#changes").innerHTML=ch.length?`<thead><tr><th>Version</th><th>Type</th><th>Class</th><th>Asset</th><th class="num">Detail</th></tr></thead><tbody>`+ch.map(c=>`<tr><td>v${c.version}</td><td class="${esc(c.type)}"><b>${esc(c.type)}</b></td><td>${esc(c.label)}</td><td>${esc(c.b||c.a||"")}</td><td class="num">${c.displacement!=null?(+c.displacement).toFixed(2)+" m":c.chamfer!=null?"Δshape "+(+c.chamfer).toFixed(2)+" m":""}</td></tr>`).join("")+"</tbody>"
 :`<tbody><tr><td class="empty">No changes logged yet. Commit a second mapping session to compare versions.</td></tr></tbody>`;
const hv=D.twin.history||[];
$("#versions").innerHTML=`<thead><tr><th>Version</th><th>Session</th><th>Time</th><th class="num">Assets</th></tr></thead><tbody>`+hv.map(h=>`<tr><td>v${h.version}</td><td>${esc(h.session)}</td><td>${esc(h.time)}</td><td class="num">${h.n_assets}</td></tr>`).join("")+"</tbody>";
// studies
$("#studies").innerHTML=D.studies.length?D.studies.map(s=>`<div class="card study"><h2>${esc(s.title)}</h2>${s.note?`<div class="note">${esc(s.note)}</div>`:""}
 ${s.image?`<img src="${s.image}" alt="${esc(s.title)} chart">`:""}${s.table?`<div class="scroll" style="margin-top:10px;max-height:320px"><table><thead><tr>${s.table.columns.map(c=>`<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${s.table.rows.map(r=>`<tr class="${r.includes(s.highlight)?"hl":""}">${r.map(v=>`<td class="${typeof v==="number"?"num":""}">${v===null?"—":esc(v)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`:""}</div>`).join("")
 :`<div class="card empty">No comparative studies attached to this dashboard.</div>`;
// performance
const st=D.perf.stages_ms||{};const mx=Math.max(1,...Object.values(st));
$("#perfbars").innerHTML=Object.keys(st).length?Object.entries(st).map(([k,v])=>`<div style="margin-bottom:10px"><div class="row" style="justify-content:space-between;margin:0"><span>${esc(k)}</span><span class="sub">${(+v).toFixed(0)} ms</span></div><div class="bar" style="width:${Math.max(2,100*v/mx)}%"></div></div>`).join(""):`<div class="empty">No timing data.</div>`;
$("#perfkv").innerHTML=Object.entries(D.perf.details||{}).map(([k,v])=>`<div>${esc(k)}</div><div>${esc(v)}</div>`).join("")||`<div class="empty">—</div>`;
renderTable();draw();
</script></body></html>"""

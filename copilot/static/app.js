/* Weld Copilot UI - vanilla JS, no build step. Libraries: uPlot 1.6.30, marked 12.0.2, DOMPurify 3.1.6 (vendored). */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) el.append(c.nodeType ? c : document.createTextNode(String(c)));
  return el;
}
async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts,
    body: opts.body && typeof opts.body !== "string" ? JSON.stringify(opts.body) : opts.body });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}
const kids = (...a) => a.flat(9).filter((x) => x != null && x !== false);
const md = (t) => DOMPurify.sanitize(marked.parse(t || "", { breaks: false, gfm: true }));
const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const lsGet = (k, d) => { try { return localStorage.getItem(k) ?? d; } catch { return d; } };
const lsSet = (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } };

const SIG = {
  current_a: { n: "Current", u: "A", d: 0 },
  current2_a: { n: "Current (trail)", u: "A", d: 0 },
  voltage_v: { n: "Voltage", u: "V", d: 1 },
  voltage2_v: { n: "Voltage (trail)", u: "V", d: 1 },
  travel_speed_mmps: { n: "Travel speed", u: "mm/s", d: 2 },
  gas_lpm: { n: "Shielding gas", u: "L/min", d: 1 },
  heat_input_kj_mm: { n: "Heat input", u: "kJ/mm", d: 2 },
  interpass_c: { n: "Interpass", u: "°C", d: 0 },
  wfs_mpm: { n: "Wire feed", u: "m/min", d: 1 },
  wfs2_mpm: { n: "Wire feed (trail)", u: "m/min", d: 1 },
};
const CHART_PARAMS = ["current_a", "voltage_v", "travel_speed_mmps", "gas_lpm", "heat_input_kj_mm", "interpass_c"];
const TRAIL = { current_a: "current2_a", voltage_v: "voltage2_v", wfs_mpm: "wfs2_mpm" };
const COL = { arc: "#5cc8ff", trail: "#c9a6ff", molten: "#ff8a3d", crit: "#ff5d5d", warn: "#f2b33d", ok: "#3ecf8e", grid: "rgba(255,255,255,0.045)", axis: "#5a6573" };
const ICON = { reject: "✕", accept: "✓", review: "!", pending: "…", indications: "✕", no_indications: "✓", not_inspected: "–" };
// NDT indication type colours (strip, timeline, hatching); unknown types get a stable hashed hue
const DEF_COL = { porosity: "#ff6b6b", lack_of_fusion: "#c792ff", slag_inclusion: "#f2b33d", slag: "#f2b33d", metallic_inclusion: "#4fd1c5", crack: "#ff4fa3",
  undercut: "#5cc8ff", incomplete_penetration: "#ff9f43", burn_through: "#ff9f43", arc_instability: "#7ee0c3" };
function defCol(t) {
  if (DEF_COL[t]) return DEF_COL[t];
  let hsh = 0; for (const ch of String(t)) hsh = (hsh * 31 + ch.charCodeAt(0)) % 360;
  return `hsl(${hsh}, 70%, 65%)`;
}
const rgba = (hex, a) => { if (!hex.startsWith("#")) return hex.replace("hsl", "hsla").replace(")", `, ${a})`); const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; };
const prettyDef = (d) => String(d || "").replace(/_/g, " ");
// rt_result -> CSS class (colour) and badge text
const RT_CLASS = { indications: "reject", no_indications: "accept", not_inspected: "pending", reject: "reject", accept: "accept", review: "review" };
function ndtText(rt, counts) {
  counts = counts || {};
  if (rt === "indications") {
    const top = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    return top.length ? "NDT: " + top.slice(0, 2).map(([k, v]) => `${prettyDef(k)} ×${v}`).join(", ") + (top.length > 2 ? " …" : "") : "NDT: indications";
  }
  if (rt === "no_indications") return "NDT clear";
  if (rt === "not_inspected") return "Not inspected";
  if (rt === "accept" || rt === "reject") return `RT ${rt}`;
  return "RT …";
}
function rtBadge(rt, counts, title, short = false) {
  const k = rt || "pending";
  const full = ndtText(k, counts);
  let txt = full;
  if (short && k === "indications") {
    const top = Object.entries(counts || {}).sort((a, b) => b[1] - a[1]);
    txt = top.length ? `${prettyDef(top[0][0]).replace("lack of fusion", "LoF").replace("slag inclusion", "slag")} ×${top[0][1]}${top.length > 1 ? " +" : ""}` : "indications";
  }
  return h("span", { class: `badge ${RT_CLASS[k] || "pending"}`, title: title || full }, `${ICON[k] || ""} ${txt}`);
}

const S = {
  passes: [], seq: null, detail: null, sig: null, view: "torch", win: lsGet("weld.win", "follow"),
  charts: [], offset: 0, dur: 0, sessionId: null, demo: [], demoDone: new Set(), busy: false, data: null,
  filters: { source: "", defectType: "", rt: "", defect: false },
};

/* ===================================================================== tabs */
$$(".tab-btn").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(name) {
  $$(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab-pane").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  if (name === "dataset") loadDataset();
  if (name === "generate") loadGenerate();
  if (name === "flow") loadFlow();
  if (name === "copilot") requestAnimationFrame(resizeCharts);
  location.hash = name === "copilot" ? "" : name;
}

/* ===================================================================== pipeline / health */
async function pollHealth() {
  try {
    const hl = await api("/api/health");
    const strip = $("#pipeline");
    strip.innerHTML = "";
    hl.stages.forEach((s, i) => {
      if (i) strip.append(h("div", { class: "arrow" }, h("i", { style: { animationDelay: `${i * 0.3}s` } })));
      strip.append(h("div", { class: `stage-node ${s.status}`, id: `node-${s.id}`, title: `${s.name}: ${s.detail}\n${s.sub || ""}` },
        h("span", { class: "st" }),
        h("div", { class: "txt" }, h("span", { class: "nm" }, h("span", { class: "num" }, `${i + 1} `), s.name), h("span", { class: "dt" }, s.detail))));
    });
    const mb = $("#modeBadge");
    const m = (hl.mode || "").toUpperCase();
    mb.className = "mode-badge " + (m.includes("LIVE") ? "live" : m.includes("NIM") ? "nim" : "replay");
    $("#modeText").textContent = m || "—";
    $("#llmBadge").textContent = `LLM · ${hl.llm.label}`;
  } catch (e) { console.warn("health", e); }
}
function pulseStage(id, ms = 2500) {
  const n = $(`#node-${id}`);
  if (!n) return;
  n.classList.add("active");
  setTimeout(() => n.classList.remove("active"), ms);
}

/* ===================================================================== shop floor */
const isReal = () => !!S.data?.real;
const srcKey = (p) => (p.synthetic ? "sim" : p.source_id || "unknown");
const SRC_LABEL = { aimen_saw: "AIMEN SAW", bam_tgmaw: "BAM tandem-GMAW", sim: "Simulated" };
async function loadPasses() {
  const r = await api("/api/passes");
  S.passes = r.passes;
  S.data = r.data || S.data;
  const real = isReal();
  $("#floorSource").textContent = real ? `${(S.data.short || []).join(" + ")} · ${r.source === "mosaico_io" ? "via mosaico_io" : "local files"}`
    : `${[...new Set(S.passes.map((p) => p.cell_id).filter(Boolean))].join(", ") || "Cell"} · ${r.source === "mosaico_io" ? "via mosaico_io" : "local files"}`;
  const srcs = [...new Set(S.passes.map(srcKey))];
  $("#fSource").innerHTML = "";
  const SHORT = { aimen_saw: "AIMEN", bam_tgmaw: "BAM", sim: "Sim" };
  if (srcs.length > 1) ["", ...srcs].forEach((v) => $("#fSource").append(h("button", { "data-v": v, class: v === S.filters.source ? "on" : "",
    title: v ? `Only ${SRC_LABEL[v] || v}` : "All datasets" }, v ? (SHORT[v] || v) : "All")));
  $("#fSource").classList.toggle("hidden", srcs.length < 2);
  const types = [...new Set(S.passes.flatMap((p) => Object.keys(p.defect_counts || {}).concat(p.defect_types || [])))].sort();
  $("#fDefectType").innerHTML = '<option value="">All defect types</option>' + types.map((t) => `<option value="${esc(t)}">${esc(prettyDef(t))}</option>`).join("");
  $("#fDefectType").value = S.filters.defectType;
  const rts = [...new Set(S.passes.map((p) => p.rt_result || "pending"))];
  const RT_OPT = { indications: "NDT indications", no_indications: "NDT clear", not_inspected: "Not inspected", reject: "RT reject", accept: "RT accept", pending: "RT pending" };
  $("#fRt").innerHTML = `<option value="">${real ? "NDT: any" : "RT: any"}</option>` + Object.keys(RT_OPT).filter((k) => rts.includes(k)).map((k) => `<option value="${k}">${RT_OPT[k]}</option>`).join("");
  $("#fRt").value = S.filters.rt;
  $("#fDefectLbl").textContent = real ? "NDT indications / excursions only" : "Defects / WPS breaks only";
  renderFloor();
}
function rtClass(p) { return RT_CLASS[p.rt_result] || "pending"; }
function passLabel(p) { return p.pass_label || `P${p.pass_no}`; }
function renderFloor() {
  const f = S.filters;
  const rows = S.passes.filter((p) => (!f.source || srcKey(p) === f.source)
    && (!f.defectType || (p.defect_counts || {})[f.defectType] || (p.defect_types || []).includes(f.defectType))
    && (!f.rt || (p.rt_result || "pending") === f.rt) && (!f.defect || p.has_defect || p.violation_count));
  const all = S.passes;
  const real = isReal();
  $("#floorKpis").innerHTML = "";
  $("#floorKpis").append(
    h("div", { class: "kpi" }, h("div", { class: "v" }, all.length), h("div", { class: "l" }, "Passes")),
    real ? h("div", { class: "kpi crit", title: "Passes where RT/UT found indications" }, h("div", { class: "v" }, all.filter((p) => p.rt_result === "indications").length), h("div", { class: "l" }, "NDT indic."))
      : h("div", { class: "kpi crit" }, h("div", { class: "v" }, all.filter((p) => p.rt_result === "reject").length), h("div", { class: "l" }, "RT reject")),
    h("div", { class: "kpi warn", title: real ? "Passes outside the published-settings window" : "" }, h("div", { class: "v" }, all.filter((p) => p.violation_count).length),
      h("div", { class: "l" }, real ? "Off settings" : "WPS breaks")));
  const list = $("#passList");
  list.innerHTML = "";
  // source -> coupon/workpiece (spool) -> pass
  const tree = {};
  rows.forEach((p) => (((tree[srcKey(p)] ||= {})[p.spool_id] ||= []).push(p)));
  const order = Object.keys(tree).sort((a, b) => (a === "sim") - (b === "sim") || a.localeCompare(b));
  for (const sk of order) {
    const coupons = tree[sk];
    const first = Object.values(coupons)[0][0];
    const sim = sk === "sim";
    if (order.length > 1 || real) list.append(h("div", { class: `src-h ${sim ? "sim" : ""}` },
      h("span", { class: "nm" }, sim ? "WeldCell simulator" : (first.source_short || SRC_LABEL[sk] || sk)),
      sim ? h("span", { class: "badge molten", title: "Synthetic data — not a real weld" }, "SIMULATED")
        : h("span", { class: "lic", title: "Licence — see Data & credits" }, first.license || "")));
    for (const sp of Object.keys(coupons).sort()) {
      const ps = coupons[sp].sort((a, b) => a.pass_no - b.pass_no);
      const nInd = ps.filter((p) => p.rt_result === "indications" || p.rt_result === "reject").length;
      const inspected = ps.filter((p) => p.ndt_inspected).length;
      const kind = sk === "aimen_saw" ? "COUPON" : sk === "bam_tgmaw" ? "WORKPIECE" : "SPOOL";
      const sum = !inspected ? h("span", { class: "badge pending", title: "No NDT on this " + kind.toLowerCase() }, "not inspected")
        : nInd ? h("span", { class: "badge reject" }, `${nInd}/${inspected} ${real ? "w/ indic." : "rej"}`) : h("span", { class: "badge accept" }, "clean");
      const box = h("div", { class: "spool" }, h("div", { class: "spool-h" },
        h("span", {}, `${kind} ${sp}`, h("small", {}, [first.material, first.thickness_mm ? `${first.thickness_mm} mm` : null].filter(Boolean).join(" · "))), sum));
      const jb = h("div", { class: "joint" });
      let lastJoint = null;
      for (const p of ps) {
        if (!real && p.joint_id !== lastJoint) { jb.append(h("div", { class: "joint-h" }, `${p.joint_id}`, h("span", { class: "sub" }, p.position || ""))); lastJoint = p.joint_id; }
        const counts = Object.entries(p.defect_counts || {});
        jb.append(h("div", { class: `prow ${rtClass(p)} ${p.sequence_name === S.seq ? "active" : ""} ${p.synthetic ? "sim" : ""}`, "data-seq": p.sequence_name,
          onclick: () => selectPass(p.sequence_name), title: p.sequence_name },
          h("div", { class: "stripe" }),
          h("div", {}, h("div", { class: "nm" }, `${passLabel(p)} ${p.pass_type || ""}`, h("small", {}, p.process),
              p.synthetic && real ? h("span", { class: "simtag" }, "SIMULATED") : null),
            h("div", { class: "meta" }, h("span", {}, `${fmt(p.duration_s, 0)} s`),
              p.has_video ? h("span", { class: "mi", title: "Process video" }, "▶ video") : null,
              p.has_photo ? h("span", { class: "mi", title: "Bead photo" }, "◩ photo") : null,
              !real && counts.length ? h("span", { style: { color: "#ff9c9c" } }, counts.map(([d]) => prettyDef(d)).join(", ")) : null)),
          h("div", { class: "tags" }, p.violation_count ? h("span", { class: "vtag", title: `Outside ${real ? "published settings" : "WPS"}: ${(p.violated_params || []).join(", ")}` }, `⚠ ${p.violation_count}`) : null,
            real ? rtBadge(p.rt_result, p.defect_counts, null, true) : h("span", { class: `badge ${rtClass(p)}` }, `${ICON[rtClass(p)]} ${rtClass(p) === "pending" ? "RT …" : "RT"}`))));
      }
      box.append(jb);
      list.append(box);
    }
  }
  if (!rows.length) list.append(h("div", { class: "placeholder" }, S.passes.length ? "No passes match" : "No passes yet — run the pipeline, or COPILOT_FIXTURES=real / =1"));
}
$("#fSource").addEventListener("click", (e) => {
  if (e.target.tagName !== "BUTTON") return;
  $$("#fSource button").forEach((b) => b.classList.toggle("on", b === e.target));
  S.filters.source = e.target.dataset.v; renderFloor();
});
$("#fDefectType").onchange = (e) => { S.filters.defectType = e.target.value; renderFloor(); };
$("#fRt").onchange = (e) => { S.filters.rt = e.target.value; renderFloor(); };
$("#fDefect").onchange = (e) => { S.filters.defect = e.target.checked; renderFloor(); };

/* ===================================================================== replay */
const vMain = $("#vMain"), vSecond = $("#vSecond");

async function selectPass(seq, t = null, view = null) {
  if (seq !== S.seq) {
    S.seq = seq;
    $$(".prow").forEach((r) => r.classList.toggle("active", r.dataset.seq === seq));
    const [detail, sig] = await Promise.all([api(`/api/passes/${encodeURIComponent(seq)}`), api(`/api/passes/${encodeURIComponent(seq)}/signals?hz=25`)]);
    S.detail = detail; S.sig = sig; S.dur = sig.t.length ? sig.t[sig.t.length - 1] : (detail.pass.duration_s || 0);
    renderReplayHead();
    renderViewToggle();
    buildCharts();
    renderTimeline();
    const views = availableViews();
    const keep = views.includes(S.view) && !(S.view === "sim2real" && !detail.sim2real) ? S.view : views[0];
    setView(view && views.includes(view) ? view : keep, true);
    $(`.prow[data-seq="${CSS.escape(seq)}"]`)?.scrollIntoView({ block: "nearest" });
  } else if (view) setView(view);
  if (t != null) seek(t);
}

function availableViews() {
  const m = S.detail?.media || {}, synth = !!S.detail?.source?.synthetic;
  const v = [];
  if (m.torch) v.push("torch");
  if (m.photos?.length) v.push("photo");
  if (m.edge) v.push("edge");
  if (m.depth) v.push("depth");
  if (synth && m.edge) v.push("sim2real");
  if (!v.length) v.push("torch");
  return v;
}
function renderViewToggle() {
  const m = S.detail.media || {};
  const real = m.video?.real;
  const lbl = { torch: real ? "Process cam" : "Torch cam", photo: `Bead photo${m.photos?.length > 1 ? "s" : ""}`, edge: "Edge", depth: "Depth", sim2real: "Sim → Real" };
  const box = $("#viewToggle");
  box.innerHTML = "";
  const views = availableViews();
  views.forEach((v) => box.append(h("button", { "data-view": v, class: `${v === "sim2real" ? "s2r" : ""} ${v === S.view ? "on" : ""}`,
    title: v === "edge" && real ? "Canny edges of the real video (Cosmos Transfer control)" : null }, lbl[v])));
  box.classList.toggle("hidden", views.length < 2 && !m.torch && !m.photos);
}

function renderReplayHead() {
  const d = S.detail, p = d.pass || {}, m = d.meta || {}, src = d.source || {};
  const real = !src.synthetic;
  $("#seqTitle").textContent = d.sequence_name;
  const chips = $("#seqChips");
  chips.innerHTML = "";
  const nv = d.violations.length;
  const basis = d.wps_basis || (real ? "" : "WPS from the shop's procedure library (simulated)");
  chips.append(...kids(
    real ? rtBadge(p.rt_result, p.defect_counts, (d.ndt?.methods || []).length ? `NDT: ${(d.ndt.methods || []).join(" + ")}${(d.ndt.report_files || []).length ? ` (${d.ndt.report_files.join(", ")})` : ""}` : "This pass was not NDT-inspected")
      : h("span", { class: `badge ${p.rt_result || "pending"}` }, `${ICON[p.rt_result || "pending"]} RT ${p.rt_result || "pending"}`),
    src.synthetic && S.data?.real ? h("span", { class: "badge molten", title: "Synthetic pass from the WeldCell simulator" }, "SIMULATED") : null,
    h("span", { class: "chip" }, `${p.process} · ${p.pass_type} · ${passLabel(p)}`),
    real ? h("span", { class: "chip" }, `${m.material || p.material}${m.thickness_mm ? ` · ${m.thickness_mm} mm` : ""}`)
      : h("span", { class: "chip" }, m.wps_id || p.wps_id),
    real ? null : h("span", { class: "chip" }, `${m.position || p.position} · ${m.material || p.material}`),
    m.pipe_od_mm ? h("span", { class: "chip" }, `Ø${m.pipe_od_mm} × ${m.wall_mm} mm`) : null,
    (m.electrodes || 1) > 1 ? h("span", { class: "chip", title: "Two wires: lead + trail electrode" }, "tandem · lead + trail") : null,
    String(m.process).toUpperCase() === "SAW" ? h("span", { class: "chip", title: "Submerged arc: the arc is covered by granular flux — no shielding gas" }, "flux-shielded") : null,
    nv ? h("span", { class: "badge warn", title: basis }, `⚠ ${nv} ${real ? "outside published settings" : `WPS violation${nv > 1 ? "s" : ""}`}`)
      : h("span", { class: "badge ok", title: basis }, real ? "✓ within published settings" : "✓ within WPS")));
  const prov = $("#seqProv");
  prov.innerHTML = "";
  if (src.synthetic) {
    prov.append(h("span", { class: "sim" }, "SIMULATED"), " WeldCell simulator — synthetic pass, not a real weld");
  } else {
    prov.append(...kids(h("span", { class: "k" }, "Data"), h("b", {}, src.org ? `${src.short}` : src.title || src.id), " · ",
      h("span", { title: src.authors || "" }, src.title || ""), " · ",
      h("span", { class: "lic" }, src.license || "licence?"),
      src.doi ? [" · ", h("a", { href: `https://doi.org/${src.doi}`, target: "_blank", rel: "noopener" }, `DOI ${src.doi}`)] : null,
      " · ", h("a", { href: "#", onclick: (e) => { e.preventDefault(); openCredits(src.id); } }, "credits"),
      m.fixture ? h("span", { class: "fx", title: "copilot/fixtures_real.py stand-in until the real conversion lands" }, "FIXTURE") : null));
  }
}

function setView(view, force = false) {
  if (!S.detail) return;
  if (view === S.view && !force) return;
  S.view = view;
  $$("#viewToggle button").forEach((b) => b.classList.toggle("on", b.dataset.view === view));
  const media = S.detail.media || {};
  const stage = $("#stage");
  const t = currentT();
  $("#s2rEmpty").classList.add("hidden");
  $(".vid.second").classList.add("hidden");
  stage.classList.remove("split");
  stage.classList.toggle("photo-on", view === "photo");
  $("#photoView").classList.toggle("hidden", view !== "photo");
  vSecond.removeAttribute("src");
  S.offset = 0;
  let src = view === "photo" ? null : media[view === "sim2real" ? "torch" : view];
  const vcap = media.video?.caption || "TORCH CAM · SIMULATOR";
  $("#capMain").textContent = { torch: vcap, edge: media.video?.real ? "CONTROL · EDGE (from the real video)" : "CONTROL · EDGE", depth: "CONTROL · DEPTH", sim2real: "SIMULATOR" }[view] || "";
  $("#capMain").classList.toggle("real", !!media.video?.real && (view === "torch" || view === "edge"));
  if (view === "photo") renderPhotoView();
  if (view === "sim2real") {
    const s2r = S.detail.sim2real;
    if (!s2r) { $("#s2rEmpty").classList.remove("hidden"); }
    else {
      stage.classList.add("split");
      $(".vid.second").classList.remove("hidden");
      src = s2r.control_url || media.edge || media.torch;
      S.offset = s2r.t0_s || 0;
      $("#capMain").textContent = `SIMULATOR · ${(s2r.control_kind || "edge").toUpperCase()} CONTROL`;
      const sn = String(s2r.source || "");
      $("#capSecond").textContent = `COSMOS 3 TRANSFER · ${sn.includes("mock") ? "MOCK" : sn.includes("heuristic") ? "PLACEHOLDER" : sn.toUpperCase()}`;
      vSecond.src = s2r.video_url;
      vMain.loop = vSecond.loop = true;
    }
  } else vMain.loop = false;
  stage.classList.toggle("hud-on", !stage.classList.contains("split") && view !== "photo");
  $("#stageEmpty").classList.toggle("hidden", !!src || view === "photo");
  $("#stageEmpty").textContent = media.photos?.length ? "No video for this pass — see the bead photo" : "No camera video for this pass — signals-only replay";
  if (src) {
    if (vMain.getAttribute("src") !== src) { vMain.src = src; vMain.load(); }
    S.pendingSeek = null;
    vMain.addEventListener("loadedmetadata", () => seek(Math.max(S.pendingSeek ?? t, S.offset)), { once: true });
  } else { if (vMain.getAttribute("src")) { vMain.pause(); vMain.removeAttribute("src"); vMain.load(); } S.vt = t; tick(true); }
}
$("#viewToggle").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b?.dataset.view) setView(b.dataset.view); });
$("#s2rGo").onclick = () => {
  if (!S.seq) return;
  sendChat(`Make photoreal training video from the simulator for ${S.seq}`);
};

/* photo viewer: thumbnails, wheel zoom about the cursor, drag to pan, double-click to fit */
const PV = { i: 0, z: 1, x: 0, y: 0, fit: 1, drag: null };
function renderPhotoView() {
  const m = S.detail.media || {}, photos = m.photos || [];
  if (PV.seq !== S.seq) { PV.seq = S.seq; PV.i = 0; }
  const th = $("#pvThumbs");
  th.innerHTML = "";
  photos.forEach((ph, i) => th.append(h("button", { class: `pv-th ${i === PV.i ? "on" : ""}`, title: ph.name, onclick: () => { PV.i = i; renderPhotoView(); } },
    h("img", { src: ph.url, alt: ph.name, loading: "lazy" }))));
  th.classList.toggle("single", photos.length < 2);
  $("#pvCap").textContent = `${m.photo_caption || "Bead photo"}${photos.length > 1 ? ` · ${PV.i + 1}/${photos.length}` : ""}`;
  const p = S.detail.pass || {}, nd = S.detail.ndt || {};
  const n = (nd.indications || []).length;
  $("#pvNdt").innerHTML = "";
  $("#pvNdt").append(rtBadge(p.rt_result, p.defect_counts),
    n ? h("span", { class: "pv-note" }, `${(nd.methods || []).join("/")} indications are internal — usually not visible on the bead surface`) : null);
  const img = $("#pvImg");
  if (!photos.length) { img.removeAttribute("src"); return; }
  if (img.getAttribute("src") !== photos[PV.i].url) { img.onload = () => pvFit(); img.src = photos[PV.i].url; }
  else pvFit();
}
function pvApply() {
  const img = $("#pvImg");
  img.style.transform = `translate(${PV.x}px, ${PV.y}px) scale(${PV.z})`;
  $("#pvZoom").textContent = Math.abs(PV.z - PV.fit) < 1e-3 ? "fit" : `${Math.round((PV.z / PV.fit) * 100)}%`;
}
function pvFit() {
  const box = $("#pvMain"), img = $("#pvImg");
  if (!img.naturalWidth) return;
  PV.fit = Math.min(box.clientWidth / img.naturalWidth, box.clientHeight / img.naturalHeight);
  PV.z = PV.fit; PV.x = (box.clientWidth - img.naturalWidth * PV.z) / 2; PV.y = (box.clientHeight - img.naturalHeight * PV.z) / 2;
  pvApply();
}
function pvZoomAt(f, cx, cy) {
  const z = Math.max(PV.fit, Math.min(PV.fit * 12, PV.z * f));
  PV.x = cx - ((cx - PV.x) * z) / PV.z; PV.y = cy - ((cy - PV.y) * z) / PV.z; PV.z = z;
  if (Math.abs(z - PV.fit) < 1e-6) return pvFit();
  pvApply();
}
$("#pvMain").addEventListener("wheel", (e) => { e.preventDefault(); const r = e.currentTarget.getBoundingClientRect(); pvZoomAt(e.deltaY < 0 ? 1.2 : 1 / 1.2, e.clientX - r.left, e.clientY - r.top); }, { passive: false });
$("#pvMain").addEventListener("mousedown", (e) => { PV.drag = { x: e.clientX - PV.x, y: e.clientY - PV.y }; e.currentTarget.classList.add("grab"); });
window.addEventListener("mousemove", (e) => { if (PV.drag) { PV.x = e.clientX - PV.drag.x; PV.y = e.clientY - PV.drag.y; pvApply(); } });
window.addEventListener("mouseup", () => { PV.drag = null; $("#pvMain").classList.remove("grab"); });
$("#pvMain").addEventListener("dblclick", (e) => { const r = e.currentTarget.getBoundingClientRect(); if (PV.z > PV.fit * 1.01) pvFit(); else pvZoomAt(3, e.clientX - r.left, e.clientY - r.top); });
$("#pvIn").onclick = () => { const b = $("#pvMain"); pvZoomAt(1.4, b.clientWidth / 2, b.clientHeight / 2); };
$("#pvOut").onclick = () => { const b = $("#pvMain"); pvZoomAt(1 / 1.4, b.clientWidth / 2, b.clientHeight / 2); };
$("#pvFit").onclick = pvFit;

// virtual clock when a pass has no video (signals-only replay)
S.vt = 0; S.vplay = false; S.vlast = 0;
function currentT() { return vMain.getAttribute("src") ? (vMain.currentTime || 0) + S.offset : S.vt; }
function seek(t) {
  t = Math.max(0, Math.min(t, S.dur || t));
  if (S.view === "sim2real" && S.detail?.sim2real) {
    const s = S.detail.sim2real;
    if (t < s.t0_s || t > s.t1_s) { setView("torch"); }
  }
  S.vt = t;
  if (vMain.getAttribute("src") && vMain.readyState < 1) S.pendingSeek = t;
  const local = t - S.offset;
  if (vMain.src) { try { vMain.currentTime = Math.max(0, local); } catch { /* not ready */ } }
  if (!vSecond.classList.contains("hidden") && vSecond.src) { try { vSecond.currentTime = Math.max(0, local); } catch { /* */ } }
  S.lastT = null;
  tick(true);
}
function togglePlay() {
  if (!vMain.getAttribute("src")) {
    S.vplay = !S.vplay; S.vlast = performance.now();
    $("#btnPlay").innerHTML = S.vplay ? '<svg viewBox="0 0 24 24"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>' : '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
    return;
  }
  if (vMain.paused) { vMain.play(); if (vSecond.src) vSecond.play(); } else { vMain.pause(); vSecond.pause(); }
}
$("#btnPlay").onclick = togglePlay;
vMain.addEventListener("play", () => { $("#btnPlay").innerHTML = '<svg viewBox="0 0 24 24"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>'; });
vMain.addEventListener("pause", () => { $("#btnPlay").innerHTML = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>'; });
vMain.addEventListener("seeked", () => { if (vSecond.src && Math.abs(vSecond.currentTime - vMain.currentTime) > 0.1) vSecond.currentTime = vMain.currentTime; });
document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, textarea, select")) return;
  if (e.code === "Space") { e.preventDefault(); togglePlay(); }
  if (e.code === "ArrowRight") seek(currentT() + 1);
  if (e.code === "ArrowLeft") seek(currentT() - 1);
});
$("#winToggle").addEventListener("click", (e) => {
  if (!e.target.dataset.w) return;
  S.win = e.target.dataset.w; lsSet("weld.win", S.win);
  $$("#winToggle button").forEach((b) => b.classList.toggle("on", b.dataset.w === S.win));
  S.lastT = null; tick(true);
});
$$("#winToggle button").forEach((b) => b.classList.toggle("on", b.dataset.w === S.win));
$("#btnAssess").onclick = () => S.seq && sendChat(`Assess ${S.seq} at ${currentT().toFixed(1)} s with the Cosmos Reasoner`);

/* timeline */
function renderTimeline() {
  const marks = $("#tlMarks");
  marks.innerHTML = "";
  const d = S.dur || 1;
  const wins = S.detail.defect_windows || [];
  const labelled = new Set();
  wins.forEach((w) => {
    const c = defCol(w.defect);
    const first = !labelled.has(w.defect); labelled.add(w.defect);
    const n = wins.filter((x) => x.defect === w.defect).length;
    marks.append(h("div", { class: "tl-mark defect", title: `${prettyDef(w.defect)} ${fmt(w.t_start)}–${fmt(w.t_end)} s${w.x_start_mm != null ? ` · x ${w.x_start_mm}–${w.x_end_mm} mm` : ""}`,
      style: { left: `${(w.t_start / d) * 100}%`, width: `${Math.max(0.5, ((w.t_end - w.t_start) / d) * 100)}%`,
        background: `repeating-linear-gradient(135deg, ${rgba(c, 0.9)} 0 3px, ${rgba(c, 0.5)} 3px 6px)` } },
      first ? h("span", { class: "lbl", style: { color: c } }, `${prettyDef(w.defect)}${n > 1 ? ` ×${n}` : ""}`) : null));
  });
  (S.detail.violations || []).forEach((v) => marks.append(h("div", { class: "tl-mark viol", title: `${v.param} ${v.value} (WPS ${v.limit_lo ?? ""}–${v.limit_hi ?? ""})`,
    style: { left: `${(v.t_start_s / d) * 100}%`, width: `${Math.max(0.6, ((v.t_end_s - v.t_start_s) / d) * 100)}%` } })));
}
$("#timeline").addEventListener("click", (e) => {
  const r = e.currentTarget.getBoundingClientRect();
  seek(((e.clientX - r.left) / r.width) * (S.dur || 0));
});

/* signal charts */
function yRange(arr, lim) {
  // robust range (0.5-99.5 percentile) so arc-off zeros / single spikes don't flatten the trace
  const vals = arr.filter((v) => v != null).sort((a, b) => a - b);
  let lo = vals.length ? vals[Math.floor(vals.length * 0.005)] : Infinity;
  let hi = vals.length ? vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.995))] : -Infinity;
  if (lim) { if (lim[0] != null) lo = Math.min(lo, lim[0]); if (lim[1] != null) hi = Math.max(hi, lim[1]); }
  if (!isFinite(lo)) return [0, 1];
  const pad = (hi - lo || Math.abs(hi) || 1) * 0.12;
  return [lo - pad, hi + pad];
}
function outOfSpec(arr, lim) {
  if (!lim) return arr.map(() => null);
  return arr.map((v) => (v != null && ((lim[0] != null && v < lim[0]) || (lim[1] != null && v > lim[1])) ? v : null));
}
function bandHook(param, lim, windows, viol) {
  return (u) => {
    const { ctx, bbox } = u;
    ctx.save();
    ctx.beginPath(); ctx.rect(bbox.left, bbox.top, bbox.width, bbox.height); ctx.clip();
    // defect windows (hatched, all charts)
    for (const w of windows) {
      const x0 = u.valToPos(w.t_start, "x", true), x1 = Math.max(u.valToPos(w.t_end, "x", true), x0 + 2 * devicePixelRatio);
      const c = defCol(w.defect);
      ctx.fillStyle = rgba(c, 0.1); ctx.fillRect(x0, bbox.top, x1 - x0, bbox.height);
      ctx.strokeStyle = rgba(c, 0.14); ctx.lineWidth = 1;
      for (let x = x0 - bbox.height; x < x1; x += 7 * devicePixelRatio) { ctx.beginPath(); ctx.moveTo(Math.max(x, x0), bbox.top + bbox.height - Math.max(0, x0 - x)); ctx.lineTo(Math.min(x + bbox.height, x1), bbox.top + Math.max(0, x + bbox.height - x1)); ctx.stroke(); }
    }
    // WPS band
    if (lim) {
      const yHi = lim[1] != null ? u.valToPos(lim[1], "y", true) : bbox.top;
      const yLo = lim[0] != null ? u.valToPos(lim[0], "y", true) : bbox.top + bbox.height;
      ctx.fillStyle = "rgba(92,200,255,0.065)";
      ctx.fillRect(bbox.left, yHi, bbox.width, yLo - yHi);
      ctx.setLineDash([4 * devicePixelRatio, 4 * devicePixelRatio]); ctx.strokeStyle = "rgba(92,200,255,0.35)"; ctx.lineWidth = devicePixelRatio;
      for (const y of [lim[0] != null ? yLo : null, lim[1] != null ? yHi : null]) if (y != null) { ctx.beginPath(); ctx.moveTo(bbox.left, y); ctx.lineTo(bbox.left + bbox.width, y); ctx.stroke(); }
      ctx.setLineDash([]);
    }
    // this param's violation windows
    for (const v of viol) {
      const x0 = u.valToPos(v.t_start_s, "x", true), x1 = u.valToPos(v.t_end_s, "x", true);
      ctx.fillStyle = "rgba(255,93,93,0.16)"; ctx.fillRect(x0, bbox.top, x1 - x0, bbox.height);
    }
    ctx.restore();
  };
}
function playheadHook(u) {
  const t = currentT();
  const x = u.valToPos(t, "x", true);
  const { ctx, bbox } = u;
  if (x < bbox.left || x > bbox.left + bbox.width) return;
  ctx.save();
  ctx.strokeStyle = COL.molten; ctx.lineWidth = 2 * devicePixelRatio; ctx.shadowColor = COL.molten; ctx.shadowBlur = 6;
  ctx.beginPath(); ctx.moveTo(x, bbox.top); ctx.lineTo(x, bbox.top + bbox.height); ctx.stroke();
  ctx.restore();
}
function chartParams() {
  const sig = S.sig;
  return CHART_PARAMS.filter((p) => sig.series[p]);
}
function buildCharts() {
  S.charts.forEach((c) => c.u.destroy());
  S.charts = [];
  const box = $("#charts");
  box.innerHTML = "";
  const sig = S.sig, wps = sig.wps || {};
  const params = chartParams();
  const notes = Object.entries(sig.notes || {}).filter(([p]) => CHART_PARAMS.includes(p));
  const wins = S.detail.defect_windows || [];
  const strip = wins.length > 0 || (S.detail.ndt && S.detail.ndt.inspected !== undefined && !S.detail.source?.synthetic);
  const stripH = strip ? 22 : 0;
  const avail = box.clientHeight - 30 - notes.length * 19 - stripH;
  const height = Math.max(40, Math.floor(avail / Math.max(params.length, 1)) - 1);
  const sync = uPlot.sync("weld");
  const basis = sig.wps_basis || S.detail.wps_basis;
  params.forEach((p, i) => {
    const last = i === params.length - 1 && !strip;
    const lim = wps[p];
    const trail = TRAIL[p] && sig.series[TRAIL[p]] ? TRAIL[p] : null;
    const viol = (S.detail.violations || []).filter((v) => v.param === p);
    const limTxt = lim ? (lim[0] == null ? `≤ ${lim[1]}` : `${fmt(lim[0], SIG[p].d)}–${fmt(lim[1], SIG[p].d)}`) : "";
    const row = h("div", { class: `chart-row ${viol.length ? "bad" : ""}` },
      h("div", { class: "chart-lbl" }, h("div", { class: "n" }, SIG[p].n,
          trail ? h("span", { class: "leg", title: "blue = lead electrode, violet = trail electrode" }, h("i", { class: "l" }), h("i", { class: "t" }), "L/T") : null),
        h("div", { class: "val", id: `val-${p}` }, "—"),
        lim ? h("div", { class: "lim", title: basis ? `WPS basis: ${basis}` : "WPS limits" }, `${basis ? "WPS*" : "WPS"} ${limTxt}`) : null),
      h("div", { class: "chart-box" }));
    box.append(row);
    const el = row.querySelector(".chart-box");
    const data = [sig.t, sig.series[p], outOfSpec(sig.series[p], lim)];
    const series = [{}, { stroke: COL.arc, width: 1.5, spanGaps: true }, { stroke: COL.crit, width: 2.4, spanGaps: false }];
    if (trail) { data.push(sig.series[trail]); series.push({ stroke: COL.trail, width: 1.2, spanGaps: true }); }
    const yr = yRange(trail ? sig.series[p].concat(sig.series[trail]) : sig.series[p], lim);
    const u = new uPlot(chartOpts(p, el, last, height, yr, series, sync, [bandHook(p, lim, wins, viol)], (u) => showVal(p, u.cursor.idx)), data, el);
    u.over.addEventListener("click", (e) => seek(u.posToVal(e.offsetX, "x")));
    u.over.addEventListener("mouseleave", () => showVal(p, null));
    S.charts.push({ p, u, el, lim, trail });
  });
  for (const [p, note] of notes) {
    box.append(h("div", { class: "chart-note" }, h("span", { class: "n" }, SIG[p]?.n || p), h("span", {}, note)));
  }
  if (strip) buildNdtStrip(box, sync, stripH);
  S.lastT = null;
  tick(true);
}
function chartOpts(p, el, last, height, yr, series, sync, drawClear, setCursor) {
  return {
    width: Math.max(100, el.clientWidth), height: height + (last ? 18 : 0),
    padding: [4, 6, 0, 0],
    cursor: { sync: { key: sync.key }, drag: { x: false, y: false }, points: { show: false }, y: false },
    legend: { show: false },
    scales: { x: { time: false, auto: false, min: 0, max: S.dur }, y: { range: () => yr } },
    axes: [
      { show: last, stroke: COL.axis, size: 18, gap: 2, grid: { stroke: COL.grid, width: 1 }, ticks: { show: false }, font: "10px IBM Plex Mono", values: (u, v) => v.map((x) => `${x}s`) },
      p ? { stroke: COL.axis, size: 40, gap: 3, grid: { stroke: COL.grid, width: 1 }, ticks: { show: false }, font: "10px IBM Plex Mono", splits: (u) => { const [a, b] = u.scales.y.range ? [u.scales.y.min, u.scales.y.max] : [0, 1]; return [a + (b - a) * 0.2, a + (b - a) * 0.8]; }, values: (u, v) => v.map((x) => fmt(x, SIG[p].d)) }
        : { size: 40, show: true, stroke: "transparent", grid: { show: false }, ticks: { show: false }, values: () => [] },
    ],
    series,
    hooks: { drawClear, draw: [playheadHook], setCursor: [setCursor] },
  };
}
/* NDT strip: one lane under the plots, each indication window drawn in its type colour, hover for details */
function buildNdtStrip(box, sync, stripH) {
  const nd = S.detail.ndt || {}, wins = S.detail.defect_windows || [];
  const types = [...new Set(wins.map((w) => w.defect))];
  const label = !nd.inspected && !wins.length ? "not inspected" : wins.length ? types.map((t) => `${prettyDef(t)} ×${wins.filter((w) => w.defect === t).length}`).join(", ") : "no indications";
  const row = h("div", { class: "chart-row ndt-row" },
    h("div", { class: "chart-lbl" }, h("div", { class: "n" }, `NDT ${(nd.methods || []).join("/") || (S.detail.source?.synthetic ? "(sim)" : "")}`),
      h("div", { class: "ndt-sum", title: label }, label)),
    h("div", { class: "chart-box" }));
  box.append(row);
  const el = row.querySelector(".chart-box");
  const draw = (u) => {
    const { ctx, bbox } = u;
    ctx.save();
    ctx.beginPath(); ctx.rect(bbox.left, bbox.top, bbox.width, bbox.height); ctx.clip();
    ctx.fillStyle = "rgba(255,255,255,0.025)"; ctx.fillRect(bbox.left, bbox.top + 3, bbox.width, bbox.height - 6);
    if (!nd.inspected && !wins.length) {
      ctx.fillStyle = "rgba(134,145,160,0.5)"; ctx.font = `${10 * devicePixelRatio}px IBM Plex Mono`;
      ctx.fillText("no NDT for this pass — internal quality unknown", bbox.left + 8 * devicePixelRatio, bbox.top + bbox.height / 2 + 3.5 * devicePixelRatio);
    }
    for (const w of wins) {
      const x0 = u.valToPos(w.t_start, "x", true), x1 = Math.max(u.valToPos(w.t_end, "x", true), x0 + 3 * devicePixelRatio);
      const c = defCol(w.defect);
      ctx.fillStyle = w === S.ndtHover ? "#fff" : c;
      ctx.fillRect(x0, bbox.top + 3, x1 - x0, bbox.height - 6);
    }
    ctx.restore();
  };
  const u = new uPlot(chartOpts(null, el, true, stripH, [0, 1], [{}, { show: false }], sync, [draw], () => {}), [S.sig.t, S.sig.t.map(() => 0.5)], el);
  const tip = $("#tip");
  u.over.addEventListener("mousemove", (e) => {
    const t = u.posToVal(e.offsetX, "x"), tol = (u.posToVal(e.offsetX + 4, "x") - t);
    const w = wins.find((x) => t >= x.t_start - tol && t <= x.t_end + tol);
    if (w !== S.ndtHover) { S.ndtHover = w; u.redraw(false, false); }
    if (!w) { tip.classList.add("hidden"); return; }
    tip.innerHTML = "";
    tip.append(h("b", { style: { color: defCol(w.defect) } }, prettyDef(w.defect)),
      h("div", {}, `${fmt(w.t_start, 2)}–${fmt(w.t_end, 2)} s from arc start`),
      w.x_start_mm != null ? h("div", {}, `x ${w.x_start_mm}–${w.x_end_mm} mm along the weld`) : null,
      h("div", { class: "sub" }, `${w.method || (nd.methods || []).join(" + ") || "NDT"} indication · click to seek`));
    tip.classList.remove("hidden");
    tip.style.left = `${Math.min(e.clientX + 12, innerWidth - 230)}px`; tip.style.top = `${e.clientY - 70}px`;
  });
  u.over.addEventListener("mouseleave", () => { S.ndtHover = null; tip.classList.add("hidden"); u.redraw(false, false); });
  u.over.addEventListener("click", (e) => { const w = S.ndtHover; seek(w ? w.t_start : u.posToVal(e.offsetX, "x")); });
  S.charts.push({ p: null, u, el, strip: true });
}
function resizeCharts() {
  if (!S.charts.length) return;
  const box = $("#charts");
  const strip = S.charts.find((c) => c.strip);
  const plots = S.charts.filter((c) => !c.strip);
  const notes = $$(".chart-note", box).length;
  const height = Math.max(40, Math.floor((box.clientHeight - 30 - notes * 19 - (strip ? 22 : 0)) / Math.max(plots.length, 1)) - 1);
  plots.forEach((c, i) => c.u.setSize({ width: Math.max(100, c.el.clientWidth), height: height + (!strip && i === plots.length - 1 ? 18 : 0) }));
  if (strip) strip.u.setSize({ width: Math.max(100, strip.el.clientWidth), height: 22 + 18 });
}
window.addEventListener("resize", () => { clearTimeout(S.rz); S.rz = setTimeout(resizeCharts, 120); });

function idxAt(t) {
  const ts = S.sig?.t || [];
  let lo = 0, hi = ts.length - 1;
  while (lo < hi) { const m = (lo + hi) >> 1; if (ts[m] < t) lo = m + 1; else hi = m; }
  return lo;
}
function valState(p, v) {
  const lim = S.sig?.wps?.[p];
  return lim && v != null && ((lim[0] != null && v < lim[0]) || (lim[1] != null && v > lim[1]));
}
function showVal(p, idx) {
  const el = $(`#val-${p}`);
  if (!el || !S.sig) return;
  const hover = idx != null;
  const i = hover ? idx : idxAt(currentT());
  const v = S.sig.series[p][i];
  const tr = TRAIL[p] && S.sig.series[TRAIL[p]] ? S.sig.series[TRAIL[p]][i] : undefined;
  el.innerHTML = `${fmt(v, SIG[p].d)}${tr !== undefined ? `<span class="tr"> / ${fmt(tr, SIG[p].d)}</span>` : ""}<small>${SIG[p].u}${hover ? ` @${fmt(S.sig.t[i])}s` : ""}</small>`;
  el.style.color = valState(p, v) ? "#ff8b8b" : "";
}
function renderHud(t) {
  if (!S.sig) return;
  const i = idxAt(t);
  const hud = $("#hud");
  const cur = S.sig.series.current_a?.[i];
  const arcOff = cur != null && cur < 5;
  const real = !S.detail.source?.synthetic;
  hud.innerHTML = (arcOff ? '<div class="r"><div class="k"><span>Arc</span></div><div class="v">OFF</div></div>' : "") +
    CHART_PARAMS.filter((p) => S.sig.series[p]).slice(0, 6).map((p) => {
    const v = S.sig.series[p][i];
    const tr = TRAIL[p] && S.sig.series[TRAIL[p]] ? S.sig.series[TRAIL[p]][i] : undefined;
    const bad = !arcOff && valState(p, v);
    return `<div class="r ${bad ? "bad" : ""}"><div class="k"><span>${SIG[p].n}${tr !== undefined ? " L/T" : ""}</span><span>${bad ? (real ? "OFF SETTINGS" : "OUT OF WPS") : ""}</span></div><div class="v">${fmt(v, SIG[p].d)}${tr !== undefined ? `<span class="tr">/${fmt(tr, SIG[p].d)}</span>` : ""}<small>${SIG[p].u}</small></div></div>`;
  }).join("");
  const w = (S.detail.defect_windows || []).find((d) => t >= d.t_start - 0.05 && t <= d.t_end + 0.05);
  const flash = $("#defectFlash");
  flash.classList.toggle("hidden", !w);
  if (w) { flash.textContent = `● ${prettyDef(w.defect)} · ${real ? `${(S.detail.ndt?.methods || ["NDT"]).join("/")} indication` : `${fmt(t - w.t_start)} s in`}`; flash.style.setProperty("--dc", defCol(w.defect)); }
}
function tick(force = false) {
  if (!S.sig) return;
  const t = currentT();
  if (!force && S.lastT != null && Math.abs(t - S.lastT) < 0.02) return;
  S.lastT = t;
  $("#timecode").textContent = `${fmt(t)} s`;
  $("#tlHead").style.left = `calc(${(t / (S.dur || 1)) * 100}% - 1px)`;
  let min = 0, max = S.dur;
  if (S.win === "follow") { min = Math.max(0, Math.min(t - 8, S.dur - 12)); max = Math.min(S.dur, min + 12); }
  for (const c of S.charts) {
    if (c.u.scales.x.min !== min || c.u.scales.x.max !== max) c.u.setScale("x", { min, max });
    else c.u.redraw(false, false);
    if (c.u.cursor.idx == null) showVal(c.p, null);
  }
  renderHud(t);
}
(function loop(now) {
  if (S.vplay && !vMain.getAttribute("src")) {
    S.vt += ((now || 0) - S.vlast) / 1000; S.vlast = now || 0;
    if (S.vt >= S.dur) { S.vt = S.dur; S.vplay = false; }
    tick();
  } else if (!vMain.paused) tick();
  requestAnimationFrame(loop);
})();
vMain.addEventListener("seeked", () => tick(true));
vMain.addEventListener("timeupdate", () => { if (vMain.paused) tick(true); });

/* ===================================================================== chat */
function botMsg() {
  const m = h("div", { class: "msg bot" }, h("div", { class: "who" }, h("span", { class: "av" }), "Weld Copilot"));
  $("#messages").append(m);
  return m;
}
function scrollChat() { const m = $("#messages"); m.scrollTop = m.scrollHeight; }
function welcome() {
  const m = botMsg();
  m.append(h("div", { class: "welcome md", html: md(isReal()
    ? "**Weld Copilot** on **real welding data** — AIMEN tandem-SAW beads on SA-516 vessel plate (RT/UT-inspected, bead photos) " +
      "and BAM tandem-GMAW layers (real process video). I query the signals in **Mosaico**, check NDT indications and the published " +
      "settings, and use **NVIDIA Cosmos 3** to inspect photos/clips and render what-ifs. When the signals show nothing, I say so.\n\n" +
      "Start with the numbered demo prompts below ↓"
    : "**Weld Copilot** for Cell 2 — ask in plain English. I query **Mosaico** sensor data, check passes against the **WPS**, " +
    "and use **NVIDIA Cosmos 3** to inspect clips, render what-ifs and turn simulator output into photoreal training video.\n\n" +
    "Start with the numbered demo prompts below ↓") }));
}
function renderSuggestions() {
  const row = $("#suggest");
  row.innerHTML = "";
  const next = S.demo.findIndex((_, i) => !S.demoDone.has(i));
  S.demo.forEach((d, i) => row.append(h("button", { class: `sugg ${i === next ? "next" : ""} ${S.demoDone.has(i) ? "done" : ""}`,
    onclick: () => { S.demoDone.add(i); sendChat(d.label); } },
    h("span", { class: "n" }, `${i + 1}`), h("span", { class: "t" }, d.label), h("span", { class: "h" }, d.hint))));
}
$("#composer").addEventListener("submit", (e) => { e.preventDefault(); const t = $("#chatInput").value.trim(); if (t) { $("#chatInput").value = ""; sendChat(t); } });
$("#chatInput").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); } });
$("#chatReset").onclick = async () => {
  if (S.sessionId) await api("/api/chat/reset", { method: "POST", body: { session_id: S.sessionId } }).catch(() => {});
  S.sessionId = null; S.demoDone.clear(); $("#messages").innerHTML = ""; welcome(); renderSuggestions();
};

async function sendChat(text) {
  if (S.busy) return;
  S.busy = true;
  const idx = S.demo.findIndex((d) => d.label === text);
  if (idx >= 0) S.demoDone.add(idx);
  renderSuggestions();
  $("#messages").append(h("div", { class: "msg user" }, text));
  const m = botMsg();
  const st = { m, textEl: null, buf: "", tools: {}, thinkEl: null, thinkBuf: "" };
  const caret = h("div", { class: "caret md" });
  m.append(caret);
  scrollChat();
  pulseStage("copilot");
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: S.sessionId }) });
    if (!r.ok) throw new Error(await r.text());
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let k;
      while ((k = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, k); buf = buf.slice(k + 2);
        const data = block.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).replace(/^ /, "")).join("\n");
        if (!data) continue;
        let ev; try { ev = JSON.parse(data); } catch { continue; }
        caret.remove();
        handleEvent(st, ev);
        m.append(caret);
        scrollChat();
      }
    }
  } catch (e) {
    m.append(h("div", { class: "toolcall err" }, "⚠ ", String(e.message || e)));
  } finally {
    caret.remove();
    S.busy = false;
    scrollChat();
  }
}
function flushText(st) { st.textEl = null; st.buf = ""; }
function handleEvent(st, ev) {
  const m = st.m;
  if (ev.session_id) { S.sessionId = ev.session_id; return; }
  switch (ev.type) {
    case "text":
      if (!st.textEl) { st.textEl = h("div", { class: "md" }); m.append(st.textEl); }
      st.buf += ev.delta;
      st.textEl.innerHTML = md(st.buf);
      break;
    case "thinking":
      if (!st.thinkEl) { st.thinkEl = h("details", { class: "thinking" }, h("summary", {}, "Thinking…"), h("div", {})); m.append(st.thinkEl); }
      st.thinkBuf += ev.delta;
      st.thinkEl.lastChild.textContent = st.thinkBuf;
      break;
    case "tool_start": {
      flushText(st);
      const args = Object.entries(ev.input || {}).map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`).join(" ");
      const row = h("div", { class: "toolcall" }, h("span", { class: "spin" }), h("span", { class: "tn" }, ev.name), h("span", { class: "ta" }, args), h("span", { class: "ts" }, "running"));
      st.tools[ev.id] = row; m.append(row);
      pulseStage(toolStage(ev.name));
      break;
    }
    case "tool_end": {
      const row = st.tools[ev.id];
      if (row) {
        row.firstChild.replaceWith(h("span", { class: ev.ok ? "ok" : "" }, ev.ok ? "✓" : "✕"));
        row.classList.toggle("err", !ev.ok);
        row.lastChild.textContent = ev.summary || "";
      }
      break;
    }
    case "card": flushText(st); m.append(renderCard(ev.card)); break;
    case "ui": if (ev.action === "show_weld" && ev.sequence) { showTab("copilot"); selectPass(ev.sequence, ev.t, ev.view); } break;
    case "error": m.append(h("div", { class: "toolcall err" }, "⚠ ", ev.message)); break;
    case "done": if (st.thinkEl) st.thinkEl.firstChild.textContent = "Reasoning"; break;
  }
}
function toolStage(name) {
  return { list_passes: "mosaico", find_anomalies: "mosaico", get_signals_summary: "mosaico", wps_check: "mosaico", show_weld: "mosaico",
    show_photo: "mosaico", ndt_summary: "mosaico",
    assess_weld: "cosmos", what_if: "cosmos", sim2real: "cosmos", dataset_stats: "curator", search_clips: "curator" }[name] || "copilot";
}

/* ===================================================================== cards */
function renderCard(c) {
  const f = { table: cardTable, chart: cardChart, verdict: cardVerdict, jump: cardJump, job: cardJob, stats: cardStats, clips: cardClips, photo: cardPhoto }[c.type];
  return f ? f(c) : h("pre", {}, JSON.stringify(c, null, 2));
}
function cellVal(col, v, row) {
  if (col.badge && col.text_key) return h("span", { class: `badge ${RT_CLASS[v] || v}` }, `${ICON[v] || ""} ${row[col.text_key] || v}`);
  if (col.badge) return h("span", { class: `badge ${RT_CLASS[v] || v}` }, `${ICON[v] || ""} ${String(v).replace(/_/g, " ")}`);
  if (Array.isArray(v)) return v.map((x) => String(x).replace(/_/g, " ")).join(", ") || "—";
  if (typeof v === "number") return Number.isInteger(v) ? v : fmt(v, 2);
  return v ?? "—";
}
function cardTable(c) {
  const tbl = h("table", { class: "grid" }, h("thead", {}, h("tr", {}, c.columns.map((col) => h("th", {}, col.label)))));
  const tb = h("tbody");
  c.rows.forEach((r) => {
    const seq = r.sequence || r.sequence_name || c.default_sequence;
    const t = c.time_key ? r[c.time_key] : null;
    tb.append(h("tr", { class: seq ? "click" : "", title: seq ? "Open in replay" : null,
      onclick: seq ? () => selectPass(seq, t) : null }, c.columns.map((col) => h("td", {}, cellVal(col, r[col.key], r),
        col.key === "sequence" && r.simulated && isReal() ? h("span", { class: "simtag" }, "SIM") : null))));
  });
  tbl.append(tb);
  return h("div", { class: "card" }, h("div", { class: "card-h" }, c.title, h("span", { class: "r" }, h("span", { class: "badge arc" }, "Mosaico"))),
    c.rows.length ? h("div", { class: "tbl-wrap" }, tbl) : h("div", { class: "card-b sub" }, "No rows"));
}
function cardChart(c) {
  const body = h("div", { class: "card-b", style: { padding: "6px 4px" } });
  const card = h("div", { class: "card" }, h("div", { class: "card-h" }, c.title, h("span", { class: "r" }, h("span", { class: "badge neutral" }, isReal() ? "band = settings window" : "shaded = WPS"))), body);
  requestAnimationFrame(() => {
    const params = Object.keys(c.series);
    params.forEach((p, i) => {
      const lim = c.wps[p];
      const vals = c.series[p];
      const bad = outOfSpec(vals, lim).some((v) => v != null);
      const holder = h("div");
      const yr = yRange(vals, lim);
      const isTrail = Object.values(TRAIL).includes(p);
      const row = h("div", { class: "spark-row" }, h("div", { class: "n" }, h("span", { style: { color: isTrail ? COL.trail : "" } }, SIG[p]?.n || p), h("b", { style: { color: bad ? "#ff8b8b" : "" } }, `${fmt(Math.min(...vals.filter((x) => x != null)), SIG[p]?.d ?? 1)}–${fmt(Math.max(...vals.filter((x) => x != null)), SIG[p]?.d ?? 1)}`)), holder);
      body.append(row);
      const last = i === params.length - 1;
      const u = new uPlot({
        width: Math.max(120, holder.clientWidth), height: 40 + (last ? 16 : 0), padding: [3, 4, 0, 0],
        cursor: { show: true, points: { show: false }, y: false, drag: { x: false, y: false } }, legend: { show: false },
        scales: { x: { time: false }, y: { range: () => yr } },
        axes: [{ show: last, stroke: COL.axis, size: 16, gap: 1, grid: { show: false }, ticks: { show: false }, font: "9px IBM Plex Mono", values: (u, v) => v.map((x) => `${x}s`) },
          { show: false }],
        series: [{}, { stroke: isTrail ? COL.trail : COL.arc, width: 1.25 }, { stroke: COL.crit, width: 2 }],
        hooks: { drawClear: [bandHook(p, lim, c.defects || [], [])] },
      }, [c.t, vals, outOfSpec(vals, lim)], holder);
      u.over.style.cursor = "pointer";
      u.over.addEventListener("click", (e) => selectPass(c.sequence, u.posToVal(e.offsetX, "x")));
    });
    const chk = c.signal_check;
    if (chk && chk.params) {
      const c0 = chk.params.current_a || Object.values(chk.params)[0];
      body.append(h("div", { class: `sigchk ${chk.signature ? "yes" : "no"}` },
        h("b", {}, chk.signature ? "Signal change inside NDT windows" : "No electrical signature"),
        ` — ${chk.n_windows} windows: ${fmt(c0.mean_in, 0)} vs ${fmt(c0.mean_clean, 0)} ${c0.unit} clean weld, ${c0.windows_outside_band}/${c0.windows_checked} outside the normal band`));
    }
  });
  return card;
}
function cardVerdict(c) {
  const v = (c.verdict || "review").toLowerCase();
  const thumb = c.video_url ? h("div", { class: "vthumb" }, h("video", { src: c.video_url, muted: true, autoplay: true, loop: true, playsinline: true }))
    : c.image_url ? h("div", { class: "vthumb", title: "Bead photo assessed", onclick: () => selectPass(c.sequence_name, null, "photo") }, h("img", { src: c.image_url, alt: "bead photo" })) : null;
  const ndtTxt = c.ndt_indications ? (Object.keys(c.ndt_indications).length ? "NDT: " + Object.entries(c.ndt_indications).map(([k, v]) => `${prettyDef(k)} ×${v}`).join(", ") : "NDT clear") : null;
  const defects = h("div", { class: "defects" }, (c.defects || []).map((d) => h("div", { class: "drow" },
    h("span", {}, (d.type || "").replace(/_/g, " ")), h("div", { class: "bar" }, h("i", { style: { width: `${Math.round((d.confidence || 0) * 100)}%` } })),
    h("span", { class: "pc" }, `${Math.round((d.confidence || 0) * 100)}%`),
    h("span", { class: "at", title: "Jump replay", onclick: () => selectPass(c.sequence_name, d.t_abs ?? c.t0_s) }, d.t_abs != null ? `@${fmt(d.t_abs)}s` : ""))));
  const findings = (c.wps_findings || []).filter((f) => !f.status || f.status !== "within").map((f) => h("li", {}, f.note || `${f.param}: ${f.value} (WPS ${f.limit_lo ?? "—"}–${f.limit_hi ?? "—"}) ${f.t_start_s != null ? `@ ${fmt(f.t_start_s)}–${fmt(f.t_end_s)} s` : ""}`));
  const src = String(c.source || "");
  const srcClass = src.includes("heuristic") || src.includes("mock") ? "warn" : src.includes("cache") ? "neutral" : "ok";
  return h("div", { class: "card verdict" },
    h("div", { class: "card-h" }, "Cosmos 3 Reasoner · weld QA", h("span", { class: "r" }, h("span", { class: "badge molten" }, "Cosmos 3"))),
    h("div", { class: "verdict-top" },
      h("div", { class: `vbig ${v}` }, ICON[v] || "", v.toUpperCase()),
      h("div", { class: "vmeta" }, h("div", {}, h("b", {}, c.sequence_name)),
        h("div", {}, c.input_kind === "photo" ? "input: bead photo + whole-pass signals" : c.input_kind === "signals" ? "input: signals only (no image)" : `clip ${fmt(c.t0_s)}–${fmt(c.t1_s)} s${c.clip_id ? ` · ${c.clip_id}` : ""}`),
        c.rt_result ? h("div", {}, ndtTxt || c.rt_result === "not_inspected" ? "NDT ground truth: " : "RT ground truth: ",
          h("span", { class: `badge ${RT_CLASS[c.rt_result] || c.rt_result}` }, ndtTxt || String(c.rt_result).replace(/_/g, " "))) : null),
      thumb),
    (c.defects || []).length ? defects : null,
    h("div", { class: "kv" },
      h("div", { class: "k" }, c.ndt_indications ? "Cause (evidence)" : "Root cause"), h("div", {}, c.root_cause || "—"),
      h("div", { class: "k" }, "Recommended action"), h("div", {}, c.recommended_action || "—"),
      findings.length ? [h("div", { class: "k" }, "WPS findings"), h("ul", {}, findings)] : null,
      c.code_basis ? [h("div", { class: "k" }, "Code basis"), h("div", {}, c.code_basis)] : null),
    c.reasoning ? h("details", { class: "trace" }, h("summary", {}, "Reasoning trace"), h("pre", {}, c.reasoning)) : null,
    h("div", { class: "card-f" }, h("span", { class: `badge ${srcClass}` }, `source · ${src || "?"}`),
      h("span", { class: "badge neutral" }, `${fmt(c.latency_s, 2)} s`),
      c.origin_source ? h("span", { class: "badge neutral" }, `orig · ${c.origin_source}`) : null,
      c.note ? h("span", { class: "sub", title: c.note }, "fallback") : null));
}
function cardJump(c) {
  return h("div", { class: "card" }, h("div", { class: "jump", onclick: () => selectPass(c.sequence, c.t) },
    h("svg", { viewBox: "0 0 24 24", html: '<path d="M8 5v14l11-7z"/>' }), h("span", {}, c.title)));
}
function mediaBlock(res) {
  if (!res) return null;
  const left = res.kind === "sim2real" ? (res.control_url || res.source_url) : res.source_url;
  const leftCap = res.kind === "sim2real" ? `${isReal() ? "CONTROL" : "SIMULATOR"} · ${(res.control_kind || "edge").toUpperCase()}` : "ORIGINAL";
  const rightCap = res.kind === "sim2real" ? "COSMOS 3 TRANSFER" : `WHAT-IF · ${(res.label || "").toUpperCase()}`;
  const vids = [];
  const mk = (src, cap) => h("figure", {}, h("video", { src, muted: true, autoplay: true, loop: true, playsinline: true, controls: false }), h("figcaption", {}, cap));
  if (left) vids.push(mk(left, leftCap));
  else if (res.source_image_url) vids.push(h("figure", {}, h("img", { src: res.source_image_url, alt: "source photo", class: "srcimg" }), h("figcaption", {}, "SOURCE · BEAD PHOTO")));
  vids.push(mk(res.video_url, rightCap));
  const wrap = h("div", { class: vids.length > 1 ? "pair" : "single" }, vids);
  if (vids.length > 1) { // keep the two loops in lock-step
    const [a, b] = wrap.querySelectorAll("video");
    b.addEventListener("play", () => { a.currentTime = b.currentTime; a.play(); });
    b.addEventListener("seeked", () => { a.currentTime = b.currentTime; });
  }
  return wrap;
}
function resultBadges(res) {
  const src = String(res.source || "");
  return [h("span", { class: `badge ${src.includes("heuristic") || src.includes("mock") ? "warn" : src.includes("cache") ? "neutral" : "ok"}` }, `source · ${src}`),
    h("span", { class: "badge neutral" }, `${fmt(res.latency_s, 1)} s`),
    res.mode ? h("span", { class: "badge arc" }, res.mode) : null,
    res.placeholder ? h("span", { class: "badge warn", title: "No Cosmos backend or cached render available: placeholder video" }, "placeholder") : null];
}
function cardJob(c) {
  const bar = h("i");
  const msg = h("span", {}, "Queued");
  const body = h("div", { class: "job-b" }, h("div", { class: "prog" }, bar), h("div", { class: "job-msg" }, h("span", { class: "spin" }), msg));
  const card = h("div", { class: "card" }, h("div", { class: "card-h" }, c.label, h("span", { class: "r" }, h("span", { class: "badge molten" }, c.kind === "sim2real" ? "Cosmos 3 Transfer" : "Cosmos 3 Generator"))), body);
  followJob(c.job_id, (j) => {
    bar.style.width = `${Math.round(j.progress * 100)}%`;
    msg.textContent = `${j.message} · ${fmt(j.elapsed_s, 0)} s`;
    if (j.status === "done") {
      body.replaceWith(h("div", { class: "card-b" }, mediaBlock(j.result)));
      card.append(h("div", { class: "card-f" }, resultBadges(j.result), j.result.kind === "sim2real"
        ? h("button", { class: "btn small", onclick: () => { selectPass(j.result.sequence_name, j.result.t0_s, "sim2real"); } }, "Open side-by-side") : null));
      onJobDone(j);
      scrollChat();
    } else if (j.status === "error") {
      body.replaceWith(h("div", { class: "card-b", style: { color: "#ff9c9c" } }, `Failed: ${j.error}`));
    }
  });
  return card;
}
function followJob(id, cb) {
  const es = new EventSource(`/api/jobs/${id}/events`);
  es.addEventListener("progress", (e) => { const j = JSON.parse(e.data); cb(j); if (j.status === "done" || j.status === "error") es.close(); });
  es.onerror = () => { es.close(); setTimeout(async () => { try { const j = await api(`/api/jobs/${id}`); cb(j); if (j.status !== "done" && j.status !== "error") followJob(id, cb); } catch { /* */ } }, 1000); };
}
async function onJobDone(j) {
  S.galleryStale = true;
  if (j.result?.kind === "sim2real" && j.result.sequence_name === S.seq) {
    S.detail = await api(`/api/passes/${encodeURIComponent(S.seq)}`);
  }
  if ($("#tab-generate").classList.contains("active")) loadGallery();
}
function cardStats(c) {
  const kp = h("div", { class: "stats-k" }, c.kpis.map((k) => h("div", { class: "kpi" }, h("div", { class: "v" }, k.value), h("div", { class: "l" }, k.label))));
  const maxT = Math.max(1, ...c.bars.map((b) => b.total));
  const bars = h("div", { class: "mini-bars" }, c.bars.map((b) => h("div", { class: `mbar ${c.title.includes(b.label) ? "hl" : ""}` },
    h("span", { class: "l" }, b.label.replace(/_/g, " ")),
    h("div", { class: "t" }, h("i", { style: { width: `${(b.value / maxT) * 100}%` } }), h("i", { class: "rej", style: { width: `${((b.total - b.value) / maxT) * 100}%` } })),
    h("span", { class: "n" }, `${b.value}/${b.total}`))));
  return h("div", { class: "card" }, h("div", { class: "card-h" }, c.title, h("span", { class: "r" }, h("span", { class: "badge arc" }, "Curator"))), kp, bars);
}
function cardPhoto(c) {
  const src = c.source || {};
  return h("div", { class: "card" }, h("div", { class: "card-h" }, c.title, h("span", { class: "r" }, rtBadge(c.rt_result, Object.fromEntries(Object.entries((c.indications || []).reduce((a, i) => { a[i.type] = (a[i.type] || 0) + 1; return a; }, {})))))),
    h("div", { class: "photo-card" }, c.photos.map((ph) => h("img", { src: ph.url, alt: ph.name, title: "Open in the replay viewer", onclick: () => { showTab("copilot"); selectPass(c.sequence, null, "photo"); } }))),
    h("div", { class: "card-f" }, h("span", { class: "sub" }, c.caption || ""), src.doi ? h("a", { class: "sub", href: `https://doi.org/${src.doi}`, target: "_blank", rel: "noopener" }, `DOI ${src.doi}`) : null));
}
function cardClips(c) {
  return h("div", { class: "card" }, h("div", { class: "card-h" }, c.title),
    h("div", { class: "clips" }, c.rows.map((r) => h("div", { class: "clip", onclick: () => selectPass(r.sequence, r.t0_s), style: { cursor: "pointer" } },
      r.video_url ? h("video", { src: r.video_url, muted: true, autoplay: true, loop: true, playsinline: true }) : null,
      h("div", { class: "c" }, h("b", {}, r.clip_id), (r.caption || "").slice(0, 110))))));
}

/* ===================================================================== dataset tab */
let DS = null;
async function loadDataset() {
  const q = $("#dsQ").value, defect = $("#dsDefect").value, kept = $("#dsKept .on")?.dataset.v || "";
  const params = new URLSearchParams();
  if (q) params.set("q", q); if (defect) params.set("defect", defect); if (kept) params.set("kept", kept);
  DS = await api(`/api/dataset?${params}`);
  const st = DS.stats;
  $("#dsKpis").innerHTML = "";
  $("#dsKpis").append(...[["Clips", st.total], ["Kept", st.kept], ["Rejected", st.rejected], ["Source passes", st.sequences]]
    .map(([l, v]) => h("div", { class: "kpi" }, h("div", { class: "v" }, v ?? 0), h("div", { class: "l" }, l))));
  const maxT = Math.max(1, ...Object.values(st.by_defect || {}).map((b) => b.total));
  $("#dsBars").innerHTML = "";
  Object.entries(st.by_defect || {}).sort((a, b) => b[1].kept - a[1].kept).forEach(([k, b]) => $("#dsBars").append(h("div", { class: "mbar" },
    h("span", { class: "l" }, k === "none" ? "no defect" : k.replace(/_/g, " ")), h("div", { class: "t" }, h("i", { style: { width: `${(b.kept / maxT) * 100}%` } }), h("i", { class: "rej", style: { width: `${((b.total - b.kept) / maxT) * 100}%` } })),
    h("span", { class: "n" }, `${b.kept}/${b.total}`))));
  const maxR = Math.max(1, ...Object.values(st.reject_reasons || {}));
  $("#dsReasons").innerHTML = "";
  Object.entries(st.reject_reasons || {}).sort((a, b) => b[1] - a[1]).forEach(([k, n]) => $("#dsReasons").append(h("div", { class: "mbar" },
    h("span", { class: "l" }, k.replace(/_/g, " ")), h("div", { class: "t" }, h("i", { style: { width: `${(n / maxR) * 100}%`, background: "var(--warn)" } })), h("span", { class: "n" }, n))));
  $("#dsCard").innerHTML = DS.card ? md(DS.card) : '<span class="sub">No dataset_card.md yet</span>';
  const dsel = $("#dsDefect");
  if (dsel.options.length <= 1) Object.keys(st.by_defect || {}).sort().forEach((k) => dsel.append(h("option", { value: k }, k.replace(/_/g, " "))));
  $("#dsCount").textContent = `${DS.count} clips`;
  const tbl = $("#dsTable");
  tbl.innerHTML = "";
  tbl.append(h("thead", {}, h("tr", {}, ["Clip", "Defects", "RT", "Status", "Caption"].map((x) => h("th", {}, x)))));
  const tb = h("tbody");
  DS.rows.forEach((r, i) => tb.append(h("tr", { class: "click", "data-i": i, onclick: (e) => previewClip(r, e.currentTarget) },
    h("td", {}, r.clip_id, h("span", { class: "w" }, `${fmt(r.t0_s)}–${fmt(r.t1_s)} s · blur ${fmt(r.quality?.blur, 2)}`)),
    h("td", {}, (r.defects || []).length ? r.defects.map((d) => h("span", { class: "badge reject", style: { marginRight: "3px" } }, d.replace(/_/g, " "))) : h("span", { class: "sub" }, "none")),
    h("td", {}, h("span", { class: `badge ${r.rt_result || "pending"}` }, r.rt_result || "pending")),
    h("td", {}, r.kept ? h("span", { class: "badge ok" }, "kept") : h("span", { class: "badge neutral", title: r.reject_reason }, (r.reject_reason || "rejected").replace(/_/g, " "))),
    h("td", { class: "cap" }, h("div", {}, r.caption || "")))));
  tbl.append(tb);
  if (DS.rows[0] && !$("#dsPreview video")) {
    const i = Math.max(0, DS.rows.findIndex((r) => (r.defects || []).length && r.kept));
    const tr = tb.children[i];
    previewClip(DS.rows[i], tr);
    tr?.scrollIntoView({ block: "center" });
  }
}
function previewClip(r, tr) {
  $$("#dsTable tr.sel").forEach((x) => x.classList.remove("sel"));
  tr?.classList.add("sel");
  let prompt = r.generator_prompt;
  try { prompt = JSON.stringify(typeof prompt === "string" ? JSON.parse(prompt) : prompt, null, 2); } catch { /* raw text */ }
  const p = $("#dsPreview");
  p.innerHTML = "";
  p.append(r.video_url ? h("video", { src: r.video_url, controls: true, autoplay: true, muted: true, loop: true }) : h("div", { class: "placeholder" }, "No clip video"),
    h("div", { style: { margin: "8px 0", display: "flex", gap: "6px", flexWrap: "wrap" } }, h("b", { style: { fontFamily: "var(--mono)" } }, r.clip_id),
      h("span", { class: `badge ${r.kept ? "ok" : "neutral"}` }, r.kept ? "kept" : `rejected · ${r.reject_reason || ""}`),
      h("span", { class: "badge neutral" }, r.caption_model || "caption")),
    h("div", { class: "md" }, r.caption || ""),
    h("h3", { style: { margin: "12px 0 6px" } }, "Generator prompt (Cosmos 3)"), h("pre", {}, prompt || "—"),
    h("div", { style: { display: "flex", gap: "8px", marginTop: "8px" } },
      h("button", { class: "btn small", onclick: () => { showTab("copilot"); selectPass(r.sequence_name, r.t0_s); } }, "Open in replay"),
      h("button", { class: "btn small", onclick: () => { showTab("copilot"); sendChat(`Assess clip ${r.clip_id} — what does the Cosmos Reasoner see?`); } }, "Assess with Reasoner")));
}
$("#dsQ").addEventListener("input", () => { clearTimeout(S.dsT); S.dsT = setTimeout(loadDataset, 250); });
$("#dsDefect").onchange = loadDataset;
$("#dsKept").addEventListener("click", (e) => { if (!e.target.dataset) return; $$("#dsKept button").forEach((b) => b.classList.toggle("on", b === e.target)); loadDataset(); });

/* ===================================================================== generate tab */
const G = { kind: "what_if", tpl: "travel_speed_-20", ctrl: "edge", inited: false };
async function loadGenerate() {
  if (!G.inited) {
    G.inited = true;
    const sel = $("#genSeq");
    sel.innerHTML = S.passes.map((p) => `<option value="${esc(p.sequence_name)}">${esc(p.sequence_name)}${p.has_defect ? " ⚠" : ""}</option>`).join("");
    sel.value = S.seq || S.passes[0]?.sequence_name || "";
    const t = await api("/api/templates");
    $("#genTpl").innerHTML = "";
    t.templates.forEach((x) => $("#genTpl").append(h("button", { class: x.key === G.tpl ? "on" : "", title: x.text, "data-k": x.key,
      onclick: (e) => { G.tpl = x.key; $$("#genTpl button").forEach((b) => b.classList.toggle("on", b === e.currentTarget)); refreshPrompt(); } }, x.label)));
    sel.onchange = refreshPrompt;
    refreshPrompt();
  } else if (S.seq) { $("#genSeq").value = S.seq; refreshPrompt(); }
  loadGallery();
}
$("#genKind").addEventListener("click", (e) => {
  if (!e.target.dataset.v) return;
  G.kind = e.target.dataset.v;
  $$("#genKind button").forEach((b) => b.classList.toggle("on", b === e.target));
  $("#genTplWrap").classList.toggle("hidden", G.kind !== "what_if");
  $("#genCtrlWrap").classList.toggle("hidden", G.kind !== "sim2real");
  refreshPrompt();
});
$("#genCtrl").addEventListener("click", (e) => { if (!e.target.dataset.v) return; G.ctrl = e.target.dataset.v; $$("#genCtrl button").forEach((b) => b.classList.toggle("on", b === e.target)); });
async function refreshPrompt() {
  const r = await api("/api/prompt", { method: "POST", body: { sequence: $("#genSeq").value, change: G.tpl, kind: G.kind } });
  $("#genPrompt").value = r.prompt;
  G.promptAuto = r.prompt;
}
$("#genGo").onclick = async () => {
  const btn = $("#genGo");
  btn.disabled = true;
  try {
    const prompt = $("#genPrompt").value.trim();
    const body = { kind: G.kind, sequence: $("#genSeq").value, change: G.tpl, control: G.ctrl, seed: +$("#genSeed").value || 0,
      t: $("#genT").value ? +$("#genT").value : null, prompt: prompt && prompt !== G.promptAuto ? prompt : null };
    const j = await api("/api/generate", { method: "POST", body });
    $("#genJobs").prepend(cardJob({ job_id: j.id, kind: j.kind, label: j.label }));
    pulseStage("cosmos", 4000);
  } catch (e) { alert(e.message); } finally { btn.disabled = false; }
};
async function loadGallery() {
  const g = await api("/api/gallery");
  const box = $("#gallery");
  box.innerHTML = "";
  $("#galCount").textContent = `${g.items.length} generated videos`;
  if (!g.items.length) box.append(h("div", { class: "empty-g" }, "Nothing generated yet. Pick a pass and a template, then Generate — or ask the copilot."));
  g.items.forEach((it) => box.append(h("div", { class: "gcard" }, mediaBlock(it),
    h("div", { class: "gb" }, h("div", { class: "gt" }, it.label || it.kind),
      h("div", { class: "gs" }, `${it.sequence_name} · ${fmt(it.t0_s)}–${fmt(it.t1_s)} s · ${new Date((it.created || 0) * 1000).toLocaleTimeString()}`),
      h("div", { class: "gbad" }, h("span", { class: "badge molten" }, it.kind === "sim2real" ? "Transfer" : "Generator"), resultBadges(it))))));
}

/* ===================================================================== boot */
/* ===================================================================== how it works */
const FLOW_STATUS = {
  live: ["ok", "Live"], cached: ["arc", "Cosmos 3 · cached"], rule: ["warn", "Rule-based"], cpu: ["warn", "CPU fallback"],
  offline: ["neutral", "Offline files"], sim: ["molten", "Simulated"], empty: ["neutral", "No data yet"], ok: ["ok", "Done"],
};
const fBadge = (k, txt) => { const [c, l] = FLOW_STATUS[k] || ["neutral", k]; return h("span", { class: `badge ${c}` }, txt || l); };
const plural = (n, w, pl = /(s|sh|ch|x)$/.test(w) ? `${w}es` : `${w}s`) => `${n} ${n === 1 ? w : pl}`;

async function loadFlow() {
  const root = $("#flow");
  let f;
  try { f = await api("/api/flow"); } catch (e) { root.innerHTML = `<div class="placeholder">Could not load: ${esc(e.message)}</div>`; return; }
  const c = f.collect, m = f.mosaico, x = f.export, cu = f.curate, co = f.cosmos;
  const realR = co.assess.real, realG = co.what_if.real, realT = co.sim2real.real;
  const curReal = cu.backend && cu.backend.startsWith("curator");
  const RD = f.data?.real ? f.real : null;
  const srcLine = (sx) => `${sx.short}: ${plural(sx.passes, sx.id === "aimen_saw" ? "bead" : "layer")} on ${sx.coupons.join(", ")}`
    + (sx.inspected ? ` · ${sx.with_indications} with NDT indications (${sx.indications})` : " · not NDT-inspected")
    + (sx.videos ? ` · ${plural(sx.videos, "video")}` : "") + (sx.photos ? ` · ${plural(sx.photos, "photo")}` : "");

  const collectReal = RD && { id: "collect", tab: "copilot", n: 1, title: "Real weld data", tech: "AIMEN SAW + BAM tandem-GMAW · CC BY 4.0",
    does: [...(f.sources || []).map(srcLine),
           "Converted to one pass dir each: arc (lead + trail) @100 Hz, travel, thermal, NDT indications on the same clock",
           RD.simulated ? `${plural(RD.simulated, "simulator pass", "simulator passes")} also loaded — marked SIMULATED` : "No simulator data mixed in"],
    big: RD.passes, unit: "real weld passes",
    foot: `${RD.inspected} NDT-inspected · ${RD.with_indications} with indications (${RD.indications} total) · ${RD.not_inspected} not inspected`,
    status: (RD.fixture || (f.sources || []).some((x) => x.fixture)) ? "sim" : "ok",
    statusText: (RD.fixture || (f.sources || []).some((x) => x.fixture)) ? "Fixture stand-ins" : "Real data", credits: true };

  const stages = [
    collectReal || { id: "collect", tab: "copilot", n: 1, title: "Weld cell sensors", tech: "WeldCell simulator",
      does: ["Robotic pipe-spool & vessel passes (GTAW root, FCAW fill/cap, SAW seam)",
             "Arc current/voltage, wire feed, gas @100 Hz · robot joints @50 Hz · IR interpass @10 Hz · torch camera 24 fps",
             "Defects injected with their physical signature; each pass gets an RT (X-ray) result"],
      big: c.passes, unit: "weld passes", foot: `${c.defect_passes} with defects · ${c.rt_reject} RT rejects · ${c.spools.join(", ")}`,
      status: c.passes ? (c.fixtures ? "sim" : "sim") : "empty",
      statusText: c.fixtures ? "Mock fixtures" : "Simulated data" },
    { id: "mosaico", tab: "copilot", n: 2, title: "Store & query by physics", tech: "Mosaico",
      does: RD ? ["One typed sequence per pass: WeldArc (lead + trail electrode), travel, thermal + camera / photo topics",
                  "NDT indications stored as defect events on the arc clock",
                  "Query on physical values, e.g. current outside the published-settings window"]
        : ["One typed sequence per pass: custom WeldArc / WeldTravel / WeldThermal ontology + JPEG camera topic",
             "Query on physical values, e.g. gas_lpm < 12, current over WPS max",
             "Synchronised 5 s windows exported with sensor sidecars"],
      big: x.clips || m.sequences, unit: x.clips ? "clips exported" : "sequences",
      foot: RD ? `${plural(m.wps_break, "pass")} outside published settings · ${x.defect_clips} defect windows`
        : `${plural(m.wps_break, "pass")} broke the WPS · ${x.defect_clips} defect windows`,
      status: m.status === "ok" ? "live" : "offline", statusText: m.status === "ok" ? "mosaicod live" : "Offline files" },
    { id: "curator", tab: "dataset", n: 3, title: "Curate training data", tech: "NVIDIA Cosmos Curator",
      does: ["Cut → transcode → quality filter (blur, motion, arc glare) → near-duplicate removal",
             "Caption every clip like a welding inspector; embed for search",
             "Write a Cosmos 3 prompt per clip for what-if / sim→real"],
      big: cu.kept, unit: `of ${cu.total} clips kept`,
      foot: (Object.entries(cu.reasons).map(([k, v]) => `${v} ${String(k).replace(/_/g, " ")}`).join(" · ") || "nothing rejected")
        + (cu.total && !curReal ? " · Cosmos Curator failed, same stages run on CPU" : ""),
      status: !cu.total ? "empty" : curReal ? "ok" : "cpu",
      statusText: !cu.total ? null : curReal ? "Cosmos Curator" : "CPU fallback" },
    { id: "cosmos", tab: "generate", n: 4, title: "World models", tech: "NVIDIA Cosmos 3",
      sub: [
        { name: "Reasoner — inspect", what: RD ? "Looks at a process clip or bead photo + reads the signals and NDT → verdict; states a cause only if the data shows one"
            : "Watches a clip + reads the sensor stats → accept / reject / review, defect, root cause, fix",
          real: realR, fb: co.assess.fallback, live: co.reasoner_live },
        { name: "Generator — what-if", what: "Image→video: “what if travel speed drops 20 %?”",
          real: realG, fb: co.what_if.fallback, live: co.generator_live },
        { name: RD ? "Transfer — new variants" : "Transfer — sim → real", what: RD ? "Edge video of the real BAM camera → new photoreal variants, same timing so labels carry over"
            : "Simulator edge/depth video → photoreal footage, labels carried over",
          real: realT, fb: co.sim2real.fallback, live: co.generator_live },
      ] },
    { id: "copilot", tab: "copilot", n: 5, title: "Ask in plain English", tech: "Weld Copilot",
      does: RD ? ["Chat model calls tools: NDT summary, signal check, settings check, bead photo, assess, what-if",
                  "Answers cite pass, time, value and NDT method — and say so when the signals show nothing",
                  "Never invents a defect cause the data doesn't support"]
        : ["Chat model calls tools: query passes, WPS check, find anomalies, assess, what-if, sim→real, dataset stats",
             "Answers cite sequence, time and value vs WPS limit",
             "Replay jumps to the exact moment in the weld"],
      big: null, unit: f.copilot.llm, foot: "chat model", status: "live", statusText: f.copilot.llm },
  ];
  const flowsOn = RD ? ["passes + signals + NDT", "5 s clips + sidecars", "captioned clips + prompts", "verdicts + videos"]
    : ["passes + 4 sensor streams", "5 s clips + sidecars", "captioned clips + prompts", "verdicts + videos"];

  const card = (s) => {
    const el = h("button", { class: `fl-card fl-${s.id}`, onclick: () => showTab(s.tab), title: `Open the ${s.tab} tab` },
      h("div", { class: "fl-top" }, h("span", { class: "fl-n" }, s.n), h("span", { class: "fl-title" }, s.title)),
      h("div", { class: "fl-tech" }, s.tech));
    if (s.sub) {
      s.sub.forEach((x) => {
        const st = x.live ? "live" : x.real ? "cached" : "rule";
        const txt = x.live ? "Live on GPU" : x.real ? `${x.real} real result${x.real === 1 ? "" : "s"} cached` :
          (x.name.startsWith("Reasoner") ? "Rule-based stand-in" : "Placeholder video");
        el.append(h("div", { class: "fl-sub" }, h("div", { class: "fl-sub-h" }, h("b", {}, x.name), fBadge(st, txt)),
          h("div", { class: "fl-sub-w" }, x.what)));
      });
    } else {
      el.append(h("ul", { class: "fl-does" }, s.does.map((d) => h("li", {}, d))),
        h("div", { class: "fl-out" }, s.big != null ? h("span", { class: "fl-big" }, s.big) : null, h("span", { class: "fl-unit" }, s.unit)),
        h("div", { class: "fl-foot" }, s.foot), h("div", { class: "fl-st" }, fBadge(s.status, s.statusText),
          s.credits ? h("span", { class: "fl-cred", onclick: (ev) => { ev.stopPropagation(); openCredits(); } }, "credits →") : null));
    }
    return el;
  };
  const lane = h("div", { class: "fl-lane" });
  stages.forEach((s, i) => {
    if (i) lane.append(h("div", { class: "fl-arrow" }, h("i", { style: { animationDelay: `${i * 0.35}s` } }), h("span", {}, flowsOn[i - 1])));
    lane.append(card(s));
  });

  // follow one weld
  const e = f.example;
  let story = null;
  if (e && e.real) {
    const counts = Object.entries(e.counts || {});
    const ctext = counts.map(([k, v]) => `${prettyDef(k)} ×${v}`).join(", ") || "indications";
    const sc = e.signal_check;
    const steps = [
      ["Real data", `${e.source} ${e.coupon} ${e.pass_label} (${e.process} ${e.pass_type}, ${e.material}): RT/UT found ${ctext}`],
      ["Mosaico", `Indications stored as defect events on the arc clock — first at ${fmt(e.defect_t, 1)} s from arc start`],
      ["Signals", sc ? (sc.signature ? `Inside the ${sc.n_windows} windows the signals do change (${fmt(sc.mean_in, 0)} vs ${fmt(sc.mean_clean, 0)} ${sc.unit}) — a correlation, not a cause`
        : `Inside the ${sc.n_windows} windows: ${fmt(sc.mean_in, 0)} vs ${fmt(sc.mean_clean, 0)} ${sc.unit} in the clean weld — no electrical signature`) : "No signal check available"],
      ["Cosmos 3", e.verdict ? `Reasoner verdict on the ${e.has_photo ? "bead photo" : "clip"}: ${String(e.verdict).toUpperCase()} (${e.verdict_real ? "real Cosmos 3 output" : e.verdict_source})`
        : `No Cosmos 3 verdict cached for this ${e.has_photo ? "bead photo" : "clip"} yet — the copilot shows an evidence-only rule check, labelled as such`],
      ["Copilot", "Answers with the numbers and says plainly that the signals don't explain the defect — no invented cause"],
    ];
    const d0 = counts[0] ? prettyDef(counts[0][0]) : "the defect";
    const ask = `Show ${d0} on ${e.coupon} ${String(e.pass_label || "").toLowerCase()} — did the arc signals change?`;
    story = h("section", { class: "fl-story" },
      h("div", { class: "fl-h" }, h("h3", {}, "Follow one real weld through it"),
        h("span", { class: "sub" }, `${e.sequence} · ${e.source} · ${e.material}`)),
      h("ol", { class: "fl-steps" }, steps.map(([k, t], i) => h("li", { class: `fl-step s${i}` }, h("b", {}, k), h("span", {}, t)))),
      h("button", { class: "btn primary", onclick: () => { showTab("copilot"); sendChat(ask); } }, `Ask the copilot: “${ask}”`));
  } else if (e) {
    const v = e.violation, sig = v && SIG[v.param];
    const lim = v ? (v.limit_lo != null && v.limit_hi != null ? `${v.limit_lo}–${v.limit_hi}` : v.limit_hi != null ? `≤ ${v.limit_hi}` : `≥ ${v.limit_lo}`) : "";
    const steps = [
      ["Sensors", v ? `${sig ? sig.n : v.param} falls to ${fmt(v.value, 1)} ${sig ? sig.u : ""} at ${fmt(v.t_start_s, 1)} s (WPS ${lim})`
                    : `${String(e.defect).replace(/_/g, " ")} starts at ${fmt(e.defect_t, 1)} s`],
      ["Mosaico", v ? `Query ${v.param} ${v.kind === "low" ? "<" : ">"} WPS limit finds this window among ${c.passes} passes` : "Window exported with sensor sidecar"],
      ["Curator", e.clip_id ? `Clip ${e.clip_id} kept and captioned (${e.caption_model}): “${(e.caption || "").slice(0, 120)}…”` : "Clip not in the curated set"],
      ["Cosmos 3", e.verdict ? `Reasoner verdict: ${String(e.verdict).toUpperCase()} (${e.verdict_real ? "real Cosmos 3 output" : e.verdict_source})`
                             : "No Cosmos 3 verdict cached for this clip yet — the copilot shows a rule-based check, labelled as such"],
      ["Radiography", `RT result on the joint: ${String(e.rt_result).toUpperCase()}`],
    ];
    const ask = `Why did ${e.sequence.split("_").slice(0, 2).join(" ")} ${e.pass_type} pass fail RT?`;
    story = h("section", { class: "fl-story" },
      h("div", { class: "fl-h" }, h("h3", {}, "Follow one weld through it"),
        h("span", { class: "sub" }, `${e.sequence} · ${e.process} ${e.pass_type} · ${e.material} · ${e.wps_id}`)),
      h("ol", { class: "fl-steps" }, steps.map(([k, t], i) => h("li", { class: `fl-step s${i}` }, h("b", {}, k), h("span", {}, t)))),
      h("button", { class: "btn primary", onclick: () => { showTab("copilot"); sendChat(ask); } }, `Ask the copilot: “${ask}”`));
  }

  const where = h("section", { class: "fl-where" },
    h("div", { class: "fl-h" }, h("h3", {}, "Where it runs")),
    h("div", { class: "fl-hosts" },
      h("div", { class: "fl-host gpu" }, h("b", {}, "Runpod GPU pod (H100)"),
        h("span", {}, "Postgres + mosaicod + MinIO · Cosmos Curator (Qwen captioner, Cosmos-Embed) · Cosmos 3 Reasoner (vLLM) · Cosmos 3 Generator (vLLM-Omni)"),
        h("small", {}, "started by runpod/launch.py, terminates itself when the run ends")),
      h("div", { class: "fl-arrow v" }, h("i"), h("span", {}, "network volume → download.py")),
      h("div", { class: "fl-host laptop" }, h("b", {}, "This computer"),
        h("span", {}, `Weld Copilot in ${co.mode} mode: Cosmos 3 answers come from the cached results of the GPU run`),
        h("small", {}, "no GPU or internet needed for the cached parts"))),
    h("div", { class: "fl-legend" }, ["live", "cached", "rule", "cpu", "offline", "sim"].map((k) => fBadge(k))));

  // the honest finding, computed from the data on disk
  let finding = null;
  const fd = f.finding;
  if (RD && fd) {
    const ex = fd.example;
    const exTxt = `${ex.sequence}: ${Object.entries(ex.counts).map(([k, v]) => `${prettyDef(k)} ×${v}`).join(", ")} — mean lead current ${fmt(ex.mean_in, 0)} ${ex.unit} inside the windows vs ${fmt(ex.mean_clean, 0)} ${ex.unit} in the clean weld (${ex.outside}/${ex.n_windows} windows outside the normal band).`;
    finding = h("section", { class: `fl-finding ${fd.passes_with_signature ? "mixed" : ""}` },
      h("div", { class: "fl-fi-k" }, "What the real data says"),
      h("div", { class: "fl-fi-big" }, fd.passes_with_signature === 0
        ? `NDT-found defects left no trace in the arc signals — none of the ${plural(fd.passes_checked, "pass", "passes")} with indications (${fd.windows} indication windows) shows a change`
        : `Only ${fd.passes_with_signature} of ${plural(fd.passes_checked, "pass", "passes")} with NDT indications show a signal change inside the windows`),
      h("p", {}, exTxt),
      h("p", { class: "sub" }, "Real defects do not necessarily leave an electrical signature. A current/voltage monitor alone would miss them, so the copilot reports this instead of inventing one — and never names a cause the data doesn't show."));
  }
  const credits = RD ? h("section", { class: "fl-credits" }, h("div", { class: "fl-h" }, h("h3", {}, "Data sources & credits"),
    h("span", { class: "sub" }, "Both datasets are CC BY 4.0 — attribution required, shown wherever their data appears")),
    creditsList(f.sources || [], true)) : null;

  root.innerHTML = "";
  root.append(
    h("header", { class: "fl-head" }, h("h2", {}, "How WeldWorld works"),
      h("p", {}, (RD ? "From real welds to a decision: public AIMEN and BAM welding data, " : "From the weld cell to a decision: ")
        + "what each step does, which component does it, and what it produced for this dataset. Click a step to open its tab.")),
    lane, finding, story, credits, where);
}

/* ===================================================================== data sources & credits */
function creditsList(sources, compact = false) {
  return h("div", { class: `cred-list ${compact ? "compact" : ""}` }, sources.map((x) => h("article", { class: `cred ${x.synthetic ? "sim" : ""}`, id: `cred-${x.id}` },
    h("div", { class: "cred-h" }, h("b", {}, x.title), x.synthetic ? h("span", { class: "badge molten" }, "SIMULATED") : h("span", { class: "badge ok" }, x.license),
      x.fixture ? h("span", { class: "badge warn", title: "Synthetic stand-ins from copilot/fixtures_real.py are loaded, not the converted dataset" }, "fixture stand-ins loaded") : null),
    x.synthetic ? h("p", {}, x.attribution) : [
      h("div", { class: "cred-kv" },
        h("span", { class: "k" }, "Authors"), h("span", {}, x.authors),
        h("span", { class: "k" }, "Publisher"), h("span", {}, [x.org, x.project].filter(Boolean).join(" · ")),
        h("span", { class: "k" }, "Repository"), h("span", {}, "Zenodo · ", h("a", { href: `https://doi.org/${x.doi}`, target: "_blank", rel: "noopener" }, `doi.org/${x.doi}`),
          " · ", h("a", { href: x.url, target: "_blank", rel: "noopener" }, x.url.replace("https://", ""))),
        h("span", { class: "k" }, "Licence"), h("span", {}, h("a", { href: x.license_url, target: "_blank", rel: "noopener" }, `${x.license} — Creative Commons Attribution 4.0 International`)),
        compact ? null : [h("span", { class: "k" }, "Process"), h("span", {}, x.process), h("span", { class: "k" }, "Material"), h("span", {}, x.material),
          h("span", { class: "k" }, "Used here"), h("span", {}, x.streams)],
        h("span", { class: "k" }, "Changes"), h("span", {}, x.modified),
        h("span", { class: "k" }, "Loaded"), h("span", {}, x.loaded ? `${x.passes} passes · ${x.coupons.join(", ")}${x.inspected ? ` · ${x.with_indications}/${x.inspected} inspected passes with indications` : ""}${x.videos ? ` · ${x.videos} videos` : ""}${x.photos ? ` · ${x.photos} photos` : ""}` : "not loaded in this session")),
      compact ? null : h("pre", { class: "cred-cite", title: "Attribution line" }, x.attribution)])));
}
async function openCredits(focus) {
  const m = $("#creditsModal");
  m.classList.remove("hidden");
  const body = $("#creditsBody");
  try {
    const r = await api("/api/sources");
    body.innerHTML = "";
    body.append(h("p", { class: "cred-intro" }, "This demo runs on public research data. Both datasets are published under ",
      h("a", { href: "https://creativecommons.org/licenses/by/4.0/", target: "_blank", rel: "noopener" }, "CC BY 4.0"),
      ", which allows commercial use with attribution and an indication of changes. Thank you to the authors at AIMEN and BAM."),
      creditsList(r.sources),
      h("p", { class: "sub" }, "Generated videos (Cosmos 3 what-if / transfer) are synthetic derivatives and are labelled as such. "
        + "Settings windows are the published nominal settings ±10 % — a demo assumption, not a qualified WPS."));
    if (focus) $(`#cred-${CSS.escape(focus)}`)?.scrollIntoView({ block: "start" });
  } catch (e) { body.innerHTML = `<div class="placeholder">Could not load: ${esc(e.message)}</div>`; }
}
$("#creditsBtn").onclick = () => openCredits();
$("#creditsClose").onclick = () => $("#creditsModal").classList.add("hidden");
$("#creditsModal").addEventListener("click", (e) => { if (e.target.id === "creditsModal") e.currentTarget.classList.add("hidden"); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#creditsModal").classList.add("hidden"); });

(async function boot() {
  pollHealth();
  setInterval(pollHealth, 5000);
  const [sug] = await Promise.all([api("/api/suggestions"), loadPasses()]);
  S.data = sug.data || S.data;
  welcome();
  S.demo = sug.demo;
  renderSuggestions();
  if (isReal()) {
    $("#brandSub").textContent = "Physical AI on real welding data · AIMEN SAW + BAM tandem-GMAW";
    $("#chatSub").textContent = "Mosaico · NDT · Cosmos 3";
    $('#genKind [data-v="sim2real"]').textContent = "Transfer";
  }
  const indic = (p) => p.rt_result === "indications" || p.rt_result === "reject";
  const first = (isReal() && (S.passes.find((p) => p.sequence_name === "PP5_B02_SAW")
      || [...S.passes].filter((p) => !p.synthetic && indic(p)).sort((a, b) => Object.values(b.defect_counts || {}).reduce((x, y) => x + y, 0) - Object.values(a.defect_counts || {}).reduce((x, y) => x + y, 0))[0]))
    || S.passes.find((p) => p.sequence_name === "SP-1043_J07_P3_FCAW")
    || S.passes.find((p) => p.has_defect && p.rt_result === "reject") || S.passes[0];
  if (first) {
    await selectPass(first.sequence_name);
    const w = (S.detail.defect_windows || [])[0];
    seek(w ? Math.max(0, w.t_start - 1.5) : 2);
  }
  const tab = location.hash.replace("#", "");
  if (["dataset", "generate", "flow"].includes(tab)) showTab(tab);
})();

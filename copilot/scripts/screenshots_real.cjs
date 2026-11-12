// Screenshot the Weld Copilot on real data (AIMEN SAW + BAM tandem-GMAW) at 1440x900 -> docs/img/real_*.png
//   uvicorn copilot.app:app --port 7860 &          (data/real)   or   COPILOT_FIXTURES=real uvicorn ... (fixtures)
//   node copilot/scripts/screenshots_real.cjs [http://localhost:7860] [docs/img] [SAW pass] [BAM pass]
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const BASE = process.argv[2] || "http://localhost:7860";
const OUT = process.argv[3] || path.join(__dirname, "..", "..", "docs", "img");
const SAW = process.argv[4] || "PP5_B02_SAW";
const BAM = process.argv[5] || "WID617_L3_TGMAW";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const exe = ["/usr/bin/google-chrome", "/usr/bin/google-chrome-stable"].find((p) => fs.existsSync(p));
  const browser = await chromium.launch({ executablePath: exe, args: ["--autoplay-policy=no-user-gesture-required"] });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  page.on("console", (m) => { if (m.type() === "error") console.log("console:", m.text()); });
  page.on("pageerror", (e) => console.log("pageerror:", e.message));
  const shot = async (name) => { await page.screenshot({ path: path.join(OUT, `real_${name}.png`) }); console.log("saved", name); };
  const waitIdle = async () => { await page.waitForFunction(() => !S.busy, null, { timeout: 120000 }); await sleep(500); };
  const say = async (text) => { await page.evaluate((t) => sendChat(t), text); await sleep(300); await waitIdle(); };
  const chip = async (i) => { await page.click(`#suggest .sugg:nth-child(${i})`); await sleep(300); await waitIdle(); };
  const scrollChat = (frac = 1) => page.evaluate((f) => { const m = document.querySelector("#messages"); m.scrollTop = (m.scrollHeight - m.clientHeight) * f; }, frac);

  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForSelector(".chart-row");
  await sleep(800);

  // 1) SAW pass: NDT strip + bead photo, after "did the arc signals change?"
  const saw = await page.evaluate((s) => S.passes.find((p) => p.sequence_name === s)?.sequence_name
    || S.passes.filter((p) => !p.synthetic && p.rt_result === "indications" && p.has_photo)[0]?.sequence_name, SAW);
  const sawRef = saw.replace(/^(PP\d+)_B0?(\d+)_.*/, "$1 bead $2");
  const d0 = await page.evaluate((s) => Object.keys(S.passes.find((p) => p.sequence_name === s).defect_counts || {})[0] || "porosity", saw);
  await say(`Show ${d0.replace(/_/g, " ")} on ${sawRef} — did the arc signals change?`);
  await page.evaluate(() => setView("photo"));
  await sleep(1200);
  // hover the first indication on the NDT strip for the tooltip
  const box = await page.evaluate(() => {
    const st = S.charts.find((c) => c.strip); const w = S.detail.defect_windows[0];
    if (!st || !w) return null;
    const r = st.u.over.getBoundingClientRect();
    return { x: r.left + st.u.valToPos((w.t_start + w.t_end) / 2, "x"), y: r.top + r.height / 2 };
  });
  if (box) { await page.mouse.move(box.x, box.y); await sleep(400); }
  await scrollChat(1);
  await shot("saw_ndt_photo");

  // 2) BAM pass with the real process video, after "compare lead vs trail"
  await page.mouse.move(10, 10);
  const bam = await page.evaluate((s) => S.passes.find((p) => p.sequence_name === s)?.sequence_name
    || S.passes.find((p) => p.has_video && !p.synthetic && p.electrodes > 1)?.sequence_name, BAM);
  const bamRef = bam.replace(/^(WID\d+)_L(\d+)_.*/, "$1 layer $2");
  await say(`Compare lead vs trail current on ${bamRef}`);
  await page.evaluate((s) => selectPass(s, 4.2, "torch"), bam);
  await sleep(1800);
  await page.evaluate(() => { document.querySelector("#vMain").pause(); });
  await scrollChat(1);
  await shot("bam_video");

  // 3) credits modal
  await page.click("#creditsBtn");
  await page.waitForSelector(".cred");
  await sleep(600);
  await shot("credits");
  await page.keyboard.press("Escape");

  // 4) How it works
  await page.click('.tab-btn[data-tab="flow"]');
  await page.waitForSelector(".fl-card");
  await sleep(1200);
  await shot("how_it_works");

  // 5) photo-based assessment + NDT list (extra, for the README)
  await page.click('.tab-btn[data-tab="copilot"]');
  await page.click("#chatReset"); await sleep(300);
  await chip(1);
  await say(`Inspect the bead photo of ${sawRef}`);
  await sleep(500);
  await scrollChat(1);
  await shot("photo_assess");
  await browser.close();
})();
